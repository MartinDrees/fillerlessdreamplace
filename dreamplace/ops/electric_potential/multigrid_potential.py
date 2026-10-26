##
# @file   multigrid_potential.py
# @brief  Electrostatic potential via nested multigrid projected-Jacobi
#         iterations, as described in:
#           "Simple Electrostatic Global Placement without Filler Cells"
#
# The algorithm replaces the FFT-based Poisson solve of ePlace/DREAMPlace.
# It operates directly on the movable-cell density map (no filler cells needed).
# A projected Jacobi iteration:
#   phi[x,y] = max(0, (d[x,y] + sum_of_neighbours) / num_neighbours)
# is applied in a nested (cascadic) fashion from the coarsest grid to the
# finest, propagating the warm-start potential up via nearest-neighbour
# prolongation after each level.
#
# The electric field (= negative gradient of potential) is computed via
# central finite differences with one-sided differences at the boundary.
##

import time
import torch
from torch.autograd import Function
import logging

from dreamplace.ops.electric_potential.electric_overflow import (
    ElectricDensityMapFunction,
    ElectricOverflow,
)

import dreamplace.ops.electric_potential.electric_potential_cpp as electric_potential_cpp
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.electric_potential.electric_potential_cuda as electric_potential_cuda

# Load the C++ multigrid extension (required — no Python fallback).
import dreamplace.ops.electric_potential.multigrid_cpp as _multigrid_cpp

# Load the CUDA multigrid extension (optional — only available when built with CUDA).
try:
    import dreamplace.ops.electric_potential.multigrid_cuda as _multigrid_cuda
    _CUDA_AVAILABLE = True
except ImportError:
    _multigrid_cuda = None
    _CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core numerics — dispatches to CPU or CUDA backend
# ---------------------------------------------------------------------------

def _multigrid_potential(d_fine, bin_size_x, bin_size_y, K, min_grid=8, num_threads=0):
    """
    Compute electric potential via nested projected-Jacobi multigrid.
    Dispatches to the CUDA extension when d_fine is on GPU, otherwise to the
    CPU extension.  num_threads is only used by the CPU backend.
    """
    if d_fine.is_cuda:
        if not _CUDA_AVAILABLE:
            raise RuntimeError(
                "multigrid_potential: input is on CUDA but the multigrid_cuda "
                "extension was not built.  Rebuild DREAMPlace with CUDA enabled.")
        return _multigrid_cuda.multigrid_potential(
            d_fine.contiguous(), bin_size_x, bin_size_y, K, min_grid)
    else:
        return _multigrid_cpp.multigrid_potential(
            d_fine.contiguous(), bin_size_x, bin_size_y, K, min_grid, num_threads)


def _electric_field_from_potential(phi, bin_size_x, bin_size_y):
    """
    Compute electric field E = -grad(phi) via central finite differences.
    Boundary bins use one-sided differences (forward/backward).

    Returns (field_x, field_y), each [num_bins_x, num_bins_y].
    field_y is scaled by bin_size_y/bin_size_x to account for non-square bins,
    consistent with the aspect-ratio weighting used in the Jacobi solver.
    """
    phi_xp = torch.cat([phi[1:, :], phi[-1:, :]], dim=0)
    phi_xm = torch.cat([phi[:1, :], phi[:-1, :]], dim=0)
    field_x = -0.5 * (phi_xp - phi_xm)

    phi_yp = torch.cat([phi[:, 1:], phi[:, -1:]], dim=1)
    phi_ym = torch.cat([phi[:, :1], phi[:, :-1]], dim=1)
    field_y = -0.5 * (phi_yp - phi_ym) * (bin_size_y / bin_size_x)

    return field_x, field_y


# ---------------------------------------------------------------------------
# PyTorch autograd Function
# ---------------------------------------------------------------------------

class MultigridPotentialFunction(Function):
    """
    Forward: compute density map -> bin deltas -> multigrid potential -> field.
    Backward: interpolate field map gradients onto cell positions via the
              electric_force kernel from electric_potential_cpp/cuda.
    """

    @staticmethod
    def forward(
        ctx,
        pos,
        node_size_x_clamped,
        node_size_y_clamped,
        offset_x,
        offset_y,
        ratio,
        bin_center_x,
        bin_center_y,
        initial_density_map,
        target_density,
        xl,
        yl,
        xh,
        yh,
        bin_size_x,
        bin_size_y,
        num_movable_nodes,
        num_filler_nodes,
        padding,
        padding_mask,
        num_bins_x,
        num_bins_y,
        num_movable_impacted_bins_x,
        num_movable_impacted_bins_y,
        num_filler_impacted_bins_x,
        num_filler_impacted_bins_y,
        deterministic_flag,
        sorted_node_map,
        K,
        num_threads
    ):
        tt = time.time()

        # 1. Build density map (movable + filler cells)
        density_map = ElectricDensityMapFunction.forward(
            pos, node_size_x_clamped, node_size_y_clamped,
            offset_x, offset_y, ratio,
            bin_center_x, bin_center_y, initial_density_map,
            target_density, xl, yl, xh, yh, bin_size_x, bin_size_y,
            num_movable_nodes, num_filler_nodes,
            padding, padding_mask,
            num_bins_x, num_bins_y,
            num_movable_impacted_bins_x, num_movable_impacted_bins_y,
            num_filler_impacted_bins_x, num_filler_impacted_bins_y,
            deterministic_flag, sorted_node_map)

        # 2. Convert to bin-density-excess d_i = density/target - 1
        bin_area = bin_size_x * bin_size_y
        d = density_map / (target_density * bin_area) - 1.0

        # 3. Multigrid projected-Jacobi solve
        phi = _multigrid_potential(d, bin_size_x, bin_size_y, K,
                                   num_threads=num_threads)

        # 4. Electric field via finite differences
        field_x, field_y = _electric_field_from_potential(phi, bin_size_x,
                                                          bin_size_y)

        ctx.field_map_x = field_x
        ctx.field_map_y = field_y
        ctx.node_size_x_clamped = node_size_x_clamped
        ctx.node_size_y_clamped = node_size_y_clamped
        ctx.offset_x = offset_x
        ctx.offset_y = offset_y
        ctx.ratio = ratio
        ctx.bin_center_x = bin_center_x
        ctx.bin_center_y = bin_center_y
        ctx.xl = xl
        ctx.yl = yl
        ctx.xh = xh
        ctx.yh = yh
        ctx.bin_size_x = bin_size_x
        ctx.bin_size_y = bin_size_y
        ctx.num_bins_x = num_bins_x
        ctx.num_bins_y = num_bins_y
        ctx.num_movable_impacted_bins_x = num_movable_impacted_bins_x
        ctx.num_movable_impacted_bins_y = num_movable_impacted_bins_y
        ctx.num_filler_impacted_bins_x = num_filler_impacted_bins_x
        ctx.num_filler_impacted_bins_y = num_filler_impacted_bins_y
        ctx.num_movable_nodes = num_movable_nodes
        ctx.num_filler_nodes = num_filler_nodes
        ctx.deterministic_flag = deterministic_flag
        ctx.pos = pos
        ctx.sorted_node_map = sorted_node_map

        logger.debug("multigrid density forward %.3f ms" %
                     ((time.time() - tt) * 1000))

        # Return dummy zero energy; gradient is carried entirely by field maps.
        return torch.zeros(1, dtype=pos.dtype, device=pos.device)

    @staticmethod
    def backward(ctx, grad_pos):
        tt = time.time()
        if grad_pos.is_cuda:
            output = -electric_potential_cuda.electric_force(
                grad_pos, ctx.num_bins_x, ctx.num_bins_y,
                ctx.num_movable_impacted_bins_x, ctx.num_movable_impacted_bins_y,
                ctx.num_filler_impacted_bins_x, ctx.num_filler_impacted_bins_y,
                ctx.field_map_x.view([-1]), ctx.field_map_y.view([-1]),
                ctx.pos, ctx.node_size_x_clamped, ctx.node_size_y_clamped,
                ctx.offset_x, ctx.offset_y, ctx.ratio,
                ctx.bin_center_x, ctx.bin_center_y,
                ctx.xl, ctx.yl, ctx.xh, ctx.yh,
                ctx.bin_size_x, ctx.bin_size_y,
                ctx.num_movable_nodes, ctx.num_filler_nodes,
                ctx.deterministic_flag, ctx.sorted_node_map)
        else:
            output = -electric_potential_cpp.electric_force(
                grad_pos, ctx.num_bins_x, ctx.num_bins_y,
                ctx.num_movable_impacted_bins_x, ctx.num_movable_impacted_bins_y,
                ctx.num_filler_impacted_bins_x, ctx.num_filler_impacted_bins_y,
                ctx.field_map_x.view([-1]), ctx.field_map_y.view([-1]),
                ctx.pos, ctx.node_size_x_clamped, ctx.node_size_y_clamped,
                ctx.offset_x, ctx.offset_y, ctx.ratio,
                ctx.bin_center_x, ctx.bin_center_y,
                ctx.xl, ctx.yl, ctx.xh, ctx.yh,
                ctx.bin_size_x, ctx.bin_size_y,
                ctx.num_movable_nodes, ctx.num_filler_nodes)

        logger.debug("multigrid density backward %.3f ms" %
                     ((time.time() - tt) * 1000))

        # One gradient per forward argument; all non-pos inputs get None.
        return output, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None


# ---------------------------------------------------------------------------
# nn.Module wrapper (drop-in replacement for ElectricPotential)
# ---------------------------------------------------------------------------

class MultigridPotential(ElectricOverflow):
    """
    Electrostatic density penalty via multigrid projected-Jacobi.

    Inherits density-map and overflow utilities from ElectricOverflow.
    Filler cells are included in the density map when num_filler_nodes > 0,
    allowing use with or without fillers (set enable_fillers=0 to skip them).
    """

    def __init__(self, node_size_x, node_size_y, bin_center_x, bin_center_y,
                 target_density, xl, yl, xh, yh, bin_size_x, bin_size_y,
                 num_movable_nodes, num_terminals, num_filler_nodes, padding,
                 deterministic_flag, sorted_node_map, movable_macro_mask=None,
                 multigrid_iters_per_level=25, num_threads=0):
        super().__init__(
            node_size_x=node_size_x,
            node_size_y=node_size_y,
            bin_center_x=bin_center_x,
            bin_center_y=bin_center_y,
            target_density=target_density,
            xl=xl, yl=yl, xh=xh, yh=yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            num_movable_nodes=num_movable_nodes,
            num_terminals=num_terminals,
            num_filler_nodes=num_filler_nodes,
            padding=padding,
            deterministic_flag=deterministic_flag,
            sorted_node_map=sorted_node_map,
            movable_macro_mask=movable_macro_mask)
        self.K = multigrid_iters_per_level
        self.num_threads = num_threads

    def forward(self, pos, mode="density"):
        assert mode in {"density", "overflow"}
        if self.initial_density_map is None:
            self.compute_initial_density_map(pos)
            logger.info(
                "fixed density map: average %g, max %g, bin area %g" %
                (self.initial_density_map.mean(),
                 self.initial_density_map.max(),
                 self.bin_size_x * self.bin_size_y))

        if mode == "density":
            return MultigridPotentialFunction.apply(
                pos,
                self.node_size_x_clamped, self.node_size_y_clamped,
                self.offset_x, self.offset_y, self.ratio,
                self.bin_center_x, self.bin_center_y,
                self.initial_density_map, self.target_density,
                self.xl, self.yl, self.xh, self.yh,
                self.bin_size_x, self.bin_size_y,
                self.num_movable_nodes, self.num_filler_nodes,
                self.padding, self.padding_mask,
                self.num_bins_x, self.num_bins_y,
                self.num_movable_impacted_bins_x,
                self.num_movable_impacted_bins_y,
                self.num_filler_impacted_bins_x,
                self.num_filler_impacted_bins_y,
                self.deterministic_flag, self.sorted_node_map,
                self.K, self.num_threads)
        else:  # overflow mode — same as parent
            density_map = ElectricDensityMapFunction.forward(
                pos, self.node_size_x_clamped, self.node_size_y_clamped,
                self.offset_x, self.offset_y, self.ratio,
                self.bin_center_x, self.bin_center_y,
                self.initial_density_map, self.target_density,
                self.xl, self.yl, self.xh, self.yh, self.bin_size_x,
                self.bin_size_y, self.num_movable_nodes, self.num_filler_nodes,
                self.padding, self.padding_mask,
                self.num_bins_x, self.num_bins_y,
                self.num_movable_impacted_bins_x,
                self.num_movable_impacted_bins_y,
                self.num_filler_impacted_bins_x,
                self.num_filler_impacted_bins_y,
                self.deterministic_flag, self.sorted_node_map)

            bin_area = self.bin_size_x * self.bin_size_y
            density_cost = (density_map -
                            self.target_density * bin_area).clamp_(min=0.0).sum()
            return density_cost, density_map.max() / bin_area
