/**
 * @file   multigrid_cuda.cpp
 * @brief  CUDA multigrid projected-Jacobi electric potential solver.
 *         Tensor management, dispatch, and pybind11 bindings.
 *
 * Exposes one function to Python:
 *   multigrid_potential(d, bin_size_x, bin_size_y, K, min_grid) -> phi
 *
 * d must be a CUDA tensor of shape [num_bins_x, num_bins_y].
 * The full nested-iteration cycle runs entirely on the GPU.
 */

#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

// Launchers declared in multigrid_cuda_kernel.cu
template <typename T>
int multigridJacobiCudaLauncher(
    T* buf_ptr, T* out_ptr, const T* d_ptr,
    int NX, int NY, T ax, T ay, int K);

template <typename T>
int multigridCoarsenCudaLauncher(
    const T* fine_ptr, T* coarse_ptr,
    int NX, int NY, int CNX, int CNY);

template <typename T>
int multigridProlongCudaLauncher(
    const T* coarse_ptr, T* fine_ptr,
    int NX, int NY, int CNY);

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
at::Tensor multigrid_potential(
    at::Tensor d,
    double bin_size_x, double bin_size_y,
    int K, int min_grid)
{
    TORCH_CHECK(d.is_cuda(),       "multigrid_potential: d must be a CUDA tensor");
    TORCH_CHECK(d.dim() == 2,      "multigrid_potential: d must be 2D");
    CHECK_CONTIGUOUS(d);

    // Build coarsened level pyramid
    std::vector<at::Tensor> levels_d;
    levels_d.push_back(d);
    while (levels_d.back().size(0) / 2 >= min_grid &&
           levels_d.back().size(1) / 2 >= min_grid) {
        const at::Tensor& prev = levels_d.back();
        int NX = prev.size(0), NY = prev.size(1);
        int CNX = NX / 2,      CNY = NY / 2;
        at::Tensor coarse = at::empty({CNX, CNY}, prev.options());

        DREAMPLACE_DISPATCH_FLOATING_TYPES(prev, "multigridCoarsenCudaLauncher", [&] {
            multigridCoarsenCudaLauncher<scalar_t>(
                DREAMPLACE_TENSOR_DATA_PTR(prev, scalar_t),
                DREAMPLACE_TENSOR_DATA_PTR(coarse, scalar_t),
                NX, NY, CNX, CNY);
        });

        levels_d.push_back(coarse);
    }

    const int L = static_cast<int>(levels_d.size());

    // Solve coarsest level from phi = 0
    at::Tensor phi;
    {
        const at::Tensor& dc = levels_d.back();
        int NX = dc.size(0), NY = dc.size(1);
        at::Tensor phi_buf = at::zeros_like(dc);
        at::Tensor phi_out = at::empty_like(dc);

        DREAMPLACE_DISPATCH_FLOATING_TYPES(dc, "multigridJacobiCudaLauncher", [&] {
            scalar_t ax = static_cast<scalar_t>(bin_size_y / bin_size_x);
            scalar_t ay = static_cast<scalar_t>(bin_size_x / bin_size_y);
            multigridJacobiCudaLauncher<scalar_t>(
                DREAMPLACE_TENSOR_DATA_PTR(phi_buf, scalar_t),
                DREAMPLACE_TENSOR_DATA_PTR(phi_out, scalar_t),
                DREAMPLACE_TENSOR_DATA_PTR(dc, scalar_t),
                NX, NY, ax, ay, K);
        });

        phi = phi_buf;
    }

    // Prolongate and smooth from coarsest to finest
    for (int lev = L - 2; lev >= 0; --lev) {
        const at::Tensor& d_cur = levels_d[lev];
        int NX  = d_cur.size(0), NY  = d_cur.size(1);
        int CNY = phi.size(1);

        // Prolongate coarse phi to fine grid
        at::Tensor phi_fine = at::empty({NX, NY}, d_cur.options());
        DREAMPLACE_DISPATCH_FLOATING_TYPES(phi, "multigridProlongCudaLauncher", [&] {
            multigridProlongCudaLauncher<scalar_t>(
                DREAMPLACE_TENSOR_DATA_PTR(phi, scalar_t),
                DREAMPLACE_TENSOR_DATA_PTR(phi_fine, scalar_t),
                NX, NY, CNY);
        });

        // Smooth the prolonged phi
        at::Tensor phi_out = at::empty_like(phi_fine);
        DREAMPLACE_DISPATCH_FLOATING_TYPES(d_cur, "multigridJacobiCudaLauncher", [&] {
            scalar_t ax = static_cast<scalar_t>(bin_size_y / bin_size_x);
            scalar_t ay = static_cast<scalar_t>(bin_size_x / bin_size_y);
            multigridJacobiCudaLauncher<scalar_t>(
                DREAMPLACE_TENSOR_DATA_PTR(phi_fine, scalar_t),
                DREAMPLACE_TENSOR_DATA_PTR(phi_out, scalar_t),
                DREAMPLACE_TENSOR_DATA_PTR(d_cur, scalar_t),
                NX, NY, ax, ay, K);
        });

        phi = phi_fine;
    }

    return phi;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multigrid_potential",
          &DREAMPLACE_NAMESPACE::multigrid_potential,
          "Multigrid projected-Jacobi electric potential (CUDA)",
          py::arg("d"), py::arg("bin_size_x"), py::arg("bin_size_y"),
          py::arg("K"), py::arg("min_grid") = 8);
}
