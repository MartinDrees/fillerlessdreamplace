/**
 * @file   multigrid.cpp
 * @brief  CPU implementation of the multigrid projected-Jacobi
 *         electric potential solver.
 *
 * Exposes one entry point to Python via pybind11:
 *   multigrid_potential(d, bin_size_x, bin_size_y, K, min_grid) -> phi
 *
 * Jacobi sweeps are parallelised over rows with OpenMP.
 */

#include <torch/extension.h>
#include <algorithm>
#include <vector>

// ---------------------------------------------------------------------------
// K projected Jacobi sweeps — dtype dispatch without AT_DISPATCH (LTO-safe)
// ---------------------------------------------------------------------------
static at::Tensor jacobi_sweeps(
    at::Tensor phi,
    at::Tensor d,
    double alpha_x,
    double alpha_y,
    int K,
    int num_threads = 0)
{
  TORCH_CHECK(phi.dim() == 2, "phi must be 2D");
  TORCH_CHECK(d.sizes() == phi.sizes(), "d must match phi shape");
  phi = phi.contiguous();
  d   = d.contiguous();

  if (num_threads <= 0) num_threads = at::get_num_threads();

  const int NX = phi.size(0);
  const int NY = phi.size(1);

  // Ping-pong buffers: buf starts with the initial phi, out is scratch.
  // Tracing the ping-pong with initial state p_cur=buf, p_nxt=out:
  //   sweep 1: writes buf→out, swap → p_cur=out, p_nxt=buf
  //   sweep 2: writes out→buf, swap → p_cur=buf, p_nxt=out
  //   K odd  → last write into out  → result in out
  //   K even → last write into buf  → result in buf
  at::Tensor buf = phi.clone();
  at::Tensor out = at::empty_like(phi);

  auto run = [&](auto* p_buf, auto* p_out, const auto* p_d) {
    using T = std::remove_pointer_t<decltype(p_buf)>;
    T ax = static_cast<T>(alpha_x);
    T ay = static_cast<T>(alpha_y);
    auto* p_cur = p_buf;
    auto* p_nxt = p_out;
    for (int k = 0; k < K; ++k) {
#pragma omp parallel for num_threads(num_threads) schedule(static)
      for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
          // S = weighted neighbour sum, D = sum of weights (degree)
          T S = 0, D = 0;
          if (x > 0)    { S += ax * p_cur[(x-1)*NY + y]; D += ax; }
          if (x < NX-1) { S += ax * p_cur[(x+1)*NY + y]; D += ax; }
          if (y > 0)    { S += ay * p_cur[x*NY + (y-1)]; D += ay; }
          if (y < NY-1) { S += ay * p_cur[x*NY + (y+1)]; D += ay; }
          // Projected Jacobi update: phi = max(0, (S + d) / D)
          p_nxt[x*NY + y] = std::max(T(0), (S + p_d[x*NY + y]) / D);
        }
      }
      std::swap(p_cur, p_nxt);
    }
  };

  if (phi.scalar_type() == at::kFloat)
    run(buf.data_ptr<float>(),  out.data_ptr<float>(),  d.data_ptr<float>());
  else
    run(buf.data_ptr<double>(), out.data_ptr<double>(), d.data_ptr<double>());

  // K odd  → result is in out;  K even → result is in buf
  return (K % 2 != 0) ? out : buf;
}

// ---------------------------------------------------------------------------
// Full multigrid nested-iteration solve
// ---------------------------------------------------------------------------

// Restrict a 2D tensor to the next coarser level by summing each 2x2 block.
static at::Tensor coarsen(const at::Tensor& fine)
{
  using S = at::indexing::Slice;
  const auto even = S(0, at::indexing::None, 2);
  const auto odd  = S(1, at::indexing::None, 2);
  return (fine.index({even, even})
        + fine.index({even, odd })
        + fine.index({odd,  even})
        + fine.index({odd,  odd })).contiguous();
}

at::Tensor multigrid_potential(
    at::Tensor d,
    double bin_size_x,
    double bin_size_y,
    int K,
    int min_grid = 8,
    int num_threads = 0)
{
  TORCH_CHECK(d.dim() == 2, "d must be 2D");
  d = d.contiguous();

  if (num_threads <= 0) num_threads = at::get_num_threads();

  const double ax = bin_size_y / bin_size_x;
  const double ay = bin_size_x / bin_size_y;

  // ---- Step 1: build coarsened delta grids (finest -> coarsest) ----
  std::vector<at::Tensor> levels_d;
  levels_d.push_back(d);
  while (levels_d.back().size(0) / 2 >= min_grid &&
         levels_d.back().size(1) / 2 >= min_grid)
    levels_d.push_back(coarsen(levels_d.back()));

  const int L = static_cast<int>(levels_d.size());

  // ---- Step 2: solve coarsest level from phi = 0 ----
  at::Tensor phi = jacobi_sweeps(
      at::zeros_like(levels_d.back()), levels_d.back(),
      ax, ay, K, num_threads);

  // ---- Step 3: prolongate and smooth up to finest ----
  for (int lev = L - 2; lev >= 0; --lev) {
    phi = phi.repeat_interleave(2, /*dim=*/0)
             .repeat_interleave(2, /*dim=*/1)
             .contiguous();
    phi = jacobi_sweeps(phi, levels_d[lev], ax, ay, K, num_threads);
  }

  return phi;
}

// ---------------------------------------------------------------------------
// pybind11 bindings
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multigrid_potential",
        &multigrid_potential,
        "Multigrid projected-Jacobi electric potential (CPU)",
        py::arg("d"), py::arg("bin_size_x"), py::arg("bin_size_y"),
        py::arg("K"), py::arg("min_grid") = 8,
        py::arg("num_threads") = 0);
}
