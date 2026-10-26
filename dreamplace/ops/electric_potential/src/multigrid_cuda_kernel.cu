/**
 * @file   multigrid_cuda_kernel.cu
 * @brief  GPU kernels and typed launchers for the multigrid projected-Jacobi
 *         electric potential solver.
 *
 * Three __global__ kernels (raw pointer interfaces):
 *   jacobi_sweep_kernel   — one projected-Jacobi sweep over a 2D grid
 *   coarsen_kernel        — 2×2 block sum (restriction)
 *   prolong_kernel        — nearest-neighbour prolongation
 *
 * Three launcher functions (called from multigrid_cuda.cpp):
 *   multigridCoarsenCudaLauncher<T>
 *   multigridProlongCudaLauncher<T>
 *   multigridJacobiCudaLauncher<T>
 *
 * Tensor management and dispatch live in multigrid_cuda.cpp.
 */

#include <cuda_runtime.h>
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

// ---------------------------------------------------------------------------
// Jacobi sweep: one projected-Jacobi iteration over a 2D grid.
// One thread per (x, y) bin.  Ping-pong between cur and nxt buffers.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void jacobi_sweep_kernel(
    const T* __restrict__ cur,
    T*       __restrict__ nxt,
    const T* __restrict__ d,
    int NX, int NY,
    T ax, T ay)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    T S = T(0), D = T(0);
    if (x > 0)    { S += ax * cur[(x-1)*NY + y]; D += ax; }
    if (x < NX-1) { S += ax * cur[(x+1)*NY + y]; D += ax; }
    if (y > 0)    { S += ay * cur[x*NY + (y-1)]; D += ay; }
    if (y < NY-1) { S += ay * cur[x*NY + (y+1)]; D += ay; }

    // Projected Jacobi update: phi = max(0, (S + d) / D)
    nxt[x*NY + y] = max(T(0), (S + d[x*NY + y]) / D);
}

// ---------------------------------------------------------------------------
// Coarsen: sum each 2×2 block of the fine grid into one coarse cell.
// fine dims: NX × NY;  coarse dims: CNX × CNY  (CNX=NX/2, CNY=NY/2)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void coarsen_kernel(
    const T* __restrict__ fine,
    T*       __restrict__ coarse,
    int NX, int NY,
    int CNX, int CNY)
{
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;
    if (cx >= CNX || cy >= CNY) return;

    int fx = cx * 2, fy = cy * 2;
    coarse[cx * CNY + cy] =
        fine[ fx      * NY + fy    ] +
        fine[ fx      * NY + fy + 1] +
        fine[(fx + 1) * NY + fy    ] +
        fine[(fx + 1) * NY + fy + 1];
}

// ---------------------------------------------------------------------------
// Prolongate: piecewise-constant prolongation (nearest-neighbour).
// Each fine cell (fx, fy) takes the value of its coarse parent (fx/2, fy/2).
// ---------------------------------------------------------------------------
template <typename T>
__global__ void prolong_kernel(
    const T* __restrict__ coarse,
    T*       __restrict__ fine,
    int NX, int NY,
    int CNY)
{
    int fx = blockIdx.x * blockDim.x + threadIdx.x;
    int fy = blockIdx.y * blockDim.y + threadIdx.y;
    if (fx >= NX || fy >= NY) return;

    fine[fx * NY + fy] = coarse[(fx / 2) * CNY + (fy / 2)];
}

// ---------------------------------------------------------------------------
// Typed launcher: K Jacobi sweeps on a single level (ping-pong).
//
// Tracing the ping-pong with initial state cur=buf, nxt=out:
//   sweep 1: writes buf→out, swap → cur=out, nxt=buf
//   sweep 2: writes out→buf, swap → cur=buf, nxt=out
//   ...
//   K odd  → last write went into out  (cur=out after last swap)
//   K even → last write went into buf  (cur=buf after last swap)
//
// The caller always reads from buf_ptr, so for K odd we copy out→buf.
// We use cudaMemcpyAsync to stay on the same stream as the kernels.
// ---------------------------------------------------------------------------
template <typename T>
int multigridJacobiCudaLauncher(
    T* buf_ptr, T* out_ptr, const T* d_ptr,
    int NX, int NY,
    T ax, T ay,
    int K)
{
    dim3 block(16, 16);
    dim3 grid((NX + 15) / 16, (NY + 15) / 16);

    T* cur = buf_ptr;
    T* nxt = out_ptr;
    for (int k = 0; k < K; ++k) {
        jacobi_sweep_kernel<T><<<grid, block>>>(cur, nxt, d_ptr, NX, NY, ax, ay);
        T* tmp = cur; cur = nxt; nxt = tmp;
    }
    // K odd  → result landed in out_ptr; copy back to buf_ptr
    // K even → result already in buf_ptr; nothing to do
    if (K % 2 != 0) {
        cudaMemcpyAsync(buf_ptr, out_ptr, sizeof(T) * NX * NY,
                        cudaMemcpyDeviceToDevice);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Typed launcher: coarsen fine → coarse
// ---------------------------------------------------------------------------
template <typename T>
int multigridCoarsenCudaLauncher(
    const T* fine_ptr, T* coarse_ptr,
    int NX, int NY, int CNX, int CNY)
{
    dim3 block(16, 16);
    dim3 grid((CNX + 15) / 16, (CNY + 15) / 16);
    coarsen_kernel<T><<<grid, block>>>(fine_ptr, coarse_ptr, NX, NY, CNX, CNY);
    return 0;
}

// ---------------------------------------------------------------------------
// Typed launcher: prolongate coarse → fine
// ---------------------------------------------------------------------------
template <typename T>
int multigridProlongCudaLauncher(
    const T* coarse_ptr, T* fine_ptr,
    int NX, int NY, int CNY)
{
    dim3 block(16, 16);
    dim3 grid((NX + 15) / 16, (NY + 15) / 16);
    prolong_kernel<T><<<grid, block>>>(coarse_ptr, fine_ptr, NX, NY, CNY);
    return 0;
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
#define REGISTER_LAUNCHER(T)                                                \
  template int multigridJacobiCudaLauncher<T>(                             \
      T*, T*, const T*, int, int, T, T, int);                              \
  template int multigridCoarsenCudaLauncher<T>(                            \
      const T*, T*, int, int, int, int);                                   \
  template int multigridProlongCudaLauncher<T>(                            \
      const T*, T*, int, int, int);

REGISTER_LAUNCHER(float);
REGISTER_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
