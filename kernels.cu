#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void add_kernel(scalar_t* __restrict__ x, scalar_t* __restrict__ y, scalar_t* __restrict__ out,  int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    out[idx] = x[idx] + y[idx];
  }
}


torch::Tensor add(torch::Tensor x, torch::Tensor y, int size)
{
  auto out = torch::zeros_like(x);

  const int block_size = 1024;
  const int grid_size = std::ceil(size/block_size);

  AT_DISPATCH_FLOATING_TYPES(x.type(), "add_cuda", ([&] {
        add_kernel<scalar_t><<<block_size, grid_size>>>
          (x.data<scalar_t>(), y.data<scalar_t>(), out.data<scalar_t>(), size);
        }));

}
