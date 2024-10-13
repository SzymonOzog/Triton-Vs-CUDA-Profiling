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
        add_kernel<scalar_t><<<grid_size, block_size>>>
          (x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), size);
        }));
  return out;
}

template <typename scalar_t>
__global__ void softmax_kernel(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < h && col < w)
  {
    scalar_t maxval = a[row*w];
    for (int i = 1; i<w; i++)
    {
      maxval = max(maxval, a[row*w + i]);
    }
    scalar_t divisor = 0.f;
    for (int i = 0; i<w; i++)
    {
      divisor += __expf(a[row*w + i] - maxval);
    }
    b[row*w + col] = __expf(a[row*w + col]-maxval)/(divisor);
  }
}

torch::Tensor softmax_cu(torch::Tensor x)
{
  auto out = torch::zeros_like(x);
  int rows = x.size(0);
  int cols = x.size(1);

  const dim3 block_size = dim3(32, 32, 1);
  const dim3 grid_size = dim3(std::ceil(rows/block_size.x), std::ceil(cols/block_size.y), 1);

  AT_DISPATCH_FLOATING_TYPES(x.type(), "softmax_cuda", ([&] {
        softmax_kernel<scalar_t><<<grid_size, block_size>>>
          (x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), rows, cols);
        }));
  return out;
}
