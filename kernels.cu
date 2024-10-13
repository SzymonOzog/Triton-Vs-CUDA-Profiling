#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_DIM_Y
#define BLOCK_DIM_Y 32 
#endif

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


template <typename scalar_t>
__global__ void softmax_kernel2(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int ty = threadIdx.y;
  int stride_b = ceil(w/blockDim.y);
  __shared__ scalar_t reduction[BLOCK_DIM_Y]; 
  if (row < h)
  {
    scalar_t maxval = 0;
    for (int i = ty; i<w; i+=stride_b)
    {
      maxval = max(maxval, a[row*w + i]);
    }

    reduction[ty] = maxval;
    for(int stride = BLOCK_DIM_Y/2; stride>=1; stride/=2)
    {
      __syncthreads();
      if (ty < stride)
      {
        reduction[ty] = max(reduction[ty], reduction[ty+stride]);
      }
    }

    __syncthreads();
    maxval = reduction[0];
    scalar_t divisor = 0.f;
    for (int i = 0; i<w; i++)
    {
      divisor += __expf(a[row*w + i] - maxval);
    }

    for (int i = ty; i<w; i+=stride_b)
    {
      b[row*w + i] = __expf(a[row*w + i]-maxval)/(divisor);
    }

  }
}

template <typename scalar_t>
__global__ void softmax_kernel3(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int ty = threadIdx.y;
  int stride_b = ceil(w/blockDim.y);
  __shared__ scalar_t reduction[BLOCK_DIM_Y]; 
  if (row < h)
  {
    scalar_t maxval = 0;
    for (int i = ty; i<w; i+=stride_b)
    {
      maxval = max(maxval, a[row*w + i]);
    }

    reduction[ty] = maxval;
    for(int stride = BLOCK_DIM_Y/2; stride>=1; stride/=2)
    {
      __syncthreads();
      if (ty < stride)
      {
        reduction[ty] = max(reduction[ty], reduction[ty+stride]);
      }
    }

    __syncthreads();
    maxval = reduction[0];

    scalar_t divisor = 0.f;
    for (int i = ty; i<w; i+=stride_b)
    {
      divisor += __expf(a[row*w + i] - maxval);
    }
    reduction[ty] = divisor;
    for(int stride = BLOCK_DIM_Y/2; stride>=1; stride/=2)
    {
      __syncthreads();
      if (ty < stride)
      {
        reduction[ty] = reduction[ty] + reduction[ty+stride];
      }
    }
    __syncthreads();
    divisor = reduction[0];

    for (int i = ty; i<w; i+=stride_b)
    {
      b[row*w + i] = __expf(a[row*w + i]-maxval)/divisor;
    }
  }
}

template <typename scalar_t>
__global__ void softmax_kernel4(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int ty = threadIdx.y;
  int stride_b = ceil(w/blockDim.y);
  __shared__ scalar_t reduction[BLOCK_DIM_Y/2]; 
  if (row < h)
  {
    scalar_t maxval = 0;
    for (int i = ty; i<w; i+=stride_b)
    {
      maxval = max(maxval, a[row*w + i]);
    }

    if (ty >= BLOCK_DIM_Y/2)
    {
      reduction[ty - BLOCK_DIM_Y/2] = maxval;
    }
    for(int stride = BLOCK_DIM_Y/2; stride>=1; stride/=2)
    {
      __syncthreads();
      if (ty < stride)
      {
        maxval = max(maxval, reduction[ty]);
        if (ty >= stride/2)
        {
          reduction[ty - stride/2] = maxval;
        }
      }
    }

    __syncthreads();
    maxval = reduction[0];

    scalar_t divisor = 0.f;
    for (int i = ty; i<w; i+=stride_b)
    {
      divisor += __expf(a[row*w + i] - maxval);
    }

    if (ty >= BLOCK_DIM_Y/2)
    {
      reduction[ty - BLOCK_DIM_Y/2] = divisor;
    }

    for(int stride = BLOCK_DIM_Y/2; stride>=1; stride/=2)
    {
      __syncthreads();
      if (ty < stride)
      {
        divisor = divisor + reduction[ty];
        if (ty >= stride/2)
        {
          reduction[ty - stride/2] = divisor;
        }
      }
    }
    __syncthreads();
    divisor = reduction[0];

    for (int i = ty; i<w; i+=stride_b)
    {
      b[row*w + i] = __expf(a[row*w + i]-maxval)/divisor;
    }
  }
}

torch::Tensor softmax_cu(torch::Tensor x)
{
  auto out = torch::zeros_like(x);
  int rows = x.size(0);
  int cols = x.size(1);

  const dim3 block_size = dim3(1, BLOCK_DIM_Y, 1);
  const dim3 grid_size = dim3(cols/block_size.x, 1, 1);

  AT_DISPATCH_FLOATING_TYPES(x.type(), "softmax_cuda", ([&] {
        softmax_kernel3<scalar_t><<<grid_size, block_size>>>
          (x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), rows, cols);
        }));
  return out;
}
