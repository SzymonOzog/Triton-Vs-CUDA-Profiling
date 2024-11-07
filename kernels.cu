#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_DIM_Y
#define BLOCK_DIM_Y 1024
#endif

#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 8
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
  __shared__ scalar_t reduction[BLOCK_DIM_Y]; 
  if (row < h)
  {
    scalar_t maxval = 0;
    for (int i = ty; i<w; i+=BLOCK_DIM_Y)
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

    for (int i = ty; i<w; i+=BLOCK_DIM_Y)
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
  __shared__ scalar_t reduction[BLOCK_DIM_Y]; 
  if (row < h)
  {
    scalar_t maxval = 0;
    for (int i = ty; i<w; i+=BLOCK_DIM_Y)
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
    for (int i = ty; i<w; i+=BLOCK_DIM_Y)
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

    for (int i = ty; i<w; i+=BLOCK_DIM_Y)
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
  __shared__ scalar_t reduction[BLOCK_DIM_Y/2]; 
  if (row < h)
  {
    scalar_t maxval = 0;
    for (int i = ty; i<w; i+=BLOCK_DIM_Y)
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
    for (int i = ty; i<w; i+=BLOCK_DIM_Y)
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

    for (int i = ty; i<w; i+=BLOCK_DIM_Y)
    {
      b[row*w + i] = __expf(a[row*w + i]-maxval)/divisor;
    }
  }
}

template <typename scalar_t>
__global__ void softmax_kernel5(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int ty = threadIdx.y;
  __shared__ scalar_t reduction[BLOCK_DIM_Y/2]; 
  if (row < h)
  {
    scalar_t maxval = 0;
    for (int i = ty; i<w/4; i+=BLOCK_DIM_Y)
    {
      float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
      maxval = max(maxval, val.x);
      maxval = max(maxval, val.y);
      maxval = max(maxval, val.z);
      maxval = max(maxval, val.w);
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
    for (int i = ty; i<w/4; i+=BLOCK_DIM_Y)
    {
      float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
      divisor += __expf(val.x - maxval);
      divisor += __expf(val.y - maxval);
      divisor += __expf(val.z - maxval);
      divisor += __expf(val.w - maxval);
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

    for (int i = ty; i<w; i+=BLOCK_DIM_Y)
    {
      b[row*w + i] = __expf(a[row*w + i]-maxval)/divisor;
    }
  }
}

template <typename scalar_t>
__global__ void softmax_kernel6(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int ty = threadIdx.y;
  int warp_id = ty/32;
  __shared__ scalar_t reduction[BLOCK_DIM_Y/32]; 
  if (row < h)
  {
    scalar_t maxval = 0;
    for (int i = ty; i<w/4; i+=BLOCK_DIM_Y)
    {
      float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
      maxval = max(maxval, val.x);
      maxval = max(maxval, val.y);
      maxval = max(maxval, val.z);
      maxval = max(maxval, val.w);
    }
    for (int mask = 16; mask>0; mask/=2)
    {
      maxval = max(maxval, __shfl_xor_sync(0xffffffff, maxval, mask, 32));
    }

    if (ty%32 == 0)
    {
      reduction[warp_id] = maxval;
    }
    __syncthreads();
    if (warp_id == 0)
    {
        maxval = ty < BLOCK_DIM_Y/32 ? reduction[ty] : 0;
        for (int mask = 16; mask>0; mask/=2)
        {
          maxval = max(maxval, __shfl_xor_sync(0xffffffff, maxval, mask, 32));
        }
    }
    if (ty == 0)
    {
        reduction[0] = maxval;
    }
    __syncthreads();
    maxval = reduction[0];
    scalar_t divisor = 0.f;
    for (int i = ty; i<w/4; i+=BLOCK_DIM_Y)
    {
      float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
      divisor += __expf(val.x - maxval);
      divisor += __expf(val.y - maxval);
      divisor += __expf(val.z - maxval);
      divisor += __expf(val.w - maxval);
    }
    for (int mask = 16; mask>0; mask/=2)
    {
      divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
    }

    if (ty%32 == 0)
    {
      reduction[warp_id] = divisor;
    }

    __syncthreads();
    if (warp_id == 0)
    {
        divisor = ty < BLOCK_DIM_Y/32 ? reduction[ty] : 0;
        for (int mask = 16; mask>0; mask/=2)
        {
          divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
        }
    }
    if (ty == 0)
    {
        reduction[0] = divisor;
    }

    __syncthreads();
    divisor = reduction[0];

    for (int i = ty; i<w; i+=BLOCK_DIM_Y)
    {
      b[row*w + i] = __expf(a[row*w + i]-maxval)/divisor;
    }
  }
}

template <typename scalar_t>
__global__ void softmax_kernel7(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int ty = threadIdx.y;
  int warp_id = ty/32;
  __shared__ scalar_t reduction[BLOCK_DIM_Y/32]; 
  if (row < h)
  {
    scalar_t maxval = 0;
    for (int i = ty; i<w/4; i+=BLOCK_DIM_Y)
    {
      float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
      maxval = fmaxf(maxval, val.x);
      maxval = fmaxf(maxval, val.y);
      maxval = fmaxf(maxval, val.z);
      maxval = fmaxf(maxval, val.w);
    }
    for (int mask = 16; mask>0; mask/=2)
    {
      maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, mask, 32));
    }

    if (ty%32 == 0)
    {
      reduction[warp_id] = maxval;
    }
    __syncthreads();
    if (warp_id == 0)
    {
        maxval = ty < BLOCK_DIM_Y/32 ? reduction[ty] : 0;
        for (int mask = 16; mask>0; mask/=2)
        {
          maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, mask, 32));
        }
    }
    if (ty == 0)
    {
        reduction[0] = maxval;
    }
    __syncthreads();
    maxval = reduction[0];
    scalar_t divisor = 0.f;
    for (int i = ty; i<w/4; i+=BLOCK_DIM_Y)
    {
      float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
      divisor += __expf(val.x - maxval);
      divisor += __expf(val.y - maxval);
      divisor += __expf(val.z - maxval);
      divisor += __expf(val.w - maxval);
    }
    for (int mask = 16; mask>0; mask/=2)
    {
      divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
    }

    if (ty%32 == 0)
    {
      reduction[warp_id] = divisor;
    }

    __syncthreads();
    if (warp_id == 0)
    {
        divisor = ty < BLOCK_DIM_Y/32 ? reduction[ty] : 0;
        for (int mask = 16; mask>0; mask/=2)
        {
          divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
        }
    }
    if (ty == 0)
    {
        reduction[0] = divisor;
    }

    __syncthreads();
    divisor = reduction[0];

    for (int i = ty; i<w/4; i+=BLOCK_DIM_Y)
    {
        float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
        val.x = __expf(val.x-maxval)/divisor;
        val.y = __expf(val.y-maxval)/divisor;
        val.z = __expf(val.z-maxval)/divisor;
        val.w = __expf(val.w-maxval)/divisor;
        reinterpret_cast<float4*>(&b[row*w + i*4])[0] = val;
    }
  }
}

    template <typename scalar_t>
__global__ void softmax_kernel8(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  int row = blockIdx.x;
  int ty = threadIdx.y;
  int warp_id = ty/32;
  __shared__ scalar_t reduction[BLOCK_DIM_Y/32]; 
  if (row < h)
  {
    scalar_t maxval = 0;
    for (int i = ty; i<w/4; i+=BLOCK_DIM_Y*UNROLL_FACTOR)
    {
      #pragma unroll
      for (int u = 0; u<UNROLL_FACTOR; u++)
      {
        float4 val = reinterpret_cast<float4*>(&a[row*w + u*BLOCK_DIM_Y*4 + i*4])[0];
        maxval = fmaxf(maxval, val.x);
        maxval = fmaxf(maxval, val.y);
        maxval = fmaxf(maxval, val.z);
        maxval = fmaxf(maxval, val.w);
      }
    }
    maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 16, 32));
    maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 8, 32));
    maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 4, 32));
    maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 2, 32));
    maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 1, 32));

    if (ty%32 == 0)
    {
      reduction[warp_id] = maxval;
    }
    __syncthreads();
    if (warp_id == 0)
    {
        maxval = ty < BLOCK_DIM_Y/32 ? reduction[ty] : 0;
        maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 16, 32));
        maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 8, 32));
        maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 4, 32));
        maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 2, 32));
        maxval = fmaxf(maxval, __shfl_down_sync(0xffffffff, maxval, 1, 32));
    }
    if (ty == 0)
    {
        reduction[0] = maxval;
    }
    __syncthreads();
    maxval = reduction[0];
    scalar_t divisor = 0.f;
    for (int i = ty; i<w/4; i+=BLOCK_DIM_Y*UNROLL_FACTOR)
    {
      #pragma unroll
      for (int u = 0; u<UNROLL_FACTOR; u++)
      {
        float4 val = reinterpret_cast<float4*>(&a[row*w + u*BLOCK_DIM_Y*4 + i*4])[0];
        divisor += __expf(val.x - maxval);
        divisor += __expf(val.y - maxval);
        divisor += __expf(val.z - maxval);
        divisor += __expf(val.w - maxval);
      }
    }

    divisor += __shfl_down_sync(0xffffffff, divisor, 16, 32);
    divisor += __shfl_down_sync(0xffffffff, divisor, 8, 32);
    divisor += __shfl_down_sync(0xffffffff, divisor, 4, 32);
    divisor += __shfl_down_sync(0xffffffff, divisor, 2, 32);
    divisor += __shfl_down_sync(0xffffffff, divisor, 1, 32);

    if (ty%32 == 0)
    {
      reduction[warp_id] = divisor;
    }

    __syncthreads();
    if (warp_id == 0)
    {
        divisor = ty < BLOCK_DIM_Y/32 ? reduction[ty] : 0;
        divisor += __shfl_down_sync(0xffffffff, divisor, 16, 32);
        divisor += __shfl_down_sync(0xffffffff, divisor, 8, 32);
        divisor += __shfl_down_sync(0xffffffff, divisor, 4, 32);
        divisor += __shfl_down_sync(0xffffffff, divisor, 2, 32);
        divisor += __shfl_down_sync(0xffffffff, divisor, 1, 32);
    }
    if (ty == 0)
    {
        reduction[0] = divisor;
    }

    __syncthreads();
    divisor = reduction[0];

    for (int i = ty; i<w/4; i+=BLOCK_DIM_Y*UNROLL_FACTOR)
    {
      #pragma unroll
      for (int u = 0; u<UNROLL_FACTOR; u++)
      {
        float4 val = reinterpret_cast<float4*>(&a[row*w + u*BLOCK_DIM_Y*4 + i*4])[0];
        val.x = __expf(val.x-maxval)/divisor;
        val.y = __expf(val.y-maxval)/divisor;
        val.z = __expf(val.z-maxval)/divisor;
        val.w = __expf(val.w-maxval)/divisor;
        reinterpret_cast<float4*>(&b[row*w + u*BLOCK_DIM_Y*4 + i*4])[0] = val;
      }
    }
  }
}

torch::Tensor softmax_cu(torch::Tensor x)
{
  auto out = torch::empty_like(x);
  int h = x.size(0);
  int w = x.size(1);

  const dim3 block_size = dim3(1, BLOCK_DIM_Y, 1);
  const dim3 grid_size = dim3(h, 1, 1);

  AT_DISPATCH_FLOATING_TYPES(x.type(), "softmax_cuda", ([&] {
        softmax_kernel8<scalar_t><<<grid_size, block_size>>>
          (x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), w, h);
        }));
  return out;
}
