import torch
import triton
import triton.language as tl
import time

from setuptools import setup
from torch.utils.cpp_extension import load
from torch.profiler import profile, ProfilerActivity, record_function
from benchmark import benchmark

add_cuda = load(name='add_cuda', sources=["interface.cpp", "kernels.cu"])

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  tl.store(output_ptr + offsets, output, mask=mask)

def add_triton(x: torch.Tensor, y: torch.Tensor):
  output = torch.empty_like(x)
  n_elements = output.nelement()
  grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
  add_kernel[grid](x, y, output, n_elements, 1024)

size = 2**20
x = torch.rand(size, device='cuda', dtype=torch.float32)
y = torch.rand(size, device='cuda', dtype=torch.float32)

benchmark(lambda: x+y, lambda: add_triton(x, y), lambda: add_cuda.add_cuda(x, y, size))

