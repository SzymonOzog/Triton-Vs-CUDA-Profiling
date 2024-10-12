import torch
import triton
import triton.language as tl
import time

from setuptools import setup
from torch.utils.cpp_extension import load
from torch.profiler import profile, ProfilerActivity, record_function

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


BENCH_STEPS=400
size = 2**20

x = torch.rand(size, device='cuda', dtype=torch.float32)
y = torch.rand(size, device='cuda', dtype=torch.float32)

torch_ms = triton.testing.do_bench(lambda: x + y)
triton_ms = triton.testing.do_bench(lambda: add_triton(x,y))
cuda_ms = triton.testing.do_bench(lambda: add_cuda.add_cuda(x, y, size))

print(f"triton reported times, torch = {torch_ms:.4f}, triton = {triton_ms:.4f}, cuda = {cuda_ms:.4f}") 

start = time.perf_counter()
for i in range(BENCH_STEPS):
  out = x+y
time_torch = (time.perf_counter() - start)/BENCH_STEPS

start = time.perf_counter()
for i in range(BENCH_STEPS):
  out = add_triton(x, y) 
time_triton = (time.perf_counter() - start)/BENCH_STEPS


start = time.perf_counter()
for i in range(BENCH_STEPS):
  out = add_cuda.add_cuda(x, y, size) 
time_cuda = (time.perf_counter() - start)/BENCH_STEPS

print(f"self evaluation results torch = {time_torch*1e3:.4f}, triton = {time_triton*1e3:.4f}, cuda = {time_cuda*1e3:.4f}")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
  with record_function("torch"):
      out = x+y
  with record_function("triton"):
      out = add_triton(x, y) 
  with record_function("cuda"):
      out = add_cuda.add_cuda(x, y, size) 
prof.export_chrome_trace("trace.json")
