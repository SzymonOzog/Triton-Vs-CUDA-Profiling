import torch
import triton
import triton.language as tl

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

@triton.testing.perf_report(
    triton.testing.Benchmark(
      x_names=["size"], 
      x_vals=[2**i for i in range(12, 28, 1)], 
      x_log=True,
      line_arg='provider',
      line_vals=['triton', 'torch'],
      line_names=['Triton', 'Torch'],
      styles=[('blue', '-'), ('green', '-')],
      ylabel='time[ms]',
      plot_name='vector addition timings',
      args={}))

def benchmark(size, provider):
  x = torch.rand(size, device='cuda', dtype=torch.float32)
  y = torch.rand(size, device='cuda', dtype=torch.float32)
  if provider == 'torch':
    ms = triton.testing.do_bench(lambda: x + y)
  if provider == 'triton':
    ms = triton.testing.do_bench(lambda: add_triton(x,y))
  return ms

benchmark.run(print_data=True, show_plots=True)
