import triton
import torch
import time
from torch.profiler import profile, ProfilerActivity, record_function

def benchmark(fn_torch, fn_triton, fn_cuda, bench_steps=400):
  torch_ms = triton.testing.do_bench(fn_torch)
  triton_ms = triton.testing.do_bench(fn_triton)
  cuda_ms = triton.testing.do_bench(fn_cuda)

  print(f"triton reported times, torch = {torch_ms:.4f}, triton = {triton_ms:.4f}, cuda = {cuda_ms:.4f}") 

  start = time.perf_counter()
  for i in range(bench_steps):
    out = fn_torch() 
  torch.cuda.synchronize()
  time_torch = (time.perf_counter() - start)/bench_steps

  start = time.perf_counter()
  for i in range(bench_steps):
    out = fn_triton() 
  torch.cuda.synchronize()
  time_triton = (time.perf_counter() - start)/bench_steps


  start = time.perf_counter()
  for i in range(bench_steps):
    out = fn_cuda()
  torch.cuda.synchronize()
  time_cuda = (time.perf_counter() - start)/bench_steps

  print(f"self evaluation results torch = {time_torch*1e3:.4f}, triton = {time_triton*1e3:.4f}, cuda = {time_cuda*1e3:.4f}")

  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("torch"):
        out = fn_torch() 
        torch.cuda.synchronize()
    with record_function("triton"):
        out =  fn_triton()
        torch.cuda.synchronize()
    with record_function("cuda"):
        out = fn_cuda() 
        torch.cuda.synchronize()
  prof.export_chrome_trace("trace.json")
