import torch
import triton
import triton.language as tl
from triton.runtime import driver
from torch.utils.cpp_extension import load
from benchmark import benchmark
torch.set_default_device('cuda')


x = torch.rand(128, 2**16, device='cuda')

best = None
best_time = float("inf")
results = {}
for dim_y in [2**x for x in range(7, 11)]:
    for unroll in [2, 4, 8, 16, 32]:
      if dim_y >= 512 and unroll == 16: continue
      if dim_y >= 1024 and unroll == 32: continue
      
      cuda = load(name='softmax_cuda', sources=["interface.cpp", "kernels.cu"], verbose=False,
                  extra_cuda_cflags=[f"-DBLOCK_DIM_Y={dim_y}", f"-UNROLL_FACTOR={unroll}" ])
      print(f"running for dim y = {dim_y}, unroll = {unroll}")
      cuda.softmax_cuda(x)
      # time=0
      _, _, time = benchmark(lambda: torch.softmax(x, dim=-1), lambda: None, lambda: cuda.softmax_cuda(x), bench_steps=50)
      results[f"y{dim_y}_unroll{unroll}"] = time
for x in sorted(results.items(), key=lambda x: x[1]):
  print(x)

