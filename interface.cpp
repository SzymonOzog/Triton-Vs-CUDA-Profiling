#include <torch/extension.h>

torch::Tensor add_cuda(torch::Tensor x, torch::Tensor y, int size)
{
  return add(x, y, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("add_cuda", &add_cuda, "Add (CUDA)");
}
