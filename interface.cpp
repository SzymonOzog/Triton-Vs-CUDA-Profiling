#include <torch/extension.h>

torch::Tensor add(torch::Tensor x, torch::Tensor y, int size);
torch::Tensor softmax_cu(torch::Tensor x);

torch::Tensor add_cuda(torch::Tensor x, torch::Tensor y, int size)
{
  return add(x, y, size);
}

torch::Tensor softmax_cuda(torch::Tensor x)
{
  return softmax_cu(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("add_cuda", &add_cuda, "Add (CUDA)");
  m.def("softmax_cuda", &softmax_cuda, "Softmax (CUDA)");
}
