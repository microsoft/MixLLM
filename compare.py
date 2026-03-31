import torch

# 加载两个 tensor
# tensor1 = torch.load("/tmp/output064-prefill-82.pth")
# tensor2 = torch.load("/workspace/vllm-benchmark/tensor/output054-prefill-82.pth")
tensor1 = torch.load("/tmp/output064-decode-97.pth")
tensor2 = torch.load("/workspace/vllm-benchmark/tensor/output054-decode-97.pth")
print("tensor1:", tensor1.shape)
print("tensor2:", tensor2.shape)
# 比较两个 tensor 是否相同
for i in range(tensor1.shape[0]):
    print("tensor1:", tensor1[i].shape)
    print("tensor2:", tensor2[i].shape)
    are_equal = torch.allclose(tensor1[i], tensor2[i], atol=1e-6)
    print("两个 tensor 第", i, "个元素是否相同:", are_equal)


print("两个 tensor 是否相同:", are_equal)
