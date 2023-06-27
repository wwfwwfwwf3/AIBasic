import torch
b = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
idx = torch.tensor([[0, 2], [1, 2]])
result = b[idx]
print(result)