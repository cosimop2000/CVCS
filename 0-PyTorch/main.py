import torch
import cv2

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
#print(X)

a = torch.zeros((5,))
print(a.shape)
a = torch.randn((5,))
print(a)
b = torch.randn((5, 5))
print(a+b)
