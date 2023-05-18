import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 6)
kW = random.randint(2, 6)
s = random.randint(2, 6)

input = torch.rand(n, iC, H, W)
kernel = torch.rand(iC, oC, kH, kW)

oH = (H - 1) * s + (kH - 1) + 1
oW = (W - 1) * s + (kW - 1) + 1
#oH = (H - 1) * s + 1
#oW = (W - 1) * s + 1


out = torch.zeros((n, oC, oH, oW), dtype=torch.float32)

print(oH, oW)

for i in range(H):
    for j in range(W):
        wind = out[:, :, i*s:i*s + kH, j*s:j*s + kW]
        print(kernel.unsqueeze(0).shape, input[:, :, i, j].unsqueeze(2).unsqueeze(3).unsqueeze(3).shape)

        prod = kernel * input[:, :, i, j].unsqueeze(2).unsqueeze(3).unsqueeze(3)
        print(wind.unsqueeze(0).shape, prod.shape)
        prod = prod.sum(dim=1)  # summation operation is performed along the second dimension (oC)
        wind = wind.unsqueeze(0) + prod
        #print(prod, wind)
        out[:, :, i*s:i*s + kH, j*s:j*s + kW] = wind

print(out)
