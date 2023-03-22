import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 6)
kW = random.randint(2, 6)

input = torch.rand(n, iC, H, W, dtype=torch.float32)
kernel = torch.rand(oC, iC, kH, kW, dtype=torch.float32)

#print(input, kernel)

# Exercise 2D convolution
# less for loops better result (2 preferred result)

oH = H - (kH - 1)
oW = W - (kW - 1)
out = torch.zeros((n, oC, oH, oW), dtype=torch.float32)
#print(out)

# Best Solution

# 2 for loops,kernel sliding over the input tensor
for i in range(oH):
    for j in range(oW):
        # shape (n, iC, H, W)
        # add an axis in position 1
        # cut the region over which I want to put my kernel, square neighbourhood on which convolution is performed
        inp = input.unsqueeze(1)[:, :, :, i:i+kH, j:j+kW]

        # shape (n, 1, iC, kH, kW)
        # add an axis in position 0
        ker = kernel.unsqueeze(0)

        # shape (1, oC, iC, kH, kW)
        # now they inp and ker can be broadcasted, 1 and oC compatible, n and 1 compatible
        # convolution
        # sum over the last three axis
        out[:, :, i, j] = (inp * ker).sum((-1, -2, -3))

print(out.shape)

# 4 for solution

'''
n_cur = -1
k_cur = -1

for image in input:
    n_cur += 1
    for ker in kernel:
        k_cur += 1
        for i in range(oH):
            for j in range(oW):
                wind = image[:, i:i + kH, j:j + kW]
                sq_wind = torch.flatten(wind)
                sq_ker = torch.flatten(ker)
                out[n_cur, k_cur, i, j] = torch.dot(sq_wind, sq_ker)
    k_cur = -1
'''

# 2 for solution


'''

for i in range(oH):
    for j in range(oW):
        wind = input[:, :, i:i + kH, j:j + kW]
        sq_wind = torch.reshape(wind, [n, 1, -1])
        sq_ker = torch.reshape(kernel, [1, oC, -1])
        out[:, :, i, j] = torch.sum(torch.mul(sq_ker, sq_wind), dim=-1)
print(out)


'''


