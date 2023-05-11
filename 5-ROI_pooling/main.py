import random
import torch
from math import floor, ceil

n = random.randint(1, 3)
C = random.randint(10, 20)
H = random.randint(5, 10)
W = random.randint(5, 10)
oH = random.randint(2, 4)
oW = random.randint(2, 4)
L = random.randint(2, 6)
input = torch.rand(n, C, H, W)
boxes = [torch.zeros(L, 4) for _ in range(n)]
for i in range(n):
    boxes[i][:, 0] = torch.rand(L) * (H-oH)       # y
    boxes[i][:, 1] = torch.rand(L) * (W-oW)       # x
    boxes[i][:, 2] = oH + torch.rand(L) * (H-oH)  # w
    boxes[i][:, 3] = oW + torch.rand(L) * (W-oW)  # h

    boxes[i][:, 2:] += boxes[i][:, :2]
    boxes[i][:, 2] = torch.clamp(boxes[i][:, 2], max=H-1)
    boxes[i][:, 3] = torch.clamp(boxes[i][:, 3], max=W-1)
output_size = (oH, oW)

output = torch.zeros((n, L, C, output_size[0], output_size[1]), dtype=torch.float32)

for i in range(n):
    img = input[i, :, :, :]
    #print(boxes[i])
    proposals = boxes[i]
    rounded = torch.round(proposals[:])
    print(rounded)

    for j in range(L):
        for h in range(oH):
            for w in range(oW):
                y1 = floor(rounded[j, 0] + h*((rounded[j, 2] - rounded[j, 0] + 1) / oH))
                y2 = ceil(rounded[j, 0] + (h+1)*((rounded[j, 2] - rounded[j, 0] + 1) / oH))
                x1 = floor(rounded[j, 1] + w * ((rounded[j, 3] - rounded[j, 1] + 1) / oW))
                x2 = ceil(rounded[j, 1] + (w+1) * ((rounded[j, 3] - rounded[j, 1] + 1) / oW))
                #print(y1, y2, x1, x2)

                output[i, j, :, h, w] = torch.amax(input[i, :, y1:y2, x1:x2], dim=(1, 2))
                #print(output[i, j, :, h, w])

