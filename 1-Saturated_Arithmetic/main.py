import random
import numpy as np
import torch
from skimage import data
from skimage.transform import resize

im = data.coffee()
im = resize(im, (im.shape[0] // 8, im.shape[1] // 8), mode='reflect', preserve_range=True, anti_aliasing=True).astype(
    np.uint8)
im = np.swapaxes(np.swapaxes(im, 0, 2), 1, 2)
im = torch.from_numpy(im)

a = random.uniform(0, 2)
b = random.uniform(-50, 50)

# First exercise

out = im.type(torch.float32)
out = a * out + b
# print(out)
out = torch.round(out)
# print(out)
out = torch.clip(out, 0, 255)
# print(out)
out = out.type(torch.uint8)
print(out)

# Second exercise

im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True).astype(
    np.uint8)
im = torch.from_numpy(im)
val = random.randint(0, 255)

out = torch.clone(im)
out = torch.where(out <= val, 0, 255)
print(out)

# Third exercise

im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True).astype(
    np.uint8)
im = torch.from_numpy(im)

p = torch.zeros(256)
for t in range(256):
    p[t] = torch.sum(im == t)
    # p[t] = torch.sum(im[im == t])
print(p)
var_max = -999
th_best = 0

for t in range(256):
    w_1 = torch.sum(p[:t + 1])
    w_2 = torch.sum(p[t + 1:])

    mu_1 = torch.sum(p[:t + 1] * torch.arange(t + 1)) / w_1
    mu_2 = torch.sum(p[t + 1:] * torch.arange(t + 1, 256)) / w_2

    var = w_1 * w_2 * ((mu_1 - mu_2) * (mu_1 - mu_2))
    print(t, var)
    if var > var_max:
        var_max = var
        th_best = t

out = th_best
print(out, var_max)
