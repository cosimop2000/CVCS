import torch
import torch.nn as nn
import torch.nn.functional as F


# residual block implementation, two functions F and G(in the shortcut)
# ReLu after convolutions
# batch normalization after each convolution operation
# conv2D, batchNorm2D


class Model(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride

        self.model = nn.Sequential(
            nn.Conv2d(inplanes, planes, (3, 3), self.stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),

            nn.Conv2d(planes, planes, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(planes)  # ,
            # nn.ReLU()
        )

        if self.inplanes != self.planes or self.stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(inplanes, planes, (1, 1), stride=self.stride, padding=0, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.residual = torch.nn.Identity()

    def forward(self, x):
        F_x = self.model(x)
        x = self.residual(x)
        return F.relu(F_x + x)


inplanes = 64
planes = 128
stride = 2

# N, inplanes, H, W
x = torch.zeros((20, 64, 10, 10))

net = Model(inplanes, planes, stride)
out = net(x)
print(out.shape)
