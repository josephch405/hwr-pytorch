import torch
import torch.nn.functional as F


class GreyNet(torch.nn.Module):
    def __init__(self):
        super(GreyNet, self).__init__()
        self.c1 = torch.nn.Conv2d(1, 16, 7, padding=3)
        self.c2a = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.c2b = torch.nn.Conv2d(32, 32, 3, padding=1)

        self.c3a = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.c3b = torch.nn.Conv2d(64, 64, 3, padding=1)

        self.c4a = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.c4b = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.c4c = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.c4d = torch.nn.Conv2d(128, 64, 3, padding=1)

        self.fc = torch.nn.Linear(32 * 32 * 64, 32 * 32 * 5)

    def forward(self, x):
        # [ 512, 512, 1 ]
        y = F.max_pool2d(F.relu(self.c1(x)), 2)
        # [ 256, 256, 16 ]
        y = F.relu(self.c2a(y))
        y = F.relu(self.c2b(y))
        y = F.max_pool2d(y, 2)
        # [ 128, 128, 32 ]
        y = F.relu(self.c3a(y))
        y = F.relu(self.c3b(y))
        y = F.max_pool2d(y, 2)
        # [ 64, 64, 64 ]
        y = F.relu(self.c4a(y))
        y = F.relu(self.c4b(y))
        y = F.relu(self.c4c(y))
        y = F.relu(self.c4d(y))
        y = F.max_pool2d(y, 2)
        # [ 32, 32, 64 ]
        y = y.view(y.shape[0], -1)
        y = self.fc(y)
        # [ 32 * 32 * 5 ]
        # TODO: make parameterized
        y = y.view(y.shape[0], 5, 32, 32)
        return y
