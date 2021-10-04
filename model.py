import torch
import torch.nn as nn


class net1(nn.Module):
    def __init__(self):
        super(net1, self).__init__()
        self.conv0 = nn.Conv2d(1, 16, (1, 1), (1, 1), (0, 0))
        self.conv1 = nn.Conv2d(1, 16, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(1, 16, (5, 5), (1, 1), (2, 2))

        self.conv3 = nn.Conv2d(16, 16, (7, 7), (1, 1), (3, 3))
        self.conv4 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0))

        self.conv6 = nn.Conv2d(256, 64, (5, 5), (1, 1), (2, 2))
        self.conv7 = nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2))
        self.conv8 = nn.Conv2d(64, 1, (1, 1), (1, 1), (0, 0))

        self.norm = nn.BatchNorm2d(256)
        self.nom = nn.BatchNorm2d(64)

    def forward(self, x, y):
        temx0 = self.conv0(x)
        temy0 = self.conv0(y)
        temx1 = self.conv1(x)
        temy1 = self.conv1(y)
        temx2 = self.conv2(x) * (1 + temx1 + temx0)
        temy2 = self.conv2(y) * (1 + temy1 + temy0)

        temx3 = self.conv3(temx0)
        temy3 = self.conv3(temy0)
        temx4 = self.conv4(torch.cat((temx2, temx3), 1))
        temy4 = self.conv4(torch.cat((temy2, temy3), 1))
        temx5 = self.conv5(torch.cat((temx2, temx3, temx4), 1))
        temy5 = self.conv5(torch.cat((temy2, temy3, temy4), 1))

        tem = torch.cat((temx2, temx3, temx4, temx5, temy2, temy3, temy4, temy5), 1)
        tem = self.norm(tem)
        res1 = self.conv6(tem) * (1 + temx5 + temy5)
        res1 = self.nom(res1)
        res2 = self.conv7(res1) * (1 + res1 + temx5 + temy5)
        res2 = self.nom(res2)
        res3 = self.conv8(res2)
        return res3
