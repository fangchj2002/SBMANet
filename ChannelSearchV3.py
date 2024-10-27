import torch
from torch import nn

class AMCS(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Sequential(nn.Conv3d(128, 2, 3, stride=1, padding=1),
                                   nn.BatchNorm3d(2),nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv3d(16, 2, 3, stride=1, padding=1),
                                   nn.BatchNorm3d(2),nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv3d(16, 16, 1, stride=1, padding="valid"),
                                   nn.BatchNorm3d(16),nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv3d(16, 16, 3, stride=1, padding=1),
                                   nn.BatchNorm3d(16),nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv3d(16, 16, 5, stride=1, padding=2),
                                   nn.BatchNorm3d(16),nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv3d(16, 16, 7, stride=1, padding=3),
                                   nn.BatchNorm3d(16),nn.ReLU(inplace=True))
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.act = nn.ReLU(inplace=False)
        self.fc = nn.Sequential(nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Linear(64, 128),
                                nn.Sigmoid())


    def forward(self, x1, x2, x3):
        attn11 = self.conv2(x1)
        attn12 = self.conv3(x1)
        attn13 = self.conv4(x1)
        attn14 = self.conv5(x1)
        attn1 = torch.cat([attn11,attn12,attn13,attn14],dim=1)
        attn21 = self.conv2(x2)
        attn22 = self.conv3(x2)
        attn23 = self.conv4(x2)
        attn24 = self.conv5(x2)
        attn2 = torch.cat([attn21,attn22,attn23,attn24],dim=1)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = self.avg_pool(attn)
        max_attn = self.max_pool(attn)
        avg_attn = avg_attn.view(avg_attn.size(0), -1)
        max_attn = max_attn.view(max_attn.size(0), -1)
        agg = avg_attn + max_attn
        channel_attention = self.fc(agg)
        channel_attention = channel_attention.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        out = attn * channel_attention
        out = self.conv0(out)
        x3 = self.conv1(x3)
        output = out + x3

        return output


