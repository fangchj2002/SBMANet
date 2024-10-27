import torch
import torch.nn as nn
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.ConvModule import BasicResBlock
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.ConvModule import DownsamolingBlock
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.ConvModule import Upsample_Layer_nearest
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.ConvModule import InputChannel_project, DecoderResBlock
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.Attention_mechanism_v3 import HHMF
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.ChannelSearchV3 import AMCS

class SBMANet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2):
        super(SBMANet, self).__init__()
        self.feature_project1 = InputChannel_project(2,16)
        self.feature_project2 = InputChannel_project(2,16)
        self.Crossmodal_Attention1 = HHMF(128,8)
        self.Crossmodal_Attention2 = HHMF(64,8)
        self.Crossmodal_Attention3 = HHMF(32,8)
        self.Crossmodal_Attention4 = HHMF(16,8)
        self.ChannelAdaptive = AMCS()
        self.act = nn.ReLU()

        self.conv1 = nn.Sequential(nn.Conv3d(32, 2, 3, stride=1, padding=1),
                                   nn.BatchNorm3d(2),nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv3d(16, 2, 3, stride=1, padding=1),
                                   nn.BatchNorm3d(2),nn.ReLU(inplace=True))

        self.encoder11 = BasicResBlock(16, 16)
        self.Down11 = DownsamolingBlock(16, 32)
        self.encoder12 = BasicResBlock(32, 32)
        self.Down12 = DownsamolingBlock(32, 64)
        self.encoder13 = BasicResBlock(64, 64)
        self.Down13 = DownsamolingBlock(64, 128)
        self.encoder14 = BasicResBlock(128, 128)
        self.Down14 = DownsamolingBlock(128, 256)
        self.encoder15 = BasicResBlock(256, 256)

        self.upsample14 = Upsample_Layer_nearest(256, 128)
        self.decoder11 = DecoderResBlock(256, 128)
        self.upsample13 = Upsample_Layer_nearest(128, 64)
        self.decoder12 = DecoderResBlock(128, 64)
        self.upsample12 = Upsample_Layer_nearest(64, 32)
        self.decoder13 = DecoderResBlock(64, 32)
        self.upsample11 = Upsample_Layer_nearest(32, 16)
        self.decoder14 = DecoderResBlock(32, 16)
        # 2
        self.encoder21 = BasicResBlock(16, 16)
        self.Down21 = DownsamolingBlock(16, 32)
        self.encoder22 = BasicResBlock(32, 32)
        self.Down22 = DownsamolingBlock(32, 64)
        self.encoder23 = BasicResBlock(64, 64)
        self.Down23 = DownsamolingBlock(64, 128)
        self.encoder24 = BasicResBlock(128, 128)
        self.Down24 = DownsamolingBlock(128, 256)
        self.encoder25 = BasicResBlock(256, 256)

        self.upsample24 = Upsample_Layer_nearest(256, 128)
        self.decoder21 = DecoderResBlock(256, 128)
        self.upsample23 = Upsample_Layer_nearest(128, 64)
        self.decoder22 = DecoderResBlock(128, 64)
        self.upsample22 = Upsample_Layer_nearest(64, 32)
        self.decoder23 = DecoderResBlock(64, 32)
        self.upsample21 = Upsample_Layer_nearest(32, 16)
        self.decoder24 = DecoderResBlock(32, 16)

        self.upsample33 = Upsample_Layer_nearest(128, 64)
        self.decoder31 = DecoderResBlock(128, 64)
        self.upsample32 = Upsample_Layer_nearest(64, 32)
        self.decoder32 = DecoderResBlock(64, 32)
        self.upsample31 = Upsample_Layer_nearest(32, 16)
        self.decoder33 = DecoderResBlock(32, 16)


    def forward(self, data):
        in1 = data[:, 0, :, :, :].unsqueeze(1)
        in2 = data[:, 1, :, :, :].unsqueeze(1)
        in3 = data[:, 2, :, :, :].unsqueeze(1)

        b1_input = torch.cat((in1,in2),dim=1)
        b2_input = torch.cat((in2,in3),dim=1)
        b1_input = self.feature_project1(b1_input)
        b2_input = self.feature_project2(b2_input)

        # Encoder 1
        out1 = self.encoder11(b1_input)
        out11 = out1
        out1 = self.Down11(out1)

        out1 = self.encoder12(out1)
        out12 = out1
        out1 = self.Down12(out1)

        out1 = self.encoder13(out1)
        out13 = out1
        out1 = self.Down13(out1)

        out1 = self.encoder14(out1)
        out14 = out1
        out1 = self.Down14(out1)

        out1 = self.encoder15(out1)
        # Encoder 2
        out2 = self.encoder21(b2_input)
        out21 = out2
        out2 = self.Down21(out2)

        out2 = self.encoder22(out2)
        out22 = out2
        out2 = self.Down22(out2)

        out2 = self.encoder23(out2)
        out23 = out2
        out2 = self.Down23(out2)

        out2 = self.encoder24(out2)
        out24 = out2
        out2 = self.Down24(out2)

        out2 = self.encoder25(out2)
        # Decoder
        #layer4
        out1 = self.upsample14(out1)
        out2 = self.upsample24(out2)

        fusion1 = self.Crossmodal_Attention1(out1,out2)
        fusion1 = self.upsample33(fusion1)

        skip14 = torch.cat((out1, out14), dim=1)
        out1 = self.decoder11(skip14)

        skip24 = torch.cat((out2, out24), dim=1)
        out2 = self.decoder21(skip24)

       #layer3
        out1 = self.upsample13(out1)
        out2 = self.upsample23(out2)

        fusion2 = self.Crossmodal_Attention2(out1,out2)
        mid = torch.cat((fusion1, fusion2),dim=1)
        fusion2 = self.decoder31(mid)
        fusion2 = self.upsample32(fusion2)

        skip13 = torch.cat((out1, out13), dim=1)
        out1 = self.decoder12(skip13)

        skip23 = torch.cat((out2, out23), dim=1)
        out2 = self.decoder22(skip23)

        #layer2
        out1 = self.upsample12(out1)
        out2 = self.upsample22(out2)

        fusion3 = self.Crossmodal_Attention3(out1,out2)
        mid = torch.cat((fusion2, fusion3),dim=1)
        fusion3 = self.decoder32(mid)
        fusion3 = self.upsample31(fusion3)

        skip12 = torch.cat((out1, out12), dim=1)
        out1 = self.decoder13(skip12)

        skip22 = torch.cat((out2, out22), dim=1)
        out2 = self.decoder23(skip22)
        #layer1
        out1 = self.upsample11(out1)
        out2 = self.upsample21(out2)

        fusion4 = self.Crossmodal_Attention4(out1,out2)
        mid = torch.cat((fusion3, fusion4),dim=1)
        out3 = self.decoder33(mid)

        skip11 = torch.cat((out1, out11), dim=1)
        out1 = self.decoder14(skip11)

        skip21 = torch.cat((out2, out21), dim=1)
        out2 = self.decoder24(skip21)

        #任务层
        out = self.ChannelAdaptive(out1,out2,out3)

        return out




