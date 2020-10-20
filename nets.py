import torch
import torch.nn.functional as F
#from torchsummary import summary
from torch import nn

from modules import DilatedResidualBlock, NLB, DGNL, DepthWiseDilatedResidualBlock



class DGNLNet_fast(nn.Module):
    def __init__(self, num_features=64):
        super(DGNLNet_fast, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False

        ############################################ Depth prediction network
        self.conv1 = nn.Sequential(
			nn.Conv2d(3, 32, 8, stride=4, padding=2),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True)
		)

        self.conv2 = nn.Sequential(
			nn.Conv2d(32, 64, 4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=64),
			nn.SELU(inplace=True)
		)

        self.conv3 = nn.Sequential(
			nn.Conv2d(64, 128, 4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=128),
			nn.SELU(inplace=True)
		)

        self.conv5 = nn.Sequential(
			nn.Conv2d(128, 128, 3, padding=2, dilation=2),
			nn.GroupNorm(num_groups=32, num_channels=128),
			nn.SELU(inplace=True)
		)

        self.conv8 = nn.Sequential(
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=64),
			nn.SELU(inplace=True)
		)

        self.conv9 = nn.Sequential(
			nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True)
		)


        self.depth_pred = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True),
			nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
			nn.Sigmoid()
		)

	############################################ Rain removal network


        self.head = nn.Sequential(
			# pw
			nn.Conv2d(3, 32, 1, 1, 0, 1, bias=False),
			nn.ReLU6(inplace=True),
			# dw
			nn.Conv2d(32, 32, kernel_size=8, stride=4, padding=2, groups=32, bias=False),
			nn.ReLU6(inplace=True),
			# pw-linear
			nn.Conv2d(32, num_features, 1, 1, 0, 1, 1, bias=False),
		)

        self.body = nn.Sequential(
			DepthWiseDilatedResidualBlock(num_features, num_features, 1),
			# DepthWiseDilatedResidualBlock(num_features, num_features, 1),
			DepthWiseDilatedResidualBlock(num_features, num_features, 2),
			DepthWiseDilatedResidualBlock(num_features, num_features, 2),
			DepthWiseDilatedResidualBlock(num_features, num_features, 4),
			DepthWiseDilatedResidualBlock(num_features, num_features, 8),
			DepthWiseDilatedResidualBlock(num_features, num_features, 4),
			DepthWiseDilatedResidualBlock(num_features, num_features, 2),
			DepthWiseDilatedResidualBlock(num_features, num_features, 2),
			# DepthWiseDilatedResidualBlock(num_features, num_features, 1),
			DepthWiseDilatedResidualBlock(num_features, num_features, 1)
		)


        self.dgnlb = DGNL(num_features)


        self.tail = nn.Sequential(
			# dw
			nn.ConvTranspose2d(num_features, 32, kernel_size=8, stride=4, padding=2, groups=32, bias=False),
			nn.ReLU6(inplace=True),
			# pw-linear
			nn.Conv2d(32, 3, 1, 1, 0, 1, bias=False),
		)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std

        ################################## depth prediction
        d_f1 = self.conv1(x)
        d_f2 = self.conv2(d_f1)
        d_f3 = self.conv3(d_f2)
        d_f5 = self.conv5(d_f3)
        d_f8 = self.conv8(d_f5)
        d_f9 = self.conv9(d_f8 + d_f2)
        depth_pred = self.depth_pred(d_f9 + d_f1)

        ################################## rain removal

        f = self.head(x)
        f = self.body(f)
        f = self.dgnlb(f, depth_pred.detach())
        r = self.tail(f)
        x = x + r

        x = (x * self.std + self.mean).clamp(min=0, max=1)

        if self.training:
            return x, F.upsample(depth_pred, size=x.size()[2:], mode='bilinear', align_corners=True)

        return x



class DGNLNet(nn.Module):
    def __init__(self, num_features=64):
        super(DGNLNet, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False

        ############################################ Depth prediction network
        self.conv1 = nn.Sequential(
			nn.Conv2d(3, 32, 4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True)
		)

        self.conv2 = nn.Sequential(
			nn.Conv2d(32, 64, 4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=64),
			nn.SELU(inplace=True)
		)

        self.conv3 = nn.Sequential(
			nn.Conv2d(64, 128, 4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=128),
			nn.SELU(inplace=True)
		)

        self.conv4 = nn.Sequential(
			nn.Conv2d(128, 256, 4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=256),
			nn.SELU(inplace=True)
		)

        self.conv5 = nn.Sequential(
			nn.Conv2d(256, 256, 3, padding=2, dilation=2),
			nn.GroupNorm(num_groups=32, num_channels=256),
			nn.SELU(inplace=True)
		)

        self.conv6 = nn.Sequential(
			nn.Conv2d(256, 256, 3, padding=4, dilation=4),
			nn.GroupNorm(num_groups=32, num_channels=256),
			nn.SELU(inplace=True)
		)

        self.conv7 = nn.Sequential(
			nn.Conv2d(256, 256, 3, padding=2, dilation=2),
			nn.GroupNorm(num_groups=32, num_channels=256),
			nn.SELU(inplace=True)
		)

        self.conv8 = nn.Sequential(
			nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=128),
			nn.SELU(inplace=True)
		)

        self.conv9 = nn.Sequential(
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=64),
			nn.SELU(inplace=True)
		)

        self.conv10 = nn.Sequential(
			nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True)
		)

        self.depth_pred = nn.Sequential(
			nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.SELU(inplace=True),
			nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
			nn.Sigmoid()
		)

	############################################ Rain removal network

        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0), nn.ReLU()
        )
        self.body = nn.Sequential(
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 4),
            DilatedResidualBlock(num_features, 8),
            DilatedResidualBlock(num_features, 4),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 1)
        )

        self.dgnlb = DGNL(num_features)

        self.tail = nn.Sequential(
            # nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std

        ################################## depth prediction
        d_f1 = self.conv1(x)
        d_f2 = self.conv2(d_f1)
        d_f3 = self.conv3(d_f2)
        d_f4 = self.conv4(d_f3)
        d_f5 = self.conv5(d_f4)
        d_f6 = self.conv6(d_f5)
        d_f7 = self.conv7(d_f6)
        d_f8 = self.conv8(d_f7)
        d_f9 = self.conv9(d_f8 + d_f3)
        d_f10 = self.conv10(d_f9 + d_f2)
        depth_pred = self.depth_pred(d_f10 + d_f1)

        ################################## rain removal

        f = self.head(x)
        f = self.body(f)
        f = self.dgnlb(f, depth_pred.detach())
        r = self.tail(f)
        x = x + r

        x = (x * self.std + self.mean).clamp(min=0, max=1)

        if self.training:
            return x, depth_pred

        return x


# device = torch.device("cuda:0")
#
# net = GDNLNet().to(device)
# summary(net, (3, 512, 1024))



class basic_NL(nn.Module):
    def __init__(self, num_features=64):
        super(basic_NL, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False

        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0), nn.ReLU()
        )
        self.body = nn.Sequential(
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 4),
            DilatedResidualBlock(num_features, 8),
            DilatedResidualBlock(num_features, 4),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 1)
        )

        self.nlb = NLB(num_features)

        self.tail = nn.Sequential(
            # nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std

        f = self.head(x)
        f = self.body(f)
        f = self.nlb(f)
        r = self.tail(f)
        x = x + r

        x = (x * self.std + self.mean).clamp(min=0, max=1)
        return x



class basic(nn.Module):
    def __init__(self, num_features=64):
        super(basic, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False

        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride = 2 ,padding=1), nn.ReLU(),
			nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0), nn.ReLU()
        )
        self.body = nn.Sequential(
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 4),
            DilatedResidualBlock(num_features, 8),
            DilatedResidualBlock(num_features, 4),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 1)
        )

        self.tail = nn.Sequential(
            #nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std

        f = self.head(x)
        f = self.body(f)
        r = self.tail(f)
        x = x + r

        x = (x * self.std + self.mean).clamp(min=0, max=1)
        return x



# device = torch.device("cuda:0")
#
# net = basic().to(device)
# summary(net, (3,512,1024))




class depth_predciton(nn.Module):
    def __init__(self):
        super(depth_predciton, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False


        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SELU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        )

        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SELU(inplace=True)
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        )

        self.conv10 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        )

        self.depth_pred = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.SELU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )


    def forward(self, x):

        x = (x - self.mean) / self.std

        d_f1 = self.conv1(x)
        d_f2 = self.conv2(d_f1)
        d_f3 = self.conv3(d_f2)
        d_f4 = self.conv4(d_f3)
        d_f5 = self.conv5(d_f4)
        d_f6 = self.conv6(d_f5)
        d_f7 = self.conv7(d_f6)
        d_f8 = self.conv8(d_f7)
        d_f9 = self.conv9(d_f8+d_f3)
        d_f10 = self.conv10(d_f9+d_f2)
        depth_pred = self.depth_pred(d_f10+d_f1)


        return depth_pred



# device = torch.device("cuda:0")
#
# dp = depth_predciton().to(device)
# summary(dp, (3,512,1024))
#
# gt_net = generator().to(device)
# summary(gt_net, (3,512,1024))

