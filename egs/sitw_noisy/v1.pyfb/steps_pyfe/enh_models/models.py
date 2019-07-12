import torch
import torch.nn as nn

    

class DFL_CAN2d_2(nn.Module):
    """
one less layer from DFL_CAN2d_1
---
        Deep Feature Loss CAN (Context-Aggregation Network) SE n/w; padding == dilation from Phani; 14 intermediate layers, total=14+input+output=16
        diff c.f. original DFL paper
            9-intermediate layers because signal is not very long here, 500? frames c.f. 1-sec
        freezing this version of model at 9 layers now
    """
    def __init__(self, num_channels=45, kernel_size=3, LeakyReLU_slope=0.2, bias=False):
        super(DFL_CAN2d_2, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size, padding=2**0, dilation=2**0, bias=True),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**1, dilation=2**1, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**2, dilation=2**2, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**3, dilation=2**3, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**4, dilation=2**4, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**5, dilation=2**5, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**6, dilation=2**6, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**7, dilation=2**7, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**8, dilation=2**8, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**9, dilation=2**9, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**10, dilation=2**10, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**11, dilation=2**11, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**12, dilation=2**12, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, 1, 1, bias=True)    # standard linear layer w/ bias
        )

    def forward(self, x):
        x = self.network(x)
        return x



    
class DFL_CAN2d_1_ReLU_BN(nn.Module):
    """
batchnorm
c.f. relu instead of leaky relu, probably this solves nan problems
--- init copy of DFL_CAN2d_1
        Deep Feature Loss CAN (Context-Aggregation Network) SE n/w; padding == dilation from Phani; 14 intermediate layers, total=14+input+output=16
        diff c.f. original DFL paper
            9-intermediate layers because signal is not very long here, 500? frames c.f. 1-sec
        freezing this version of model at 9 layers now
    """
    def __init__(self, num_channels=45, kernel_size=3, bias=False):
        super(DFL_CAN2d_1_ReLU_BN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size, padding=2**0, dilation=2**0, bias=True),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**1, dilation=2**1, bias=bias),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**2, dilation=2**2, bias=bias),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**3, dilation=2**3, bias=bias),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**4, dilation=2**4, bias=bias),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**5, dilation=2**5, bias=bias),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**6, dilation=2**6, bias=bias),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**7, dilation=2**7, bias=bias),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, 1, 1, bias=True)    # standard linear layer w/ bias
        )

    def forward(self, x):
        x = self.network(x)
        return x


class DFL_CAN2d_1_ReLU(nn.Module):
    """
c.f. relu instead of leaky relu, probably this solves nan problems
--- init copy of DFL_CAN2d_1
        Deep Feature Loss CAN (Context-Aggregation Network) SE n/w; padding == dilation from Phani; 14 intermediate layers, total=14+input+output=16
        diff c.f. original DFL paper
            9-intermediate layers because signal is not very long here, 500? frames c.f. 1-sec
        freezing this version of model at 9 layers now
    """
    def __init__(self, num_channels=45, kernel_size=3, bias=False):
        super(DFL_CAN2d_1_ReLU, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size, padding=2**0, dilation=2**0, bias=True),
            AdaBN2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**1, dilation=2**1, bias=bias),
            AdaBN2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**2, dilation=2**2, bias=bias),
            AdaBN2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**3, dilation=2**3, bias=bias),
            AdaBN2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**4, dilation=2**4, bias=bias),
            AdaBN2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**5, dilation=2**5, bias=bias),
            AdaBN2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**6, dilation=2**6, bias=bias),
            AdaBN2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**7, dilation=2**7, bias=bias),
            AdaBN2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, 1, 1, bias=True)    # standard linear layer w/ bias
        )

    def forward(self, x):
        x = self.network(x)
        return x


class DFL_CAN2d_1_BN(nn.Module):
    """
--- with BN instead of adaBN
        Deep Feature Loss CAN (Context-Aggregation Network) SE n/w; padding == dilation from Phani; 14 intermediate layers, total=14+input+output=16
        diff c.f. original DFL paper
            9-intermediate layers because signal is not very long here, 500? frames c.f. 1-sec
        freezing this version of model at 9 layers now
    """
    def __init__(self, num_channels=45, kernel_size=3, LeakyReLU_slope=0.2, bias=False):
        super(DFL_CAN2d_1_BN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size, padding=2**0, dilation=2**0, bias=True),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**1, dilation=2**1, bias=bias),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**2, dilation=2**2, bias=bias),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**3, dilation=2**3, bias=bias),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**4, dilation=2**4, bias=bias),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**5, dilation=2**5, bias=bias),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**6, dilation=2**6, bias=bias),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**7, dilation=2**7, bias=bias),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, 1, 1, bias=True)    # standard linear layer w/ bias
        )

    def forward(self, x):
        x = self.network(x)
        return x


class DFL_CAN2d_1(nn.Module):
    """
        Deep Feature Loss CAN (Context-Aggregation Network) SE n/w; padding == dilation from Phani; 14 intermediate layers, total=14+input+output=16
        diff c.f. original DFL paper
            9-intermediate layers because signal is not very long here, 500? frames c.f. 1-sec
        freezing this version of model at 9 layers now
    """
    def __init__(self, num_channels=45, kernel_size=3, LeakyReLU_slope=0.2, bias=False):
        super(DFL_CAN2d_1, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size, padding=2**0, dilation=2**0, bias=True),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**1, dilation=2**1, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**2, dilation=2**2, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**3, dilation=2**3, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**4, dilation=2**4, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**5, dilation=2**5, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**6, dilation=2**6, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**7, dilation=2**7, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**8, dilation=2**8, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**9, dilation=2**9, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**10, dilation=2**10, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**11, dilation=2**11, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**12, dilation=2**12, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, 1, 1, bias=True)    # standard linear layer w/ bias
        )

    def forward(self, x):
        x = self.network(x)
        return x

class Res_DFL_CAN2d_1(DFL_CAN2d_1):
    def forward(self, x):
        x = x + self.network(x)
        return x


class DFL_CAN2d_1_LogSigMask(DFL_CAN2d_1):
    def forward(self, x):
        x = x + torch.nn.functional.logsigmoid(self.network(x))
        return x

class DFL_CAN2d_1_LinMask(DFL_CAN2d_1):
    def forward(self, x):
        x = torch.log(torch.exp(x) * torch.nn.functional.sigmoid(self.network(x)) + 0.00001)
        return x


#smaller context networks

class DFL_CAN2d_SmallContext(nn.Module):
    """
        Deep Feature Loss CAN (Context-Aggregation Network) SE n/w; padding == dilation from Phani; 14 intermediate layers, total=14+input+output=16
        diff c.f. original DFL paper
            9-intermediate layers because signal is not very long here, 500? frames c.f. 1-sec
        freezing this version of model at 9 layers now
    """
    def __init__(self, num_channels=45, kernel_size=3, LeakyReLU_slope=0.2, bias=False):
        super(DFL_CAN2d_SmallContext, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size, padding=1, dilation=1, bias=True),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2, dilation=2, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=3, dilation=3, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=4, dilation=4, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=5, dilation=5, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=6, dilation=6, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=7, dilation=7, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=8, dilation=8, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**8, dilation=2**8, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**9, dilation=2**9, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**10, dilation=2**10, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**11, dilation=2**11, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**12, dilation=2**12, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, 1, 1, bias=True)    # standard linear layer w/ bias
        )

    def forward(self, x):
        x = self.network(x)
        return x




class DFL_CAN2d_MediumContext(nn.Module):
    """
        Deep Feature Loss CAN (Context-Aggregation Network) SE n/w; padding == dilation from Phani; 14 intermediate layers, total=14+input+output=16
        diff c.f. original DFL paper
            9-intermediate layers because signal is not very long here, 500? frames c.f. 1-sec
        freezing this version of model at 9 layers now
    """
    def __init__(self, num_channels=45, kernel_size=3, LeakyReLU_slope=0.2, bias=False):
        super(DFL_CAN2d_MediumContext, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size, padding=1, dilation=1, bias=True),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=3, dilation=3, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=5, dilation=5, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=7, dilation=7, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=9, dilation=9, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=11, dilation=11, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=13, dilation=13, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=15, dilation=15, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**8, dilation=2**8, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**9, dilation=2**9, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**10, dilation=2**10, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**11, dilation=2**11, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**12, dilation=2**12, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, 1, 1, bias=True)    # standard linear layer w/ bias
        )

    def forward(self, x):
        x = self.network(x)
        return x



class DFL_CAN2d_SmallContext_LogSigMask(DFL_CAN2d_SmallContext):
    def forward(self, x):
        x = x + torch.nn.functional.logsigmoid(self.network(x))
        return x



class DFL_CAN2d_MediumContext_LogSigMask(DFL_CAN2d_MediumContext):
    def forward(self, x):
        x = x + torch.nn.functional.logsigmoid(self.network(x))
        return x



class DFL_CAN2d_SmallContext_BNIn(nn.Module):
    """
        Deep Feature Loss CAN (Context-Aggregation Network) SE n/w; padding == dilation from Phani; 14 intermediate layers, total=14+input+output=16
        diff c.f. original DFL paper
            9-intermediate layers because signal is not very long here, 500? frames c.f. 1-sec
        freezing this version of model at 9 layers now
    """
    def __init__(self, num_channels=45, kernel_size=3, LeakyReLU_slope=0.2, bias=False):
        super(DFL_CAN2d_SmallContext_BNIn, self).__init__()
        self.network = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, num_channels, kernel_size, padding=1, dilation=1, bias=True),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2, dilation=2, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=3, dilation=3, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=4, dilation=4, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=5, dilation=5, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=6, dilation=6, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=7, dilation=7, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=8, dilation=8, bias=bias),
            AdaBN2d(num_channels),
            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**8, dilation=2**8, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**9, dilation=2**9, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**10, dilation=2**10, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**11, dilation=2**11, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**12, dilation=2**12, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, 1, 1, bias=True)    # standard linear layer w/ bias
        )

    def forward(self, x):
        x = self.network(x)
        return x

class DFL_CAN2d_SmallContext_LogSigMask_BNIn(DFL_CAN2d_SmallContext_BNIn):
    def forward(self, x):
        x = x + torch.nn.functional.logsigmoid(self.network(x))
        return x




class DFL_ResCAN2d_SmallContext_BNIn(nn.Module):
    """
        Deep Feature Loss CAN (Context-Aggregation Network) SE n/w; padding == dilation from Phani; 14 intermediate layers, total=14+input+output=16
        diff c.f. original DFL paper
            9-intermediate layers because signal is not very long here, 500? frames c.f. 1-sec
        freezing this version of model at 9 layers now
    """
    def __init__(self, num_channels=45, kernel_size=3, LeakyReLU_slope=0.2, bias=False):
        super(DFL_ResCAN2d_SmallContext_BNIn, self).__init__()
        self.network = nn.ModuleList([
            nn.BatchNorm2d(1), #0
            nn.Conv2d(1, num_channels, kernel_size, padding=1, dilation=1, bias=True), #1
            AdaBN2d(num_channels), #2
            nn.LeakyReLU(negative_slope=LeakyReLU_slope), #3
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2, dilation=2, bias=bias), #4
            AdaBN2d(num_channels), #5
            nn.LeakyReLU(negative_slope=LeakyReLU_slope), #6
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=3, dilation=3, bias=bias), #7
            AdaBN2d(num_channels), #8
            nn.LeakyReLU(negative_slope=LeakyReLU_slope), #9
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=4, dilation=4, bias=bias), #10
            AdaBN2d(num_channels), #11
            nn.LeakyReLU(negative_slope=LeakyReLU_slope), #12
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=5, dilation=5, bias=bias), #13
            AdaBN2d(num_channels), #14
            nn.LeakyReLU(negative_slope=LeakyReLU_slope), #15
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=6, dilation=6, bias=bias), #16
            AdaBN2d(num_channels), #17
            nn.LeakyReLU(negative_slope=LeakyReLU_slope), #18
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=7, dilation=7, bias=bias), #19
            AdaBN2d(num_channels), #20
            nn.LeakyReLU(negative_slope=LeakyReLU_slope), #21
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=8, dilation=8, bias=bias), #22
            AdaBN2d(num_channels), #23
            nn.LeakyReLU(negative_slope=LeakyReLU_slope), #24
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**8, dilation=2**8, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**9, dilation=2**9, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**10, dilation=2**10, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**11, dilation=2**11, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
#            nn.Conv2d(num_channels, num_channels, kernel_size, padding=2**12, dilation=2**12, bias=bias),
#            AdaBN2d(num_channels),
#            nn.LeakyReLU(negative_slope=LeakyReLU_slope),
            nn.Conv2d(num_channels, 1, 1, bias=True)    # standard linear layer w/ bias #25
        ])

    def forward(self, x):
        h1 = x
        for i in range(4):
            h1 = self.network[i](h1)

        h2 = h1
        for i in range(4,9):
            h2 = self.network[i](h2)
        h2 = self.network[9](h1+h2)

        h3 = h2
        for i in range(10,15):
            h3 = self.network[i](h3)
        h3 = self.network[15](h2+h3)

        h4 = h3
        for i in range(16,21):
            h4 = self.network[i](h4)
        h4 = self.network[21](h3+h4)

        h5 = h4
        for i in range(22,len(self.network)):
            h5 = self.network[i](h5)
        
        return h5
    

class DFL_ResCAN2d_SmallContext_LogSigMask_BNIn(DFL_ResCAN2d_SmallContext_BNIn):
    def forward(self, x):
        y = super(DFL_ResCAN2d_SmallContext_LogSigMask_BNIn, self).forward(x)
        x = x + torch.nn.functional.logsigmoid(y)
        return x

    
    
# caffeenet benchmark - better performance is possible w/ this c.f. DFL_CAN2d
#class DFL_CAN2d_BN_after_activation(nn.Module):

# +/- 2 layers from DFL_CAN2d

class AdaBN2d(nn.Module):
    """
        https://discuss.pytorch.org/t/adaptive-normalization/9157/3?u=saurabh_kataria
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(AdaBN2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.a = nn.Parameter(torch.tensor(0.1))
        self.b = nn.Parameter(torch.tensor(0.9))

    def forward(self, x):
        return self.a * x + self.b * self.bn(x)


class AdaBN1d(nn.Module):
    """
        UNTESTED
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(AdaBN1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.a = nn.Parameter(torch.FloatTensor(1, 1, 1))
        self.b = nn.Parameter(torch.FloatTensor(1, 1, 1))

    def forward(self, x):
        return self.a * x + self.b * self.bn(x)
