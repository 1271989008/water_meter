import torch.nn as nn
import torch.nn.functional as F

class res_block_A(nn.Module):
    def __init__(self,in_features):
        super(res_block_A,self).__init__()
        self.conv_block=nn.Sequential(
            nn.Conv2d(in_features,in_features,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_features,0.8),
            nn.ReLU(),
            nn.Conv2d(in_features,in_features,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_features,0.8)
        )

    def forward(self, x):
        x=x+self.conv_block(x)
        return nn.ReLU()(x)

class res_block_B(nn.Module):
    def __init__(self,in_features):
        super(res_block_B,self).__init__()
        self.conv_block=nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.ReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_features, 0.8)
        )
        self.short_conv_path=nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_features, 0.8)
        )

    def forward(self,x):
        x=self.short_conv_path(x)+self.conv_block(x)
        return nn.ReLU()(x)

class FCSRN(nn.Module):
    def __init__(self,in_channels=3, out_channels=3, n_residual_blocks=16):
        super(FCSRN, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels,16,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        #初始化残差块
        res_block=[]
        for _ in range(3):
            res_block.append(res_block_A(16))
        res_block.append(res_block_B(24))
        for _ in range(3):
            res_block.append(res_block_A(24))
        res_block.append(res_block_B(32))
        for _ in range(3):
            res_block.append(res_block_A(32))
        res_block.append(res_block_B(48))
        for _ in range(3):
            res_block.append(res_block_A(48))

    def forward(self,x):
        return x

class mnist_net(nn.Module):
    def __init__(self):
        super(mnist_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1, 2)  # 卷积层1: 二维卷积层, 1x28x28,16x28x28, 卷积核大小为5x5
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)  # 卷积层2: 二维卷积层, 16x14x14,32x14x14, 卷积核大小为5x5
        # an affine(仿射) operation: y = Wx + b # 全连接层1: 线性层, 输入维度32x7x7,输出维度128

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 全连接层2: 线性层, 输入维度128,输出维度10

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # 先卷积,再池化
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 再卷积,再池化
        x = x.view(x.size(0), -1)  # 将conv3_out展开成一维(扁平化)
        x = F.relu(self.fc1(x))  # 全连接1
        x = self.fc2(x)  # 全连接2
        # return out
        return F.log_softmax(x,dim=0)





