# In [1]: 导入必要的库和模块
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

# In [2]: 加载MNIST数据集
# 训练集
train_dataset = datasets.MNIST(
    root='./',           # 数据集存储的根目录（当前目录）
    train=True,          # 是否为训练集（True表示训练集）
    transform=transforms.ToTensor(),  # 将图像转换为PyTorch张量，并归一化到[0,1]
    download=True        # 如果数据集不存在则自动下载
)
# 测试集
test_dataset = datasets.MNIST(
    root='./',           # 数据集存储的根目录
    train=False,         # 是否为测试集（False表示测试集）
    transform=transforms.ToTensor(),  # 同样的转换
    download=True        # 如果数据集不存在则自动下载
)

# In [3]: 创建数据加载器（DataLoader）
# 批次大小：每次训练或测试时使用的样本数量
batch_size = 64

# 训练集数据加载器
train_loader = DataLoader(
    dataset=train_dataset,  # 要加载的数据集
    batch_size=batch_size,  # 每个批次的大小
    shuffle=True            # 是否在每个epoch开始时打乱数据（训练时通常打乱）
)

# 测试集数据加载器
test_loader = DataLoader(
    dataset=test_dataset,   # 要加载的数据集
    batch_size=batch_size,  # 每个批次的大小
    shuffle=True            # 测试时也可打乱，但这不是必须的
)

# In [4]: 查看数据形状（验证数据加载是否正确）
for i, data in enumerate(train_loader):
    inputs, labels = data  # 解构数据批次：inputs是图像数据，labels是标签
    print(inputs.shape)    # 打印输入数据的形状：[batch_size, channels, height, width]
    print(labels.shape)    # 打印标签的形状：[batch_size]
    break  # 只查看第一个批次的数据

# In [5]: 定义卷积神经网络（CNN）结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # 调用父类nn.Module的初始化方法
        
        # 第一层卷积序列
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,      # 输入通道数：灰度图为1，RGB图为3
                out_channels=32,    # 输出通道数：32个卷积核，产生32个特征图
                kernel_size=5,      # 卷积核大小：5×5
                stride=1,           # 卷积步长：每次移动1像素
                padding=2           # 边缘填充：保持输入输出尺寸相同（28->28）
            ),
            nn.ReLU(),              # ReLU激活函数：引入非线性
            nn.MaxPool2d(
                kernel_size=2,      # 池化窗口大小：2×2
                stride=2            # 池化步长：每次移动2像素（28->14）
            )
        )
        
        # 第二层卷积序列
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,     # 输入通道数：来自上一层conv1的32个特征图
                out_channels=64,    # 输出通道数：64个卷积核
                kernel_size=5,      # 卷积核大小：5×5
                stride=1,           # 卷积步长：1
                padding=2           # 填充：保持尺寸（14->14）
            ),
            nn.ReLU(),              # ReLU激活函数
            nn.MaxPool2d(2, 2)      # 最大池化：2×2窗口，步长2（14->7）
        )
        
        # 第一个全连接层（从卷积特征到全连接层）
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=64 * 7 * 7,  # 输入特征数：64个7×7的特征图展平
                out_features=1000        # 输出特征数：1000个神经元
            ),
            nn.Dropout(p=0.5),           # Dropout层：防止过拟合，以0.5概率丢弃神经元
            nn.ReLU()                    # ReLU激活函数
        )
        
        # 第二个全连接层（输出层）
        self.fc2 = nn.Sequential(
            nn.Linear(
                in_features=1000,        # 输入特征数：来自上一层fc1的1000个特征
                out_features=10          # 输出特征数：10个数字类别（0-9）
            ),
            nn.Softmax(dim=1)           # Softmax函数：将输出转换为概率分布（沿第1维）
        )
    
    def forward(self, x):
        # 输入x的形状：[batch_size, 1, 28, 28]
        x = self.conv1(x)               # 通过第一层卷积序列：[batch_size, 32, 14, 14]
        x = self.conv2(x)               # 通过第二层卷积序列：[batch_size, 64, 7, 7]
        x = x.view(x.size(0), -1)       # 展平操作：将三维特征图展平为一维向量 [batch_size, 64*7*7]
        x = self.fc1(x)                 # 通过第一个全连接层：[batch_size, 1000]
        x = self.fc2(x)                 # 通过第二个全连接层：[batch_size, 10]
        return x                        # 返回输出：每个类别的概率分布

# In [6]: 定义训练所需的组件
LR = 0.001  # 学习率：控制参数更新的步长

# 实例化模型
model = Net()  # 创建Net类的实例

# 定义损失函数：交叉熵损失（适用于多分类问题）
# 注意：这里变量名是mse_loss，但实际使用的是CrossEntropyLoss
mse_loss = nn.CrossEntropyLoss()

# 定义优化器：Adam优化器（自适应矩估计）
optimizer = optim.Adam(
    params=model.parameters(),  # 要优化的参数：模型的所有可学习参数
    lr=LR                       # 学习率
)

# In [7]: 定义训练和测试函数
def train():
    """训练函数：执行一个epoch的训练"""
    model.train()  # 将模型设置为训练模式（启用Dropout等训练特定层）
    
    for i, data in enumerate(train_loader):
        # 从数据加载器中获取一个批次的数据
        inputs, labels = data  # inputs: [64, 1, 28, 28], labels: [64]
        
        # 前向传播：计算模型输出
        out = model(inputs)  # 输出形状：[64, 10]，表示每个样本在10个类别上的概率分布
        
        # 计算损失：比较预测输出和真实标签
        loss = mse_loss(out, labels)
        
        # 梯度清零：清除上一批次的梯度，避免累积
        optimizer.zero_grad()
        
        # 反向传播：计算损失函数相对于每个参数的梯度
        loss.backward()
        
        # 参数更新：根据梯度更新模型参数
        optimizer.step()

def test():
    """测试函数：评估模型在测试集和训练集上的准确率"""
    model.eval()  # 将模型设置为评估模式（禁用Dropout等训练特定层）
    correct = 0   # 正确预测的样本数
    
    # 测试集评估
    for i, data in enumerate(test_loader):
        inputs, labels = data
        out = model(inputs)
        
        # 获取预测类别：找到最大概率对应的索引
        _, predicted = torch.max(out, 1)  # 返回最大值和最大值的索引，这里只取索引
        
        # 统计正确预测的数量
        correct += (predicted == labels).sum()
    
    # 计算测试集准确率
    test_acc = correct.item() / len(test_dataset)
    print('Test acc: {0}'.format(test_acc))
    
    # 训练集评估
    correct = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        out = model(inputs)
        _, predicted = torch.max(out, 1)
        correct += (predicted == labels).sum()
    
    # 计算训练集准确率
    train_acc = correct.item() / len(train_dataset)
    print('Train acc: {0}'.format(train_acc))

# In [8]: 主训练循环
for epoch in range(20):  # 训练20个epoch
    print('epoch:', epoch)  # 打印当前epoch编号
    train()                 # 执行训练
    test()                  # 执行测试（评估）