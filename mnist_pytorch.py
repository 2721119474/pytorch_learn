from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor  # 转换为张量
from torch.utils.data import DataLoader  # 数据集封装预处理
import torch.nn as nn  # 关于网络相关的包
import torch.optim as optim  # 优化器

# 加载数据集
dataset=MNIST(root='../data/mnist',download=True,train=True,transform=ToTensor())
dl = DataLoader(dataset,batch_size=256,shuffle=True)

# 构建模型
model=nn.Sequential(
    nn.Linear(in_features=784,out_features=100),  # 隐藏层
    nn.ReLU(),
    nn.Linear(in_features=100,out_features=10)
)

# 构建优化器和损失函数
optimizer=optim.SGD(model.parameters(),lr=1e-2)  # 随机梯度下降
loss_fn = nn.CrossEntropyLoss()

# 训练
for epoch in range(4):
    for img,lbl in dl:
        img = img.reshape(-1,784)  # 训练批次需要和总数能整除，不确定可以用-1，但是仅能有一个-1，
        logits=model(img)  # forward
        loss=loss_fn(logits,lbl)  # 预测值，真实值
        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        # 清理梯度 在pytorch中，如果不清理梯度，则会累加起来，
        model.zero_grad()  # 或者用optimizer.zero_grad()
        print(f"epoch:{epoch+1},train loss:{loss.item():.3f}")
