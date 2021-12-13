import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import random
from torch.utils.data.dataloader import DataLoader

DATASET_DIR = './datasets/Garbage classification'
TRAIN_MODE = False
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 2e-4
MOMENTUM = 0.9
WD = 5e-4
TO_SAVE = True
SAVE_DIR = './checkpoint/2021-12-13-11-00-resnet50-pretrained.pth'
CHECKPOINT_DIR = './checkpoint/2021-12-13-11-00-resnet50-pretrained.pth'



# 获取GPU/CPU
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


# 设置准确率计算
def acc(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(labels == preds).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader, device):
    val_t = 0
    model.eval()
    acc_all = 0
    with torch.no_grad():
        for X, y in val_loader:
            val_t += 1
            X = X.to(device)
            y = y.to(device)
            preds = model(X)
            acc_all += acc(preds, y)
    return acc_all / val_t


def main():
    transform = transforms.Compose([
        # 定义对于所有数据集的预处理操作，用于数据增广，作为卷积神经网络的输入
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                     ratio=(3.0 / 4.0, 4.0 / 3.0)),
        transforms.RandomHorizontalFlip(),
        # 随机更改亮度，对比度和饱和度
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4),
        transforms.ToTensor(),
        # 标准化图像的每个通道
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        # 从图像中心裁切224x224大小的图片
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    # 利用torchvision封装好的ImageFolder加载数据集中所有图片路径
    dataset = ImageFolder(DATASET_DIR, transform=transform)
    # 查看数据集的信息
    print(dataset)
    # 使用随机索引通过plt随即展示初始数据集
    # rand_img = random.sample(dataset.samples, 1)[0][0]
    # plt.imshow(plt.imread(rand_img))
    # plt.show()
    # 划分训练集，验证集，测试集
    train_ds, val_ds, test_ds = random_split(dataset, [1593, 176, 758])
    val_ds.transform = transform_test
    batch_size = BATCH_SIZE
    # 获得用于训练和验证的Datlaloader
    if TRAIN_MODE:
        train_iter = DataLoader(train_ds, batch_size, shuffle=True)
        val_iter = DataLoader(val_ds, batch_size * 2)
    else:
        val_iter = DataLoader(test_ds, batch_size * 2)
    device = get_default_device()
    # 输出使用GPU/CPU训练
    if TRAIN_MODE:
        print("training on", device)
        # 利用torchvision获得resnet50卷积神经网络模型
        net = resnet50(pretrained=True)
        # 将模型参数放入device
        net.to(device)
        # 定义优化器为SGD,loss函数为交叉熵损失函数
        optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM,
                                    weight_decay=WD)
        loss = torch.nn.CrossEntropyLoss()
        # 迭代进行训练
        for epoch in range(EPOCHS):
            print("Epoch", epoch + 1)
            net.train()
            for i, (X, y) in enumerate(train_iter):
                X, y = X.to(device), y.to(device)
                # 将梯度清零
                optimizer.zero_grad()
                # outputs为模型输出结果(前向传播)
                outputs = net(X)
                # 与真实标签计算loss
                l = loss(outputs, y)
                # loss计算反向传播求解梯度
                l.backward()
                # 执行梯度下降进行权重参数更新
                optimizer.step()
                # 计算训练准确率
                train_acc = acc(outputs, y)
            # 计算验证准确率
            val_acc = evaluate(net, val_iter, device)
            print("loss {:.3f}, train acc {:.3f}, test acc {:.3f}".format(l.item(), train_acc.item(), val_acc.item()))
            # 如果要保存训练的模型参数，则将结果保存至SAVE_DIR
            if TO_SAVE:
                torch.save(net.state_dict(), SAVE_DIR)
                print("权重已保存至",SAVE_DIR)
    else:
        # 测试模式，加载模型参数
        params = torch.load(CHECKPOINT_DIR)
        net = resnet50(pretrained=False)
        net.load_state_dict(params)
        net.to(device)
        # 计算测试准确率
        test_acc = evaluate(net, val_iter, device)
        print("test acc {:.3f}".format(test_acc.item()))


if __name__ == '__main__':
    main()
