import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from model import CIFAR10Net  # 导入模型结构

classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# 反标准化图像用于可视化
def unnormalize(img):
    img = img * 0.5 + 0.5  # [0, 1] 范围
    return img

# 加载模型
def load_model(pth_file, activation='relu', device='cuda'):
    model = CIFAR10Net(activation=activation)
    model.load_state_dict(torch.load(pth_file, map_location=device))
    model.to(device)
    model.eval()
    return model

# 显示预测图像（每张图配一个 label）
def show_images_with_predictions(images, predictions):
    images = images.cpu()
    predictions = predictions.cpu()

    fig, axes = plt.subplots(1, len(images), figsize=(15, 2.5))

    for i in range(len(images)):
        img = unnormalize(images[i])  # [C, H, W]
        npimg = img.numpy().transpose((1, 2, 0))  # [H, W, C]
        axes[i].imshow(npimg)
        axes[i].axis('off')
        axes[i].set_title(classes[predictions[i]], fontsize=9)

    plt.tight_layout()
    plt.savefig("prediction_grid.png")
    plt.show()

# 主程序：加载数据 + 预测 + 显示
def predict_and_show(pth_file, activation='relu', device='cuda'):
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True, num_workers=2)

    model = load_model(pth_file, activation, device)

    # 获取一批数据进行预测
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # 显示结果
    show_images_with_predictions(images[:8], predicted[:8])

# 示例调用
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predict_and_show('cifar10_model_adam_relu.pth', activation='relu', device=device)
