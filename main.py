import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dataset import get_dataloaders
from model import CIFAR10Net
from train import train
from evaluate import evaluate
from visualization import plot_metrics
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Train CNN on CIFAR-10")

    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'], help='Optimizer type')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'leakyrelu', 'elu'], help='Activation function')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    trainloader, testloader = get_dataloaders()
    model = CIFAR10Net(activation=args.activation)
    loss_list, acc_list = train(model, trainloader, testloader, epochs=args.epochs, optimizer_type=args.optimizer)
    evaluate(model, testloader)

    # 保存 loss 和 acc 到 .npy 文件
    np.save(f"loss_{args.optimizer}_{args.activation}.npy", loss_list)
    np.save(f"acc_{args.optimizer}_{args.activation}.npy", acc_list)
