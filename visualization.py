import matplotlib.pyplot as plt


def plot_metrics(losses, accs, name):
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(10, 5))

    # Loss 图
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)

    # Accuracy 图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in accs], label='Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'training_metrics_{name}.png')  # 保存图像到文件
    plt.show()
