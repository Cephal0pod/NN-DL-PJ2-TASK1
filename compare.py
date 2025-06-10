import numpy as np
import matplotlib.pyplot as plt

activations = 'relu'
optimizer = ['adam', 'sgd']


# 设置颜色映射
colors = {
    'adam': 'red',
    'sgd': 'purple',
}

# 绘图：Loss 和 Accuracy 同图双 y 轴
plt.figure(figsize=(10, 6))
ax1 = plt.gca()  # 左 y 轴（Loss）
ax2 = ax1.twinx()  # 右 y 轴（Accuracy）

for opt in optimizer:
    loss = np.load(f'loss_{opt}_{activations}.npy')
    acc = np.load(f'acc_{opt}_{activations}.npy')
    acc = [a * 100 for a in acc]  # 精度转百分比

    epochs = range(1, len(loss)+1)
    ax1.plot(epochs, loss, label=f'{opt} loss', color=colors[opt], linestyle='-')
    ax2.plot(epochs, acc, label=f'{opt} acc', color=colors[opt], linestyle='--')

# 设置标签和图例
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax2.set_ylabel('Accuracy (%)')
plt.title(f'Loss & Accuracy Comparison (activation = {activations})')

# 合并两个轴的图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

plt.grid(True)
plt.tight_layout()
plt.savefig(f'loss_acc_compare_{activations}.png')
plt.show()