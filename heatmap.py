import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 创建一个数据集
data = np.array([
    [96.7, 69.4, 54.7, 59.5, 75.0],
    [59.5, 95.3, 48.2, 61.4, 72.7],
    [51.4, 60.0, 97.4, 69.9, 54.1],
    [61.1, 80.9, 72.4, 95.3, 70.3],
    [72.2, 79.6, 55.3, 66.1, 96.6]
])

diagonal_mean = np.mean(np.diagonal(data))
print("对角线元素的均值是: ", diagonal_mean)

nondiagonal_mean = np.mean(data[np.where(~np.eye(data.shape[0],dtype=bool))])
print("非对角线元素的均值是: ", nondiagonal_mean)

x_labels = ["P1", "P2", "P3", "P4", "P5"]

y_labels = ["T1", "T2", "T3", "T4", "T5"]

# 创建一个新的图形并指定其大小为10x8
plt.figure(figsize=(4, 2), constrained_layout=True)

# 创建热力图，cmap参数设为蓝色系列，注释保留一位小数，颜色为白色
heatmap = sns.heatmap(
    data, xticklabels=x_labels, yticklabels=y_labels, cmap='Blues', annot=True, fmt=".1f",
    linewidths=1
)

# 旋转y轴的标签
plt.yticks(rotation=0)
plt.xlabel("Prompts")
plt.ylabel("Triggers")

# 显示图形
# plt.show()
plt.savefig('heatmap.svg')
