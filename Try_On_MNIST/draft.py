# 真实的示例，创建自己的数据集并加载，用于训练
# torch.utils.data.Dataset 的自定义数据集类，是一个可迭代对象

import torch
import numpy as np

# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetDataset(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是方便DataLoader划分，
    # 如果不知道Dataset大小，DataLoader无法工作
    def __len__(self):
        return len(self.data)

# 随机生成数据，大小为10 * 20列
source_data = np.random.rand(10, 20)
# 随机生成标签，大小为10 * 1列
source_label = np.random.randint(0,2,(10, 1))
# 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
my_dataset = GetDataset(source_data, source_label)

print(my_dataset)
print("len =",len(my_dataset))
for data in my_dataset:
    print(data[0].shape) # data是tuple类型
# 输出如下。这个for可以循环，是因为可以直接循环这个可迭代对象Dataset。
# <__main__.GetDataset object at 0x7f266e2e29a0>
# len = 10
# (20,)
# (20,)
# (20,)
# (20,)
# (20,)
# (20,)
# (20,)
# (20,)
# (20,)
# (20,)

# 下一步利用Dataloader进一步划分Dataset。
# torch.utils.data.DataLoader(dataset,batch_size,shuffle,drop_last，num_workers)

from torch.utils.data import DataLoader
# 读取数据
data_loader = DataLoader(my_dataset, batch_size=6, shuffle=True, drop_last=False, num_workers=2)
# 这个时候的data_loader仍然是一个可迭代对象。

print(data_loader)
# print(next(data_loader))
# 输出
# <torch.utils.data.dataloader.DataLoader object at 0x7f2501019dc0>
# Traceback (most recent call last):
#   File "draft.py", line 57, in <module>
#     print(next(data_loader))
# TypeError: 'DataLoader' object is not an iterator
# 报错是因为此时的data_loader是可迭代对象DataLoader，不是迭代器

# 最后一步才是将其转化成迭代器，达到节省内存的效果
for idx, batch in enumerate(data_loader):
    datas, labels = batch
    print(datas.shape)
    print(labels.shape)
    break # 输出第一个数据看看
# 输出
# torch.Size([6, 20])
# torch.Size([6, 1]) 