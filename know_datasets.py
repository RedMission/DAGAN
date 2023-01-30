"""
查看数据集结构
"""

import numpy as np

if __name__ == '__main__':
    # 加载数据
    # raw_data = np.load("datasets/omniglot_data.npy")
    raw_data = np.load("datasets/IITDdata.npy",allow_pickle=True)
    print("数据集类型：",type(raw_data))
    print("数据集形状：",raw_data.shape)
    print("一类数据类型：",type(raw_data[0]))
    print("一类数据形状：",raw_data[0].shape)

    print("一张图片的类型：",type(raw_data[0][2]))
    print("一张图片的形状：",raw_data[0][2].shape)
