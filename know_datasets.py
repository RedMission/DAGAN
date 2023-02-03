"""
查看数据集结构
"""

import numpy as np
from torchvision.transforms import transforms

from dataset import create_dagan_dataloader
from utils.parser import get_dagan_args

if __name__ == '__main__':
    # 加载数据
    raw_data = np.load("datasets/omniglot_data.npy")
    # raw_data = np.load("datasets/IITDdata_left.npy",allow_pickle=True)
    # print("数据集类型：",type(raw_data))
    # print("数据集形状：",raw_data.shape)
    # print("一类数据类型：",type(raw_data[0]))
    # print("一类数据形状：",raw_data[0].shape)
    #
    # print("一张图片的类型：",type(raw_data[0][2]))
    # print("一张图片的形状：",raw_data[0][2].shape)

    # Load input args
    args = get_dagan_args()
    # 加载数据集路径
    dataset_path = args.dataset_path
    # 加载数据
    # raw_data = np.load(dataset_path, allow_pickle=True).copy()
    # 输入通道数
    in_channels = raw_data.shape[-1]
    # 找图像尺寸
    img_size = args.img_size or raw_data.shape[2]
    # 训练的类别数
    num_training_classes = 1200
    # 测试的类别数
    num_val_classes = args.num_val_classes
    batch_size = args.batch_size
    # 最大？？？
    max_pixel_value = args.max_pixel_value
    should_display_generations = not args.suppress_generations
    mid_pixel_value = max_pixel_value / 2

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (mid_pixel_value,) * in_channels, (mid_pixel_value,) * in_channels
            ),
        ]
    )

    # 创建训练数据加载器
    train_dataloader = create_dagan_dataloader(
        raw_data, num_training_classes, train_transform, batch_size
    )

    print(type(train_dataloader))
    count = 0
    for i, data in enumerate(train_dataloader):
        if i % 50 == 0:
            print("Iteration {}".format(i))
        count+=1
    print(count)
