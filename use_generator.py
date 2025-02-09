import time

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt, pyplot


def render_img(arr):
    # 归一化
    arr = (arr * 0.5) + 0.5
    arr = np.uint8(arr * 255)

    # 转换格式为PIL.Image.Image
    img = Image.fromarray(arr, mode='L')
    plt.imshow(img, cmap='gray')
    plt.show()

def display_transform(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(g.dim), # 调整格式
        transforms.Resize(g.dim), # 调整格式
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    return transform(img)

def norm_transform(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(g.dim), # 调整格式
        transforms.Resize(g.dim), # 调整格式
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    img = transform(img).numpy()
    img = img.reshape(img.shape[1], img.shape[2], 1)
    return  img


def display_generations(self, data_loader):
    train_idx = torch.randint(0, len(data_loader.dataset), (1,))[0]
    train_img = display_transform(data_loader.dataset.x1_examples[train_idx])
    self.render_img(train_img[0])

    z = torch.randn((1, self.g.z_dim)).to(self.device)
    inp = train_img.unsqueeze(0).to(self.device)
    train_gen = self.g(inp, z).cpu()[0]
    self.render_img(train_gen[0])

def generate_arr(raw_inp):
    '''
    :param raw_inp: 输入的array
    :return: 生成的格式为array [150,150,1]的模拟图像
    '''
    # 1.转换成张量
    inp = display_transform(raw_inp).to(device)
    # 2.利用模型生成张量
    with torch.no_grad():
        res = g(inp.unsqueeze(0), z)[0]
    # 3.张量转成数组
    test_raw = res.to('cpu').numpy()
    test = test_raw.reshape(test_raw.shape[1], test_raw.shape[2], 1)
    return test
def generate_dataset(generator_sample_num):
    '''
    :param generator_sample_num: 要增加的样本数
    :return: 新的数据集
    '''
    generate_dataset = np.array([generate_arr(c) for c in raw_data[:,0,]])[:, np.newaxis]
    col_num = raw_data.shape[1]
    for num in range(1,generator_sample_num):
        # 0.切割一列出来
        raw_inps = raw_data[:,num%col_num,]
        # # 1.每个类别的都取出一张进行生成张量
        #  取单个图片放入训练好的模型
        # # 2.将生成的存到一个矩阵中
        generate_dataset_arr = np.array([generate_arr(c) for c in raw_inps])
        # 3.合并npy 数据集形状：np.newaxis增加维度，合并【修改原始矩阵大小！】
        generate_dataset_arr = generate_dataset_arr[:, np.newaxis]
        # print(generate_dataset_arr.shape) # (230, 1, 150, 150, 1)
        generate_dataset = np.concatenate((generate_dataset, generate_dataset_arr), axis=1)

    for num in range(0,raw_data.shape[1]):
        # 修改原始图像形状
        raw_inps = raw_data[:, num, ]
        generate_dataset_arr = np.array([norm_transform(c) for c in raw_inps])[:, np.newaxis]
        generate_dataset = np.concatenate((generate_dataset, generate_dataset_arr), axis=1) # 所以生成的是前面几列

    return generate_dataset

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # 加载训练好的模型
    # model_name = "IITD(PSA_2+Unet+SC)_generator.pt"
    model_name = "IITD(Z3_PSA_2+Unet+SC)_generator.pt"
    g = torch.load("model_path/" + model_name, map_location=torch.device(device))
    # model.eval()不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化，
    # pytorch框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值
    g.eval()
    tmp = filter(lambda x: x.requires_grad, g.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', num)
    # 加载数据
    data_name = "IITDdata_left"
    # data_name = "Tongji_session2"
    # data_name = "PolyUROI")
    raw_data = np.load("datasets/"+ data_name +".npy", allow_pickle=True).copy()
    # print(g.z_dim) # 100
    # print(g.dim) # 84
    # 噪声
    z = torch.randn((1, g.z_dim)).to(device)
    generator_sample_num = 6
    start = time.time()
    new_data = generate_dataset(generator_sample_num)
    end = time.time()
    print("times:",(end - start)/(new_data.shape[0]*new_data.shape[1]))
    print("已生成扩充数据：",new_data.shape)
    np.save('datasets/'+data_name+"_Z3(3)_PSA2+SC+W_"+str(generator_sample_num)+".npy", new_data)





