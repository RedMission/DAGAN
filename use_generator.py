import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

def render_img(arr):
    # 归一化
    print("展示变换前：",arr)
    arr = (arr * 0.5) + 0.5
    arr = np.uint8(arr * 255)
    print("展示变换后：",arr)

    # 转换格式为PIL.Image.Image
    img = Image.fromarray(arr, mode='L')
    plt.imshow(img, cmap='gray')
    plt.show()

def display_transform(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(g.dim),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    return transform(img)

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
    :return: 生成的array [150,150,1]
    '''
    # 1.转换成张量
    inp = display_transform(raw_inp)
    # 2.利用模型生成张量
    with torch.no_grad():
        res = g(inp.unsqueeze(0), z)[0]
    # 3.张量转成数组
    test_raw = res.numpy()
    test = test_raw.reshape(test_raw.shape[1], test_raw.shape[2], 1)
    return test
def generate_dataset(generator_sample_num):
    '''
    :param generator_sample_num: 要增加的样本数
    :return: 新的数据集
    '''
    for num in range(generator_sample_num):
        # 0.切割一列出来
        raw_inps = raw_data[:, num,]
        num += 1
        # 1.每个类别的都取出一张进行生成张量
        res_inps = []  # 修改保存类
        for i in range(raw_inps.shape[0]):
            res_inp = generate_arr(raw_inps[i])
            res_inps.append(res_inp)
        # 2.将生成的存到一个矩阵中
        generate_dataset_arr = np.array(res_inps)
        # 3.合并npy 数据集形状： 增加维度，合并
        generate_dataset_arr = generate_dataset_arr[:, np.newaxis]
        generate_dataset = np.concatenate((raw_data, generate_dataset_arr), axis=1)

    return generate_dataset

if __name__ == '__main__':
    # 加载训练好的模型
    g = torch.load("model_path/IITD_20220202_generator.pt", map_location=torch.device('cpu'))
    # model.eval()不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化，
    # pytorch框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值
    g.eval()

    # 加载数据
    # raw_data = np.load("datasets/omniglot_data.npy")
    raw_data = np.load("datasets/IITDdata_right.npy", allow_pickle=True).copy()
    # raw_data = np.load("datasets/Tongji_session1.npy", allow_pickle=True).copy()
    # 噪声
    z = torch.randn((1, g.z_dim))
    # render_img(raw_inp.reshape(150,150))

    generator_sample_num = 3
    new_data = generate_dataset(2)
    print(new_data.shape)

    render_img(new_data[2][0].reshape(150,150))
    render_img(new_data[2][-1].reshape(150,150))



