import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

def render_img(arr):
    arr = (arr * 0.5) + 0.5
    arr = np.uint8(arr * 255)
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

if __name__ == '__main__':
    # 加载训练好的模型
    g = torch.load("model_path/final_omniglot_generator.pt", map_location=torch.device('cpu'))
    # model.eval()不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化，
    # pytorch框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值
    g.eval()

    # 加载数据
    raw_data = np.load("datasets/omniglot_data.npy")
    # raw_data = np.load("datasets/IITDdata.npy",allow_pickle=True).copy()

    z = torch.randn((1, g.z_dim))
    i = 5
    j = 0
    raw_inp = raw_data[i][j]
    inp = display_transform(raw_inp)
    with torch.no_grad():
        res = g(inp.unsqueeze(0), z)[0]
    # 输入图像
    render_img(inp[0])
    # 生成图像
    render_img(res[0])
