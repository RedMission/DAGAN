import math

import torch
import torchvision.transforms as transforms
import numpy as np
from scipy import linalg
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

# 定义预处理操作
from scipy.stats import entropy

def preprocess(img):
    process = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return process(img).reshape(-1) # 在计算图像的协方差矩阵时，需要将图像展平为一维数组

def torch_cov(m, rowvar=False):
    if rowvar:
        m = m.t()
    m = m - m.mean(dim=1, keepdim=True)
    return 1 / m.size(1) * m.mm(m.t())

# 定义计算弗雷谢特距离的函数
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # Compute the squared Frobenius norm between two covariance matrices.
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f'fid calculation produces singular product; adding {eps} to diagonal of cov estimates'
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # Calculate the trace of the product of covariance matrices.
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=2e-1):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def get_FID(generated_images,real_images):
    # 对生成的图像和真实图像进行预处理 stack拼接
    generated_images = torch.stack([preprocess(np.squeeze(image))
                                    for images_class in generated_images for image in images_class])
    real_images = torch.stack([preprocess(np.squeeze(image))
                               for images_class in real_images for image in images_class])

    # 计算生成的图像的均值和协方差矩阵
    mu_generated = torch.mean(generated_images, dim=0).to(torch.float)
    sigma_generated = torch_cov(generated_images, rowvar=False).to(torch.float)

    # 计算真实图像的均值和协方差矩阵
    mu_real = torch.mean(real_images, dim=0).to(torch.float)
    sigma_real = torch_cov(real_images, rowvar=False).to(torch.float)

    # 计算弗雷谢特距离
    fid = calculate_frechet_distance(mu_generated.numpy(), sigma_generated.numpy(), mu_real.numpy(), sigma_real.numpy())
    print(f'FID: {fid}')
    return
def get_inception_score(images, device, batch_size=32, splits=5):
    """
    计算Inception Score

    Args:
        images: torch.Tensor, 图像数据张量，维度为(N, C, H, W)，范围[0, 1] 初步猜测应该是将所有生成图像（抹去类别）排列输入;
        device: str or torch.device, 使用的设备
        batch_size: int, 计算Inception Score时的批大小
        resize: bool, 是否对图像进行缩放到(299, 299)
        splits: int, Inception Score的拆分次数

    Returns:
        inception_score: float, 计算得到的Inception Score
    """
    # 将所有生成图像（抹去类别）排列输入;
    images = np.reshape(images, (-1,images.shape[-1],images.shape[2],images.shape[3]), order='C')
    # 转换数据类型并将数据缩放到(-1, 1)的范围
    images = images * 2 - 1
    # 加载预训练的Inception v3模型
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    if images.shape[1]==1:
        # 网络原始参数要求图像是3通道,找到需要修改的卷积层并修改其权值
        for name, param in inception_model.named_parameters():
            if 'Conv2d_1a_3x3.conv.weight' in name:
                conv_weight = param.data
                new_conv_weight = torch.mean(conv_weight, dim=1, keepdim=True).repeat(1, 1, 1, 1)
                inception_model.Conv2d_1a_3x3.conv.weight.data = new_conv_weight

    inception_model.to(device)
    inception_model.eval()

    # 定义数据加载器
    data_loader = DataLoader(images, batch_size=batch_size)

    # 计算所有图像的softmax预测值
    preds = []
    for batch in data_loader:
        batch = batch.to(device)
        with torch.no_grad():
            pred = F.softmax(inception_model(batch), dim=1)
        preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    # 计算Inception Score
    scores = []
    for k in range(splits):
        part = preds[k * (preds.shape[0] // splits): (k + 1) * (preds.shape[0] // splits), :]
        p_y = np.mean(part, axis=0)
        scores.append(entropy(p_y, base=2))
    inception_score = np.exp(np.mean(scores))
    print(f'IS: {inception_score}')
    return
def get_SSIM(generated_images,real_images):
    # 假设有num_classes个类别，每个类别生成了num_images_per_class张图像，且真实图像和生成图像的大小均为(H,W,C)
    num_classes = generated_images.shape[0]
    num_images_per_class = generated_images.shape[1]
    ssim_values = torch.zeros(num_classes)
    for i in range(num_classes):
        ssim_sum = 0.0
        for j in range(num_images_per_class):
            # 计算第i个类别下第j张生成图像和真实图像之间的SSIM值
            ssim_sum += ssim(real_images[i,j],
                             generated_images[i,j],
                             multichannel=True)
        # 计算第i个类别下所有图像的平均SSIM值
        ssim_values[i] = ssim_sum / num_images_per_class

    # 计算所有类别下的平均SSIM值作为整个生成图像集合的质量评价指标
    avg_ssim = torch.mean(ssim_values)
    print(f'avg_ssim: {avg_ssim}')
    return
def get_PSNR(generated_images, real_images):
    def psnr2(img1, img2):
        mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
        if mse < 1e-10:
            return 100
        psnr2 = 20 * math.log10(1 / math.sqrt(mse))
        return psnr2
    generated_images = generated_images.reshape(-1,generated_images.shape[2],generated_images.shape[3],generated_images.shape[4])
    real_images = real_images.reshape(-1,real_images.shape[2],real_images.shape[3],real_images.shape[4])
    ret = []
    for i,j in zip(generated_images,real_images):
        ret.append(psnr2(i,j))
    ret = np.array(ret)
    print(f'PSNR_avg: {np.mean(ret)}')

if __name__ == '__main__':
    # 加载数据
    data_name = "IITDdata_left_PSA+SC+W(2)_6.npy"
    # data_name = "PolyUROI_PSA+SC_6.npy"
    # data_name = "IITDdata_left_PSA+SC+MC+W_6.npy"
    num = data_name.split(".")[0].split("_")[-1] # 取位数
    raw_data = np.load("../datasets/"+ data_name, allow_pickle=True).copy()

    # 加载生成的图像和真实图像的array
    generated_images = raw_data[:, 0:int(num), ]
    real_images = raw_data[:, int(num):int(num)*2, ]
    get_FID(generated_images,real_images) # 越小越相似
    get_inception_score(generated_images,'cuda') # 越大越多样
    get_SSIM(generated_images, real_images) #SSIM值越接近1表示生成的图像与真实图像越相似，质量越高
    # get_MMD(generated_images, real_images) #MMD值越大表示两个分布之间差异越大，值越小表示两个分布之间差异越小
    get_PSNR(generated_images, real_images)

