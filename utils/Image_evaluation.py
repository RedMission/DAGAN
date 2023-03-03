import torch
import torchvision.transforms as transforms
import numpy as np
from scipy import linalg
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader

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
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def cal_FID(generated_images,real_images):
    # 对生成的图像和真实图像进行预处理 stack拼接
    generated_images = torch.stack([preprocess(np.squeeze(image))
                                    for images_class in generated_images for image in images_class])
    real_images = torch.stack([preprocess(np.squeeze(image))
                               for images_class in real_images for image in images_class])

    # 计算生成的图像的均值和协方差矩阵
    mu_generated = torch.mean(generated_images, dim=0)
    sigma_generated = torch_cov(generated_images, rowvar=False)

    # 计算真实图像的均值和协方差矩阵
    mu_real = torch.mean(real_images, dim=0)
    sigma_real = torch_cov(real_images, rowvar=False)

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
    # 加载预训练的Inception v3模型
    # inception_model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
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


if __name__ == '__main__':
    # 加载数据
    data_name = "IITDdata_left_6"
    num = data_name[-1]
    raw_data = np.load("../datasets/"+ data_name +".npy", allow_pickle=True).copy()

    # 加载生成的图像和真实图像的array
    generated_images = raw_data[:, 0:int(num), ]
    real_images = raw_data[:, int(num):, ]
    # cal_FID(generated_images,real_images) # 越小越相似
    print(generated_images.shape)
    aa = np.reshape(generated_images, (-1,1,84,84), order='C')

    get_inception_score(aa,'cuda')
