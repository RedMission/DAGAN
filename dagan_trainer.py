import imageio
import numpy as np
import torch
import time
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from PIL import Image
import PIL
import warnings

from torch.utils.tensorboard import SummaryWriter

# from IPython.core.display import display


class DaganTrainer:
    def __init__(
        self,
        generator,
        discriminator,
        gen_optimizer,
        dis_optimizer,
        batch_size,
        device="cpu",
        gp_weight=10, # Penalty coefficient 按照经验取10
        critic_iterations=5,
        print_every=50,
        num_tracking_images=0,
        save_checkpoint_path=None,
        load_checkpoint_path=None,
        display_transform=None,
        should_display_generations=True,
         # tensorboard
        writer = SummaryWriter('/runs')
    ):
        self.device = device
        self.g = generator.to(device)
        self.g_opt = gen_optimizer
        self.d = discriminator.to(device)
        self.d_opt = dis_optimizer
        self.losses = {"G": [0.0], "D": [0.0], "GP": [0.0], "gradient_norm": [0.0]}
        self.num_steps = 0
        self.epoch = 0
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.num_tracking_images = num_tracking_images
        self.display_transform = display_transform or transforms.ToTensor()
        self.checkpoint_path = save_checkpoint_path
        self.should_display_generations = should_display_generations
        self.writer = writer

        # Track progress of fixed images throughout the training
        # 在整个训练过程中，跟踪固定图像的进度
        self.tracking_images = None
        self.tracking_z = None
        self.tracking_images_gens = None
        # 如果有检查点，从检查点加载模型
        if load_checkpoint_path:
            self.hydrate_checkpoint(load_checkpoint_path)

    def _critic_train_iteration(self, x1, x2):
        """
        判别器更新迭代
        """
        # Get generated data 生成器生成样本
        generated_data = self.sample_generator(x1)
        # 判别真实样本对
        d_real = self.d(x1, x2)
        # 判别生成样本对
        d_generated = self.d(x1, generated_data)

        # Get gradient penalty 	获得梯度惩罚值
        gradient_penalty = self._gradient_penalty(x1, x2, generated_data)
        # 记录梯度惩罚损失
        self.losses["GP"].append(gradient_penalty.item())
        self.writer.add_scalar('train/loss_GP', gradient_penalty.item(),self.num_steps)

        # Create total loss and optimize
        self.d_opt.zero_grad()
        # 计算判别损失、反向传播
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        self.d_opt.step()

        # Record loss 记录判别损失
        self.losses["D"].append(d_loss.item())
        self.writer.add_scalar('train/loss_D', d_loss.item(), self.num_steps)


    def _generator_train_iteration(self, x1):
        """
        生成器更新迭代
        """
        # 梯度清零
        self.g_opt.zero_grad()

        # Get generated data 获取生成数据
        generated_data = self.sample_generator(x1)

        # Calculate loss and optimize 输入判别器计算
        d_generated = self.d(x1, generated_data)
        # 获得生成损失、反向传播
        g_loss = -d_generated.mean()
        g_loss.backward()
        self.g_opt.step()

        # Record loss
        self.losses["G"].append(g_loss.item())
        self.writer.add_scalar('train/loss_G', g_loss.item(), self.num_steps)


    def _gradient_penalty(self, x1, x2, generated_data):
        # Calculate interpolation 计算插值
        # alpha.size()=(x1.shape[0], 1, 1, 1)
        alpha = torch.rand(x1.shape[0], 1, 1, 1)
        # 把一个alpha变成和x2一样形状的tensor
        alpha = alpha.expand_as(x2).to(self.device)
        # （计算差距）生成数据-alpha*(x2-生成)
        interpolated = alpha * x2.data + (1 - alpha) * generated_data.data
        # 转换为计算图的张量
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.d(x1, interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(x1.shape[0], -1)
        self.losses["gradient_norm"].append(gradients.norm(2, dim=1).mean().item())
        self.writer.add_scalar('train/loss_gradient_norm', gradients.norm(2, dim=1).mean().item())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader, val_images):
        for i, data in enumerate(data_loader):
            # 到设定值就进行执行打印
            if i % self.print_every == 0:
                print("Iteration {}".format(i))
                self.print_progress(data_loader, val_images)
            self.num_steps += 1
            x1, x2 = data[0].to(self.device), data[1].to(self.device)
            # 判别器进行迭代（每步都跑）
            self._critic_train_iteration(x1, x2)
            # Only update generator every |critic_iterations| iterations
            # 到达设定要求才进行 生成器迭代
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(x1)

    def train(self, data_loader, epochs, val_images=None, save_training_gif=True):
        '''
        :param data_loader: 训练数据加载器
        :param epochs:
        :param val_images: 测试数据矩阵
        :param save_training_gif:
        :return:
        '''
        if self.tracking_images is None and self.num_tracking_images > 0:
            self.tracking_images = self.sample_val_images(
                self.num_tracking_images // 2, val_images
            )
            self.tracking_images.extend(
                self.sample_train_images(
                    self.num_tracking_images - len(self.tracking_images), data_loader
                )
            )
            self.tracking_images = torch.stack(self.tracking_images).to(self.device)
            # 噪声
            self.tracking_z = torch.randn((self.num_tracking_images, self.g.z_dim)).to(
                self.device
            )
            self.tracking_images_gens = []

        # Save checkpoint once before training to catch errors
        self._save_checkpoint()

        start_time = int(time.time())

        while self.epoch < epochs:
            print("\nEpoch {}".format(self.epoch))
            print(f"Elapsed time: {(time.time() - start_time) / 60:.2f} minutes\n")
            # 传入训练数据加载器和测试数据
            self._train_epoch(data_loader, val_images)
            self.epoch += 1
            self._save_checkpoint()

    def sample_generator(self, input_images, z=None):
        # 生成伪造图像
        if z is None:
            # 生成伪造图像
            z = torch.randn((input_images.shape[0], self.g.z_dim)).to(self.device)
        # 返回生成器对象的生成
        return self.g(input_images, z)

    def render_img(self, arr):
        arr = (arr * 0.5) + 0.5
        arr = np.uint8(arr * 255)
        # 在终端可视化
        # display(Image.fromarray(arr, mode="L").transpose(PIL.Image.TRANSPOSE))

    def sample_train_images(self, n, data_loader):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return [
                self.display_transform(data_loader.dataset.x1_examples[idx]) #转为张量
                for idx in torch.randint(0, len(data_loader.dataset), (n,))
            ]

    def sample_val_images(self, n, val_images):
        if val_images is None:
            return []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return [
                self.display_transform(val_images[idx])
                for idx in torch.randint(0, len(val_images), (n,))
            ]

    def display_generations(self, data_loader, val_images):
        n = 5
        images = self.sample_train_images(n, data_loader) + self.sample_val_images(
            n, val_images
        )
        img_size = images[0].shape[-1]
        images.append(torch.tensor(np.ones((1, img_size, img_size))).float())
        images.append(torch.tensor(np.ones((1, img_size, img_size))).float() * -1)
        self.render_img(torch.cat(images, 1)[0])
        z = torch.randn((len(images), self.g.z_dim)).to(self.device)
        inp = torch.stack(images).to(self.device)
        train_gen = self.g(inp, z).cpu()
        self.render_img(train_gen.reshape(-1, train_gen.shape[-1]))

    def print_progress(self, data_loader, val_images):
        self.g.eval()
        with torch.no_grad():
            if self.should_display_generations:
                self.display_generations(data_loader, val_images)
            if self.num_tracking_images > 0:
                self.tracking_images_gens.append(
                    self.g(self.tracking_images, self.tracking_z).cpu()
                )
        self.g.train()
        print("D: {}".format(self.losses["D"][-1]))
        print("Raw D: {}".format(self.losses["D"][-1] - self.losses["GP"][-1]))
        print("GP: {}".format(self.losses["GP"][-1]))
        print("Gradient norm: {}".format(self.losses["gradient_norm"][-1]))
        if self.num_steps > self.critic_iterations:
            print("G: {}".format(self.losses["G"][-1]))

    def _save_checkpoint(self):
        # 保存检查点
        if self.checkpoint_path is None:
            return
        checkpoint = {
            "epoch": self.epoch,
            "num_steps": self.num_steps,
            "g": self.g.state_dict(),
            "g_opt": self.g_opt.state_dict(),
            "d": self.d.state_dict(),
            "d_opt": self.d_opt.state_dict(),
            "tracking_images": self.tracking_images,
            "tracking_z": self.tracking_z,
            "tracking_images_gens": self.tracking_images_gens,
        }
        torch.save(checkpoint, self.checkpoint_path)

    def hydrate_checkpoint(self, checkpoint_path):
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.epoch = checkpoint["epoch"]
        self.num_steps = checkpoint["num_steps"]

        self.g.load_state_dict(checkpoint["g"])
        self.g_opt.load_state_dict(checkpoint["g_opt"])
        self.d.load_state_dict(checkpoint["d"])
        self.d_opt.load_state_dict(checkpoint["d_opt"])

        self.tracking_images = checkpoint["tracking_images"]
        self.tracking_z = checkpoint["tracking_z"]
        self.tracking_images_gens = checkpoint["tracking_images_gens"]
