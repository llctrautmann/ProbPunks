import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from math import log2, sqrt
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from scipy.stats import truncnorm
from tqdm import tqdm
import os
import sys

# Model

factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]


class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,gain=2):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels + kernel_size ** 2)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0],1,1)


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x * torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
    

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,use_pixel_norm=True):
        super(ConvBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels,out_channels)
        self.conv2 = WSConv2d(out_channels,out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()
        self.use_pixel_norm = use_pixel_norm

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pixel_norm else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pixel_norm else x
        return x
    

class Generator(nn.Module):
    def __init__(self, z_dim,in_channels,img_channels=3):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0), # 1 x 1 -> 4 x 4
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1,padding=0)
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList([self.initial_rgb])

        for i in range(len(factors) - 1):
            # factors[i] = factors[i] + 1
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1,padding=0))
            

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, z, alpha, steps):
        out = self.initial(z)

        if steps == 0:
            return self.initial_rgb(out)
        
        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode='nearest')
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out) # <-- potential bug note: look here if steps is correct


        return self.fade_in(alpha, final_upscaled, final_out)



class Discriminator(nn.Module):
    def __init__(self,in_channels,img_channels=3):
        super(Discriminator, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList()
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1,0,-1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c,use_pixel_norm=False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in_c, kernel_size=1, stride=1, padding=0))


        self.initial_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1,padding=0) # potential bug note: look here if steps is correct
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2,stride=2)

        # Block for 4 x 4 resolution
        self.final_block = nn.Sequential(
            WSConv2d(in_channels+1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, stride=1 , padding=0)
        )

    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled
    
    def minibatch_std(self, x):
        batch_stats = torch.std(x,dim=0).mean().repeat(x.shape[0],1,x.shape[2],x.shape[3])
        return torch.cat([x,batch_stats],dim=1)

    def forward(self, x, alpha, steps):
        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0],-1)
            
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1,len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0],-1)


# Training Loop Hyperparameters
START_TRAIN_IMG_SIZE = 4
DATASET = 'cryptopunks'
CHECKPOINT_GEN = '../runs/ProGAN/checkpoints/generator.pth'
CHECKPOINT_DIS = '../runs/ProGAN/checkpoints/discriminator.pth'

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
SAVE_MODEL = False
LOAD_MODEL = False
LEARNING_RATE = 3e-4
Z_DIM = 512  # should be 512 in original paper
IN_CHANNELS = 512  # should be 512 in original paper
CRITIC_ITERATIONS = 1
BATCH_SIZE = [32, 16, 16, 8, 8, 8, 8, 8, 4]
CHANNELS = 3
LAMBDA_GP = 10

PROGRESSIVE_EPOCHS = [50] * len(BATCH_SIZE)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 1


def plot_to_tensorboard(writer, loss_critic, loss_gen,real,fake, tb_step, run_name):
    writer.add_scalar(f'loss_critic', loss_critic, tb_step)
    writer.add_scalar(f'loss_gen', loss_gen, tb_step)

    with torch.no_grad():
        img_grid_fake = torchvision.utils.make_grid(fake[:4], normalize=True)
        img_grid_real = torchvision.utils.make_grid(real[:4], normalize=True)

        writer.add_image(f'img_grid_fake', img_grid_fake, tb_step)
        writer.add_image(f'img_grid_real', img_grid_real, tb_step)

def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="mps")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def generate_examples(gen, steps, truncation=0.7, n=100):
    """
    Tried using truncation trick here but not sure it actually helped anything, you can
    remove it if you like and just sample from torch.randn
    """
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.tensor(truncnorm.rvs(-truncation, truncation, size=(1, Z_DIM, 1, 1)), device=DEVICE, dtype=torch.float32)
            img = gen(noise, alpha, steps)
            save_image(img*0.5+0.5, f"saved_examples/img_{i}.png")
    gen.train()


def get_loader(image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNELS)], [0.5 for _ in range(CHANNELS)]),
        ])

    batch_size = BATCH_SIZE[int(log2(image_size / 4))]

    print(batch_size,)
    dataset = datasets.ImageFolder(root='./data/cryptopunks/',transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=NUM_WORKERS,pin_memory=True)

    return loader, dataset


def train(critic,
            gen,
            loader,
            dataset,
            step,
            alpha,
            opt_critic,
            opt_gen,
            tensorboard_step,
            writer,
            run_name
            ):
    loop = tqdm(loader, leave=True)
    for idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]

        noise = torch.randn(cur_batch_size, Z_DIM, 1, 1, device=DEVICE)

        fake = gen(noise, alpha, step)
        critic_real = critic(real,alpha,step)
        critic_fake = critic(fake.detach(),alpha,step)

        gp = gradient_penalty(critic, real, fake,alpha, step,device=DEVICE)

        loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp + (0.001 * torch.mean(critic_real.pow(2)))

        opt_critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()


        # Train Generator max E[critic(gen_fake)]
        gen_fake = critic(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        alpha  += cur_batch_size / (PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset) 
        alpha = min(1, alpha)


        if idx % 100 == 0:
            with torch.no_grad():
                fixed_fakes = gen(FIXED_NOISE, alpha, step) * 0.5 + 0.5

            plot_to_tensorboard(writer,
                                loss_critic.item(),
                                loss_gen.item(),
                                real.detach(),
                                fixed_fakes.detach(),
                                tb_step=tensorboard_step,
                                run_name=run_name)
            
            tensorboard_step += 1

    return tensorboard_step, alpha




def main(identifier='',load_gen='', load_cri='',run_name=''):
    gen = Generator(z_dim=Z_DIM, in_channels=IN_CHANNELS, img_channels=CHANNELS).to(DEVICE)
    critic = Discriminator(in_channels=IN_CHANNELS, img_channels=CHANNELS).to(DEVICE)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE,betas=(0.0,0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE,betas=(0.0,0.99))


    # Tensorboard
    writer_fake = SummaryWriter(f'./data/runs/ProGAN/{run_name}/fake')
    writer_real = SummaryWriter(f'./data/runs/ProGAN/{run_name}/real')
    writer = SummaryWriter(f'./data/runs/ProGAN/{run_name}/log')

    if LOAD_MODEL:
        load_checkpoint(
            load_gen, gen, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            load_cri, critic, opt_critic, LEARNING_RATE,
        )


    gen.train()
    critic.train()


    tensorboard_step = 0
    step = int(log2(START_TRAIN_IMG_SIZE / 4))

    for num_epochs in PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5 
        loader, dataset = get_loader(4 * 2 ** step)
        print(f"Current image size: {4 * 2 ** step}")

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            tensorboard_step, alpha = train(
                critic,
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                tensorboard_step,
                writer,
                run_name
            )

            if SAVE_MODEL:
                save_checkpoint(gen, opt_gen, filename=CHECKPOINT_GEN + f'step {step} epoch {epoch}')
                save_checkpoint(critic, opt_critic, filename=CHECKPOINT_DIS + f'step {step} epoch {epoch}')

        step += 1


if __name__ == "__main__":
    output_from_bash = sys.argv[1]
    print(f"Run tag: {output_from_bash}")
    main(load_gen='',load_cri='', run_name=output_from_bash)