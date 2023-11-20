import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Model
class vae(nn.Module):
    def __init__(self,im_width, im_height, filter_size=[32, 64, 128, 256, 512], lantent_dim=512):
        super(vae, self).__init__()
        self.filter_size = [32, 64, 128, 256, 512]
        self.lantent_dim = 512
        self.im_width = im_width
        self.im_height = im_height

        self.encoder_conv = nn.Sequential(
            self.conv_block(3, self.filter_size[0], (5,5), 1, 'same'),
            self.conv_block(self.filter_size[0], self.filter_size[1], (5,5), 1, 'same'),
            self.conv_block(self.filter_size[1], self.filter_size[2], (5,5), 1, 'same'),
            self.conv_block(self.filter_size[2], self.filter_size[3], (5,5), 1, 'same'),
            self.conv_block(self.filter_size[3], self.filter_size[4], (5,5), 1, 'same'),
        )

        self.encoder_mu = nn.Sequential(
            nn.Linear(in_features=512 * self.im_width // 2 ** (len(self.filter_size) -1)  * self.im_width // 2 ** (len(self.filter_size) -1), out_features=1024),
            nn.Linear(1024, self.lantent_dim))

        self.encoder_var = nn.Sequential(
            nn.Linear(in_features=512 * self.im_width // 2 ** (len(self.filter_size) -1)  * self.im_width // 2 ** (len(self.filter_size) -1), out_features=1024),
            nn.Linear(1024, self.lantent_dim))

        self.decoder = nn.Sequential(
            nn.Linear(self.lantent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.02),
            nn.Linear(1024, 512 * self.im_width // 2 ** (len(self.filter_size) -1)  * self.im_height // 2 ** (len(self.filter_size) -1)),
            nn.BatchNorm1d(512 * self.im_width // 2 ** (len(self.filter_size) -1)  * self.im_height // 2 ** (len(self.filter_size) -1)),
            nn.LeakyReLU(0.02),
        )

        self.decoder_conv = nn.Sequential(
            self.deconv_block(self.filter_size[4], self.filter_size[3], (5,5), 1),
            self.deconv_block(self.filter_size[3], self.filter_size[2], (5,5), 2),
            self.deconv_block(self.filter_size[2], self.filter_size[1], (5,5), 2),
            self.deconv_block(self.filter_size[1], self.filter_size[0], (5,5), 2),
            self.deconv_block(self.filter_size[0], 3, (5,5), 1),
        )



    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.MaxPool2d(2, 2) if out_channels != 512 else nn.Identity(),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.02)
        )

    def deconv_block(self, in_channels, out_channels, kernel_size, stride,padding=1):
        output_padding = 1 if stride == 2 else 0
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.02) if out_channels != 3 else nn.Sigmoid()

        )

    def encoder(self, x):
        x = self.encoder_conv(x)
        x = torch.flatten(x, 1)
        return self.encoder_mu(x), self.encoder_var(x)
        

    def forward(self, x):
        mu, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        z = self.decoder(z)
        z = z.view(-1, self.filter_size[4], 8, 8)
        z = self.decoder_conv(z)

        return z, mu, log_var

