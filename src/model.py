import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

# --- Extrator de Características (Otimizado para Radiografias) ---
class FeatureExtractorVGG(nn.Module):
    def __init__(self):
        super(FeatureExtractorVGG, self).__init__()
        vgg19_model = vgg19(weights=VGG19_Weights.DEFAULT)
        # Modo Pro: Extrai apenas até a camada 18 (relu3_4) 
        # Preserva melhor texturas e bordas estruturais de materiais, ignorando semântica profunda
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18]).eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img):
        img_rgb = img.repeat(1, 3, 1, 1)
        img_norm = (img_rgb - self.mean) / self.std
        return self.feature_extractor(img_norm)

# --- Bloco Residual ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

# --- Gerador (SRResNet com Upsampling 4x) ---
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_res_blocks=16):
        super(Generator, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        
        self.middle = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # Modo Pro: Blocos de Upsampling (PixelShuffle) para escalar 4x (2x e depois 2x)
        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), # Transforma 256 canais em 64 canais com 2x a resolução espacial
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), # Transforma 256 canais em 64 canais com +2x a resolução espacial
            nn.PReLU()
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        initial = self.initial(x)
        x = self.res_blocks(initial)
        x = self.middle(x)
        # Skip connection ocorre ANTES do upsampling
        x = x + initial 
        x = self.upsampling(x)
        return self.final(x)

# --- Discriminador (Patch/Logit) ---
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        def discriminator_block(in_f, out_f, stride):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_f),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            discriminator_block(64, 64, 2),
            discriminator_block(64, 128, 1),
            discriminator_block(128, 128, 2),
            discriminator_block(128, 256, 1),
            discriminator_block(256, 256, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
            # Sigmoid removida para uso de BCEWithLogitsLoss no treino!
        )

    def forward(self, img):
        return self.model(img)