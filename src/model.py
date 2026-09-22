import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

# Profundidades de extração disponíveis para a content loss. O valor é o fim
# da fatia em vgg19.features (verificado contra a topologia da rede).
#
# A escolha NÃO é neutra: camadas rasas respondem a textura e borda e produzem
# uma loss de magnitude muito maior; camadas profundas respondem a estrutura
# semântica e produzem loss pequena. Como a perda total é
# content_weight * loss_content + adv_weight * loss_GAN, a profundidade
# determina o peso EFETIVO do termo adversarial — ver docs/resolution_metrics.md
# e o experimento comparativo descrito no README.
CAMADAS_VGG = {
    "relu2_2": 9,    # VGG22 de Ledig et al. (2017)
    "relu3_4": 18,
    "relu4_4": 27,
    "relu5_4": 36,   # VGG54 de Ledig et al. (2017)
}

# --- Extrator de Características (Otimizado para Radiografias) ---
class FeatureExtractorVGG(nn.Module):
    def __init__(self, camada="relu3_4"):
        super(FeatureExtractorVGG, self).__init__()
        if camada not in CAMADAS_VGG:
            raise ValueError(
                f"camada VGG desconhecida: {camada!r}. "
                f"Disponíveis: {', '.join(CAMADAS_VGG)}"
            )
        self.camada = camada
        vgg19_model = vgg19(weights=VGG19_Weights.DEFAULT)
        # relu3_4 (padrão do projeto) preserva texturas e bordas estruturais de
        # materiais; relu5_4 é o VGG54 do artigo original da SRGAN.
        fim = CAMADAS_VGG[camada]
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:fim]).eval()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img):
        # ATENÇÃO: 'img' DEVE estar no intervalo [0, 1] — as estatísticas
        # mean/std da ImageNet pressupõem esse domínio. Como o pipeline de
        # treino trabalha em [-1, 1] (Tanh do Gerador / Normalize do dataset),
        # o chamador precisa desnormalizar antes (ver utils.denormalize).
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
    """
    Duas variantes:

    "compacto" (padrão do projeto): 6 convoluções até 256 canais, seguidas de
        AdaptiveAvgPool2d(1). 1,4 M parâmetros.

    "artigo": as 8 convoluções de Ledig et al. (2017), até 512 canais, com
        cabeça densa sobre um mapa 6x6. 23,6 M parâmetros.

    A diferença de custo é menor do que a de parâmetros sugere: medido em
    batch 8 / patch 256, a memória de ATIVAÇÃO é praticamente a mesma (712 MB
    contra 773 MB) — o que cresce são os pesos e os estados do otimizador.

    O AdaptiveAvgPool2d(1) da variante compacta colapsa toda a informação
    espacial num único valor por canal antes da cabeça densa. Para um
    discriminador que deve julgar TEXTURA, isso descarta justamente o sinal de
    interesse; a variante do artigo preserva um mapa 6x6. O pooling adaptativo
    (em vez do flatten direto do artigo) mantém a rede independente do
    --patch-size usado no treino.
    """
    def __init__(self, in_channels=1, variante="compacto"):
        super(Discriminator, self).__init__()
        if variante not in ("compacto", "artigo"):
            raise ValueError(f"variante de Discriminador desconhecida: {variante!r}")
        self.variante = variante

        def discriminator_block(in_f, out_f, stride):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_f),
                nn.LeakyReLU(0.2, inplace=True)
            )

        if variante == "compacto":
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
        else:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                discriminator_block(64, 64, 2),
                discriminator_block(64, 128, 1),
                discriminator_block(128, 128, 2),
                discriminator_block(128, 256, 1),
                discriminator_block(256, 256, 2),
                discriminator_block(256, 512, 1),
                discriminator_block(512, 512, 2),
                nn.AdaptiveAvgPool2d(6),
                nn.Flatten(),
                nn.Linear(512 * 6 * 6, 1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(1024, 1)
            )

    def forward(self, img):
        return self.model(img)