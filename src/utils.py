import torch
import numpy as np
from torchvision.utils import save_image
import os

def denormalize(tensors):
    """
    Converte tensores de [-1, 1] de volta para [0, 1] para visualização e cálculo de métricas.
    """
    return (tensors + 1.0) / 2.0

def calculate_psnr(img1, img2):
    """
    Calcula o Peak Signal-to-Noise Ratio (PSNR).
    Alvo da IC: Melhoria mínima de 5 dB.
    """
    # 1. Desnormaliza as imagens para garantir o domínio [0, 1] e o cálculo correto do MAX_I
    img1_norm = denormalize(img1)
    img2_norm = denormalize(img2)
    
    # 2. Calcula o MSE nas imagens corrigidas
    mse = torch.mean((img1_norm - img2_norm) ** 2)
    
    if mse == 0:
        return float('inf')
        
    # MAX_I é implicitamente 1.0 agora
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def save_samples(epoch, lr_imgs, hr_imgs, fake_imgs, save_dir="samples"):
    """
    Salva uma grade de comparação: Baixa Resolução | Gerada | Original (Alta Resolução).
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Desnormaliza para [0, 1]
    lr_imgs = denormalize(lr_imgs)
    fake_imgs = denormalize(fake_imgs)
    hr_imgs = denormalize(hr_imgs)
    
    # Redimensiona a LR para o tamanho da HR para comparação visual direta
    lr_resized = torch.nn.functional.interpolate(lr_imgs, size=hr_imgs.shape[2:], mode='bicubic')
    
    # Concatena as imagens horizontalmente (Batch de 1 para o exemplo)
    comparison = torch.cat((lr_resized[0:1], fake_imgs[0:1], hr_imgs[0:1]), 3)
    
    save_path = os.path.join(save_dir, f"epoch_{epoch}.png")
    ssave_image(comparison, save_path, normalize=True, scale_each=True)
    print(f"Amostras da época {epoch} salvas em {save_path}")

def save_model_weights(generator, discriminator, epoch, save_dir="weights"):
    """
    Salva os pesos para posterior inferência ou integração no sistema do IEA-R1.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Correção do typo 'state_state_dict' -> 'state_dict'
    torch.save(generator.state_dict(), os.path.join(save_dir, f"gen_epoch_{epoch}.pth"))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, f"disc_epoch_{epoch}.pth"))