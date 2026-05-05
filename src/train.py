import torch
import torch.nn as nn
import torch.optim as optim
import os

# Importando as métricas "Pro" oficiais do PyTorch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# Importações dos seus módulos locais
from src.model import Generator, Discriminator, FeatureExtractorVGG
from src.data_loader import get_dataloader
from src.utils import denormalize, save_samples, save_model_weights

def train():
    # --- 1. Configurações ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando treinamento usando: {device}")
    
    DATA_DIR = "data/raw" 
    EPOCHS = 100
    BATCH_SIZE = 8
    LR = 1e-4 
    
    # --- Inicialização das Métricas Científicas ---
    # data_range=1.0 porque vamos desnormalizar as imagens de [-1, 1] para [0, 1] antes de medir
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # --- 2. Inicialização dos Modelos ---
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    feature_extractor = FeatureExtractorVGG().to(device)
    
    # --- 3. Funções de Custo ---
    criterion_GAN = nn.BCEWithLogitsLoss().to(device)
    criterion_content = nn.MSELoss().to(device) 
    
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.9, 0.999))
    
    # --- 4. Carregamento de Dados ---
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        print(f"AVISO: Nenhuma imagem encontrada em '{DATA_DIR}'. Coloque imagens de teste para rodar.")
        return

    dataloader = get_dataloader(DATA_DIR, batch_size=BATCH_SIZE)
    
    # --- 5. Loop de Treinamento ---
    for epoch in range(EPOCHS):
        for i, batch in enumerate(dataloader):
            imgs_lr = batch["lr"].to(device)
            imgs_hr = batch["hr"].to(device)
            
            valid = torch.ones((imgs_lr.size(0), 1), requires_grad=False).to(device)
            fake = torch.zeros((imgs_lr.size(0), 1), requires_grad=False).to(device)
            
            # --- Treino do Gerador ---
            optimizer_G.zero_grad()
            gen_hr = generator(imgs_lr)
            
            pred_fake = discriminator(gen_hr)
            loss_GAN = criterion_GAN(pred_fake, valid)
            
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_features, real_features.detach())
            
            loss_G = loss_content + 1e-3 * loss_GAN
            loss_G.backward()
            optimizer_G.step()
            
            # --- Treino do Discriminador ---
            optimizer_D.zero_grad()
            pred_real = discriminator(imgs_hr)
            loss_real = criterion_GAN(pred_real, valid)
            
            pred_fake = discriminator(gen_hr.detach())
            loss_fake = criterion_GAN(pred_fake, fake)
            
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()
            
            # --- 6. Logs e Métricas de Validação ---
            if i % 10 == 0:
                with torch.no_grad():
                    # Desnormaliza para [0, 1] apenas para o cálculo exato das métricas
                    gen_eval = denormalize(gen_hr)
                    real_eval = denormalize(imgs_hr)
                    
                    current_psnr = psnr_metric(gen_eval, real_eval).item()
                    current_ssim = ssim_metric(gen_eval, real_eval).item()
                
                print(f"[Época {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}] "
                      f"[PSNR: {current_psnr:.2f} dB] [SSIM: {current_ssim:.4f}]")
        
        # --- 7. Checkpoints ---
        if (epoch + 1) % 5 == 0 or epoch == 0:
            save_samples(epoch, imgs_lr, imgs_hr, gen_hr)
            save_model_weights(generator, discriminator, epoch)

if __name__ == "__main__":
    train()