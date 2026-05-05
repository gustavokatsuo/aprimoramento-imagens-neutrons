import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Importações dos seus módulos locais
from src.model import Generator, Discriminator, FeatureExtractorVGG
from src.data_loader import get_dataloader
from src.utils import calculate_psnr, save_samples, save_model_weights

def train():
    # --- 1. Configurações e Hiperparâmetros ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando treinamento usando: {device}")
    
    # Ajuste os caminhos conforme sua pasta de dados
    DATA_DIR = "data/raw" 
    EPOCHS = 100
    BATCH_SIZE = 8
    LR = 1e-4 # Taxa de aprendizado padrão para SRGAN
    
    # --- 2. Inicialização dos Modelos ---
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    feature_extractor = FeatureExtractorVGG().to(device)
    
    # --- 3. Funções de Custo (Loss) e Otimizadores ---
    # Usamos BCEWithLogitsLoss porque removemos a Sigmoid do Discriminador (Modo Pro)
    criterion_GAN = nn.BCEWithLogitsLoss().to(device)
    # Usamos L1 (Mean Absolute Error) ou MSE para a Content Loss
    criterion_content = nn.MSELoss().to(device) 
    
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.9, 0.999))
    
    # --- 4. Carregamento de Dados ---
    # Se a pasta estiver vazia, o script avisará
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        print(f"AVISO: Nenhuma imagem encontrada em '{DATA_DIR}'. Coloque imagens de teste para rodar.")
        return

    dataloader = get_dataloader(DATA_DIR, batch_size=BATCH_SIZE)
    
    # --- 5. Loop de Treinamento ---
    for epoch in range(EPOCHS):
        for i, batch in enumerate(dataloader):
            # Move as imagens para a GPU
            imgs_lr = batch["lr"].to(device)
            imgs_hr = batch["hr"].to(device)
            
            # Cria tensores de "labels" (1 = Real, 0 = Falso) para o Discriminador
            valid = torch.full((imgs_lr.size(0), 1), 0.9, requires_grad=False).to(device)
            fake = torch.zeros((imgs_lr.size(0), 1), requires_grad=False).to(device)
            
            # -------------------------
            # Treinamento do Gerador (G)
            # -------------------------
            optimizer_G.zero_grad()
            
            # Gera uma imagem de alta resolução a partir da baixa
            gen_hr = generator(imgs_lr)
            
            # Adversarial Loss: O gerador quer que o discriminador ache que a gen_hr é 'valid' (1)
            pred_fake = discriminator(gen_hr)
            loss_GAN = criterion_GAN(pred_fake, valid)
            
            # Content Loss: Compara as características da VGG da imagem gerada vs real
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_features, real_features.detach())
            
            # Perda Total do Gerador (Peso de 1e-3 para a GAN Loss estabiliza o treino)
            loss_G = loss_content + 1e-3 * loss_GAN
            
            loss_G.backward()
            optimizer_G.step()
            
            # -----------------------------
            # Treinamento do Discriminador (D)
            # -----------------------------
            optimizer_D.zero_grad()
            
            # Avalia as imagens reais
            pred_real = discriminator(imgs_hr)
            loss_real = criterion_GAN(pred_real, valid)
            
            # Avalia as imagens falsas geradas (usando detach para não atualizar o Gerador aqui)
            pred_fake = discriminator(gen_hr.detach())
            loss_fake = criterion_GAN(pred_fake, fake)
            
            # Perda Média do Discriminador
            loss_D = (loss_real + loss_fake) / 2
            
            loss_D.backward()
            optimizer_D.step()
            
            # --- 6. Logs e Métricas ---
            if i % 10 == 0:
                # Calcula o PSNR do batch atual (sem calcular gradientes para economizar memória)
                with torch.no_grad():
                    current_psnr = calculate_psnr(gen_hr, imgs_hr).item()
                
                print(f"[Época {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}] "
                      f"[PSNR: {current_psnr:.2f} dB]")
        
        # --- 7. Checkpoints (Fim de cada época) ---
        # Salva amostras visuais e os pesos do modelo
        if (epoch + 1) % 5 == 0 or epoch == 0:
            save_samples(epoch, imgs_lr, imgs_hr, gen_hr)
            save_model_weights(generator, discriminator, epoch)

if __name__ == "__main__":
    train()