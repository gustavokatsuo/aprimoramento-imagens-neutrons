import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Importações dos seus módulos locais
from src.model import Generator, Discriminator, FeatureExtractorVGG
from src.data_loader import get_dataloader
from src.utils import (calculate_psnr, save_samples, save_model_weights,
                       save_checkpoint, load_checkpoint, denormalize)

def parse_args():
    """
    Configuração via linha de comando (substitui as constantes fixas de antes).
    Os defaults reproduzem os valores originais do script.
    """
    parser = argparse.ArgumentParser(description="Treinamento SRGAN para radiografias de nêutrons")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Pasta com as radiografias de treino (TIFF 16-bit, FITS, PNG, JPG)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Número TOTAL de épocas (pré-treino + adversarial)")
    parser.add_argument("--pretrain-epochs", type=int, default=5,
                        help="Épocas iniciais só com loss de conteúdo MSE pixel-a-pixel "
                             "(protocolo SRGAN, Ledig et al. 2017); 0 desativa")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Taxa de aprendizado padrão para SRGAN")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="Tamanho do recorte HR para treino")
    parser.add_argument("--lr-scale", type=int, default=4,
                        help="Fator de super-resolução (deve casar com o Gerador: 4x)")
    parser.add_argument("--adv-weight", type=float, default=1e-3,
                        help="Peso da GAN loss na perda total do Gerador")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Norma máxima do gradiente (0 = sem clipping)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semente para reprodutibilidade")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None,
                        help="Caminho de um checkpoint salvo (weights/checkpoint_last.pth) para retomar")
    parser.add_argument("--sample-interval", type=int, default=5,
                        help="Intervalo (épocas) para salvar amostras, pesos e validação de bordas")
    parser.add_argument("--weights-dir", type=str, default="weights")
    parser.add_argument("--fits-normalization", type=str, default="minmax",
                        choices=["minmax", "range", "none"],
                        help="Normalização para arquivos .fits (ver data_loader.load_image_as_array)")
    parser.add_argument("--fits-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Faixa explícita (lo hi) quando --fits-normalization=range")
    return parser.parse_args()

def set_seed(seed):
    """Controle de semente para reprodutibilidade (python, numpy, torch, cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(args):
    # --- 1. Configurações e Hiperparâmetros ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando treinamento usando: {device}")

    set_seed(args.seed)

    # --- 2. Inicialização dos Modelos ---
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    feature_extractor = FeatureExtractorVGG().to(device)

    # --- 3. Funções de Custo (Loss) e Otimizadores ---
    # Usamos BCEWithLogitsLoss porque removemos a Sigmoid do Discriminador (Modo Pro)
    criterion_GAN = nn.BCEWithLogitsLoss().to(device)
    # Usamos L1 (Mean Absolute Error) ou MSE para a Content Loss
    criterion_content = nn.MSELoss().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # --- 4. Carregamento de Dados ---
    # Se a pasta estiver vazia, o script avisará
    if not os.path.exists(args.data_dir) or len(os.listdir(args.data_dir)) == 0:
        print(f"AVISO: Nenhuma imagem encontrada em '{args.data_dir}'. Coloque imagens de teste para rodar.")
        return

    dataloader = get_dataloader(args.data_dir, batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                patch_size=args.patch_size, lr_scale=args.lr_scale,
                                fits_normalization=args.fits_normalization,
                                fits_range=args.fits_range)

    # --- Retomada de checkpoint (opcional) ---
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, generator, discriminator,
                                      optimizer_G, optimizer_D, device)

    # --- 5. Loop de Treinamento ---
    for epoch in range(start_epoch, args.epochs):
        # Fase de pré-treino: só a content loss (MSE) treina o Gerador, sem
        # Discriminador — evita que o D domine antes do G aprender o básico
        pretraining = epoch < args.pretrain_epochs

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

            if pretraining:
                # Pré-treino: MSE pixel-a-pixel direto (sem VGG, sem adversarial)
                loss_content = criterion_content(gen_hr, imgs_hr)
                loss_GAN = torch.zeros((), device=device)
                loss_G = loss_content
            else:
                # Adversarial Loss: O gerador quer que o discriminador ache que a gen_hr é 'valid' (1)
                pred_fake = discriminator(gen_hr)
                loss_GAN = criterion_GAN(pred_fake, valid)

                # Content Loss: Compara as características da VGG da imagem gerada vs real.
                # CORREÇÃO: a VGG espera entrada em [0, 1] (estatísticas ImageNet),
                # mas gen_hr/imgs_hr estão em [-1, 1] — denormalize antes de extrair
                gen_features = feature_extractor(denormalize(gen_hr))
                real_features = feature_extractor(denormalize(imgs_hr))
                loss_content = criterion_content(gen_features, real_features.detach())

                # Perda Total do Gerador (Peso de 1e-3 para a GAN Loss estabiliza o treino)
                loss_G = loss_content + args.adv_weight * loss_GAN

            loss_G.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip)
            optimizer_G.step()

            # -----------------------------
            # Treinamento do Discriminador (D)
            # -----------------------------
            if pretraining:
                # D não é atualizado no pré-treino
                loss_D = torch.zeros(())
            else:
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
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(discriminator.parameters(), args.grad_clip)
                optimizer_D.step()

            # --- 6. Logs e Métricas ---
            if i % 10 == 0:
                # Calcula o PSNR do batch atual (sem gradientes, para economizar memória)
                with torch.no_grad():
                    current_psnr = calculate_psnr(gen_hr, imgs_hr).item()

                phase = "PRÉ" if pretraining else "GAN"
                print(f"[{phase}][Época {epoch}/{args.epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}] "
                      f"[PSNR: {current_psnr:.2f} dB]")

        # --- 7. Checkpoints (Fim de cada época) ---
        # Checkpoint retomável (modelos + otimizadores + época + RNG) toda época
        save_checkpoint(os.path.join(args.weights_dir, "checkpoint_last.pth"),
                        epoch, generator, discriminator, optimizer_G, optimizer_D)

        # Salva amostras visuais e os pesos do modelo
        if (epoch + 1) % args.sample_interval == 0 or epoch == 0:
            save_samples(epoch, imgs_lr, imgs_hr, gen_hr)
            save_model_weights(generator, discriminator, epoch, save_dir=args.weights_dir)

if __name__ == "__main__":
    train(parse_args())
