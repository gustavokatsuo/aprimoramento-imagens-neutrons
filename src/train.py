import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Importações dos seus módulos locais
from src.model import (Generator, Discriminator, FeatureExtractorVGG,
                       CAMADAS_VGG)
from src.data_loader import (get_dataloader, get_step_edge_loader,
                             list_supported_images, SUPPORTED_EXTENSIONS)
from src.utils import (calculate_psnr, calculate_ssim, save_samples,
                       save_model_weights, save_checkpoint, load_checkpoint,
                       log_epoch_csv, save_run_config, denormalize,
                       mtf10_from_edge)

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
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch TOTAL; com múltiplas GPUs o DataParallel "
                             "divide este valor entre elas")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Taxa de aprendizado padrão para SRGAN")
    parser.add_argument("--lr-decay-epochs", type=int, nargs="+", default=None,
                        metavar="EPOCA",
                        help="Épocas em que a taxa de aprendizado é multiplicada "
                             "por --lr-decay-gamma (ex.: 50 80). Omitido, a taxa "
                             "fica constante")
    parser.add_argument("--lr-decay-gamma", type=float, default=0.1,
                        help="Fator aplicado à taxa de aprendizado em cada época "
                             "de --lr-decay-epochs")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="Tamanho do recorte HR para treino")
    parser.add_argument("--patches-per-image", type=int, default=1,
                        help="Recortes aleatórios que cada radiografia rende por "
                             "época. Com 1 (padrão histórico), uma época tem "
                             "tantas amostras quanto imagens — para treinar de "
                             "verdade use algo na casa das dezenas ou centenas")
    parser.add_argument("--lr-scale", type=int, default=4,
                        help="Fator de super-resolução (deve casar com o Gerador: 4x)")
    parser.add_argument("--adv-weight", type=float, default=1e-3,
                        help="Peso da GAN loss na perda total do Gerador")
    parser.add_argument("--vgg-layer", type=str, default="relu3_4",
                        choices=sorted(CAMADAS_VGG),
                        help="Profundidade da VGG na content loss. relu5_4 é o "
                             "VGG54 do artigo da SRGAN; camadas rasas produzem "
                             "loss de magnitude muito maior e reduzem o peso "
                             "efetivo do termo adversarial")
    parser.add_argument("--content-weight", type=float, default=1.0,
                        help="Peso da content loss da VGG. Ledig et al. reescalam "
                             "as features por 1/12.75, equivalente a 0.006 aqui; "
                             "1.0 mantém o comportamento histórico do projeto")
    parser.add_argument("--discriminator", type=str, default="compacto",
                        choices=["compacto", "artigo"],
                        help="Arquitetura do Discriminador: 'compacto' (6 convs "
                             "até 256 canais) ou 'artigo' (8 convs até 512, "
                             "Ledig et al. 2017)")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Norma máxima do gradiente (0 = sem clipping)")
    parser.add_argument("--amp", action="store_true",
                        help="Precisão mista (float16) na GPU: reduz a memória de "
                             "ativação e acelera o treino. Ignorado sem CUDA")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semente para reprodutibilidade")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-data-parallel", action="store_true",
                        help="Não usa múltiplas GPUs mesmo quando há mais de uma "
                             "no nó (útil para isolar um experimento por GPU)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Caminho de um checkpoint salvo (weights/checkpoint_last.pth) para retomar")
    parser.add_argument("--sample-interval", type=int, default=5,
                        help="Intervalo (épocas) para salvar amostras, pesos e validação de bordas")
    parser.add_argument("--log-file", type=str, default="logs/training_log.csv",
                        help="CSV append-only com as métricas por época")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Pasta desta execução: config, logs, pesos e amostras "
                             "vão todos para dentro dela (ex.: runs/$SLURM_JOB_ID). "
                             "Omitida, cada saída usa seu próprio argumento")
    parser.add_argument("--weights-dir", type=str, default="weights")
    parser.add_argument("--samples-dir", type=str, default="samples",
                        help="Pasta das grades de comparação LR|SR|HR salvas por época")
    parser.add_argument("--step-edge-dir", type=str, default="data/step_edges",
                        help="Pasta com phantoms de borda para validação MTF (opcional)")
    parser.add_argument("--fits-normalization", type=str, default="minmax",
                        choices=["minmax", "range", "none"],
                        help="Normalização para arquivos .fits (ver data_loader.load_image_as_array)")
    parser.add_argument("--fits-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Faixa explícita (lo hi) quando --fits-normalization=range")
    args = parser.parse_args()

    # Validações que precisam falhar AQUI, e não no meio do treino:
    # o Gerador reconstrói 4x de forma fixa (dois PixelShuffle de 2x em
    # model.py); com outro fator o erro só aparece no broadcast da loss, com
    # uma mensagem de shape que não aponta para a causa.
    if args.lr_scale != 4:
        parser.error(f"--lr-scale={args.lr_scale} não é suportado: o Gerador "
                     f"reconstrói 4x (dois PixelShuffle de 2x em model.py). "
                     f"Outro fator exige alterar a arquitetura.")

    # Idem para a faixa dos FITS: sem isto o erro só surge ao ler o primeiro
    # .fits, possivelmente minutos depois de o job começar.
    if args.fits_normalization == "range" and args.fits_range is None:
        parser.error("--fits-normalization=range exige --fits-range LO HI.")

    # --run-dir agrupa tudo de uma execução num lugar só. É o modo pensado para
    # o cluster: cada job escreve em runs/<id> e nenhum pisa no outro.
    if args.run_dir:
        args.weights_dir = os.path.join(args.run_dir, "weights")
        args.log_file = os.path.join(args.run_dir, "logs", "training_log.csv")
        args.samples_dir = os.path.join(args.run_dir, "samples")

    return args

def set_seed(seed):
    """Controle de semente para reprodutibilidade (python, numpy, torch, cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def validate_step_edges(generator, loader, device, epoch, log_path="logs/step_edge_mtf.csv"):
    """
    Validação com os phantoms de borda: mede a frequência de corte (MTF10) da
    imagem super-resolvida, da referência HR e da MESMA entrada LR levada ao
    tamanho HR por interpolação bicúbica.

    A curva bicúbica é a linha de base que torna o número interpretável: ela é
    o que se obtém sem nenhum aprendizado. MTF10 da SR acima da bicúbica é
    evidência de recuperação de frequência; igual ou abaixo significa que a
    rede não entregou resolução além da interpolação, por melhor que estejam
    PSNR e SSIM (ver docs/resolution_metrics.md).

    Cada uma das três medidas é feita de forma independente, com seu próprio
    estado registrado: a falha de uma não descarta as outras, e a linha vai
    para o CSV de qualquer maneira, para que a lacuna fique visível.
    """
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            imgs_lr = batch["lr"].to(device)
            imgs_hr = batch["hr"].to(device)
            name = batch["name"][0]

            gen_hr = generator(imgs_lr)

            # Interpolação bicúbica da LR no domínio do modelo, [-1, 1];
            # o clamp contém o overshoot característico do bicúbico.
            bic_hr = torch.nn.functional.interpolate(
                imgs_lr, size=imgs_hr.shape[2:], mode="bicubic",
                align_corners=False).clamp(-1.0, 1.0)

            def em_2d(tensor):
                return denormalize(tensor).clamp(0, 1).squeeze().cpu().numpy()

            medidas, pendencias = {}, []
            for rotulo, tensor in (("lr_bicubic", bic_hr),
                                   ("sr", gen_hr),
                                   ("hr", imgs_hr)):
                valor, estado = mtf10_from_edge(em_2d(tensor))
                medidas[rotulo] = valor
                if estado != "ok":
                    pendencias.append(f"{rotulo}={estado}")

            num = lambda v: "" if math.isnan(v) else f"{v:.4f}"
            log_epoch_csv(log_path, {
                "epoch": epoch,
                "sample": name,
                "mtf10_lr_bicubic": num(medidas["lr_bicubic"]),
                "mtf10_sr": num(medidas["sr"]),
                "mtf10_hr": num(medidas["hr"]),
                "psnr_lr_bicubic": f"{calculate_psnr(bic_hr, imgs_hr).item():.3f}",
                "psnr_sr": f"{calculate_psnr(gen_hr, imgs_hr).item():.3f}",
                "status": ";".join(pendencias) if pendencias else "ok",
            })

            txt = lambda v: "n/d" if math.isnan(v) else f"{v:.3f}"
            print(f"[Bordas] {name}: MTF10 bicúbica={txt(medidas['lr_bicubic'])} | "
                  f"SR={txt(medidas['sr'])} | HR={txt(medidas['hr'])} ciclos/px"
                  + (f"  [{';'.join(pendencias)}]" if pendencias else ""), flush=True)
    generator.train()

def train(args):
    # --- 1. Configurações e Hiperparâmetros ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando treinamento usando: {device}", flush=True)

    set_seed(args.seed)

    # Procedência ANTES de qualquer treino: se o job morrer no meio, o registro
    # do que foi pedido já está gravado.
    save_run_config(os.path.join(os.path.dirname(args.log_file) or ".",
                                 "config.json"), args)

    # --- 2. Inicialização dos Modelos ---
    generator = Generator().to(device)
    discriminator = Discriminator(variante=args.discriminator).to(device)
    feature_extractor = FeatureExtractorVGG(camada=args.vgg_layer).to(device)
    print(f"Discriminador '{args.discriminator}' | content loss em "
          f"VGG {args.vgg_layer} (peso {args.content_weight})", flush=True)

    # Múltiplas GPUs no mesmo nó: DataParallel replica os modelos e divide o
    # batch entre elas. É a abordagem usada nos scripts do grupo no Coaraci e
    # não exige torchrun, DistributedSampler nem process group — ao custo de
    # concentrar a agregação na GPU 0. Os checkpoints seguem gravados sem o
    # prefixo 'module.' (ver utils.unwrap), portáveis para 1 GPU ou CPU.
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and not args.no_data_parallel:
        print(f"Usando {n_gpus} GPUs via DataParallel "
              f"(batch total {args.batch_size} dividido entre elas)", flush=True)
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        feature_extractor = nn.DataParallel(feature_extractor)

    # --- 3. Funções de Custo (Loss) e Otimizadores ---
    # Usamos BCEWithLogitsLoss porque removemos a Sigmoid do Discriminador (Modo Pro)
    criterion_GAN = nn.BCEWithLogitsLoss().to(device)
    # Usamos L1 (Mean Absolute Error) ou MSE para a Content Loss
    criterion_content = nn.MSELoss().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # Precisão mista: só faz sentido em CUDA. Os GradScaler são separados
    # porque G e D têm seus próprios backward e podem divergir de escala.
    usar_amp = args.amp and device.type == "cuda"
    if args.amp and not usar_amp:
        print("AVISO: --amp pedido, mas não há CUDA disponível; "
              "seguindo em precisão simples.", flush=True)
    scaler_G = torch.amp.GradScaler(device.type, enabled=usar_amp)
    scaler_D = torch.amp.GradScaler(device.type, enabled=usar_amp)
    if usar_amp:
        print("Precisão mista (AMP) ativa", flush=True)

    # --- 4. Carregamento de Dados ---
    if not os.path.exists(args.data_dir):
        print(f"AVISO: A pasta '{args.data_dir}' não existe. Coloque as radiografias lá para rodar.", flush=True)
        return

    # Valida o dataset ANTES de construir o loader: com dataset vazio o próprio
    # DataLoader estoura (o sampler exige num_samples > 0) e, com menos imagens
    # que o batch, drop_last=True produz zero batches — nesse caso o erro só
    # apareceria no fim da primeira época, tarde demais num job de cluster.
    # Contar arquivos da pasta não serve: conta qualquer arquivo, não só imagens.
    n_imgs = len(list_supported_images(args.data_dir))
    if n_imgs == 0:
        print(f"AVISO: Nenhuma imagem suportada em '{args.data_dir}' "
              f"(extensões aceitas: {', '.join(SUPPORTED_EXTENSIONS)}).", flush=True)
        return

    n_amostras = n_imgs * args.patches_per_image
    if n_amostras < args.batch_size:
        print(f"AVISO: {n_imgs} imagem(ns) x {args.patches_per_image} patch(es) = "
              f"{n_amostras} amostra(s) para --batch-size {args.batch_size}. Como o "
              f"loader usa drop_last=True, nenhum batch se forma. Reduza o batch ou "
              f"aumente --patches-per-image.", flush=True)
        return

    # Uma época curta demais não é erro, mas quase sempre é engano: com poucas
    # radiografias e 1 patch por imagem, "100 épocas" somam algumas centenas de
    # amostras e o treino não sai do lugar.
    if n_amostras < 100:
        print(f"AVISO: a época tem só {n_amostras} amostra(s) "
              f"({n_imgs} imagem(ns) x {args.patches_per_image} patch(es)). "
              f"Considere --patches-per-image maior.", flush=True)

    dataloader = get_dataloader(args.data_dir, batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                pin_memory=device.type == "cuda",
                                patch_size=args.patch_size, lr_scale=args.lr_scale,
                                patches_per_image=args.patches_per_image,
                                fits_normalization=args.fits_normalization,
                                fits_range=args.fits_range)

    # Phantoms de borda (opcional): usados só em validação MTF, nunca na loss
    step_edge_loader = get_step_edge_loader(args.step_edge_dir,
                                            lr_scale=args.lr_scale,
                                            fits_normalization=args.fits_normalization,
                                            fits_range=args.fits_range)
    if step_edge_loader is not None:
        print(f"Validação de bordas ativa: {len(step_edge_loader.dataset)} phantom(s) em '{args.step_edge_dir}'", flush=True)

    # O CSV da MTF acompanha --log-file: duas execuções simultâneas no cluster
    # apontando para pastas de log diferentes não podem sobrescrever uma à outra.
    mtf_log_path = os.path.join(os.path.dirname(args.log_file) or ".",
                                "step_edge_mtf.csv")

    # Decaimento da taxa de aprendizado: o artigo original reduz a taxa depois
    # de um número fixo de iterações. Aqui as marcas são em épocas, e o
    # scheduler acompanha os dois otimizadores para que G e D não fiquem com
    # taxas descasadas.
    schedulers = None
    if args.lr_decay_epochs:
        schedulers = {
            "G": optim.lr_scheduler.MultiStepLR(optimizer_G, args.lr_decay_epochs,
                                                gamma=args.lr_decay_gamma),
            "D": optim.lr_scheduler.MultiStepLR(optimizer_D, args.lr_decay_epochs,
                                                gamma=args.lr_decay_gamma),
        }
        print(f"Taxa de aprendizado decai x{args.lr_decay_gamma} nas épocas "
              f"{args.lr_decay_epochs}", flush=True)

    # --- Retomada de checkpoint (opcional) ---
    # Escolhas que mudam o modelo ou o objetivo viajam com o checkpoint
    arquitetura = {"discriminator": args.discriminator, "vgg_layer": args.vgg_layer}

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, generator, discriminator,
                                      optimizer_G, optimizer_D, device,
                                      arquitetura=arquitetura,
                                      scalers={"G": scaler_G, "D": scaler_D},
                                      schedulers=schedulers)

    # --- 5. Loop de Treinamento ---
    for epoch in range(start_epoch, args.epochs):
        # Fase de pré-treino: só a content loss (MSE) treina o Gerador, sem
        # Discriminador — evita que o D domine antes do G aprender o básico
        pretraining = epoch < args.pretrain_epochs

        # Acumuladores para o log estruturado por época
        sums = {"loss_D": 0.0, "loss_G": 0.0, "loss_content": 0.0,
                "loss_GAN": 0.0, "d_real": 0.0, "d_fake": 0.0, "psnr": 0.0}
        n_batches = 0

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

            with torch.amp.autocast(device.type, enabled=usar_amp):
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

                    # Perda Total do Gerador. O peso EFETIVO do termo adversarial
                    # depende da magnitude da content loss, que por sua vez depende
                    # de --vgg-layer: com relu3_4 e content_weight=1.0 a parcela
                    # adversarial fica na casa de 0,01% da perda total.
                    loss_G = args.content_weight * loss_content + args.adv_weight * loss_GAN

            scaler_G.scale(loss_G).backward()
            if args.grad_clip > 0:
                # O clipping precisa ver os gradientes na escala real, não na
                # escala inflada pelo GradScaler.
                scaler_G.unscale_(optimizer_G)
                nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip)
            scaler_G.step(optimizer_G)
            scaler_G.update()

            # -----------------------------
            # Treinamento do Discriminador (D)
            # -----------------------------
            if pretraining:
                # D não é atualizado no pré-treino
                loss_D = torch.zeros(())
                d_real_prob = float("nan")
                d_fake_prob = float("nan")
            else:
                optimizer_D.zero_grad()

                with torch.amp.autocast(device.type, enabled=usar_amp):
                    # Avalia as imagens reais
                    pred_real = discriminator(imgs_hr)
                    loss_real = criterion_GAN(pred_real, valid)

                    # Avalia as imagens falsas geradas (usando detach para não atualizar o Gerador aqui)
                    pred_fake = discriminator(gen_hr.detach())
                    loss_fake = criterion_GAN(pred_fake, fake)

                    # Perda Média do Discriminador
                    loss_D = (loss_real + loss_fake) / 2

                scaler_D.scale(loss_D).backward()
                if args.grad_clip > 0:
                    scaler_D.unscale_(optimizer_D)
                    nn.utils.clip_grad_norm_(discriminator.parameters(), args.grad_clip)
                scaler_D.step(optimizer_D)
                scaler_D.update()

                # Probabilidades médias D(real) e D(fake): sinal direto de colapso
                # (D(real)->1 e D(fake)->0 constantes = D dominando; ambos ~0.5 = equilíbrio)
                with torch.no_grad():
                    d_real_prob = torch.sigmoid(pred_real).mean().item()
                    d_fake_prob = torch.sigmoid(pred_fake).mean().item()

            # --- 6. Logs e Métricas ---
            with torch.no_grad():
                # .float(): sob AMP gen_hr sai em float16 e as métricas precisam
                # da precisão simples para não introduzir erro próprio
                current_psnr = calculate_psnr(gen_hr.float(), imgs_hr).item()

            sums["loss_D"] += loss_D.item()
            sums["loss_G"] += loss_G.item()
            sums["loss_content"] += loss_content.item()
            sums["loss_GAN"] += loss_GAN.item()
            if not math.isnan(d_real_prob):
                sums["d_real"] += d_real_prob
                sums["d_fake"] += d_fake_prob
            sums["psnr"] += current_psnr
            n_batches += 1

            if i % 10 == 0:
                phase = "PRÉ" if pretraining else "GAN"
                print(f"[{phase}][Época {epoch}/{args.epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}] "
                      f"[PSNR: {current_psnr:.2f} dB]", flush=True)

        # --- Log estruturado por época (CSV append-only) ---
        if n_batches > 0:
            with torch.no_grad():
                epoch_ssim = calculate_ssim(gen_hr.float(), imgs_hr).item()
            n_gan = n_batches if not pretraining else 1  # evita divisão por zero
            # loss_content vai para colunas SEPARADAS por fase. No pré-treino
            # ela é MSE pixel-a-pixel; na fase GAN é MSE sobre features da VGG.
            # São grandezas de escalas diferentes (medidas: ~0.29 e ~9.5 nos
            # mesmos dados) e, numa coluna única, a troca de definição aparece
            # como um salto de 30x que parece piora do modelo sem ser.
            # PSNR e SSIM continuam comparáveis entre as duas fases.
            media = lambda k: f"{sums[k] / n_batches:.6f}"
            log_epoch_csv(args.log_file, {
                "epoch": epoch,
                "phase": "pretrain" if pretraining else "gan",
                "loss_D": media("loss_D"),
                "loss_G": media("loss_G"),
                "loss_content_mse": media("loss_content") if pretraining else "",
                "loss_content_vgg": "" if pretraining else media("loss_content"),
                "loss_GAN": media("loss_GAN"),
                "D_real_prob": f"{sums['d_real'] / n_gan:.4f}" if not pretraining else "",
                "D_fake_prob": f"{sums['d_fake'] / n_gan:.4f}" if not pretraining else "",
                "psnr": f"{sums['psnr'] / n_batches:.3f}",
                "ssim_last_batch": f"{epoch_ssim:.4f}",
            })

        # Decaimento da taxa, uma vez por época
        if schedulers is not None:
            for s in schedulers.values():
                s.step()
            if (epoch + 1) in args.lr_decay_epochs:
                print(f"Taxa de aprendizado agora em "
                      f"{optimizer_G.param_groups[0]['lr']:.3e}", flush=True)

        # --- 7. Checkpoints (Fim de cada época) ---
        # Checkpoint retomável (modelos + otimizadores + época + RNG) toda época
        save_checkpoint(os.path.join(args.weights_dir, "checkpoint_last.pth"),
                        epoch, generator, discriminator, optimizer_G, optimizer_D,
                        arquitetura=arquitetura,
                        scalers={"G": scaler_G, "D": scaler_D},
                        schedulers=schedulers)

        # Salva amostras visuais e os pesos do modelo
        if (epoch + 1) % args.sample_interval == 0 or epoch == 0:
            # imgs_lr/imgs_hr/gen_hr vêm do loop de batches: só existem se a
            # época processou ao menos um
            if n_batches > 0:
                save_samples(epoch, imgs_lr, imgs_hr, gen_hr.float(),
                             save_dir=args.samples_dir)
            save_model_weights(generator, discriminator, epoch, save_dir=args.weights_dir)
            if step_edge_loader is not None:
                validate_step_edges(generator, step_edge_loader, device, epoch,
                                    log_path=mtf_log_path)

if __name__ == "__main__":
    train(parse_args())
