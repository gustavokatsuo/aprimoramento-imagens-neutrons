import argparse
import json
import math
import os
import random
import signal
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Importações dos seus módulos locais
from src.model import (Generator, Discriminator, FeatureExtractorVGG,
                       CAMADAS_VGG)
from src.data_loader import (get_dataloader, get_step_edge_loader,
                             list_supported_images, dividir_pilha,
                             SUPPORTED_EXTENSIONS)
from src.utils import (calculate_psnr, calculate_ssim, save_samples,
                       save_model_weights, save_checkpoint, load_checkpoint,
                       log_epoch_csv, save_run_config, denormalize,
                       mtf10_from_edge, energia_grade, unwrap)

# Interrupção pedida de fora (SIGTERM do SLURM no walltime, ou SIGUSR1 do
# script do job). O laço confere a marca a cada batch e encerra de forma
# limpa: a época em andamento é descartada e valem o checkpoint_last.pth e o
# best.pth da última época COMPLETA. Sem isto o processo morria no meio de
# uma gravação qualquer.
PARAR = {"sinal": None}

def _pedir_parada(signum, frame):
    PARAR["sinal"] = signal.Signals(signum).name
    print(f"\n[{PARAR['sinal']}] Parada pedida: encerrando ao fim do batch atual.", flush=True)

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
                        help="Peso da GAN loss na perda total do Gerador. Com 0, o "
                             "Discriminador nem é avaliado nem treinado (fase GAN "
                             "vira treino só com conteúdo: ablação 'perceptual')")
    parser.add_argument("--adv-razao", type=float, default=0.0,
                        help="Peso adversarial ADAPTATIVO (0 = desligado, vale "
                             "--adv-weight fixo). A cada 50 batches o peso é "
                             "reajustado para que o gradiente adversarial na imagem "
                             "tenha esta fração da norma do gradiente de conteúdo "
                             "(VGG + pixel), como em Esser et al. (2021). Necessário "
                             "com 'patch_sn': a normalização espectral limita o "
                             "gradiente do D, e a escala dele muda ordens de grandeza "
                             "durante o treino (scripts/diagnostico_1879476.py "
                             "calibrar-adv). --adv-weight vira só o valor inicial")
    parser.add_argument("--pixel-weight", type=float, default=0.0,
                        help="Peso da MSE de pixel na fase GAN, SOMADA à content "
                             "loss da VGG. 0 (padrão histórico) = protocolo de Ledig "
                             "et al.: a MSE sai na fase GAN. No job 1879476 isso "
                             "custou ~1,4 dB e criou um padrão em grade "
                             "(docs/diagnostico_1879476.md)")
    parser.add_argument("--init-generator", type=str, default=None,
                        help="Pesos de Gerador (state_dict, ex.: best.pth de outra "
                             "execução) para iniciar o treino. Otimizadores começam "
                             "do zero. Com --pretrain-epochs 0, é o modo de partir "
                             "de uma SRResNet já treinada direto para a fase GAN")
    parser.add_argument("--max-horas", type=float, default=0.0,
                        help="Orçamento de tempo desta execução, em horas (0 = sem "
                             "limite). Uma nova época só começa se couber no que "
                             "resta, estimado pela duração da última. Permite "
                             "dividir um job do SLURM em fases pelo relógio")
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
                        choices=["compacto", "artigo", "patch_sn"],
                        help="Arquitetura do Discriminador: 'compacto' (6 convs "
                             "até 256 canais), 'artigo' (8 convs até 512, "
                             "Ledig et al. 2017) ou 'patch_sn' (PatchGAN local com "
                             "normalização espectral, sem BatchNorm)")
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
    parser.add_argument("--tiff-normalization", type=str, default="dtype",
                        choices=["dtype", "range", "minmax"],
                        help="Normalização para .tif/.tiff. 'dtype' divide pelo teto "
                             "do tipo (65535 em uint16); 'range' usa --tiff-range e é "
                             "o modo correto para uma pilha tomográfica; 'minmax' "
                             "normaliza cada fatia por si e NÃO preserva a "
                             "comparabilidade entre fatias")
    parser.add_argument("--tiff-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Faixa explícita (lo hi) quando --tiff-normalization=range")
    parser.add_argument("--val-fraction", type=float, default=0.0,
                        help="Fração das imagens reservada para validação, tomada "
                             "como BLOCO CONTÍGUO ao final da pilha. Divisão "
                             "aleatória não serve: fatias vizinhas de um volume "
                             "são quase idênticas (SSIM 0,97 nestes dados)")
    parser.add_argument("--test-fraction", type=float, default=0.0,
                        help="Fração reservada para teste, em bloco contíguo. Fica "
                             "intocada durante o treino")
    parser.add_argument("--split-gap", type=int, default=0,
                        help="Fatias descartadas nas fronteiras entre os blocos, "
                             "para que a mesma estrutura não apareça em dois "
                             "conjuntos. Precisa exceder a extensão em z das "
                             "estruturas de interesse")
    parser.add_argument("--cache-images", type=int, default=0,
                        help="Imagens normalizadas mantidas em memória (0 = sem "
                             "cache, -1 = todas). Evita renormalizar a imagem "
                             "inteira a cada recorte. Com --num-workers > 0 a "
                             "memória é multiplicada pelo número de workers")
    parser.add_argument("--min-nonzero", type=float, default=0.0,
                        help="Fração mínima de pixels não-nulos num recorte de treino. "
                             "Reconstruções tomográficas têm zeros fora do círculo de "
                             "reconstrução; 0.5 descarta recortes majoritariamente vazios")
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
    if args.tiff_normalization == "range" and args.tiff_range is None:
        parser.error("--tiff-normalization=range exige --tiff-range LO HI.")
    if args.init_generator and not os.path.isfile(args.init_generator):
        parser.error(f"--init-generator '{args.init_generator}' não existe.")
    if args.init_generator and args.resume:
        parser.error("--init-generator e --resume são exclusivos: a retomada já "
                     "traz os pesos do Gerador.")

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

def avaliar(generator, loader, device, usar_amp=False):
    """
    PSNR e SSIM médios sobre um conjunto que NÃO entra no gradiente.

    É a única métrica do CSV que mede generalização: as colunas 'psnr' e
    'ssim_last_batch' são calculadas sobre os mesmos patches que treinaram o
    Gerador, e sobem mesmo quando o modelo apenas decora. Depende de o conjunto
    ter sido separado em bloco contíguo (ver data_loader.dividir_pilha).

    Devolve também a energia de grade média do resíduo (utils.energia_grade):
    o artefato do job 1879476 passa a ser medido a cada época, não só visto
    nas amostras.
    """
    generator.eval()
    soma_psnr = soma_ssim = soma_grade = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            imgs_lr = batch["lr"].to(device)
            imgs_hr = batch["hr"].to(device)
            with torch.amp.autocast(device.type, enabled=usar_amp):
                gen_hr = generator(imgs_lr)
            gen_hr = gen_hr.float()
            soma_psnr += calculate_psnr(gen_hr, imgs_hr).item()
            soma_ssim += calculate_ssim(gen_hr, imgs_hr).item()
            soma_grade += energia_grade(gen_hr - imgs_hr)
            n += 1
    generator.train()
    nan = float("nan")
    return (soma_psnr / n, soma_ssim / n, soma_grade / n) if n else (nan, nan, nan)

def peso_adv_adaptativo(gen_hr, imgs_hr, discriminator, feature_extractor, args):
    """
    Peso adversarial que faz |grad adv| = args.adv_razao x |grad conteúdo|,
    com os gradientes tomados em relação à imagem gerada (2 imagens, float32).

    A ideia é a do peso adaptativo de Esser et al. (2021, "Taming
    Transformers"): em vez de fixar um número cuja força depende da escala do
    D naquele momento, fixa-se a FORÇA relativa do termo. O limite [1e-6, 1]
    impede que um D ainda sem sinal (gradiente ~0 no início) receba um peso
    absurdo que só amplificaria ruído.
    """
    x = gen_hr[:2].detach().float().requires_grad_(True)
    alvo = imgs_hr[:2].float()
    conteudo = (args.content_weight * nn.functional.mse_loss(
                    feature_extractor(denormalize(x)), feature_extractor(denormalize(alvo)))
                + args.pixel_weight * nn.functional.mse_loss(x, alvo))
    g_cont, = torch.autograd.grad(conteudo, x)
    discriminator.eval()   # sem mexer na BatchNorm nem na norma espectral
    pred = discriminator(x)
    discriminator.train()
    adv = nn.functional.binary_cross_entropy_with_logits(pred, torch.full_like(pred, 0.9))
    g_adv, = torch.autograd.grad(adv, x)
    peso = args.adv_razao * g_cont.norm() / (g_adv.norm() + 1e-12)
    return float(peso.clamp(1e-6, 1.0))

def normas_gradiente(gen_hr, imgs_hr, discriminator, feature_extractor, args, usar_D,
                     peso_adv=None):
    """
    Norma do gradiente de cada termo da loss_G (já multiplicado pelo seu peso)
    em relação à imagem gerada. Mostra quanto cada termo de fato puxa o
    Gerador. No job 1879476 a VGG puxava ~350-750x mais que o adversarial, e
    isso só se descobriu depois (docs/diagnostico_1879476.md §4).

    Calculada uma vez por época, em 2 imagens, em float32 e fora do autocast:
    em float16 os gradientes da MSE (~1e-8 por pixel) zerariam por underflow.
    Não interfere no treino: parte de uma cópia desligada do grafo.
    """
    x = gen_hr[:2].detach().float().requires_grad_(True)
    alvo = imgs_hr[:2].float()
    termos = {"grad_pixel": args.pixel_weight * nn.functional.mse_loss(x, alvo)}
    termos["grad_vgg"] = args.content_weight * nn.functional.mse_loss(
        feature_extractor(denormalize(x)), feature_extractor(denormalize(alvo)))
    if usar_D:
        # Em eval: em modo de treino esta chamada atualizaria as estatísticas
        # da BatchNorm (compacto) e a iteração da norma espectral (patch_sn)
        discriminator.eval()
        pred = discriminator(x)
        discriminator.train()
        peso = args.adv_weight if peso_adv is None else peso_adv
        termos["grad_adv"] = peso * nn.functional.binary_cross_entropy_with_logits(
            pred, torch.full_like(pred, 0.9))
    # autograd.grad devolve só o gradiente em x e não toca nos .grad dos
    # parâmetros: o passo de otimização não é afetado
    return {nome: torch.autograd.grad(termo, x, retain_graph=True)[0].norm().item()
            for nome, termo in termos.items()}

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
    inicio = time.time()
    signal.signal(signal.SIGTERM, _pedir_parada)
    signal.signal(signal.SIGUSR1, _pedir_parada)

    set_seed(args.seed)
    print(f"Semente: {args.seed}", flush=True)

    # Procedência ANTES de qualquer treino: se o job morrer no meio, o registro
    # do que foi pedido já está gravado.
    save_run_config(os.path.join(os.path.dirname(args.log_file) or ".",
                                 "config.json"), args)

    # --- 2. Inicialização dos Modelos ---
    generator = Generator().to(device)
    discriminator = Discriminator(variante=args.discriminator).to(device)
    feature_extractor = FeatureExtractorVGG(camada=args.vgg_layer).to(device)
    print(f"Discriminador '{args.discriminator}' | content loss em "
          f"VGG {args.vgg_layer} (peso {args.content_weight}) | MSE de pixel na "
          f"fase GAN (peso {args.pixel_weight}) | adversarial (peso {args.adv_weight})",
          flush=True)
    # Sem peso adversarial, o D não influencia nada: nem avaliá-lo nem
    # treiná-lo poupa ~1/3 do custo de cada batch da fase GAN
    usar_D = args.adv_weight > 0

    if args.init_generator:
        generator.load_state_dict(torch.load(args.init_generator, map_location=device,
                                             weights_only=True))
        print(f"Gerador iniciado de {args.init_generator} (otimizadores novos)", flush=True)

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

    # --- Divisão treino / validação / teste, em blocos contíguos ---
    todos = list_supported_images(args.data_dir)
    arq_treino, arq_val, arq_teste = dividir_pilha(
        todos, args.val_fraction, args.test_fraction, args.split_gap)
    if (arq_val or arq_teste) and args.split_gap == 0:
        print("AVISO: --split-gap 0 com divisão ativa. Fatias na fronteira entre "
              "os blocos são quase idênticas (SSIM ~0,97 entre vizinhas nestes "
              "dados) e a mesma estrutura aparecerá em dois conjuntos, inflando "
              "a métrica de validação.", flush=True)
    if arq_val or arq_teste:
        print(f"Divisão contígua: {len(arq_treino)} treino | {len(arq_val)} validação | "
              f"{len(arq_teste)} teste | {len(todos)-len(arq_treino)-len(arq_val)-len(arq_teste)} "
              f"descartadas nas margens", flush=True)
    n_imgs = len(arq_treino)

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
                                arquivos=arq_treino,
                                patches_per_image=args.patches_per_image,
                                min_nonzero=args.min_nonzero,
                                cache_images=args.cache_images,
                                fits_normalization=args.fits_normalization,
                                fits_range=args.fits_range,
                                tiff_normalization=args.tiff_normalization,
                                tiff_range=args.tiff_range)

    # Loader de validação: mesmo pipeline, sem embaralhar, para que a métrica
    # seja comparável entre épocas.
    val_loader = None
    if arq_val:
        # semente_fixa: os MESMOS recortes em toda época, senão o val_psnr
        # oscila pelo sorteio e o best.pth sai de uma época com recortes fáceis
        val_loader = get_dataloader(args.data_dir, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers,
                                    pin_memory=device.type == "cuda",
                                    arquivos=arq_val, drop_last=False,
                                    semente_fixa=args.seed,
                                    patch_size=args.patch_size, lr_scale=args.lr_scale,
                                    patches_per_image=args.patches_per_image,
                                    min_nonzero=args.min_nonzero,
                                    cache_images=args.cache_images,
                                    fits_normalization=args.fits_normalization,
                                    fits_range=args.fits_range,
                                    tiff_normalization=args.tiff_normalization,
                                    tiff_range=args.tiff_range)

    # Phantoms de borda (opcional): usados só em validação MTF, nunca na loss
    step_edge_loader = get_step_edge_loader(args.step_edge_dir,
                                            lr_scale=args.lr_scale,
                                            fits_normalization=args.fits_normalization,
                                            fits_range=args.fits_range,
                                            tiff_normalization=args.tiff_normalization,
                                            tiff_range=args.tiff_range)
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

    # Peso adversarial em uso: fixo, ou o valor inicial do modo adaptativo
    peso_adv = args.adv_weight

    # Estado do laço que precisa sobreviver à retomada
    estado = {"melhor_val_psnr": -math.inf, "melhor_epoca": None}
    caminho_best = os.path.join(args.weights_dir, "best.pth")

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, generator, discriminator,
                                      optimizer_G, optimizer_D, device,
                                      arquitetura=arquitetura,
                                      scalers={"G": scaler_G, "D": scaler_D},
                                      schedulers=schedulers, extra=estado)
        # Retomada numa pasta nova (cada job do SLURM grava em runs/<job>):
        # o best.pth antigo não veio junto, então o recorde também não vale
        if not os.path.isfile(caminho_best):
            estado = {**estado, "melhor_val_psnr": -math.inf, "melhor_epoca": None}
        peso_adv = estado.get("peso_adv", peso_adv)

    duracao_ultima = None   # segundos da última época completa (orçamento)

    # --- 5. Loop de Treinamento ---
    motivo_fim = "épocas concluídas"
    for epoch in range(start_epoch, args.epochs):
        # Orçamento de tempo: só começa a época se ela couber no que resta,
        # estimando pela duração da última (com 10% de folga)
        if args.max_horas > 0 and duracao_ultima is not None:
            restante = args.max_horas * 3600 - (time.time() - inicio)
            if restante < 1.1 * duracao_ultima:
                motivo_fim = (f"orçamento de {args.max_horas:.2f} h esgotado "
                              f"(restavam {restante / 60:.1f} min; a última época "
                              f"levou {duracao_ultima / 60:.1f} min)")
                break
        if PARAR["sinal"]:
            break
        t_epoca = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Fase de pré-treino: só a content loss (MSE) treina o Gerador, sem
        # Discriminador — evita que o D domine antes do G aprender o básico
        pretraining = epoch < args.pretrain_epochs

        # Acumuladores para o log estruturado por época
        sums = {"loss_D": 0.0, "loss_G": 0.0, "loss_content": 0.0, "loss_pixel": 0.0,
                "loss_GAN": 0.0, "d_real": 0.0, "d_fake": 0.0, "psnr": 0.0,
                "peso_adv": 0.0}
        n_batches = 0
        normas = {}

        for i, batch in enumerate(dataloader):
            if PARAR["sinal"]:
                break
            # Move as imagens para a GPU
            imgs_lr = batch["lr"].to(device)
            imgs_hr = batch["hr"].to(device)

            # Os rótulos do Discriminador (0,9 = real com label smoothing,
            # 0 = falso) são criados com o shape da SAÍDA do D, via full_like:
            # o 'patch_sn' devolve um mapa de logits, não um número por imagem
            real_alvo = lambda pred: torch.full_like(pred, 0.9)
            falso_alvo = lambda pred: torch.zeros_like(pred)

            # -------------------------
            # Treinamento do Gerador (G)
            # -------------------------
            optimizer_G.zero_grad()

            with torch.amp.autocast(device.type, enabled=usar_amp):
                # Gera uma imagem de alta resolução a partir da baixa
                gen_hr = generator(imgs_lr)

                # MSE de pixel: é a perda inteira no pré-treino e, na fase GAN,
                # entra com --pixel-weight (e é registrada mesmo com peso 0)
                loss_pixel = criterion_content(gen_hr, imgs_hr)

                if pretraining:
                    # Pré-treino: MSE pixel-a-pixel direto (sem VGG, sem adversarial)
                    loss_content = loss_pixel
                    loss_GAN = torch.zeros((), device=device)
                    loss_G = loss_content
                else:
                    # Adversarial Loss: O gerador quer que o discriminador ache que a gen_hr é real (0,9)
                    if usar_D:
                        pred_fake = discriminator(gen_hr)
                        loss_GAN = criterion_GAN(pred_fake, real_alvo(pred_fake))
                    else:
                        loss_GAN = torch.zeros((), device=device)

                    # Content Loss: Compara as características da VGG da imagem gerada vs real.
                    # CORREÇÃO: a VGG espera entrada em [0, 1] (estatísticas ImageNet),
                    # mas gen_hr/imgs_hr estão em [-1, 1] — denormalize antes de extrair
                    gen_features = feature_extractor(denormalize(gen_hr))
                    real_features = feature_extractor(denormalize(imgs_hr))
                    loss_content = criterion_content(gen_features, real_features.detach())

                    # Perda Total do Gerador. Com Adam só a RAZÃO entre os pesos
                    # importa, não a escala absoluta. No job 1879476 (VGG 1.0,
                    # adversarial 1e-3, sem pixel) a VGG puxava o Gerador
                    # ~350-750x mais que o adversarial e criou um padrão em grade
                    # (docs/diagnostico_1879476.md).
                    loss_G = (args.content_weight * loss_content
                              + args.pixel_weight * loss_pixel
                              + peso_adv * loss_GAN)

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
            if pretraining or not usar_D:
                # D não é atualizado no pré-treino nem sem peso adversarial
                loss_D = torch.zeros(())
                d_real_prob = float("nan")
                d_fake_prob = float("nan")
            else:
                optimizer_D.zero_grad()

                with torch.amp.autocast(device.type, enabled=usar_amp):
                    # Avalia as imagens reais
                    pred_real = discriminator(imgs_hr)
                    loss_real = criterion_GAN(pred_real, real_alvo(pred_real))

                    # Avalia as imagens falsas geradas (usando detach para não atualizar o Gerador aqui)
                    pred_fake = discriminator(gen_hr.detach())
                    loss_fake = criterion_GAN(pred_fake, falso_alvo(pred_fake))

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

            # Peso adversarial adaptativo: reajustado a cada 50 batches, já com
            # o D deste passo atualizado. Vale a partir do batch seguinte.
            if args.adv_razao > 0 and usar_D and not pretraining and i % 50 == 0:
                peso_adv = peso_adv_adaptativo(gen_hr, imgs_hr, discriminator,
                                               feature_extractor, args)

            # Quanto cada termo puxa o Gerador: uma vez por época, no 1º batch
            if i == 0 and not pretraining:
                normas = normas_gradiente(gen_hr, imgs_hr, discriminator,
                                          feature_extractor, args, usar_D, peso_adv)

            # --- 6. Logs e Métricas ---
            with torch.no_grad():
                # .float(): sob AMP gen_hr sai em float16 e as métricas precisam
                # da precisão simples para não introduzir erro próprio
                current_psnr = calculate_psnr(gen_hr.float(), imgs_hr).item()

            sums["loss_D"] += loss_D.item()
            sums["loss_G"] += loss_G.item()
            sums["loss_content"] += loss_content.item()
            sums["loss_pixel"] += loss_pixel.item()
            sums["loss_GAN"] += loss_GAN.item()
            sums["peso_adv"] += peso_adv
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

        # Época interrompida por sinal: descartada inteira (sem log, sem
        # checkpoint), para que o CSV só tenha épocas completas
        if PARAR["sinal"]:
            print(f"Época {epoch} interrompida e descartada. Valem o "
                  f"checkpoint_last.pth e o best.pth da última época completa.",
                  flush=True)
            break

        # --- Log estruturado por época (CSV append-only) ---
        if n_batches > 0:
            with torch.no_grad():
                epoch_ssim = calculate_ssim(gen_hr.float(), imgs_hr).item()
            # D_real/D_fake só existem na fase GAN com D ativo
            com_D = not pretraining and usar_D

            # Métrica de generalização: conjunto separado, fora do gradiente
            val_psnr, val_ssim, val_grade = (avaliar(generator, val_loader, device, usar_amp)
                                             if val_loader is not None else (float("nan"),) * 3)
            # loss_content vai para colunas SEPARADAS por fase. No pré-treino
            # ela é MSE pixel-a-pixel; na fase GAN é MSE sobre features da VGG.
            # São grandezas de escalas diferentes (medidas: ~0.29 e ~9.5 nos
            # mesmos dados) e, numa coluna única, a troca de definição aparece
            # como um salto de 30x que parece piora do modelo sem ser.
            # A coluna mse traz a MSE de pixel nas DUAS fases (na GAN ela é
            # registrada mesmo com --pixel-weight 0). PSNR e SSIM continuam
            # comparáveis entre as duas fases.
            media = lambda k: f"{sums[k] / n_batches:.6f}"
            num = lambda v, f: "" if v is None or math.isnan(v) else format(v, f)
            log_epoch_csv(args.log_file, {
                "epoch": epoch,
                "phase": "pretrain" if pretraining else "gan",
                "loss_D": media("loss_D"),
                "loss_G": media("loss_G"),
                "loss_content_mse": media("loss_pixel"),
                "loss_content_vgg": "" if pretraining else media("loss_content"),
                "loss_GAN": media("loss_GAN"),
                "D_real_prob": f"{sums['d_real'] / n_batches:.4f}" if com_D else "",
                "D_fake_prob": f"{sums['d_fake'] / n_batches:.4f}" if com_D else "",
                "psnr": f"{sums['psnr'] / n_batches:.3f}",
                "ssim_last_batch": f"{epoch_ssim:.4f}",
                "val_psnr": num(val_psnr, ".3f"),
                "val_ssim": num(val_ssim, ".4f"),
                # % da energia do resíduo em períodos de 2/4 px; com recortes
                # de 256 px, ~0,2 = sem grade (utils.piso_grade)
                "val_grade_pct": num(val_grade, ".2f"),
                # norma do gradiente de cada termo (já com peso) na imagem
                "grad_pixel": num(normas.get("grad_pixel"), ".3e"),
                "grad_vgg": num(normas.get("grad_vgg"), ".3e"),
                "grad_adv": num(normas.get("grad_adv"), ".3e"),
                # peso adversarial médio da época (= --adv-weight se fixo)
                "adv_weight_efetivo": f"{sums['peso_adv'] / n_batches:.3e}" if com_D else "",
                # treino + validação desta época (sem gravações em disco)
                "tempo_epoca_s": f"{time.time() - t_epoca:.1f}",
                "mem_pico_gb": (f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f}"
                                if device.type == "cuda" else ""),
            })
            if not math.isnan(val_psnr):
                print(f"[Validação] época {epoch}: PSNR {val_psnr:.2f} dB | "
                      f"SSIM {val_ssim:.4f} | grade {val_grade:.2f}%  "
                      f"({len(val_loader.dataset)} amostras) | "
                      f"época em {(time.time() - t_epoca) / 60:.1f} min",
                      flush=True)

            # best.pth: o Gerador da melhor época em val_psnr, avaliado a cada
            # época. No job 1879476 o melhor só existia porque calhou de cair
            # numa época múltipla do intervalo de salvamento.
            if not math.isnan(val_psnr) and val_psnr > estado["melhor_val_psnr"]:
                estado.update(melhor_val_psnr=val_psnr, melhor_epoca=epoch)
                os.makedirs(args.weights_dir, exist_ok=True)
                torch.save(unwrap(generator).state_dict(), caminho_best + ".tmp")
                os.replace(caminho_best + ".tmp", caminho_best)
                with open(os.path.join(args.weights_dir, "best.json"), "w") as f:
                    json.dump({"melhor_val_psnr": val_psnr, "melhor_epoca": epoch, "val_ssim": val_ssim, "val_grade_pct": val_grade,
                               "phase": "pretrain" if pretraining else "gan"}, f, indent=2)
                print(f"Novo best.pth: época {epoch}, val_psnr {val_psnr:.3f} dB", flush=True)

            # Baseline SRResNet: o Gerador ao fim do pré-treino, antes de
            # qualquer perda VGG ou adversarial
            if pretraining and epoch == args.pretrain_epochs - 1:
                os.makedirs(args.weights_dir, exist_ok=True)
                torch.save(unwrap(generator).state_dict(),
                           os.path.join(args.weights_dir, "srresnet_fim_pretreino.pth"))
                print("Baseline SRResNet salvo em srresnet_fim_pretreino.pth", flush=True)

        # Decaimento da taxa, uma vez por época
        if schedulers is not None:
            for s in schedulers.values():
                s.step()
            if (epoch + 1) in args.lr_decay_epochs:
                print(f"Taxa de aprendizado agora em "
                      f"{optimizer_G.param_groups[0]['lr']:.3e}", flush=True)

        # --- 7. Checkpoints (Fim de cada época) ---
        estado["peso_adv"] = peso_adv   # o modo adaptativo retoma do último valor
        # Checkpoint retomável (modelos + otimizadores + época + RNG) toda época
        save_checkpoint(os.path.join(args.weights_dir, "checkpoint_last.pth"),
                        epoch, generator, discriminator, optimizer_G, optimizer_D,
                        arquitetura=arquitetura,
                        scalers={"G": scaler_G, "D": scaler_D},
                        schedulers=schedulers, extra=estado)
        # Gerador da última época completa, no formato de inferência (o
        # checkpoint_last.pth tem o mesmo, mas embrulhado com otimizadores)
        torch.save(unwrap(generator).state_dict(),
                   os.path.join(args.weights_dir, "gen_ultimo.pth.tmp"))
        os.replace(os.path.join(args.weights_dir, "gen_ultimo.pth.tmp"),
                   os.path.join(args.weights_dir, "gen_ultimo.pth"))

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

        duracao_ultima = time.time() - t_epoca

    # --- 8. Resumo da execução ---
    # Lido pelo script do job para montar o relatório final
    if PARAR["sinal"]:
        motivo_fim = f"interrompido por {PARAR['sinal']}"
    resumo = {
        "motivo_fim": motivo_fim,
        "horas": round((time.time() - inicio) / 3600, 3),
        "melhor_epoca": estado["melhor_epoca"],
        "melhor_val_psnr": (None if math.isinf(estado["melhor_val_psnr"])
                            else round(estado["melhor_val_psnr"], 3)),
    }
    caminho_resumo = os.path.join(os.path.dirname(args.log_file) or ".", "resumo.json")
    with open(caminho_resumo, "w") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    print(f"Fim do treino: {motivo_fim}. Melhor época {resumo['melhor_epoca']} "
          f"(val_psnr {resumo['melhor_val_psnr']}). Resumo em {caminho_resumo}", flush=True)
    if args.adv_razao > 0:
        print(f"Peso adversarial adaptativo terminou em {peso_adv:.3e}", flush=True)
    if estado["melhor_epoca"] is None and val_loader is None:
        print("AVISO: sem conjunto de validação não há best.pth. Use --val-fraction > 0.",
              flush=True)

if __name__ == "__main__":
    train(parse_args())
