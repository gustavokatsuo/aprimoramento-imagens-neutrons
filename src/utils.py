import torch
import numpy as np
from torchvision.utils import save_image
from torchmetrics.functional.image import structural_similarity_index_measure
import os
import csv
import random

def denormalize(tensors):
    """
    Converte tensores de [-1, 1] de volta para [0, 1] para visualização e cálculo de métricas.
    """
    return (tensors + 1.0) / 2.0

def calculate_psnr(img1, img2):
    """
    Calcula o Peak Signal-to-Noise Ratio (PSNR).
    Alvo da IC: Melhoria mínima de 5 dB.

    Espera tensores no domínio do modelo, [-1, 1]. Após denormalize, o domínio é
    [0, 1] e MAX_I = 1.0 — matematicamente equivalente a calcular em uint16 com
    data_range=65535, pois o PSNR é invariante a reescala linear conjunta.
    O clamp protege contra reconstruções (ex.: baselines bicúbicos) que
    ultrapassem levemente o intervalo por overshoot de interpolação.
    """
    # 1. Desnormaliza as imagens para garantir o domínio [0, 1] e o cálculo correto do MAX_I
    img1_norm = denormalize(img1).clamp(0.0, 1.0)
    img2_norm = denormalize(img2).clamp(0.0, 1.0)

    # 2. Calcula o MSE nas imagens corrigidas
    mse = torch.mean((img1_norm - img2_norm) ** 2)

    if mse == 0:
        return float('inf')

    # MAX_I é implicitamente 1.0 agora
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    """
    Calcula o Structural Similarity Index (SSIM) via torchmetrics.
    Espera tensores 4D (B, C, H, W) no domínio do modelo, [-1, 1];
    desnormaliza para [0, 1] e usa data_range=1.0, coerente com calculate_psnr.
    """
    img1_norm = denormalize(img1).clamp(0.0, 1.0)
    img2_norm = denormalize(img2).clamp(0.0, 1.0)
    return structural_similarity_index_measure(img1_norm, img2_norm, data_range=1.0)

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
    save_image(comparison, save_path, normalize=True, scale_each=True)
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

def save_checkpoint(path, epoch, generator, discriminator, optimizer_G, optimizer_D):
    """
    Salva um checkpoint COMPLETO e retomável do treinamento: modelos, otimizadores,
    época atual e estado dos geradores de números aleatórios (RNG). Diferente de
    save_model_weights (que serve apenas para inferência), este permite retomar
    o treino exatamente de onde parou via --resume.
    """
    save_dir = os.path.dirname(path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state = {
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict(),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    torch.save(state, path)

def load_checkpoint(path, generator, discriminator, optimizer_G, optimizer_D, device):
    """
    Restaura um checkpoint salvo por save_checkpoint e retorna a época em que
    o treino deve recomeçar (época salva + 1). Também restaura o estado dos RNGs
    para reprodutibilidade da sequência de dados/augmentation.
    """
    # weights_only=False é necessário pois o checkpoint contém estados de RNG (numpy/python)
    state = torch.load(path, map_location=device, weights_only=False)

    generator.load_state_dict(state["generator"])
    discriminator.load_state_dict(state["discriminator"])
    optimizer_G.load_state_dict(state["optimizer_G"])
    optimizer_D.load_state_dict(state["optimizer_D"])

    rng = state.get("rng")
    if rng is not None:
        torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])

    print(f"Checkpoint '{path}' carregado. Retomando da época {state['epoch'] + 1}.")
    return state["epoch"] + 1

def log_epoch_csv(log_path, row):
    """
    Registro estruturado (append-only) das métricas de uma época em CSV.
    'row' é um dicionário; o cabeçalho é escrito apenas na criação do arquivo.
    Inclui D(real)/D(fake) médios — o sinal direto para diagnosticar colapso da GAN.
    """
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def mtf_from_edge(edge_image, oversample=4, window_halfwidth=16):
    """
    Calcula a MTF (Modulation Transfer Function) a partir de uma imagem de borda
    inclinada (método slanted-edge, no espírito da ISO 12233):

        1. ESF (Edge Spread Function): estima a posição sub-pixel da borda em cada
           linha (centroide da derivada horizontal), ajusta uma reta à borda e
           projeta todos os pixels na direção normal à borda, acumulando-os em
           bins superamostrados (oversample bins por pixel).
        2. LSF (Line Spread Function): derivada numérica da ESF, recortada em uma
           janela simétrica ao redor do pico e apodizada com janela de Hann para
           reduzir vazamento espectral do ruído das caudas.
        3. MTF: módulo da FFT da LSF, normalizado para MTF(0) = 1.

    Parâmetros:
        edge_image: array 2D (numpy) em escala de cinza, contendo UMA borda de
            degrau aproximadamente VERTICAL e levemente inclinada (~2-10 graus).
        oversample: fator de superamostragem da ESF (4 é o usual na ISO 12233).
        window_halfwidth: meia-largura da janela da LSF, em pixels NATIVOS.

    Retorna:
        freq: frequências espaciais em ciclos/pixel (Nyquist nativo = 0.5).
        mtf:  MTF correspondente, adimensional em [0, 1].
    """
    img = np.asarray(edge_image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("mtf_from_edge espera uma imagem 2D em escala de cinza.")

    # --- 1. Posição sub-pixel da borda em cada linha (centroide da derivada) ---
    # O centroide é calculado apenas numa vizinhança (+-10 px) do gradiente
    # máximo de cada linha, para não ser enviesado por ruído ou por bordas
    # secundárias longe da borda principal (prática padrão, ex. sfrmat/ISO 12233)
    deriv = np.abs(np.diff(img, axis=1))
    cols = np.arange(deriv.shape[1]) + 0.5
    peak_cols = np.argmax(deriv, axis=1)
    local = np.abs(cols[None, :] - (peak_cols[:, None] + 0.5)) <= 10
    deriv = deriv * local
    weights = deriv.sum(axis=1)
    valid = weights > 0
    if valid.sum() < 2:
        raise ValueError("Nenhuma borda detectável na imagem fornecida.")
    edge_pos = (deriv[valid] * cols).sum(axis=1) / weights[valid]
    rows = np.arange(img.shape[0])[valid]

    # Ajuste linear da borda: x = a*y + b (borda ~vertical)
    a, b = np.polyfit(rows, edge_pos, 1)

    # --- Projeção: distância perpendicular de cada pixel à reta da borda ---
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    dist = (xx - (a * yy + b)) / np.sqrt(1.0 + a**2)

    # --- Binning superamostrado da ESF ---
    bin_width = 1.0 / oversample
    bins = np.floor(dist.ravel() / bin_width).astype(np.int64)
    bins -= bins.min()
    counts = np.bincount(bins)
    sums = np.bincount(bins, weights=img.ravel())
    filled = counts > 0
    esf = np.empty(len(counts))
    esf[filled] = sums[filled] / counts[filled]
    # Bins vazios (possíveis quando a inclinação é pequena) são interpolados
    x_all = np.arange(len(counts))
    esf[~filled] = np.interp(x_all[~filled], x_all[filled], esf[filled])

    # --- 2. LSF: derivada da ESF + janela de Hann centrada no pico ---
    lsf = np.gradient(esf, bin_width)
    peak = int(np.argmax(np.abs(lsf)))
    half = window_halfwidth * oversample
    lo, hi = max(0, peak - half), min(len(lsf), peak + half)
    lsf_win = lsf[lo:hi] * np.hanning(hi - lo)

    # --- 3. MTF: |FFT| normalizada em DC ---
    mtf = np.abs(np.fft.rfft(lsf_win))
    if mtf[0] == 0:
        raise ValueError("LSF degenerada: componente DC nula.")
    mtf = mtf / mtf[0]
    freq = np.fft.rfftfreq(hi - lo, d=bin_width)

    return freq, mtf

def cutoff_frequency(freq, mtf, threshold=0.1):
    """
    Frequência espacial de corte: primeira frequência em que a MTF cai abaixo
    de 'threshold' (padrão MTF10 = 0.1), obtida por interpolação linear.
    Retorna np.nan se a MTF nunca cruzar o limiar no intervalo medido.
    """
    below = np.where(mtf < threshold)[0]
    # Ignora o ponto DC e exige um cruzamento real
    below = below[below > 0]
    if len(below) == 0:
        return float("nan")
    k = below[0]
    # Interpolação linear entre (k-1) e k
    f1, f2 = freq[k - 1], freq[k]
    m1, m2 = mtf[k - 1], mtf[k]
    if m1 == m2:
        return float(f2)
    return float(f1 + (threshold - m1) * (f2 - f1) / (m2 - m1))