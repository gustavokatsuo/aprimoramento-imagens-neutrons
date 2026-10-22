"""
Baselines clássicos de super-resolução para comparação justa com a SRGAN.

Roda o MESMO pipeline degradação -> reconstrução -> métrica do treino da GAN:
  - Degradação: gaussian blur (k=3) + downsample bicúbico 4x (idêntico ao
    NeutronDataset, para que todos os métodos partam da mesma imagem LR).
  - Reconstrução: bicúbica, Lanczos, bicúbica + unsharp masking e,
    opcionalmente, o Gerador da SRGAN (--gen-weights).
  - Métricas: as MESMAS funções calculate_psnr/calculate_ssim de src/utils.py
    usadas no treino (que esperam o domínio [-1, 1] do modelo), garantindo
    comparação justa entre métodos clássicos e a GAN.
  - Conjunto: por padrão, SÓ o bloco de teste, reconstruído com a mesma
    dividir_pilha do treino. Avaliar a pasta inteira misturaria fatias que o
    Gerador viu no treino, e a linha 'srgan' da tabela deixaria de medir
    generalização. --val-fraction, --test-fraction e --split-gap precisam ser
    os MESMOS do treino (estão em logs/config.json da execução).

Uso:
    python -m src.classical_baselines --data-dir data/raw \\
        --val-fraction 0.10 --test-fraction 0.10 --split-gap 30 \\
        --tiff-normalization range --tiff-range 0 8300 \\
        --gen-weights runs/<job>/weights/gen_epoch_99.pth
"""

import argparse
import csv
import os

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

from src.data_loader import (load_image_as_array, list_supported_images,
                             dividir_pilha)
from src.model import Generator
from src.utils import calculate_psnr, calculate_ssim

def parse_args():
    parser = argparse.ArgumentParser(description="Baselines clássicos vs SRGAN")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--lr-scale", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=512,
                        help="Recorte central para avaliação (0 = imagem inteira)")
    parser.add_argument("--gen-weights", type=str, default=None,
                        help="Pesos do Gerador (ex.: weights/gen_epoch_99.pth) para incluir a SRGAN na tabela")
    parser.add_argument("--unsharp-amount", type=float, default=1.0,
                        help="Intensidade do unsharp masking")
    parser.add_argument("--output", type=str, default="results/classical_baselines.csv")
    parser.add_argument("--fits-normalization", type=str, default="minmax",
                        choices=["minmax", "range", "none"])
    parser.add_argument("--fits-range", type=float, nargs=2, default=None)
    parser.add_argument("--tiff-normalization", type=str, default="dtype",
                        choices=["dtype", "range", "minmax"])
    parser.add_argument("--tiff-range", type=float, nargs=2, default=None)
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "val", "train", "all"],
                        help="Conjunto avaliado. 'test' (padrão) é o único que vale "
                             "para o relatório; 'all' avalia a pasta inteira, "
                             "incluindo fatias de treino")
    parser.add_argument("--val-fraction", type=float, default=0.0,
                        help="Igual ao do treino (ver logs/config.json)")
    parser.add_argument("--test-fraction", type=float, default=0.0,
                        help="Igual ao do treino (ver logs/config.json)")
    parser.add_argument("--split-gap", type=int, default=0,
                        help="Igual ao do treino (ver logs/config.json)")
    return parser.parse_args()

def to_model_range(img):
    """[0, 1] -> [-1, 1], o domínio esperado pelas métricas de utils.py."""
    return img * 2.0 - 1.0

def degrade(img_hr, lr_scale):
    """Degradação idêntica à do NeutronDataset (blur + bicúbico)."""
    _, _, h, w = img_hr.shape
    img_lr = TF.gaussian_blur(img_hr, kernel_size=3)
    img_lr = TF.resize(img_lr, (h // lr_scale, w // lr_scale),
                       interpolation=transforms.InterpolationMode.BICUBIC,
                       antialias=True).clamp(0.0, 1.0)
    return img_lr

def upscale_bicubic(img_lr, size):
    return TF.resize(img_lr, size,
                     interpolation=transforms.InterpolationMode.BICUBIC,
                     antialias=True).clamp(0.0, 1.0)

def upscale_lanczos(img_lr, size):
    """
    Upsampling Lanczos via PIL em modo 'F' (float32) — o modo 'F' preserva a
    precisão total, diferente da conversão implícita para 8 bits do modo 'L'.
    """
    arr = img_lr.squeeze().numpy().astype(np.float32)
    pil = Image.fromarray(arr, mode="F")
    pil = pil.resize((size[1], size[0]), resample=Image.Resampling.LANCZOS)
    out = torch.from_numpy(np.array(pil, dtype=np.float32))
    return out.unsqueeze(0).unsqueeze(0).clamp(0.0, 1.0)

def unsharp_mask(img, amount=1.0, kernel_size=5):
    """
    Unsharp masking: realce = img + amount * (img - blur(img)).
    Aumenta o contraste local nas bordas, mas NÃO recupera frequências
    espaciais perdidas — por isso é um baseline importante (ver docs).
    """
    blurred = TF.gaussian_blur(img, kernel_size=kernel_size)
    return (img + amount * (img - blurred)).clamp(0.0, 1.0)

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ordem natural e a mesma divisão do treino: o bloco de teste aqui é
    # exatamente o bloco que o treino deixou intocado.
    todos = list_supported_images(args.data_dir)
    if not todos:
        print(f"AVISO: Nenhuma imagem encontrada em '{args.data_dir}'.", flush=True)
        return

    if args.split == "all":
        files = todos
        print("AVISO: --split all inclui fatias de treino; os números não medem "
              "generalização e não servem para o relatório.", flush=True)
    else:
        treino, validacao, teste = dividir_pilha(
            todos, args.val_fraction, args.test_fraction, args.split_gap)
        files = {"train": treino, "val": validacao, "test": teste}[args.split]
        if not files:
            raise SystemExit(
                f"ERRO: o conjunto '{args.split}' ficou vazio. Passe os mesmos "
                f"--val-fraction, --test-fraction e --split-gap do treino "
                f"(estão em logs/config.json da execução).")

    print(f"Conjunto '{args.split}': {len(files)} imagem(ns), de "
          f"{os.path.basename(files[0])} a {os.path.basename(files[-1])}", flush=True)

    # Carrega o Gerador da SRGAN, se solicitado
    generator = None
    if args.gen_weights:
        generator = Generator().to(device)
        generator.load_state_dict(torch.load(args.gen_weights, map_location=device,
                                             weights_only=True))
        generator.eval()

    methods = ["bicubic", "lanczos", "bicubic+unsharp"] + (["srgan"] if generator else [])
    rows = []

    for img_path in files:
        name = os.path.basename(img_path)
        arr = load_image_as_array(img_path, args.fits_normalization, args.fits_range,
                                  args.tiff_normalization, args.tiff_range)
        img = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W), [0, 1]

        # Recorte central determinístico, múltiplo do fator de escala
        _, _, h, w = img.shape
        crop = min(h, w) if args.patch_size == 0 else min(args.patch_size, h, w)
        crop -= crop % args.lr_scale
        img_hr = TF.center_crop(img, (crop, crop))

        img_lr = degrade(img_hr, args.lr_scale)
        hr_size = (crop, crop)

        # --- Reconstruções ---
        recon = {
            "bicubic": upscale_bicubic(img_lr, hr_size),
            "lanczos": upscale_lanczos(img_lr, hr_size),
        }
        recon["bicubic+unsharp"] = unsharp_mask(recon["bicubic"], amount=args.unsharp_amount)

        if generator is not None:
            with torch.no_grad():
                gen_out = generator(to_model_range(img_lr).to(device)).cpu()
            # Saída do Gerador já está em [-1, 1] (Tanh); volta para [0, 1]
            recon["srgan"] = ((gen_out + 1.0) / 2.0).clamp(0.0, 1.0)

        # --- Métricas (mesmas funções do treino, domínio [-1, 1]) ---
        hr_model = to_model_range(img_hr)
        for method in methods:
            sr_model = to_model_range(recon[method])
            psnr = calculate_psnr(sr_model, hr_model).item()
            ssim = calculate_ssim(sr_model, hr_model).item()
            rows.append({"image": name, "method": method,
                         "psnr": psnr, "ssim": ssim})

    # --- Tabela-resumo (média por método) ---
    print(f"\n{'Método':<20} {'PSNR médio (dB)':>16} {'SSIM médio':>12}   (n={len(files)} imagens)", flush=True)
    print("-" * 55, flush=True)
    for method in methods:
        m_rows = [r for r in rows if r["method"] == method]
        mean_psnr = sum(r["psnr"] for r in m_rows) / len(m_rows)
        mean_ssim = sum(r["ssim"] for r in m_rows) / len(m_rows)
        print(f"{method:<20} {mean_psnr:>16.3f} {mean_ssim:>12.4f}", flush=True)

    # --- CSV com os resultados por imagem ---
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "method", "psnr", "ssim"])
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "psnr": f"{r['psnr']:.4f}", "ssim": f"{r['ssim']:.5f}"})
    print(f"\nResultados por imagem salvos em {args.output}", flush=True)

if __name__ == "__main__":
    evaluate(parse_args())
