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
  - Resolução: MTF10 de cada método (e da HR) sobre os phantoms de
    --step-edge-dir, com o mesmo recorte e degradação da validação do treino.
  - Conjunto: por padrão, SÓ o bloco de teste, reconstruído com a mesma
    dividir_pilha do treino. Avaliar a pasta inteira misturaria fatias que o
    Gerador viu no treino, e a linha 'srgan' da tabela deixaria de medir
    generalização.
  - Configuração herdada: normalização, divisão e fator de escala vêm do
    logs/config.json da execução que gerou os pesos (ver CHAVES_HERDADAS).
    Avaliar com outra normalização não dá erro nenhum: a SRGAN recebe uma
    entrada que nunca viu e o PSNR de todos os métodos muda de escala, e a
    tabela sai com cara de válida e conclusão invertida.

Uso:
    # com pesos: config.json encontrado ao lado deles, nada a repetir
    python -m src.classical_baselines --data-dir data/raw \\
        --gen-weights runs/<job>/weights/gen_epoch_99.pth

    # só os clássicos, com a configuração de uma execução
    python -m src.classical_baselines --data-dir data/raw \\
        --config runs/<job>/logs/config.json
"""

import argparse
import csv
import json
import math
import os

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

from src.data_loader import (load_image_as_array, list_supported_images,
                             dividir_pilha, StepEdgeDataset)
from src.model import Generator
from src.utils import calculate_psnr, calculate_ssim, denormalize, mtf10_from_edge

# Argumentos que precisam ser idênticos aos do treino, com o valor usado
# quando não há config.json (os mesmos padrões de src/train.py). data_dir
# fica de fora de propósito: no cluster ele aponta para o scratch do nó.
CHAVES_HERDADAS = {
    "lr_scale": 4,
    "fits_normalization": "minmax",
    "fits_range": None,
    "tiff_normalization": "dtype",
    "tiff_range": None,
    "val_fraction": 0.0,
    "test_fraction": 0.0,
    "split_gap": 0,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Baselines clássicos vs SRGAN")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--patch-size", type=int, default=512,
                        help="Recorte central para avaliação (0 = imagem inteira)")
    parser.add_argument("--gen-weights", type=str, default=None,
                        help="Pesos do Gerador (ex.: runs/<job>/weights/gen_epoch_99.pth) "
                             "para incluir a SRGAN na tabela")
    parser.add_argument("--config", type=str, default=None,
                        help="config.json da execução de treino. Omitido, é procurado "
                             "em <execução>/logs/config.json a partir de --gen-weights")
    parser.add_argument("--sem-config", action="store_true",
                        help="Avalia os pesos sem config.json, só com os argumentos "
                             "passados. Use apenas para execuções antigas que não o gravaram")
    parser.add_argument("--unsharp-amount", type=float, default=1.0,
                        help="Intensidade do unsharp masking")
    parser.add_argument("--output", type=str, default="results/classical_baselines.csv",
                        help="CSV de PSNR/SSIM por imagem; a MTF vai para <output>_mtf.csv")
    parser.add_argument("--step-edge-dir", type=str, default="data/step_edges",
                        help="Phantoms de borda para a MTF de cada método. Não vem do "
                             "config.json: no cluster aquele caminho é o scratch do nó")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "val", "train", "all"],
                        help="Conjunto avaliado. 'test' (padrão) é o único que vale "
                             "para o relatório; 'all' avalia a pasta inteira, "
                             "incluindo fatias de treino")

    # Os argumentos abaixo vêm do config.json quando há um. Passá-los só é
    # necessário sem config; passá-los com valor diferente do config é erro.
    # O default None distingue "não passado" de "passado com o valor padrão".
    herdados = parser.add_argument_group(
        "herdados do treino", "Lidos do config.json; informe só sem config")
    herdados.add_argument("--lr-scale", type=int, default=None)
    herdados.add_argument("--fits-normalization", type=str, default=None,
                          choices=["minmax", "range", "none"])
    herdados.add_argument("--fits-range", type=float, nargs=2, default=None)
    herdados.add_argument("--tiff-normalization", type=str, default=None,
                          choices=["dtype", "range", "minmax"])
    herdados.add_argument("--tiff-range", type=float, nargs=2, default=None)
    herdados.add_argument("--val-fraction", type=float, default=None)
    herdados.add_argument("--test-fraction", type=float, default=None)
    herdados.add_argument("--split-gap", type=int, default=None)
    return parser.parse_args()

def _localizar_config(args):
    """
    Caminho do config.json a usar, ou None. Com --run-dir o treino grava
    <execução>/weights/gen_epoch_N.pth e <execução>/logs/config.json; sem ele,
    weights/ e logs/ relativos ao diretório de trabalho — a mesma conta
    (dois níveis acima dos pesos, depois logs/) cobre os dois casos.
    """
    if args.config:
        if not os.path.isfile(args.config):
            raise SystemExit(f"ERRO: --config '{args.config}' não existe.")
        return args.config
    if args.gen_weights:
        execucao = os.path.dirname(os.path.dirname(os.path.abspath(args.gen_weights)))
        candidato = os.path.join(execucao, "logs", "config.json")
        if os.path.isfile(candidato):
            return candidato
        if not args.sem_config:
            raise SystemExit(
                f"ERRO: não achei o config.json da execução dos pesos (procurei "
                f"'{candidato}'). Sem ele não há como garantir que a avaliação use "
                f"a normalização e a divisão do treino. Passe --config <caminho> "
                f"ou, para uma execução antiga sem config, --sem-config com os "
                f"argumentos do treino.")
    return None

def _mesmo_valor(a, b):
    """Compara valores do CLI e do JSON (listas e int/float misturados)."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_mesmo_valor(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b

def resolver_configuracao(args):
    """
    Preenche em 'args' os argumentos de CHAVES_HERDADAS a partir do config.json
    da execução. Um argumento passado na linha de comando com valor DIFERENTE
    do config é erro: significaria avaliar o modelo em condições distintas das
    do treino. Sem config, valem os argumentos passados ou os padrões.
    """
    caminho = _localizar_config(args)
    gravados = {}
    if caminho:
        with open(caminho) as f:
            gravados = json.load(f).get("argumentos", {})
        print(f"Configuração do treino lida de {caminho}", flush=True)

    conflitos = []
    for chave, padrao in CHAVES_HERDADAS.items():
        passado = getattr(args, chave)
        if chave in gravados:
            if passado is not None and not _mesmo_valor(passado, gravados[chave]):
                conflitos.append(f"--{chave.replace('_', '-')}: passado {passado}, "
                                 f"treino usou {gravados[chave]}")
            setattr(args, chave, gravados[chave])
        elif passado is None:
            setattr(args, chave, padrao)

    if conflitos:
        raise SystemExit("ERRO: argumentos diferentes dos do treino:\n  "
                         + "\n  ".join(conflitos)
                         + "\nRemova-os para usar os do config.json.")
    if args.lr_scale != 4:
        raise SystemExit(f"ERRO: --lr-scale={args.lr_scale} não é suportado: o "
                         f"Gerador reconstrói 4x (ver src/model.py).")
    if args.tiff_normalization == "range" and args.tiff_range is None:
        raise SystemExit("ERRO: tiff_normalization='range' exige tiff_range.")
    if args.fits_normalization == "range" and args.fits_range is None:
        raise SystemExit("ERRO: fits_normalization='range' exige fits_range.")

    print("  " + " | ".join(f"{k}={getattr(args, k)}" for k in CHAVES_HERDADAS),
          flush=True)
    return args

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

def reconstruir(img_lr, hr_size, generator, device, unsharp_amount):
    """
    Todas as reconstruções a partir da MESMA imagem LR, em [0, 1]. Usada nas
    fatias e nos phantoms, para que PSNR/SSIM e MTF vejam exatamente as mesmas
    saídas de cada método.
    """
    recon = {
        "bicubic": upscale_bicubic(img_lr, hr_size),
        "lanczos": upscale_lanczos(img_lr, hr_size),
    }
    recon["bicubic+unsharp"] = unsharp_mask(recon["bicubic"], amount=unsharp_amount)

    if generator is not None:
        with torch.no_grad():
            gen_out = generator(to_model_range(img_lr).to(device)).cpu()
        # Saída do Gerador já está em [-1, 1] (Tanh); volta para [0, 1]
        recon["srgan"] = ((gen_out + 1.0) / 2.0).clamp(0.0, 1.0)
    return recon

def avaliar_bordas(args, generator, device, methods):
    """
    MTF10 de cada método sobre os phantoms de borda, mais a da referência HR.

    Usa o StepEdgeDataset do treino — mesmo recorte central e mesma
    degradação de validate_step_edges —, de modo que as linhas 'srgan' e 'hr'
    daqui coincidem com mtf10_sr e mtf10_hr do step_edge_mtf.csv do treino.
    A 'bicubic' difere um pouco de mtf10_lr_bicubic (0,1488 contra 0,1486
    num phantom sintético): validate_step_edges amplia com F.interpolate
    (núcleo a=-0,75) e este script com TF.resize antialias=True (a=-0,5, o
    do PIL).

    Só phantoms de borda servem: sobre fatias da amostra, mtf_from_edge acha
    um gradiente qualquer e não uma borda de degrau (ver docs/dataset.md §8).

    Cada linha leva também o PSNR no phantom. MTF10 alta com PSNR baixo é
    ruído ou artefato sustentando a curva, não resolução: o unsharp, por
    exemplo, pode subir a MTF10 sem recuperar frequência alguma.
    """
    if not os.path.isdir(args.step_edge_dir):
        print(f"\nAVISO: '{args.step_edge_dir}' não existe — MTF não medida.", flush=True)
        return []
    dataset = StepEdgeDataset(args.step_edge_dir, lr_scale=args.lr_scale,
                              fits_normalization=args.fits_normalization,
                              fits_range=args.fits_range,
                              tiff_normalization=args.tiff_normalization,
                              tiff_range=args.tiff_range)
    if len(dataset) == 0:
        print(f"\nAVISO: nenhum phantom de borda em '{args.step_edge_dir}' — "
              f"MTF não medida.", flush=True)
        return []

    rows = []
    for amostra in dataset:
        # O dataset entrega [-1, 1]; as reconstruções trabalham em [0, 1]
        img_lr = denormalize(amostra["lr"]).unsqueeze(0).clamp(0.0, 1.0)
        img_hr = denormalize(amostra["hr"]).unsqueeze(0).clamp(0.0, 1.0)
        recon = reconstruir(img_lr, img_hr.shape[2:], generator, device,
                            args.unsharp_amount)
        recon["hr"] = img_hr

        hr_model = to_model_range(img_hr)
        for method in methods + ["hr"]:
            valor, estado = mtf10_from_edge(recon[method].squeeze().numpy())
            psnr = (float("nan") if method == "hr" else
                    calculate_psnr(to_model_range(recon[method]), hr_model).item())
            rows.append({"image": amostra["name"], "method": method,
                         "mtf10": valor, "status": estado, "psnr": psnr})
    return rows

def resumir_bordas(rows, methods):
    """Tabela de MTF10 por método, com a bicúbica como linha de base."""
    def media(m, chave):
        vals = [r[chave] for r in rows if r["method"] == m
                and r["status"] == "ok" and not math.isnan(r[chave])]
        return (sum(vals) / len(vals)) if vals else float("nan")

    n_phantoms = len({r["image"] for r in rows})
    print(f"\n{'Método':<20} {'MTF10 (c/px)':>13} {'medidas ok':>11} "
          f"{'PSNR phantom':>13}   (n={n_phantoms} phantom(s))", flush=True)
    print("-" * 62, flush=True)
    base_mtf, base_psnr = media("bicubic", "mtf10"), media("bicubic", "psnr")
    for m in methods + ["hr"]:
        mtf, psnr = media(m, "mtf10"), media(m, "psnr")
        n_ok = sum(1 for r in rows if r["method"] == m and r["status"] == "ok")
        txt = lambda v, f: "n/d" if math.isnan(v) else format(v, f)
        # Ganho de MTF com PSNR pior que a bicúbica não é resolução
        alerta = ("  <- MTF acima da bicúbica com PSNR abaixo: suspeito"
                  if m not in ("bicubic", "hr") and mtf > base_mtf and psnr < base_psnr
                  else "")
        print(f"{m:<20} {txt(mtf, '.4f'):>13} {f'{n_ok}/{n_phantoms}':>11} "
              f"{txt(psnr, '.3f'):>13}{alerta}", flush=True)

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
                f"ERRO: o conjunto '{args.split}' ficou vazio com val_fraction="
                f"{args.val_fraction}, test_fraction={args.test_fraction} e "
                f"split_gap={args.split_gap}. Indique a execução de treino com "
                f"--gen-weights ou --config, ou passe esses argumentos.")

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
        recon = reconstruir(img_lr, (crop, crop), generator, device, args.unsharp_amount)

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

    # --- MTF nos phantoms de borda ---
    rows_mtf = avaliar_bordas(args, generator, device, methods)
    if rows_mtf:
        resumir_bordas(rows_mtf, methods)
        saida_mtf = os.path.splitext(args.output)[0] + "_mtf.csv"
        num = lambda v, f: "" if math.isnan(v) else format(v, f)
        with open(saida_mtf, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "method", "mtf10",
                                                   "status", "psnr"])
            writer.writeheader()
            for r in rows_mtf:
                writer.writerow({**r, "mtf10": num(r["mtf10"], ".4f"),
                                 "psnr": num(r["psnr"], ".4f")})
        print(f"MTF por phantom salva em {saida_mtf}", flush=True)

if __name__ == "__main__":
    evaluate(resolver_configuracao(parse_args()))
