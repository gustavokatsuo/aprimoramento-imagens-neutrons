"""
Gera phantoms SINTÉTICOS de borda inclinada para a validação de MTF.

Enquanto não há phantoms adquiridos no equipamento, estes destravam a medida
de MTF no treino (validate_step_edges) e na comparação com os métodos
clássicos (classical_baselines). Servem para comparar MÉTODOS sob a
degradação sintética do projeto; NÃO medem a resolução do sistema real, que
só um phantom físico mede.

Cada phantom é uma borda de degrau quase vertical, inclinada de alguns graus
(o que o método slanted-edge de utils.mtf_from_edge exige), construída assim:

  1. Degrau entre dois níveis desfocado por uma gaussiana contínua de desvio
     --sigmas (em pixels), que faz o papel da PSF do sistema de alta
     resolução. É uma SUPOSIÇÃO: a PSF real do 3D-XRM não foi medida. Por
     isso há mais de um valor.
  2. Integração na área de cada pixel por superamostragem (abertura do
     detector).
  3. Ruído gaussiano opcional (--ruidos, em contagens).

A MTF pré-amostragem resultante é conhecida: gaussiana x sinc da abertura.
Para sigma = 1 px, a MTF10 teórica é ~0,325 ciclos/px, o que serve de
conferência de utils.mtf_from_edge.

Os níveis padrão (1400 e 5500) são os percentis 1 e 99 do centro da fatia
2029, e ficam dentro da faixa de treino --tiff-range 0 8300. O ruído de 150
contagens é da ordem do desvio medido entre pixels vizinhos na mesma região
(~156, que inclui textura e portanto superestima o ruído).

Uso:
    python scripts/gerar_phantoms.py                 # padrão, em data/step_edges/
    python scripts/gerar_phantoms.py --saida outra/pasta --angulos 4 6
"""
import argparse
import os

import numpy as np
import tifffile
import torch   # só para torch.erf: o projeto não depende de scipy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    p = argparse.ArgumentParser(description="Phantoms sintéticos de borda inclinada")
    p.add_argument("--saida", default=os.path.join(REPO, "data", "step_edges"))
    p.add_argument("--tamanho", type=int, default=768,
                   help="Lado da imagem em pixels. O StepEdgeDataset usa o recorte "
                        "central de 512, então precisa ser >= 512")
    p.add_argument("--angulos", type=float, nargs="+", default=[3.0, 5.0, 8.0],
                   help="Inclinação da borda em relação à vertical, em graus (2-10)")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.5, 1.0],
                   help="Desvio da PSF gaussiana do sistema HR, em pixels")
    p.add_argument("--ruidos", type=float, nargs="+", default=[0.0, 150.0],
                   help="Desvio do ruído gaussiano, em contagens")
    p.add_argument("--niveis", type=float, nargs=2, default=[1400.0, 5500.0],
                   metavar=("ESCURO", "CLARO"))
    p.add_argument("--superamostragem", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def borda(tamanho, angulo_graus, sigma, escuro, claro, superamostragem):
    """
    Borda inclinada desfocada por uma gaussiana CONTÍNUA de desvio sigma e
    integrada na área de cada pixel.

    O desfoque é aplicado analiticamente (perfil erf ao longo da normal à
    borda), não por convolução na grade de pixels: um núcleo gaussiano
    amostrado com sigma ~0,5 px tem resposta em frequência bem diferente da
    gaussiana contínua, e a MTF medida não corresponderia ao sigma declarado.
    Assim a MTF pré-amostragem é exatamente gaussiana x abertura do pixel.
    """
    s = superamostragem
    c = tamanho / 2.0
    t = np.tan(np.deg2rad(angulo_graus))
    centros = np.arange(tamanho, dtype=np.float64)
    yy, xx = np.meshgrid(centros, centros, indexing="ij")
    soma = np.zeros((tamanho, tamanho))
    # Média sobre s x s sub-pixels de cada pixel (abertura do detector)
    for dy in (np.arange(s) + 0.5) / s:
        for dx in (np.arange(s) + 0.5) / s:
            # Distância com sinal à reta x = c + t*(y - c); positiva = lado claro
            d = ((xx + dx - c) - t * (yy + dy - c)) / np.sqrt(1.0 + t * t)
            if sigma > 0:
                frac = 0.5 * (1.0 + torch.erf(torch.from_numpy(d / (sigma * np.sqrt(2.0)))).numpy())
            else:
                frac = (d > 0).astype(np.float64)
            soma += escuro + (claro - escuro) * frac
    return soma / (s * s)

def main():
    a = parse_args()
    if a.tamanho < 512:
        raise SystemExit("--tamanho precisa ser >= 512 (recorte central do StepEdgeDataset).")
    os.makedirs(a.saida, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    escuro, claro = a.niveis

    n = 0
    for angulo in a.angulos:
        for sigma in a.sigmas:
            limpa = borda(a.tamanho, angulo, sigma, escuro, claro, a.superamostragem)
            for ruido in a.ruidos:
                img = limpa + (rng.normal(0.0, ruido, limpa.shape) if ruido > 0 else 0.0)
                img = np.clip(np.rint(img), 0, np.iinfo(np.uint16).max).astype(np.uint16)
                nome = f"sintetico_ang{angulo:g}_sigma{sigma:g}_ruido{ruido:g}.tiff"
                tifffile.imwrite(os.path.join(a.saida, nome), img)
                n += 1
    print(f"{n} phantom(s) gravado(s) em {a.saida}")

if __name__ == "__main__":
    main()
