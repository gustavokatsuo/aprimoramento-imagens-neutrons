"""
Mede a faixa de valores da pilha COMPLETA e quanto dela a normalização corta.

O treino usa --tiff-normalization range --tiff-range 0 8300, faixa estimada
com só 3 fatias (pendência P2 em docs/decisoes.md). Valores acima do teto
viram 1,0 sem aviso. Este script lê todas as fatias, nos valores brutos do
detector, e responde: qual é o máximo global, que percentis a pilha ocupa e
que fração dos pixels fica acima do teto (isto é, é cortada).

Roda dentro do job (scripts/train_coaraci.slurm), logo depois da cópia dos
dados, e grava o resultado em runs/<job>/ambiente/faixa_pilha.json. Não muda
a normalização: só mede, para o relatório dizer com números se houve corte.

Uso:
    python scripts/medir_faixa.py <pasta_fatias> <saida.json> [--teto 8300]
"""
import argparse
import json
import os
import sys

import numpy as np
import tifffile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.data_loader import list_supported_images


def main():
    p = argparse.ArgumentParser(description="Faixa de valores da pilha completa")
    p.add_argument("pasta")
    p.add_argument("saida")
    p.add_argument("--teto", type=float, default=8300.0,
                   help="Teto da normalização usada no treino (--tiff-range)")
    a = p.parse_args()

    arquivos = [f for f in list_supported_images(a.pasta)
                if f.lower().endswith((".tif", ".tiff"))]
    if not arquivos:
        raise SystemExit(f"Nenhum TIFF em {a.pasta}")

    # Histograma inteiro de 16 bits: exato e de memória fixa, qualquer que
    # seja o tamanho da pilha (percentis sem guardar os pixels)
    hist = np.zeros(65536, dtype=np.int64)
    max_por_fatia = []
    for caminho in arquivos:
        img = tifffile.imread(caminho)
        if img.dtype != np.uint16:
            raise SystemExit(f"{os.path.basename(caminho)}: dtype {img.dtype}; "
                             f"este script espera uint16")
        hist += np.bincount(img.ravel(), minlength=65536)
        max_por_fatia.append(int(img.max()))

    total = int(hist.sum())
    nao_nulos = total - int(hist[0])
    acumulado = np.cumsum(hist)

    def percentil(q, sem_zero=False):
        # sem_zero: ignora o fundo fora do círculo de reconstrução
        base = int(hist[0]) if sem_zero else 0
        n = nao_nulos if sem_zero else total
        alvo = base + q / 100.0 * n
        return int(np.searchsorted(acumulado, alvo))

    teto = int(a.teto)
    acima = int(hist[teto + 1:].sum()) if teto < 65535 else 0
    fatias_acima = sum(1 for m in max_por_fatia if m > teto)
    resultado = {
        "fatias": len(arquivos),
        "pixels": total,
        "maximo_global": int(np.nonzero(hist)[0].max()),
        "teto_normalizacao": teto,
        "pixels_acima_do_teto": acima,
        "fracao_acima_do_teto_nao_nulos": (acima / nao_nulos) if nao_nulos else 0.0,
        "fatias_com_pixels_acima_do_teto": fatias_acima,
        "percentis_nao_nulos": {f"p{q}": percentil(q, sem_zero=True)
                                for q in (1, 50, 99, 99.9, 99.99)},
        "fracao_zeros": int(hist[0]) / total,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.saida)), exist_ok=True)
    with open(a.saida, "w") as f:
        json.dump(resultado, f, indent=2)

    r = resultado
    print(f"Faixa da pilha: {r['fatias']} fatias, máximo global {r['maximo_global']} "
          f"(teto do treino {teto}); p99,99 dos não nulos = "
          f"{r['percentis_nao_nulos']['p99.99']}")
    if acima:
        print(f"AVISO: {acima} pixels ({100 * r['fracao_acima_do_teto_nao_nulos']:.4f}% "
              f"dos não nulos, em {fatias_acima} fatias) ficam acima de {teto} e são "
              f"cortados pela normalização. Registrar no relatório.")
    else:
        print(f"Nenhum pixel acima de {teto}: a normalização não corta nada.")


if __name__ == "__main__":
    main()
