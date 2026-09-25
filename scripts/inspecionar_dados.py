"""
Diagnóstico de um conjunto de imagens antes de treinar com ele.

Responde, sobre cada arquivo: se é página única ou pilha, shape e dtype, a
faixa de valores efetivamente ocupada, quanto o pipeline preserva na leitura,
a extensão da amostra em pixels e se há borda aproveitável para medir a MTF.

Foi escrito ao receber as primeiras reconstruções do 3D-XRM, e o que ele
mostrou está registrado em docs/dataset.md. Vale repassá-lo em qualquer
conjunto novo — em particular nas radiografias de nêutrons do IEA-R1, cujas
características ainda são desconhecidas.

Uso:
    python scripts/inspecionar_dados.py [pasta] [--voxel-um VALOR]

    pasta       padrão: data/raw
    --voxel-um  tamanho de voxel para converter MTF10 em lp/mm e estimar
                dimensões físicas (padrão: 3.647937, do conjunto atual)
"""
import sys, os, glob, argparse
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

_p = argparse.ArgumentParser(add_help=True)
_p.add_argument("pasta", nargs="?", default=os.path.join(REPO, "data", "raw"))
_p.add_argument("--voxel-um", type=float, default=3.647937)
_a = _p.parse_args()
PX_UM = _a.voxel_um
alvo = _a.pasta
arquivos = sorted(sum([glob.glob(os.path.join(alvo, e)) for e in
                       ('*.tif','*.tiff','*.TIF','*.TIFF','*.fits','*.png','*.jpg')], []))
if not arquivos:
    print(f"nenhum arquivo em {alvo}"); sys.exit(1)

print(f"{len(arquivos)} arquivo(s) em {alvo}\n" + "="*78)

for caminho in arquivos[:6]:
    nome = os.path.basename(caminho)
    tam = os.path.getsize(caminho)
    print(f"\n### {nome}  ({tam/1024**2:.2f} MiB)")

    # --- leitura crua, antes de qualquer normalização do projeto ---
    ext = os.path.splitext(caminho)[1].lower()
    if ext in ('.tif', '.tiff'):
        import tifffile
        with tifffile.TiffFile(caminho) as tf:
            n_pag = len(tf.pages)
            bruto = tf.asarray()
        print(f"  páginas no arquivo : {n_pag}  -> {'PILHA' if n_pag > 1 else 'página única'}")
    elif ext == '.fits':
        from astropy.io import fits
        with fits.open(caminho) as h:
            bruto = np.asarray([x.data for x in h if x.data is not None][0])
        print("  formato            : FITS")
    else:
        from PIL import Image
        bruto = np.asarray(Image.open(caminho))

    print(f"  shape / dtype      : {bruto.shape} / {bruto.dtype}")

    f = bruto.astype(np.float64).ravel()
    lo, hi = f.min(), f.max()
    print(f"  faixa efetiva      : [{lo:.4g}, {hi:.4g}]")
    if np.issubdtype(bruto.dtype, np.integer):
        teto = np.iinfo(bruto.dtype).max
        print(f"  ocupação do dtype  : {(hi-lo)/teto*100:.1f}% dos {teto+1} níveis "
              f"(máximo usado: {hi/teto*100:.1f}%)")
    print(f"  percentis 1/50/99  : {np.percentile(f,1):.4g} / "
          f"{np.percentile(f,50):.4g} / {np.percentile(f,99):.4g}")
    print(f"  níveis distintos   : {len(np.unique(f)):,}")

    # transmitância costuma ficar em [0,1] com moda perto de 1 (fundo transparente)
    if 0 <= lo and hi <= 1.01:
        print("  -> valores em [0,1]: compatível com TRANSMITÂNCIA já corrigida")
    elif np.issubdtype(bruto.dtype, np.integer):
        print("  -> inteiro sem faixa [0,1]: contagens brutas ou reconstrução reescalada")

    # --- leitura pelo pipeline do projeto ---
    try:
        from src.data_loader import load_image_as_array
        a = load_image_as_array(caminho)
        print(f"  load_image_as_array: OK  {a.shape} {a.dtype} "
              f"[{a.min():.4f}, {a.max():.4f}]  ({len(np.unique(a)):,} níveis)")
        if a.ndim == 2:
            h, w = a.shape
            print(f"  patch 256 cabe?    : {'sim' if min(h,w) >= 256 else 'NÃO'}"
                  f"   FOV {w*PX_UM/1000:.2f} x {h*PX_UM/1000:.2f} mm")
    except Exception as e:
        print(f"  load_image_as_array: FALHOU -> {type(e).__name__}: {e}")

    # --- mede o tubo de kapton para deduzir o voxel ---
    # Se o diâmetro físico do porta-amostra for conhecido, a extensão medida em
    # pixels dá o tamanho do voxel por divisão. Foi assim que se desempatou a
    # divergência entre as duas fontes de parâmetros (ver docs/dataset.md §2).
    try:
        img2d = a if a.ndim == 2 else a[a.shape[0] // 2]
        h, w = img2d.shape
        fundo = np.percentile(img2d, 2)
        pico = np.percentile(img2d, 98)
        limiar = fundo + 0.25 * (pico - fundo)
        # varre 9 linhas ao redor do centro e toma a mediana da extensão
        larguras = []
        for lin in range(h // 2 - 40, h // 2 + 41, 10):
            acesos = np.where(img2d[lin] > limiar)[0]
            if len(acesos) > 20:
                larguras.append(acesos[-1] - acesos[0] + 1)
        if larguras:
            d_px = float(np.median(larguras))
            print(f"  extensão da amostra: {d_px:.0f} px (mediana de {len(larguras)} linhas centrais)")
            print(f"    se isso for o tubo de 5,0 mm -> voxel = {5000/d_px:.3f} um")
            for cand, rot in ((3.647937, '.txt'), (1.74, 'relatório')):
                print(f"    com voxel {cand:.4f} um ({rot}): a extensão seria "
                      f"{d_px*cand/1000:.2f} mm")
        else:
            print("  extensão da amostra: não detectada (contraste insuficiente)")
    except Exception as e:
        print(f"  medição do tubo   : {type(e).__name__}: {e}")

    # --- há borda aproveitável para MTF? ---
    try:
        from src.utils import mtf10_from_edge
        img2d = a if a.ndim == 2 else a[0]
        v, estado = mtf10_from_edge(img2d)
        extra = f" = {v/PX_UM*1000:.1f} lp/mm" if not np.isnan(v) else ""
        print(f"  MTF10 na imagem    : {'n/d' if np.isnan(v) else f'{v:.4f} c/px'}{extra}"
              f"   [{estado[:44]}]")
    except Exception as e:
        print(f"  MTF10              : {type(e).__name__}: {e}")

if len(arquivos) > 6:
    print(f"\n(+{len(arquivos)-6} arquivos não mostrados)")
