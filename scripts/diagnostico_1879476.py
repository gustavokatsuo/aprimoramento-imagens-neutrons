"""
Diagnóstico do job 1879476 (100 épocas, Coaraci): colapso da fase GAN e
padrão em grade nas amostras. Os resultados estão em docs/diagnostico_1879476.md.

Tudo roda em CPU, sobre os pesos que o job devolveu e as fatias locais
2028-2030. Essas fatias ficam no BLOCO DE TESTE (topo da pilha), então o
gerador nunca as viu no treino; em compensação são só 3 fatias, e os números
aqui são indicativos, não o resultado do trabalho.

Subcomandos:

  evolucao     Para cada gerador salvo (épocas 0, 4, 9, 14, ..., 99): PSNR em 5
               recortes fixos, energia de grade do resíduo SR-HR e o cosseno
               entre o passo do termo adversarial e a grade (o D empurra a
               favor ou contra ela?).
  gradientes   Na época 99: norma e espectro do gradiente de cada termo da
               loss_G (VGG, adversarial, MSE de pixel) em relação à imagem.
  ajuste-fino  Parte do gerador da época 4 (fim do pré-treino, sem grade) e
               treina alguns passos com UMA perda escolhida, SEM discriminador.
               É o teste causal: quem cria a grade?
  calibrar-adv Treina cada variante de D por alguns passos contra o G da
               época 4 congelado e mede a razão entre o gradiente adversarial
               e o da VGG x0,006: base para escolher --adv-weight por variante.
  bicubica     PSNR/SSIM da bicúbica, da época 4 e da época 99 em 24 recortes
               aleatórios, com o mesmo critério do treino (--min-nonzero 0.5),
               e a separação por fração de fundo.

"Energia de grade": fração da energia do espectro do resíduo que cai nas
frequências múltiplas de 1/4 ciclo/px (períodos de 2 e 4 px, janela 3x3 em
torno de cada pico). Num resíduo de espectro plano ela fica em 135/(H*W):
0,2% nos recortes de 256 px, 0,8% nos de 128 px do ajuste fino. Também é dada a amplitude RMS dessa componente, em milésimos da escala
cheia [0, 1] (1 milésimo = 8,3 níveis de cinza na faixa 0-8300).

Uso (da raiz do repositório; os pesos vêm de resultados_cluster.zip):
    unzip resultados_cluster.zip '1879476/*' -d ../resultados_cluster
    python scripts/diagnostico_1879476.py evolucao   --pesos ../resultados_cluster/1879476/weights
    python scripts/diagnostico_1879476.py gradientes --pesos ../resultados_cluster/1879476/weights
    python scripts/diagnostico_1879476.py bicubica   --pesos ../resultados_cluster/1879476/weights
    python scripts/diagnostico_1879476.py ajuste-fino --pesos ... --perda vgg --camada relu3_4
    python scripts/diagnostico_1879476.py ajuste-fino --pesos ... --perda pix
    python scripts/diagnostico_1879476.py ajuste-fino --pesos ... --perda vgg+pix
    python scripts/diagnostico_1879476.py ajuste-fino --pesos ... --perda vgg --avgpool

O ajuste fino leva cerca de 3 a 10 minutos por configuração em CPU.
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.model import Generator, Discriminator, FeatureExtractorVGG, CAMADAS_VGG
from src.data_loader import load_image_as_array
from src.utils import calculate_psnr, calculate_ssim, denormalize

# Mesma normalização do job 1879476 (config.json): --tiff-range 0 8300
FAIXA = (0.0, 8300.0)
FATIAS = (2028, 2029, 2030)


# --- Dados: mesma degradação de data_loader.TomographyDataset ----------------
def carregar(fatia):
    caminho = os.path.join(REPO, "data", "raw", f"{fatia}.tiff")
    arr = load_image_as_array(caminho, tiff_normalization="range", tiff_range=FAIXA)
    return torch.from_numpy(arr).unsqueeze(0)

def degradar(hr):
    """hr em [0, 1], (1, P, P) -> par (lr, hr) no domínio do modelo, [-1, 1]."""
    p = hr.shape[-1]
    lr = TF.gaussian_blur(hr, kernel_size=3)
    lr = TF.resize(lr, (p // 4, p // 4), interpolation=transforms.InterpolationMode.BICUBIC,
                   antialias=True).clamp(0.0, 1.0)
    return lr * 2 - 1, hr * 2 - 1

def recortes_fixos(p=256):
    """5 recortes da fatia 2029, todos dentro do círculo de reconstrução."""
    img = carregar(2029)
    h, w = img.shape[1:]
    pares = []
    for di, dj in ((0, 0), (-300, 0), (0, -300), (300, 0), (0, 300)):
        i0, j0 = h // 2 - p // 2 + di, w // 2 - p // 2 + dj
        pares.append(degradar(img[:, i0:i0 + p, j0:j0 + p]))
    return torch.stack([a for a, _ in pares]), torch.stack([b for _, b in pares])

def recortes_aleatorios(imgs, rng, n, p, min_nonzero):
    pares = []
    while len(pares) < n:
        img = imgs[rng.randrange(len(imgs))]
        h, w = img.shape[1:]
        i, j = rng.randint(0, h - p), rng.randint(0, w - p)
        hr = img[:, i:i + p, j:j + p]
        if (hr > 0).float().mean().item() >= min_nonzero:
            pares.append(degradar(hr))
    return torch.stack([a for a, _ in pares]), torch.stack([b for _, b in pares])

def carregar_gerador(pesos, epoca):
    g = Generator()
    g.load_state_dict(torch.load(os.path.join(pesos, f"gen_epoch_{epoca}.pth"),
                                 map_location="cpu", weights_only=True))
    return g.eval()

def carregar_discriminador(pesos, epoca):
    d = Discriminator(variante="compacto")   # variante do job 1879476
    d.load_state_dict(torch.load(os.path.join(pesos, f"disc_epoch_{epoca}.pth"),
                                 map_location="cpu", weights_only=True))
    return d.eval()


# --- Medida da grade ---------------------------------------------------------
def mascara_grade(n):
    """Janelas 3x3 em torno de todas as frequências (fy, fx) múltiplas de 1/4."""
    m = np.zeros((n, n), bool)
    for fy in (0, .25, .5, .75):
        for fx in (0, .25, .5, .75):
            if fy == fx == 0:
                continue   # o DC não é grade
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    m[(int(fy * n) + a) % n, (int(fx * n) + b) % n] = True
    return m

def energia_grade(x):
    """
    x: (B, 1, n, n), resíduo no domínio [-1, 1].
    Devolve (% da energia nas frequências de grade, RMS da componente em
    milésimos da escala [0, 1]).
    """
    r = x.squeeze(1).double().numpy()
    r = r - r.mean(axis=(1, 2), keepdims=True)
    b, n, _ = r.shape
    S = np.abs(np.fft.fft2(r)) ** 2
    m = mascara_grade(n)
    frac = 100 * S[:, m].sum() / S.sum()
    # Parseval: energia espacial = soma |F|^2 / n^2; RMS por pixel; x0,5 leva
    # de [-1, 1] para [0, 1]
    rms = np.sqrt(S[:, m].sum() / (n * n) / (b * n * n)) * 0.5 * 1000
    return frac, rms

def componente_grade(x):
    """Só a parte de x nas frequências de grade (filtro no espectro)."""
    Fx = np.fft.fft2(x.squeeze(1).double().numpy())
    Fx[:, ~mascara_grade(Fx.shape[-1])] = 0
    return np.real(np.fft.ifft2(Fx))

def grad_adversarial(D, sr):
    """Gradiente da loss_GAN do Gerador (alvo 0,9, como em train.py) na imagem."""
    x = sr.clone().requires_grad_(True)
    logit = D(x)
    loss = F.binary_cross_entropy_with_logits(logit, torch.full_like(logit, 0.9))
    g, = torch.autograd.grad(loss, x)
    return g, logit, loss


# --- Subcomandos -------------------------------------------------------------
def evolucao(args):
    lr, hr = recortes_fixos()
    print("época | PSNR (5 recortes) | grade no resíduo         | cos(passo adv, grade)")
    for ep in [0, 4] + list(range(9, 100, 5)):
        g = carregar_gerador(args.pesos, ep)
        with torch.no_grad():
            sr = g(lr)
        frac, rms = energia_grade(sr - hr)
        cos = ""
        if ep >= 5:   # só há D treinado na fase GAN
            gr, _, _ = grad_adversarial(carregar_discriminador(args.pesos, ep), sr)
            # O passo de descida é -grad. Cosseno NEGATIVO com a grade do
            # resíduo significa que o passo REDUZ a grade.
            a, b = componente_grade(-gr).ravel(), componente_grade(sr - hr).ravel()
            cos = f"{a @ b / (np.linalg.norm(a) * np.linalg.norm(b)):+.3f}"
        print(f"{ep:5d} | {calculate_psnr(sr, hr).item():8.2f} dB      | "
              f"{frac:5.1f}% (RMS {rms:5.2f} ‰)    | {cos}", flush=True)

def gradientes(args):
    lr, hr = recortes_fixos()
    with torch.no_grad():
        sr = carregar_gerador(args.pesos, 99)(lr)
    vgg = FeatureExtractorVGG("relu3_4").eval()

    x = sr.clone().requires_grad_(True)
    l_vgg = F.mse_loss(vgg(denormalize(x)), vgg(denormalize(hr)))
    g_vgg, = torch.autograd.grad(l_vgg, x)
    g_adv, logit, l_adv = grad_adversarial(carregar_discriminador(args.pesos, 99), sr)
    x = sr.clone().requires_grad_(True)
    g_mse, = torch.autograd.grad(F.mse_loss(x, hr), x)

    print(f"Época 99, 5 recortes: D_fake_prob médio {torch.sigmoid(logit).mean():.2e} | "
          f"loss_GAN {l_adv.item():.2f} | loss_VGG {l_vgg.item():.3f}")
    print("Norma do gradiente na imagem, já com os pesos do job 1879476:")
    print(f"  VGG relu3_4 x 1.0   : {g_vgg.norm():.3e}")
    print(f"  adversarial x 1e-3  : {1e-3 * g_adv.norm():.3e}")
    print(f"  MSE de pixel (fora da loss_G na fase GAN): {g_mse.norm():.3e}")
    print(f"  razão VGG / adversarial: {g_vgg.norm() / (1e-3 * g_adv.norm()):.0f}x")
    print("Fração do gradiente nas frequências de grade:")
    for nome, g in (("VGG", g_vgg), ("adversarial", g_adv), ("MSE", g_mse)):
        print(f"  {nome:12s} {energia_grade(g)[0]:5.1f}%")

def ajuste_fino(args):
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    imgs = [carregar(k) for k in FATIAS]
    # recortes de 128 px quase sem fundo: o fundo não tem grade a medir
    lr_v, hr_v = recortes_aleatorios(imgs, random.Random(123), 8, 128, 0.95)

    g = carregar_gerador(args.pesos, 4).train()
    vgg = None
    if args.perda != "pix":
        vgg = FeatureExtractorVGG(args.camada).eval()
        if args.avgpool:
            # Hipótese testada: o retrocesso do max-pooling concentra o
            # gradiente numa rede de passo 2/4. Com AvgPool isso não ocorre.
            for k, mod in enumerate(vgg.feature_extractor):
                if isinstance(mod, nn.MaxPool2d):
                    vgg.feature_extractor[k] = nn.AvgPool2d(2, 2)
    # Adam NOVO, como seria numa fase GAN que reinicia o otimizador
    opt = torch.optim.Adam(g.parameters(), lr=1e-4)

    rotulo = args.perda + ("" if args.perda == "pix" else f" {args.camada}") + \
        (" avgpool" if args.avgpool else "")
    t0 = time.time()

    def medir(passo):
        g.eval()
        with torch.no_grad():
            sr = g(lr_v)
        g.train()
        frac, rms = energia_grade(sr - hr_v)
        print(f"[{rotulo}] passo {passo:4d}: PSNR {calculate_psnr(sr, hr_v).item():6.2f} dB | "
              f"grade {frac:5.1f}% (RMS {rms:5.2f} ‰) | {time.time() - t0:4.0f} s", flush=True)

    medir(0)
    rng = random.Random(0)
    for passo in range(1, args.passos + 1):
        lr, hr = recortes_aleatorios(imgs, rng, 8, 128, 0.95)
        sr = g(lr)
        l_pix = F.mse_loss(sr, hr)
        if args.perda == "pix":
            loss = l_pix
        else:
            l_vgg = F.mse_loss(vgg(denormalize(sr)), vgg(denormalize(hr)))
            # vgg+pix: VGG na escala do artigo (0,006) mais o MSE de pixel
            loss = l_vgg if args.perda == "vgg" else 0.006 * l_vgg + l_pix
        opt.zero_grad()
        loss.backward()
        opt.step()
        if passo % 50 == 0 or passo == args.passos:
            medir(passo)

def bicubica(args):
    imgs = [carregar(k) for k in FATIAS]
    lr, hr = recortes_aleatorios(imgs, random.Random(7), 24, 256, 0.5)
    bic = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(-1, 1)
    with torch.no_grad():
        sr = {ep: carregar_gerador(args.pesos, ep)(lr) for ep in (4, 99)}

    def por_recorte(x, metrica):
        return [metrica(x[i:i + 1], hr[i:i + 1]).item() for i in range(len(hr))]

    print("24 recortes aleatórios do bloco de teste (--min-nonzero 0.5)")
    for nome, x in (("bicúbica", bic), ("época 4 ", sr[4]), ("época 99", sr[99])):
        p, s = por_recorte(x, calculate_psnr), por_recorte(x, calculate_ssim)
        print(f"  {nome}  PSNR {np.mean(p):6.2f} dB  SSIM {np.mean(s):.4f}")

    print("\nPor recorte: fração não-nula | PSNR bicúbica | PSNR época 4 | ganho | "
          "valor médio da época 4 no fundo (níveis de 0..8300; o correto é 0)")
    pb, ps = por_recorte(bic, calculate_psnr), por_recorte(sr[4], calculate_psnr)
    linhas = []
    for i in range(len(hr)):
        nz = (hr[i] > -1).float().mean().item()
        fundo = hr[i] <= -1
        val = ((sr[4][i][fundo] + 1) / 2 * FAIXA[1]).mean().item() if fundo.any() else float("nan")
        linhas.append((nz, pb[i], ps[i], val))
    for nz, b, s, val in sorted(linhas):
        fundo = "" if np.isnan(val) else f"{val:7.1f}"
        print(f"  {nz:5.2f} | {b:6.2f} | {s:6.2f} | {s - b:+5.2f} | {fundo}")


def calibrar_adv(args):
    """
    Quanto o termo adversarial puxa o Gerador, comparado à VGG x0,006, para
    cada variante de D. Um D recém-criado não serve para isso (ainda não
    discrimina nada), então cada D é treinado por --passos contra o Gerador
    da época 4 CONGELADO, como no início da fase GAN da próxima rodada, e a
    razão é medida ao longo do treino.

    Motivo: com normalização espectral (patch_sn) o gradiente do D em relação
    à imagem é limitado por construção, e o mesmo --adv-weight do D compacto
    deixaria o adversarial abafado de novo.
    """
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    imgs = [carregar(k) for k in FATIAS]
    g = carregar_gerador(args.pesos, 4)
    vgg = FeatureExtractorVGG("relu3_4").eval()
    lr_m, hr_m = recortes_aleatorios(imgs, random.Random(123), 4, 256, 0.95)
    with torch.no_grad():
        sr_m = g(lr_m)
    x = sr_m.clone().requires_grad_(True)
    g_vgg, = torch.autograd.grad(0.006 * F.mse_loss(vgg(denormalize(x)), vgg(denormalize(hr_m))), x)
    g_pix, = torch.autograd.grad(F.mse_loss(x, hr_m), x)
    print(f"Referência (G da época 4): |grad VGG x0,006| = {g_vgg.norm():.3e} | "
          f"|grad MSE x1| = {g_pix.norm():.3e}")

    for variante in args.variantes:
        torch.manual_seed(0)
        D = Discriminator(variante=variante).train()
        opt = torch.optim.Adam(D.parameters(), lr=1e-4)
        rng = random.Random(0)
        for passo in range(args.passos + 1):
            if passo % 50 == 0:
                gr, logit, _ = grad_adversarial(D.eval(), sr_m)
                D.train()
                p_falso = torch.sigmoid(logit).mean().item()
                print(f"  [{variante:8s}] passo {passo:4d}: D_fake {p_falso:.3f} | "
                      f"|grad adv| sem peso = {gr.norm():.3e} | razão adv/VGG com "
                      f"adv-weight 1e-3: {1e-3 * gr.norm() / g_vgg.norm():.3f} | "
                      f"peso p/ razão 1: {g_vgg.norm() / gr.norm():.2e}", flush=True)
            if passo == args.passos:
                break
            lr, hr = recortes_aleatorios(imgs, rng, 4, 256, 0.5)
            with torch.no_grad():
                sr = g(lr)
            pr, pf = D(hr), D(sr)
            loss = (F.binary_cross_entropy_with_logits(pr, torch.full_like(pr, 0.9))
                    + F.binary_cross_entropy_with_logits(pf, torch.zeros_like(pf))) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = p.add_subparsers(dest="comando", required=True)
    for nome in ("evolucao", "gradientes", "ajuste-fino", "bicubica", "calibrar-adv"):
        s = sub.add_parser(nome)
        s.add_argument("--pesos", required=True,
                       help="pasta weights/ do job 1879476 (gen_epoch_*.pth, disc_epoch_*.pth)")
        if nome == "ajuste-fino":
            s.add_argument("--perda", choices=["vgg", "pix", "vgg+pix"], required=True)
            s.add_argument("--camada", choices=sorted(CAMADAS_VGG), default="relu3_4")
            s.add_argument("--avgpool", action="store_true",
                           help="troca os MaxPool da VGG por AvgPool")
            s.add_argument("--passos", type=int, default=100)
            s.add_argument("--threads", type=int, default=4)
        if nome == "calibrar-adv":
            s.add_argument("--variantes", nargs="+", default=["compacto", "patch_sn"])
            s.add_argument("--passos", type=int, default=200)
            s.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    {"evolucao": evolucao, "gradientes": gradientes,
     "ajuste-fino": ajuste_fino, "bicubica": bicubica,
     "calibrar-adv": calibrar_adv}[args.comando](args)

if __name__ == "__main__":
    main()
