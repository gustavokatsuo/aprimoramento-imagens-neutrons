"""Roda src/train.py com o D 'patch_sn' SEM normalização espectral (só para o
ensaio): a função é trocada por uma identidade antes de src.model ser lido."""
import sys, os, runpy
import torch.nn as nn
nn.utils.parametrizations.spectral_norm = lambda modulo, *a, **k: modulo
repo = sys.argv[1]; sys.argv = ["src.train"] + sys.argv[2:]
sys.path.insert(0, repo); os.chdir(repo)
from src.model import Discriminator
d = Discriminator(variante="patch_sn")
assert not any("parametrizations" in n for n, _ in d.named_parameters()), "SN ainda ativa"
print("[sem_sn] PatchGAN sem normalização espectral e sem BatchNorm", flush=True)
runpy.run_module("src.train", run_name="__main__")
