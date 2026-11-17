"""
Testes em CPU da rodada de variantes, com dados SINTÉTICOS e poucas fatias.

Cobrem o que precisa dar certo na primeira tentativa no Coaraci:
retomada de checkpoint, interrupção por sinal (aviso de walltime), corte pelo
orçamento de tempo, best.pth, procedência do código, a rodada inteira em
miniatura até o relatório final, e o próprio scripts/train_coaraci.slurm.

Uso (da raiz do repositório):
    python -m unittest tests.test_rodada_cpu -v            # tudo, ~15 min
    RAPIDO=1 python -m unittest tests.test_rodada_cpu -v   # sem os de ponta a ponta

Os números de PSNR aqui não significam nada: são modelos de poucas épocas em
dados de brinquedo. O que se testa é o encanamento.
"""
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import tifffile
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
PY = sys.executable
RAPIDO = os.environ.get("RAPIDO") == "1"

# Argumentos que encolhem o treino para CPU. Os últimos valem no argparse,
# então podem vir depois dos da rodada.
PEQUENO = ["--patch-size", "64", "--batch-size", "4", "--patches-per-image", "2",
           "--tiff-normalization", "range", "--tiff-range", "0", "8300",
           "--val-fraction", "0.1", "--test-fraction", "0.1", "--split-gap", "2",
           "--num-workers", "0", "--cache-images", "0", "--sample-interval", "1"]
EXTRA_RODADA = ("--patch-size 64 --batch-size 4 --patches-per-image 2 --split-gap 2 "
                "--num-workers 0 --cache-images 0 --sample-interval 1")


def fatias_sinteticas(pasta, n=40, lado=160):
    """Pilha de n fatias uint16 com círculo de reconstrução e fundo zero."""
    os.makedirs(pasta, exist_ok=True)
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:lado, :lado]
    circ = (yy - lado / 2) ** 2 + (xx - lado / 2) ** 2 < (0.48 * lado) ** 2
    for k in range(n):
        img = 3000 + 1500 * np.sin(xx / 7 + k / 5) * np.cos(yy / 9) + rng.normal(0, 100, (lado, lado))
        tifffile.imwrite(os.path.join(pasta, f"{k:04d}.tiff"),
                         np.clip(img * circ, 0, 8300).astype(np.uint16))
    return pasta


def phantom_pequeno(pasta):
    subprocess.run([PY, "scripts/gerar_phantoms.py", "--saida", pasta, "--tamanho", "512",
                    "--angulos", "5", "--sigmas", "1", "--ruidos", "0"],
                   cwd=REPO, check=True, capture_output=True)
    return pasta


def treinar(run_dir, dados, *extra, esperar=True, timeout=600):
    cmd = [PY, "-u", "-m", "src.train", "--data-dir", dados,
           "--step-edge-dir", os.path.join(run_dir, "_sem_bordas"),
           "--run-dir", run_dir, *PEQUENO, *extra]
    if not esperar:
        # saída em arquivo: um PIPE não lido enche e trava o processo
        os.makedirs(run_dir, exist_ok=True)
        log = open(os.path.join(run_dir, "_saida.txt"), "w")
        return subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True)
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise AssertionError(f"treino falhou ({r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r.stdout


def linhas_csv(run_dir):
    with open(os.path.join(run_dir, "logs", "training_log.csv")) as f:
        return list(csv.DictReader(f))


def resumo(run_dir):
    with open(os.path.join(run_dir, "logs", "resumo.json")) as f:
        return json.load(f)


def esperar_arquivo(caminho, timeout, condicao=lambda c: True):
    fim = time.time() + timeout
    while time.time() < fim:
        if os.path.exists(caminho):
            with open(caminho) as f:
                if condicao(f.read()):
                    return
        time.sleep(0.5)
    raise AssertionError(f"tempo esgotado esperando {caminho}")


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="teste_rodada_")
        cls.dados = fatias_sinteticas(os.path.join(cls.tmp, "raw"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
class TestUnidades(Base):
    def test_validacao_com_recortes_fixos(self):
        """semente_fixa: o mesmo índice dá o mesmo recorte; sem ela, sorteia."""
        from src.data_loader import NeutronDataset, list_supported_images
        arqs = list_supported_images(self.dados)[:4]
        kw = dict(patch_size=64, arquivos=arqs, patches_per_image=3, min_nonzero=0.5,
                  tiff_normalization="range", tiff_range=(0, 8300))
        a = NeutronDataset(self.dados, semente_fixa=42, **kw)
        b = NeutronDataset(self.dados, semente_fixa=42, **kw)
        for idx in range(len(a)):
            self.assertTrue(torch.equal(a[idx]["hr"], b[idx]["hr"]))
            self.assertTrue(torch.equal(a[idx]["hr"], a[idx]["hr"]))
        # índices diferentes da mesma imagem continuam sendo recortes diferentes
        self.assertFalse(torch.equal(a[0]["hr"], a[4]["hr"]))
        livre = NeutronDataset(self.dados, **kw)
        self.assertFalse(all(torch.equal(livre[0]["hr"], livre[0]["hr"]) for _ in range(5)))

    def test_discriminadores(self):
        from src.model import Discriminator
        x = torch.randn(2, 1, 64, 64)
        self.assertEqual(tuple(Discriminator(variante="compacto")(x).shape), (2, 1))
        mapa = Discriminator(variante="patch_sn")(x)
        self.assertEqual(mapa.dim(), 4)   # mapa de logits, um por região
        with self.assertRaises(ValueError):
            Discriminator(variante="inexistente")

    def test_energia_grade(self):
        """Ruído branco fica no piso; uma grade de período 4 px domina a medida."""
        from src.utils import energia_grade, piso_grade
        torch.manual_seed(0)
        ruido = torch.randn(4, 1, 64, 64)
        self.assertLess(abs(energia_grade(ruido) - piso_grade(64, 64)), 1.0)
        self.assertAlmostEqual(piso_grade(256, 256), 0.206, places=3)
        yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
        grade = torch.cos(2 * torch.pi * xx / 4) * torch.cos(2 * torch.pi * yy / 4)
        self.assertGreater(energia_grade(ruido * 0.1 + grade), 90.0)

    def test_psnr_com_mascara(self):
        from src.utils import calculate_psnr
        hr = torch.zeros(1, 1, 8, 8) - 1
        hr[..., 2:6, 2:6] = 0.5
        sr = hr.clone()
        sr[..., 0, 0] = 1.0   # erro só no fundo
        mascara = hr > -1
        self.assertTrue(torch.isfinite(calculate_psnr(sr, hr)))
        self.assertEqual(calculate_psnr(sr, hr, mascara=mascara), float("inf"))

    def test_procedencia_fora_do_git(self):
        """Cópia solta (como no job 1879476): sem commit, com hashes e aviso."""
        copia = os.path.join(self.tmp, "copia_solta")
        shutil.copytree(os.path.join(REPO, "src"), os.path.join(copia, "src"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        codigo = ("import argparse; from src.utils import save_run_config, _git_revision; "
                  "a = argparse.Namespace(seed=7); save_run_config('c.json', a); "
                  "print('REV', _git_revision())")
        r = subprocess.run([PY, "-c", codigo], cwd=copia, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("NÃO veio de um clone git", r.stdout)
        self.assertIn("REV None", r.stdout)
        with open(os.path.join(copia, "c.json")) as f:
            cfg = json.load(f)
        self.assertIsNone(cfg["git_commit"])
        self.assertEqual(cfg["semente"], 7)
        self.assertIn("src/train.py", cfg["codigo_sha256"])
        self.assertEqual(cfg["ambiente"]["pacotes"]["torch"].split("+")[0],
                         torch.__version__.split("+")[0])

    def test_procedencia_git_archive(self):
        """git archive grava o commit em src/_versao.py (export-subst)."""
        if shutil.which("git") is None:
            self.skipTest("git ausente")
        repo = os.path.join(self.tmp, "repo_archive")
        os.makedirs(os.path.join(repo, "src"))
        for arq in ("src/_versao.py", "src/utils.py", "src/__init__.py", ".gitattributes"):
            shutil.copy(os.path.join(REPO, arq), os.path.join(repo, arq))
        g = lambda *a: subprocess.run(["git", "-C", repo, *a], check=True, capture_output=True, text=True)
        g("init", "-q")
        g("-c", "user.name=teste", "-c", "user.email=teste@exemplo", "add", ".")
        g("-c", "user.name=teste", "-c", "user.email=teste@exemplo", "commit", "-q", "-m", "teste")
        commit = g("rev-parse", "HEAD").stdout.strip()
        destino = os.path.join(self.tmp, "extraido")
        os.makedirs(destino)
        arquivo = os.path.join(self.tmp, "codigo.tar")
        g("archive", "-o", arquivo, "HEAD")
        subprocess.run(["tar", "-xf", arquivo, "-C", destino], check=True)
        r = subprocess.run([PY, "-c", "from src.utils import _git_revision; print(_git_revision())"],
                           cwd=destino, capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), commit + "-archive", r.stderr)

    def test_avaliacao_recusa_configs_diferentes(self):
        """Dois Geradores treinados com divisões diferentes não vão à mesma tabela."""
        from src.model import Generator
        pesos = []
        for nome, gap in (("A", 2), ("B", 5)):
            d = os.path.join(self.tmp, f"cfg_{nome}")
            os.makedirs(os.path.join(d, "logs"))
            os.makedirs(os.path.join(d, "weights"))
            torch.save(Generator().state_dict(), os.path.join(d, "weights", "best.pth"))
            with open(os.path.join(d, "logs", "config.json"), "w") as f:
                json.dump({"argumentos": {"tiff_normalization": "range", "tiff_range": [0, 8300],
                                          "val_fraction": 0.1, "test_fraction": 0.1,
                                          "split_gap": gap, "lr_scale": 4}}, f)
            pesos.append(f"{nome}={os.path.join(d, 'weights', 'best.pth')}")
        r = subprocess.run([PY, "-m", "src.classical_baselines", "--data-dir", self.dados,
                            "--gen-weights", *pesos, "--output", os.path.join(self.tmp, "x.csv")],
                           cwd=REPO, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("split_gap", r.stdout + r.stderr)


# ---------------------------------------------------------------------------
class TestTreino(Base):
    def test_best_e_colunas_novas(self):
        run = os.path.join(self.tmp, "best")
        treinar(run, self.dados, "--epochs", "3", "--pretrain-epochs", "1",
                "--pixel-weight", "1", "--content-weight", "0.006")
        linhas = linhas_csv(run)
        self.assertEqual([int(l["epoch"]) for l in linhas], [0, 1, 2])
        for col in ("val_grade_pct", "grad_vgg", "grad_adv", "adv_weight_efetivo",
                    "tempo_epoca_s", "mem_pico_gb"):
            self.assertIn(col, linhas[0])
        # a fase GAN registra a MSE de pixel e as normas de gradiente
        self.assertNotEqual(linhas[2]["loss_content_mse"], "")
        self.assertNotEqual(linhas[2]["grad_adv"], "")
        # best.pth = melhor val_psnr do CSV, sem depender do intervalo de salvamento
        melhor = max(linhas, key=lambda l: float(l["val_psnr"]))
        with open(os.path.join(run, "weights", "best.json")) as f:
            best = json.load(f)
        self.assertEqual(best["melhor_epoca"], int(melhor["epoch"]))
        self.assertTrue(os.path.isfile(os.path.join(run, "weights", "best.pth")))
        self.assertTrue(os.path.isfile(os.path.join(run, "weights", "gen_ultimo.pth")))
        self.assertTrue(os.path.isfile(os.path.join(run, "weights", "srresnet_fim_pretreino.pth")))
        self.assertEqual(resumo(run)["melhor_epoca"], best["melhor_epoca"])

    def test_retomada_reproduz_execucao_continua(self):
        """3 épocas seguidas == 2 épocas + retomada até 3 (pesos idênticos)."""
        comum = ["--pretrain-epochs", "1", "--pixel-weight", "1", "--content-weight", "0.006",
                 "--adv-razao", "0.5", "--discriminator", "patch_sn"]
        continua = os.path.join(self.tmp, "continua")
        treinar(continua, self.dados, "--epochs", "3", *comum)
        partida = os.path.join(self.tmp, "partida")
        treinar(partida, self.dados, "--epochs", "2", *comum)
        saida = treinar(partida, self.dados, "--epochs", "3", *comum,
                        "--resume", os.path.join(partida, "weights", "checkpoint_last.pth"))
        self.assertIn("Retomando da época 2", saida)
        self.assertEqual([int(l["epoch"]) for l in linhas_csv(partida)], [0, 1, 2])
        a = torch.load(os.path.join(continua, "weights", "gen_ultimo.pth"))
        b = torch.load(os.path.join(partida, "weights", "gen_ultimo.pth"))
        dif = max((a[k].float() - b[k].float()).abs().max().item() for k in a)
        self.assertLess(dif, 1e-5, f"retomada diverge da execução contínua (máx {dif:.2e})")
        # o recorde do best sobreviveu à retomada
        self.assertEqual(resumo(continua)["melhor_epoca"], resumo(partida)["melhor_epoca"])
        # o peso adaptativo também
        self.assertEqual(linhas_csv(continua)[2]["adv_weight_efetivo"],
                         linhas_csv(partida)[2]["adv_weight_efetivo"])

    def test_sinal_interrompe_limpo(self):
        """SIGUSR1/SIGTERM (aviso de walltime): sai com 0, sem época pela metade."""
        for sinal in (signal.SIGUSR1, signal.SIGTERM):
            with self.subTest(sinal=sinal.name):
                run = os.path.join(self.tmp, f"sinal_{sinal.name}")
                p = treinar(run, self.dados, "--epochs", "1000", "--pretrain-epochs", "1",
                            "--adv-weight", "1e-3", esperar=False)
                esperar_arquivo(os.path.join(run, "logs", "training_log.csv"), 300,
                                lambda c: c.count("\n") >= 3)   # cabeçalho + 2 épocas
                p.send_signal(sinal)
                p.wait(timeout=120)
                with open(os.path.join(run, "_saida.txt")) as f:
                    saida = f.read()
                self.assertEqual(p.returncode, 0, saida[-2000:])
                self.assertIn("interrompida e descartada", saida)
                self.assertEqual(resumo(run)["motivo_fim"], f"interrompido por {sinal.name}")
                # o checkpoint é a última época COMPLETA, igual à última linha do CSV
                ck = torch.load(os.path.join(run, "weights", "checkpoint_last.pth"),
                                weights_only=False)
                self.assertEqual(ck["epoch"], int(linhas_csv(run)[-1]["epoch"]))

    def test_orcamento_de_tempo(self):
        run = os.path.join(self.tmp, "orcamento")
        orcamento_h = 0.004   # 14,4 s
        treinar(run, self.dados, "--epochs", "100000", "--pretrain-epochs", "100000",
                "--max-horas", str(orcamento_h))
        r = resumo(run)
        self.assertIn("orçamento", r["motivo_fim"])
        self.assertGreaterEqual(len(linhas_csv(run)), 1)
        # não estoura o orçamento por mais que a duração de uma época
        duracao = max(float(l["tempo_epoca_s"]) for l in linhas_csv(run))
        self.assertLess(r["horas"] * 3600, orcamento_h * 3600 + 2 * duracao + 5)


# ---------------------------------------------------------------------------
@unittest.skipIf(RAPIDO, "RAPIDO=1")
class TestRodada(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.bordas = phantom_pequeno(os.path.join(cls.tmp, "bordas"))

    def rodar(self, saida, prazo_s, interromper_em=None):
        env = {**os.environ, "PYTHON": PY, "PRAZO_FINAL": str(int(time.time()) + prazo_s),
               "RESERVA_MIN": "1", "EXTRA_TREINO": EXTRA_RODADA, "WORKERS": "0", "CACHE": "0"}
        log = saida + "_saida.txt"
        with open(log, "w") as f:
            p = subprocess.Popen(["bash", "scripts/rodada_variantes.sh", self.dados, self.bordas,
                                  saida], cwd=REPO, env=env, stdout=f,
                                 stderr=subprocess.STDOUT, text=True)
        if interromper_em:
            esperar_arquivo(os.path.join(saida, interromper_em, "logs", "config.json"), 600)
            time.sleep(3)
            p.send_signal(signal.SIGUSR1)
        p.wait(timeout=1500)
        with open(log) as f:
            return p.returncode, f.read()

    def test_rodada_completa(self):
        saida = os.path.join(self.tmp, "rodada")
        rc, out = self.rodar(saida, prazo_s=5 * 60)
        self.assertEqual(rc, 0, out[-4000:])
        with open(os.path.join(saida, "estado_fases.txt")) as f:
            estado = f.read()
        for fase in ("V0_srresnet", "V1_perceptual", "V2_adv_forte", "V3_patch_sn"):
            self.assertIn(f"{fase}: código 0", estado)
            self.assertTrue(os.path.isfile(os.path.join(saida, fase, "weights", "best.pth")), fase)
        # V1-V3 partiram do best.pth do V0
        for fase in ("V1_perceptual", "V2_adv_forte", "V3_patch_sn"):
            with open(os.path.join(saida, fase, "logs", "config.json")) as f:
                cfg = json.load(f)["argumentos"]
            self.assertTrue(cfg["init_generator"].endswith("V0_srresnet/weights/best.pth"))
        self.assertEqual(json.load(open(os.path.join(saida, "V3_patch_sn", "logs",
                                                     "config.json")))["argumentos"]["adv_razao"], 0.5)
        with open(os.path.join(saida, "avaliacao", "relatorio_final.txt")) as f:
            rel = f.read()
        for linha in ("bicubic", "lanczos", "bicubic+unsharp", "V0_srresnet ", "V1_perceptual ",
                      "V2_adv_forte ", "V3_patch_sn ", "V3_patch_sn_ultimo", "MTF10",
                      "DEGRADAÇÃO SINTÉTICA", "PSNR fov", "Avaliação terminou com código 0"):
            self.assertIn(linha, rel)
        self.assertTrue(os.path.isfile(os.path.join(saida, "avaliacao", "teste.csv")))
        self.assertTrue(os.path.isfile(os.path.join(saida, "avaliacao", "teste_mtf.csv")))
        self.assertTrue(os.path.isfile(os.path.join(saida, "ambiente", "pip_freeze.txt")))

    def test_rodada_interrompida_ainda_avalia(self):
        """Aviso de walltime durante o V1: pula V2/V3 e o relatório sai mesmo assim."""
        saida = os.path.join(self.tmp, "rodada_int")
        rc, out = self.rodar(saida, prazo_s=30 * 60, interromper_em="V1_perceptual")
        self.assertEqual(rc, 0, out[-4000:])
        with open(os.path.join(saida, "estado_fases.txt")) as f:
            estado = f.read()
        self.assertIn("V2_adv_forte: pulada (interrupção)", estado)
        self.assertIn("V3_patch_sn: pulada (interrupção)", estado)
        with open(os.path.join(saida, "avaliacao", "relatorio_final.txt")) as f:
            rel = f.read()
        self.assertIn("V0_srresnet ", rel)
        self.assertIn("Avaliação terminou com código 0", rel)


# ---------------------------------------------------------------------------
@unittest.skipIf(RAPIDO, "RAPIDO=1")
class TestSlurm(Base):
    """
    Roda scripts/train_coaraci.slurm inteiro numa CÓPIA do projeto, com o
    ambiente do SLURM simulado: 'module' vira no-op, sem sbcast (cai no cp),
    TESTE_CPU=1 pula a checagem de GPU. A cópia não é um clone git, como no
    job 1879476, então o aviso de procedência precisa aparecer.
    """
    def test_script_slurm_de_ponta_a_ponta(self):
        proj = os.path.join(self.tmp, "projeto")
        for pasta in ("src", "scripts"):
            shutil.copytree(os.path.join(REPO, pasta), os.path.join(proj, pasta),
                            ignore=shutil.ignore_patterns("__pycache__"))
        venv = os.path.dirname(os.path.dirname(PY))
        os.symlink(venv, os.path.join(proj, "venv"))
        shutil.copytree(self.dados, os.path.join(proj, "data", "raw"))
        os.makedirs(os.path.join(proj, "logs"))
        job = "999001"
        env = {**os.environ, "SLURM_SUBMIT_DIR": proj, "SLURM_JOB_ID": job,
               "SLURM_JOB_END_TIME": str(int(time.time()) + 6 * 60),
               "SCRATCH_RAIZ": os.path.join(self.tmp, "scratch"), "MODULOS": "",
               "TESTE_CPU": "1", "RESERVA_MIN": "2", "EXTRA_TREINO": EXTRA_RODADA,
               "WORKERS": "0", "CACHE": "0", "BASH_FUNC_module%%": "() { :; }"}
        r = subprocess.run(["bash", os.path.join(proj, "scripts", "train_coaraci.slurm")],
                           cwd=proj, env=env, capture_output=True, text=True, timeout=1800)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-5000:])
        self.assertIn("gerando os sintéticos", out)
        self.assertIn("NÃO veio de um clone git", out)
        res = os.path.join(proj, "runs", job)
        for item in ("V0_srresnet/weights/best.pth", "V3_patch_sn/weights/best.pth",
                     "avaliacao/relatorio_final.txt", "ambiente/pip_freeze.txt",
                     "estado_fases.txt"):
            self.assertTrue(os.path.exists(os.path.join(res, item)), item)
        with open(os.path.join(res, "V0_srresnet", "logs", "config.json")) as f:
            cfg = json.load(f)
        self.assertIsNone(cfg["git_commit"])
        self.assertTrue(cfg["codigo_sha256"])
        # o scratch do nó é limpo depois da cópia
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "scratch", f"srgan-{job}")))


if __name__ == "__main__":
    unittest.main()
