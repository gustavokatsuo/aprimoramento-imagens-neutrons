# Aprimoramento de Imagens para Tomografia com Nêutrons Utilizando Deep Learning

Este repositório contém o código-fonte, protótipos e documentação referentes ao projeto de Iniciação Científica focado na restauração e aprimoramento de imagens obtidas por tomografia com nêutrons (NT). 

O projeto visa superar as limitações de capacidade de generalização de algoritmos clássicos de filtragem (como o BM3D) perante ruídos e distorções complexas, estruturando uma metodologia baseada em Inteligência Artificial para o sistema de imageamento do Reator Multipropósito Brasileiro (RMB).

## Autores e Instituições
* **Pesquisador:** Gustavo Katsuo Tsutsui (IME-USP)
* **Orientador:** Prof. Dr. Frederico A. Genezini (CERPq - IPEN/USP)
* **Instituições:** Universidade de São Paulo (USP) | Instituto de Pesquisas Energéticas e Nucleares (IPEN/CNEN)

---

## Arquitetura do Modelo
A abordagem principal deste projeto utiliza uma **SRGAN** (Super-Resolution Generative Adversarial Network). O modelo é composto por duas redes neurais que "interagem" entre si:
1. **Gerador:** Uma rede convolucional profunda focada em reconstruir imagens de alta resolução a partir de radiografias ruidosas ou de baixa qualidade. Reconstrói em **4x**, por dois blocos de `PixelShuffle` de 2x.
2. **Discriminador:** Uma rede que tenta distinguir entre as imagens reais de alta qualidade e as imagens geradas (falsas).

A função de perda (*Loss*) combina a perda de conteúdo (garantindo que os detalhes físicos da amostra são mantidos) e a perda adversarial (garantindo que a textura e a nitidez pareçam reais).

O treino segue o protocolo de Ledig et al. (2017) em duas fases: as primeiras épocas ajustam apenas o Gerador com MSE pixel-a-pixel (`--pretrain-epochs`), e só então o Discriminador entra. Sem essa etapa o Discriminador domina antes de o Gerador aprender a reconstrução básica.

### Integridade radiométrica
As radiografias são TIFF de 16 bits (65.536 níveis) e o pipeline preserva essa profundidade de ponta a ponta: arquivo → `float32` em [0, 1] → `Normalize(0.5, 0.5)` → [-1, 1], domínio da saída `Tanh` do Gerador. Nenhuma etapa passa por PIL em modo `'L'`, que quantizaria para 8 bits. As métricas desnormalizam de volta para [0, 1] e usam `data_range = 1.0`.

---

## Métricas

PSNR e SSIM medem fidelidade à referência, não resolução — uma imagem suavizada pode pontuar bem em ambos. Para sustentar a meta do projeto, o repositório mede também a **MTF** (Modulation Transfer Function) pelo método da borda inclinada, sobre phantoms de borda de degrau colocados em `data/step_edges/` e usados **somente em validação**, nunca na loss.

A cada validação são medidas três frequências de corte (MTF10) sobre o mesmo phantom:

| coluna | o que é |
|---|---|
| `mtf10_lr_bicubic` | entrada LR levada ao tamanho HR por interpolação bicúbica |
| `mtf10_sr` | saída do Gerador |
| `mtf10_hr` | referência de alta resolução |

A bicúbica é a linha de base que torna o número interpretável: é o que se obtém sem nenhum aprendizado. MTF10 da SR **acima** da bicúbica é evidência de recuperação real de frequência espacial.

O detalhamento de cada métrica, suas limitações e o protocolo de comparação estão em [`docs/resolution_metrics.md`](docs/resolution_metrics.md).

---

## Estrutura do Repositório

```
aprimoramento-imagens-neutrons/
├── data/                       # Ficheiros de dados (conteúdo não versionado)
│   ├── raw/                    # Radiografias originais (TIFF 16-bit, FITS, PNG, JPG)
│   ├── processed/              # Dados após pré-processamento
│   └── step_edges/             # Phantoms de borda para a validação de MTF
├── docs/
│   ├── plano_ic_gustavo.pdf
│   └── resolution_metrics.md   # Métricas de resolução: definição e limitações
├── notebooks/
│   └── arquitetura_ic.ipynb    # Protótipo inicial (ver nota abaixo)
├── scripts/
│   └── train_coaraci.slurm     # Submissão no cluster Coaraci
├── src/                        # Código-fonte modularizado
│   ├── __init__.py
│   ├── data_loader.py          # Leitura científica, degradação e datasets
│   ├── model.py                # Gerador, Discriminador e extrator VGG
│   ├── train.py                # Loop de treino e validação de bordas
│   ├── utils.py                # Métricas, checkpoints, MTF e logs
│   └── classical_baselines.py  # Bicúbico, Lanczos e unsharp para comparação
├── runs/                       # Saídas por execução (não versionadas)
├── requirements.txt
└── README.md                   # Este arquivo
```

> **Nota sobre o notebook:** `notebooks/arquitetura_ic.ipynb` é o protótipo inicial da arquitetura e **não** é o caminho de treino. Ele redefine as redes com assinaturas incompatíveis com `src/model.py` e está mantido apenas como registro histórico do desenvolvimento. O código canônico é o de `src/`.

---

## Pré-requisitos e Instalação

Para executar este projeto localmente, é recomendável a utilização de um ambiente virtual (como `venv` ou `conda`) com suporte para aceleração por GPU (CUDA).

1. **Clonar o repositório:**
   ```bash
   git clone https://github.com/gustavokatsuo/aprimoramento-imagens-neutrons.git
   cd aprimoramento-imagens-neutrons
   ```

2. **Instalar as dependências:**
   ```bash
   python -m venv venv
   venv/bin/python -m pip install -r requirements.txt
   ```
   *(Certifique-se de que as bibliotecas `torch` e `torchvision` estão configuradas corretamente para a sua versão do CUDA — pode ser necessário instalá-las a partir do índice correspondente, ex.: `--index-url https://download.pytorch.org/whl/cu124`).*

3. **Colocar os dados** em `data/raw/` (e, opcionalmente, os phantoms de borda em `data/step_edges/`). As imagens precisam ser maiores ou iguais ao `--patch-size` usado no treino.

---

## Como Executar

### Treino do modelo

O treino é feito por linha de comando, a partir da raiz do repositório:

```bash
python -m src.train --data-dir data/raw --epochs 100 --batch-size 8
```

Argumentos mais usados:

| argumento | padrão | o que faz |
|---|---|---|
| `--data-dir` | `data/raw` | pasta das radiografias de treino |
| `--epochs` | `100` | épocas totais (pré-treino + adversarial) |
| `--pretrain-epochs` | `5` | épocas iniciais só com MSE; `0` desativa |
| `--batch-size` | `8` | batch total (dividido entre GPUs, se houver mais de uma) |
| `--patch-size` | `256` | tamanho do recorte HR |
| `--lr` | `1e-4` | taxa de aprendizado |
| `--seed` | `42` | semente para reprodutibilidade |
| `--run-dir` | — | agrupa config, logs, pesos e amostras desta execução |
| `--resume` | — | retoma de um `checkpoint_last.pth` |
| `--step-edge-dir` | `data/step_edges` | phantoms para a validação de MTF (opcional) |

`python -m src.train --help` lista todos.

**Saídas.** Com `--run-dir runs/experimento-1`, a execução grava:

```
runs/experimento-1/
├── logs/config.json           # argumentos, commit do código, versões, GPU, SLURM
├── logs/training_log.csv      # perdas, PSNR e SSIM por época
├── logs/step_edge_mtf.csv     # MTF10 da bicúbica, da SR e da HR
├── samples/epoch_N.png        # comparação LR | SR | HR
└── weights/
    ├── checkpoint_last.pth    # retomável (modelos, otimizadores, época, RNG)
    └── gen_epoch_N.pth        # pesos para inferência
```

O `config.json` registra os argumentos, o commit do código e o ambiente, de modo que qualquer número do relatório possa ser rastreado até a execução que o produziu.

### Comparação com métodos clássicos

```bash
python -m src.classical_baselines --data-dir data/raw \
    --gen-weights runs/experimento-1/weights/gen_epoch_99.pth
```

Avalia bicúbico, Lanczos, bicúbico+unsharp e a SRGAN no **mesmo** pipeline degradação → reconstrução → métrica, com as mesmas funções de PSNR/SSIM do treino. O unsharp masking entra de propósito: realça contraste sem recuperar frequência espacial, servindo de controle para distinguir ganho real de mero realce visual.

### Treino no cluster (Coaraci)

```bash
sbatch scripts/train_coaraci.slurm
```

O script copia os dados para o scratch do nó, treina, e traz os resultados de volta para `runs/$SLURM_JOB_ID` — inclusive se o job for interrompido por walltime. Múltiplas GPUs do nó são usadas automaticamente via `DataParallel`.

Para retomar um treino interrompido:

```bash
sbatch --export=ALL,RESUME=runs/<job>/weights/checkpoint_last.pth scripts/train_coaraci.slurm
```

> O script ainda tem placeholders (`<PARTICAO>`, `<CONTA>`, `<N_GPUS>`, `<TEMPO_MAXIMO>`, `<MODULOS>`, `<RAIZ_SCRATCH>`) a preencher com os dados do cluster antes da primeira submissão.

---

## Resultados (Em progresso)
*(Esta secção será atualizada com métricas de desempenho — PSNR, SSIM e frequência de corte MTF10 — e exemplos visuais das reconstruções tomográficas à medida que os treinos avançarem).*

---

## Referências Principais
* ALMAU, O.; ALARCÓN, T. E. *A residual dense u-net neural network for image denoising*. IEEE Access, 2021.
* LEDIG, C. et al. *Photo-realistic single image super-resolution using a generative adversarial network*. CVPR, 2017.
* MA, Y. et al. *Enhancing the spatial resolution of neutron radiography with generative adversarial networks*. Journal of Instrumentation, 2025.
* SCHOUERI, R. M. et al. *The new facility for neutron tomography of ipen-cnen/sp and its potential to investigate hydrogenous substances*. Applied Radiation and Isotopes, 2014.

---
*Projeto desenvolvido no âmbito do programa de Iniciação Científica (IC).*
