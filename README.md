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

**Faixa de normalização.** Dividir pelo teto do dtype só é adequado quando a aquisição ocupa a faixa toda. Nas reconstruções deste projeto o valor máximo é ~8.300 de 65.535 — **12,7%** —, de modo que o modo `dtype` comprime tudo em [0, 0.13] e desperdiça 87% da faixa de saída do Gerador. Para uma pilha tomográfica o modo correto é:

```bash
--tiff-normalization range --tiff-range 0 8300
```

A mesma faixa em todas as fatias preserva a comparabilidade radiométrica entre elas, que é o que dá sentido físico aos tons de cinza — claros para a fase metálica, intermediários para os óxidos, escuros para os poros. O modo `minmax` normaliza cada fatia por si e **destrói** essa comparabilidade: medido nas fatias 2028–2030, as três saem com máximo exatamente 1,0, apagando a diferença real entre elas.

**Área útil.** Reconstruções tomográficas têm um círculo útil inscrito na imagem e zeros nos cantos — nestes dados, 24,7% da área. Sem filtro, 11,3% dos recortes aleatórios de 256 px caem majoritariamente fora do círculo e ensinariam o Gerador a reproduzir vazio. `--min-nonzero 0.5` resorteia o recorte até que ao menos metade tenha conteúdo, levando esse número a zero.

### Divisão treino / validação / teste

**Divisão aleatória invalida o resultado.** Fatias vizinhas de um mesmo volume tomográfico são quase a mesma imagem — medido nas fatias 2028–2030 deste conjunto:

| par | distância | correlação | SSIM |
|---|---|---|---|
| 2028 vs 2029 | 1 fatia | 0,9962 | **0,9688** |
| 2028 vs 2030 | 2 fatias | 0,9875 | 0,9165 |
| par não relacionado | — | −0,0003 | 0,0055 |

Sorteadas, o conjunto de teste conteria quase-duplicatas do treino e a métrica final não mediria generalização alguma. A divisão é feita em **blocos contíguos** ao longo de z:

```bash
python -m src.train --data-dir data/raw \
    --val-fraction 0.10 --test-fraction 0.10 --split-gap 30
```

`--split-gap` descarta fatias nas fronteiras para que a mesma estrutura não apareça em dois conjuntos. A margem precisa exceder a extensão em z das estruturas de interesse: com voxel de 3,65 µm, as partículas de 35,7 µm do segundo pico da amostra atravessam cerca de 10 fatias.

As colunas `val_psnr` e `val_ssim` do CSV são as **únicas** que medem generalização — `psnr` e `ssim_last_batch` são calculadas sobre os mesmos patches que treinaram o Gerador e sobem mesmo quando o modelo apenas decora. O bloco de teste fica intocado durante o treino.

> A ordenação dos arquivos é **natural**, não alfabética: `10.tiff` vem depois de `9.tiff`. Sem isso, um bloco contíguo na lista seria descontíguo no volume.

**Cache.** Sem cache, cada recorte relê e renormaliza a imagem inteira para extrair 256×256 pixels dela: 4,8 ms de leitura (servida pelo cache de página do sistema) e 28,3 ms normalizando 4,1 milhões de pixels. Com `--patches-per-image` alto, essa renormalização repetida domina o carregamento. `--cache-images` guarda as imagens já normalizadas e reduz o custo por recorte de **29,3 ms para 1,8 ms — 16×**.

O cache vive em cada processo do DataLoader, então a memória é `cache × num_workers`. Nas 2030 fatias, o cache completo ocupa 31,1 GiB por worker. Como o cache remove o gargalo, poucos workers com cache grande rendem mais que muitos workers sem cache — por exemplo `--num-workers 2 --cache-images 500` cabe em ~15 GiB no total.

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

O detalhamento de cada métrica, suas limitações e o protocolo de comparação estão em [`docs/resolution_metrics.md`](docs/resolution_metrics.md). A caracterização do conjunto de dados — escala física, faixa dinâmica, geometria da reconstrução, correlação entre fatias e as decisões de pipeline que decorrem de cada uma — está em [`docs/dataset.md`](docs/dataset.md).

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
│   ├── dataset.md              # Caracterização dos dados e decisões que decorrem
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

2. **Instalar as dependências** (testado em Python 3.11):
   ```bash
   python -m venv venv
   venv/bin/python -m pip install -r requirements.txt
   ```

   **Em máquina com GPU**, instale `torch` e `torchvision` *antes*, a partir do
   índice correspondente ao runtime CUDA do sistema — caso contrário o pip traz
   o build do índice padrão, que pode não casar com os drivers:
   ```bash
   venv/bin/python -m pip install torch==2.13.0 torchvision==0.28.0 \
       --index-url https://download.pytorch.org/whl/cu124
   venv/bin/python -m pip install -r requirements.txt
   ```
   Confirme a versão de CUDA com `nvidia-smi` ou com o suporte do cluster
   (`cu121`, `cu124`, `cu126`… variam por instalação).

3. **Colocar os dados** em `data/raw/` (e, opcionalmente, os phantoms de borda em `data/step_edges/`). As imagens precisam ser maiores ou iguais ao `--patch-size` usado no treino.

> **Tamanho da época.** Cada radiografia rende `--patches-per-image` recortes aleatórios por época. Com o padrão `1`, um conjunto de 30 imagens dá uma época de 30 amostras — longe do necessário para treinar uma GAN. Para treino real, use dezenas ou centenas; o script avisa quando a época tem menos de 100 amostras.

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
| `--patches-per-image` | `1` | recortes por radiografia em cada época |
| `--tiff-normalization` | `dtype` | `dtype`, `range` ou `minmax` (ver abaixo) |
| `--min-nonzero` | `0.0` | fração mínima de pixels não-nulos num recorte |
| `--cache-images` | `0` | imagens normalizadas mantidas em memória (`-1` = todas) |
| `--val-fraction` | `0.0` | fração reservada para validação, em bloco contíguo |
| `--test-fraction` | `0.0` | fração reservada para teste, em bloco contíguo |
| `--split-gap` | `0` | fatias descartadas entre os blocos |
| `--lr` | `1e-4` | taxa de aprendizado |
| `--seed` | `42` | semente para reprodutibilidade |
| `--run-dir` | — | agrupa config, logs, pesos e amostras desta execução |
| `--resume` | — | retoma de um `checkpoint_last.pth` |
| `--step-edge-dir` | `data/step_edges` | phantoms para a validação de MTF (opcional) |
| `--vgg-layer` | `relu3_4` | profundidade da VGG na content loss |
| `--content-weight` | `1.0` | peso da content loss (`0.006` = rescale do artigo) |
| `--discriminator` | `compacto` | `compacto` ou `artigo` (Ledig et al. 2017) |
| `--amp` | desligado | precisão mista na GPU (menos memória, mais rápido) |
| `--lr-decay-epochs` | — | épocas em que a taxa decai (ex.: `50 80`) |

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

O script foi feito para submeter **sem editar o arquivo**: o que varia por instalação vem da linha de comando ou do ambiente, e as diretivas `#SBATCH` são apenas defaults, que qualquer opção passada ao `sbatch` sobrepõe.

```bash
sbatch scripts/train_coaraci.slurm                            # tenta com os defaults
sbatch -p gpu -A meu-projeto scripts/train_coaraci.slurm      # partição e conta
sbatch --export=ALL,MODULOS="python/3.11 cuda/12.4" scripts/train_coaraci.slurm
```

Partição e conta ficam fora do arquivo de propósito: seus nomes variam por cluster e um valor errado faz o `sbatch` recusar o job de imediato. Sem a diretiva, o SLURM usa a partição padrão do site e dispensa a conta quando ela não é exigida. A raiz do scratch é descoberta na ordem `$SCRATCH_RAIZ`, `$SCRATCH`, `$SLURM_TMPDIR`, `$TMPDIR` e, em último caso, uma pasta dentro do projeto.

Se a primeira submissão for recusada, o próprio SLURM diz o que falta (`Invalid partition`, `Invalid account`); os nomes corretos saem de `sinfo -o "%P %G %l"` e `sacctmgr show assoc user=$USER format=account`.

### Experimento: profundidade da VGG e peso do termo adversarial

A perda do Gerador é `content_weight * loss_content + adv_weight * loss_GAN`. A
magnitude de `loss_content` depende fortemente da profundidade da VGG usada,
de modo que a camada escolhida determina o peso **efetivo** do termo
adversarial — isto é, o quanto o treino é de fato uma GAN e não uma SRResNet.

Medido com `adv_weight = 1e-3`, a parcela adversarial na perda total fica:

| `--vgg-layer` | `--content-weight 1.0` | `--content-weight 0.006` |
|---|---|---|
| `relu2_2` | 0,005 % | 0,88 % |
| `relu3_4` *(padrão do projeto)* | 0,013 % | 2,06 % |
| `relu4_4` | 0,164 % | 21,5 % |
| `relu5_4` *(VGG54 do artigo)* | 1,100 % | 65,0 % |

As duas configurações de referência, com a mesma semente e os mesmos dados:

```bash
# configuração histórica do projeto
python -m src.train --data-dir data/raw --seed 42 \
    --vgg-layer relu3_4 --content-weight 1.0 \
    --run-dir runs/vgg-relu3_4

# configuração do artigo (Ledig et al. 2017)
python -m src.train --data-dir data/raw --seed 42 \
    --vgg-layer relu5_4 --content-weight 0.006 --discriminator artigo \
    --run-dir runs/vgg-relu5_4
```

A comparação se faz pelo `mtf10_sr` de cada execução **contra a coluna
`mtf10_lr_bicubic`** — que é idêntica nas duas, por ser determinística — e não
pelas perdas, que não são comparáveis entre objetivos diferentes. O
`config.json` de cada pasta registra qual código e quais argumentos produziram
cada número.

> Os valores da tabela acima foram medidos com tensores aleatórios e servem
> para mostrar a ordem de grandeza do desequilíbrio. Os números com as
> radiografias reais precisam ser remedidos.

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
