#!/bin/bash
#===============================================================================
# Script de treinamento SRGAN no cluster Coaraci (SLURM)
#
# ANTES DE SUBMETER, preencha os campos <ENTRE_COLCHETES> abaixo com as
# informações do cluster (obtenha com a equipe de suporte do Coaraci ou via
# `sinfo`, `sacctmgr show assoc user=$USER` e a documentação do cluster):
#
#   <PARTICAO>        -> nome da partição/fila com GPU (ex.: gpu, paralela)
#                        Descubra com: sinfo -o "%P %G %l"
#   <CONTA>           -> conta/projeto de alocação (se o cluster exigir)
#                        Descubra com: sacctmgr show assoc user=$USER format=account
#   <N_GPUS>          -> número de GPUs por job (1 é suficiente para este treino)
#   <TEMPO_MAXIMO>    -> walltime no formato HH:MM:SS (ex.: 24:00:00)
#   <MODULO_PYTHON>   -> módulo de Python/CUDA a carregar (veja: module avail)
#                        ex.: python/3.11, cuda/12.4 — depende do que o Coaraci oferece
#   <CAMINHO_PROJETO> -> caminho absoluto do repositório no filesystem do cluster
#
# Submissão:   sbatch scripts/train_coaraci.slurm
# Fila:        squeue -u $USER
# Retomada:    adicione --resume weights/checkpoint_last.pth aos argumentos
#===============================================================================

#SBATCH --job-name=srgan-neutrons
#SBATCH --partition=<PARTICAO>
#SBATCH --account=<CONTA>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:<N_GPUS>
#SBATCH --time=<TEMPO_MAXIMO>
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# --- Ambiente ---
module purge
module load <MODULO_PYTHON>
# module load <MODULO_CUDA>   # descomente se CUDA for um módulo separado no Coaraci

cd <CAMINHO_PROJETO>

# Ambiente virtual do projeto (crie uma vez no cluster com:
#   python -m venv venv && venv/bin/pip install -r requirements.txt)
source venv/bin/activate

mkdir -p logs

# --- Treinamento (single-GPU; ver nota sobre DDP abaixo) ---
python -m src.train \
    --data-dir data/raw \
    --epochs 100 \
    --pretrain-epochs 5 \
    --batch-size 8 \
    --lr 1e-4 \
    --patch-size 256 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --seed 42 \
    --log-file logs/training_log.csv
    # --resume weights/checkpoint_last.pth   # descomente para retomar um treino

#===============================================================================
# NOTA SOBRE MULTI-GPU (DDP): o treino atual é single-GPU, o que é adequado
# para o tamanho do dataset e do modelo. Se os experimentos exigirem múltiplas
# GPUs no futuro, o caminho é torchrun + DistributedDataParallel, o que requer
# mudanças em src/train.py (DistributedSampler, inicialização do process group).
# Decisão em aberto — confirmar com o orientador se vale a complexidade.
#===============================================================================
