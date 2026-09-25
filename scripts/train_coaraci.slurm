#!/bin/bash
#===============================================================================
# Treinamento SRGAN no cluster Coaraci (SLURM)
#
# Os valores de partição, GPUs, módulos e scratch foram tirados de um script
# que já roda no Coaraci (script_gpu.slurm, do grupo). Qualquer opção passada
# ao sbatch sobrepõe as diretivas abaixo.
#
#   sbatch scripts/train_coaraci.slurm                       # como está
#   sbatch --gpus 1 --time 8:00:00 scripts/train_coaraci.slurm
#   sbatch --export=ALL,RESUME=runs/<job>/weights/checkpoint_last.pth \
#          scripts/train_coaraci.slurm                       # retomar
#
# Antes da primeira submissão:
#   1. criar o venv no cluster:  python -m venv venv
#                                venv/bin/python -m pip install -r requirements.txt
#   2. colocar as fatias em data/raw/ (ver nota sobre o .zip mais abaixo)
#
# Fila:  squeue -u $USER      Cancelar:  scancel <job>
#===============================================================================

#SBATCH --job-name=srgan-neutrons
#SBATCH --partition=gpu-x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --output=out_%j.txt
#SBATCH --error=err_%j.txt
#
# O Coaraci não exige --account: o script de referência do grupo não traz a
# diretiva. Se alguma fila passar a exigir, o sbatch recusa dizendo o que falta.
# Limite de cores: 32 por nó ao usar duas GPUs.
# Para receber aviso por e-mail, acrescente na submissão:
#   sbatch --mail-user=SEU_EMAIL --mail-type=END,FAIL scripts/train_coaraci.slurm

set -euo pipefail

# Módulos do Coaraci, na ordem usada pelo script de referência do grupo.
MODULOS="${MODULOS:-gnu12 hwloc cuda python/3.11.7-gcc-12.2.0-jm3y27n}"

# Scratch local do nó. A convenção do Coaraci é /scratch/local/$USER/$SLURM_JOB_ID.
SCRATCH_RAIZ="${SCRATCH_RAIZ:-/scratch/local/${USER}}"

PROJETO="${SLURM_SUBMIT_DIR}"
TRABALHO="${SCRATCH_RAIZ}/srgan-${SLURM_JOB_ID}"

# Com --ntasks-per-node (em vez de --cpus-per-task) o SLURM não define
# SLURM_CPUS_PER_TASK. Os workers do DataLoader são dimensionados pelo que
# estiver disponível, com teto de 8: acima disso o ganho é pequeno e cada
# worker carrega sua própria cópia do cache de imagens.
CORES="${SLURM_CPUS_PER_TASK:-${SLURM_NTASKS_PER_NODE:-8}}"
WORKERS=$(( CORES < 8 ? CORES : 8 ))

echo "[$(hostname)] job ${SLURM_JOB_ID}"
echo "  projeto : ${PROJETO}"
echo "  scratch : ${TRABALHO}"
echo "  GPUs    : ${CUDA_VISIBLE_DEVICES:-nenhuma} | cores ${CORES} | workers ${WORKERS}"

# --- Ambiente ---------------------------------------------------------------
module purge
module load ${MODULOS}
source "${PROJETO}/venv/bin/activate"

# Explícito, em vez de contar com o SLURM iniciar no diretório de submissão:
# 'python -m src.train' precisa da raiz do projeto no caminho de import.
cd "${PROJETO}"

# --- Dados para o scratch do nó ---------------------------------------------
# Ler as fatias do filesystem compartilhado durante horas de treino o castiga,
# e são milhares de arquivos pequenos. Duas formas, em ordem de preferência:
#
#   1. data/raw.zip + sbcast  — um arquivo só viaja pela rede, descompactado
#      localmente. É o caminho usado pelo grupo. Criar com:
#          cd data && zip -r -0 raw.zip raw/     (-0: sem compressão, TIFF já é)
#   2. cp -r                  — funciona, mas copia arquivo a arquivo.
mkdir -p "${TRABALHO}/data"

if [[ -f "${PROJETO}/data/raw.zip" ]] && command -v sbcast >/dev/null 2>&1; then
    echo "Enviando data/raw.zip para o scratch via sbcast..."
    sbcast -f "${PROJETO}/data/raw.zip" "${TRABALHO}/raw.zip"
    unzip -q "${TRABALHO}/raw.zip" -d "${TRABALHO}/data/"
    rm -f "${TRABALHO}/raw.zip"
else
    echo "Copiando data/raw para o scratch (sem raw.zip; ver nota no script)..."
    cp -r "${PROJETO}/data/raw" "${TRABALHO}/data/"
fi

cp -r "${PROJETO}/data/step_edges" "${TRABALHO}/data/" 2>/dev/null || \
    echo "AVISO: sem data/step_edges — o treino roda, mas sem a validação de MTF"

echo "Fatias no scratch: $(ls -1 "${TRABALHO}/data/raw" | wc -l)"

# O checkpoint de retomada também precisa estar no scratch
ARG_RESUME=""
if [[ -n "${RESUME:-}" ]]; then
    cp "${PROJETO}/${RESUME}" "${TRABALHO}/retomar.pth"
    ARG_RESUME="--resume ${TRABALHO}/retomar.pth"
    echo "Retomando de ${RESUME}"
fi

# Traz os resultados de volta mesmo se o job for interrompido por walltime ou
# erro: sem isto, um treino de horas some junto com o scratch do nó.
recuperar_resultados() {
    if [[ ! -d "${TRABALHO}/run" ]]; then
        echo "AVISO: nada a recuperar — o treino não chegou a gravar ${TRABALHO}/run"
        return
    fi
    mkdir -p "${PROJETO}/runs"
    if cp -r "${TRABALHO}/run" "${PROJETO}/runs/${SLURM_JOB_ID}"; then
        echo "Resultados em ${PROJETO}/runs/${SLURM_JOB_ID}"
        rm -rf "${TRABALHO}"
    else
        echo "ERRO: falha ao copiar os resultados; eles seguem em ${TRABALHO}/run"
    fi
}
trap recuperar_resultados EXIT

# --- Treinamento ------------------------------------------------------------
# --run-dir agrupa config.json, logs, pesos e amostras desta execução num lugar
# só, de modo que jobs simultâneos não escrevam uns por cima dos outros.
# As GPUs do nó são usadas automaticamente (DataParallel) e --batch-size é o
# batch TOTAL, dividido entre elas.
# python -u complementa o flush=True dos prints: nada fica preso no buffer e o
# out_%j.txt acompanha o treino em tempo real.
#
# Os valores de --tiff-range, --split-gap e --min-nonzero vêm da caracterização
# em docs/dataset.md; leia-a antes de alterá-los.
python -u -m src.train \
    --data-dir      "${TRABALHO}/data/raw" \
    --step-edge-dir "${TRABALHO}/data/step_edges" \
    --run-dir       "${TRABALHO}/run" \
    --epochs 100 \
    --pretrain-epochs 5 \
    --batch-size 16 \
    --lr 1e-4 \
    --lr-decay-epochs 50 80 \
    --patch-size 256 \
    --patches-per-image 100 \
    --tiff-normalization range --tiff-range 0 8300 \
    --min-nonzero 0.5 \
    --val-fraction 0.10 \
    --test-fraction 0.10 \
    --split-gap 30 \
    --cache-images 500 \
    --amp \
    --num-workers "${WORKERS}" \
    --seed 42 \
    ${ARG_RESUME}

echo "Treinamento concluído."
