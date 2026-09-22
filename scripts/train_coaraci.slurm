#!/bin/bash
#===============================================================================
# Treinamento SRGAN no cluster Coaraci (SLURM)
#
# Ainda falta preencher (peça ao suporte do Coaraci, ou copie do script SLURM
# do orientador, que já roda no cluster):
#
#   <PARTICAO>       fila com GPU           -> sinfo -o "%P %G %l"
#   <CONTA>          conta de alocação      -> sacctmgr show assoc user=$USER format=account
#   <N_GPUS>         GPUs por job
#   <TEMPO_MAXIMO>   walltime HH:MM:SS      (ex.: 24:00:00)
#   <MODULOS>        módulos a carregar     -> module avail
#   <RAIZ_SCRATCH>   raiz do scratch do nó  (ex.: /scratch/$USER)
#
# Submissão:  sbatch scripts/train_coaraci.slurm
# Fila:       squeue -u $USER
# Retomada:   sbatch --export=ALL,RESUME=runs/<job>/weights/checkpoint_last.pth \
#                    scripts/train_coaraci.slurm
#===============================================================================

#SBATCH --job-name=srgan-neutrons
#SBATCH --partition=<PARTICAO>
#SBATCH --account=<CONTA>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:<N_GPUS>
#SBATCH --time=<TEMPO_MAXIMO>
#SBATCH --output=out_%j.txt
#SBATCH --error=err_%j.txt

set -euo pipefail

MODULOS="<MODULOS>"
SCRATCH_RAIZ="<RAIZ_SCRATCH>"

PROJETO="${SLURM_SUBMIT_DIR}"
TRABALHO="${SCRATCH_RAIZ}/srgan-${SLURM_JOB_ID}"

echo "[$(hostname)] job ${SLURM_JOB_ID} | projeto ${PROJETO} | scratch ${TRABALHO}"

# --- Ambiente ---------------------------------------------------------------
module purge
module load ${MODULOS}
source "${PROJETO}/venv/bin/activate"

# Explícito, em vez de contar com o SLURM iniciar no diretório de submissão:
# 'python -m src.train' precisa da raiz do projeto no caminho de import.
cd "${PROJETO}"

# --- Dados no scratch -------------------------------------------------------
# Ler as radiografias direto do home castiga o filesystem compartilhado durante
# todo o treino. Copia-se uma vez para o scratch e lê-se de lá.
mkdir -p "${TRABALHO}/data"
cp -r "${PROJETO}/data/raw" "${TRABALHO}/data/"
cp -r "${PROJETO}/data/step_edges" "${TRABALHO}/data/" 2>/dev/null || \
    echo "AVISO: sem data/step_edges — o treino roda, mas sem a validação de MTF"

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
    --amp \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --seed 42 \
    ${ARG_RESUME}

echo "Treinamento concluído."
