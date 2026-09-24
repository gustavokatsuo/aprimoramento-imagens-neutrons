#!/bin/bash
#===============================================================================
# Treinamento SRGAN no cluster Coaraci (SLURM)
#
# Feito para ser submetido SEM editar o arquivo: os valores que variam por
# instalação vêm da linha de comando ou do ambiente, e as diretivas abaixo são
# só defaults. Opções passadas ao sbatch têm precedência sobre elas.
#
#   sbatch scripts/train_coaraci.slurm                      # tenta com os defaults
#   sbatch -p gpu -A meu-projeto scripts/train_coaraci.slurm
#   sbatch --export=ALL,MODULOS="python/3.11 cuda/12.4" scripts/train_coaraci.slurm
#
# Se a primeira submissão for recusada, o próprio SLURM diz o que falta
# ("Invalid partition", "Invalid account"); descubra os nomes com:
#
#   sinfo -o "%P %G %l"                              partições, GPUs e walltime
#   sacctmgr show assoc user=$USER format=account    contas de alocação
#   module avail                                     módulos disponíveis
#   echo "$SCRATCH"                                  raiz do scratch, se definida
#
# Fila:      squeue -u $USER
# Retomada:  sbatch --export=ALL,RESUME=runs/<job>/weights/checkpoint_last.pth \
#                   scripts/train_coaraci.slurm
#===============================================================================

#SBATCH --job-name=srgan-neutrons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=out_%j.txt
#SBATCH --error=err_%j.txt
#
# Partição e conta ficam DE FORA de propósito: seus nomes variam por cluster e
# um valor errado faz o sbatch recusar o job na hora. Sem a diretiva, o SLURM
# usa a partição padrão do site e dispensa a conta quando ela não é exigida.
# Passe-as no sbatch quando for necessário:  sbatch -p <fila> -A <conta> ...

set -euo pipefail

# Módulos a carregar. Vazio = nenhum, o que basta quando o venv já traz tudo.
MODULOS="${MODULOS:-}"

# Raiz do scratch, na ordem de preferência: o que foi passado no ambiente, as
# variáveis que os sites costumam definir, e por último o diretório do projeto
# (funciona, mas lê do filesystem compartilhado durante todo o treino).
SCRATCH_RAIZ="${SCRATCH_RAIZ:-${SCRATCH:-${SLURM_TMPDIR:-${TMPDIR:-${SLURM_SUBMIT_DIR}/scratch-local}}}}"

PROJETO="${SLURM_SUBMIT_DIR}"
TRABALHO="${SCRATCH_RAIZ}/srgan-${SLURM_JOB_ID}"

echo "[$(hostname)] job ${SLURM_JOB_ID} | projeto ${PROJETO} | scratch ${TRABALHO}"
echo "GPUs visíveis: ${CUDA_VISIBLE_DEVICES:-nenhuma} | CPUs: ${SLURM_CPUS_PER_TASK}"

# --- Ambiente ---------------------------------------------------------------
if [[ -n "${MODULOS}" ]]; then
    module purge
    module load ${MODULOS}
fi
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
    --val-fraction 0.10 \
    --test-fraction 0.10 \
    --split-gap 30 \
    --min-nonzero 0.5 \
    --cache-images 500 \
    --amp \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --seed 42 \
    ${ARG_RESUME}

echo "Treinamento concluído."
