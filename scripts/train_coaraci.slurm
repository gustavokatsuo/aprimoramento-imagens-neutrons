#!/bin/bash
#===============================================================================
# Treinamento SRGAN no cluster Coaraci (SLURM)
#
# Antes da primeira submissão (no nó de login, a partir da raiz do repo):
#   1. module load gnu12 hwloc cuda python/3.11.7-gcc-12.2.0-jm3y27n
#      python -m venv venv && venv/bin/python -m pip install -r requirements.txt
#   2. baixar os pesos da VGG19 (os nós de computação podem não ter internet):
#      venv/bin/python -c 'from torchvision.models import vgg19, VGG19_Weights; vgg19(weights=VGG19_Weights.DEFAULT)'
#   3. colocar as fatias em data/raw/ (de preferência também data/raw.zip)
#
#   mkdir -p logs                                  # uma vez: o SLURM não cria
#   sbatch scripts/train_coaraci.slurm
#   sbatch --export=ALL,RESUME=runs/<job>/weights/checkpoint_last.pth scripts/train_coaraci.slurm
#   sbatch --export=ALL,WORKERS=2,CACHE=500 scripts/train_coaraci.slurm
#
# Encadear jobs (treino maior que o walltime):
#   J1=$(sbatch --parsable scripts/train_coaraci.slurm)
#   J2=$(sbatch --parsable --dependency=afterany:$J1 \
#        --export=ALL,RESUME=runs/$J1/weights/checkpoint_last.pth scripts/train_coaraci.slurm)
#===============================================================================

#SBATCH --job-name=srgan-neutrons
#SBATCH --partition=gpu-x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
# #SBATCH --mem=64G        # descomente se a fila exigir; confirme o limite com o suporte

# Sem -u: o Lmod (module) e o activate do venv leem variáveis possivelmente
# não definidas e abortariam o job com "unbound variable".
set -eo pipefail

PROJETO="${SLURM_SUBMIT_DIR}"
if [[ ! -f "${PROJETO}/src/train.py" ]]; then
    echo "ERRO: submeta a partir da raiz do repositório (não achei src/train.py em ${PROJETO})"
    exit 1
fi

MODULOS="${MODULOS:-gnu12 hwloc cuda python/3.11.7-gcc-12.2.0-jm3y27n}"
SCRATCH_RAIZ="${SCRATCH_RAIZ:-/scratch/local/${USER}}"
TRABALHO="${SCRATCH_RAIZ}/srgan-${SLURM_JOB_ID}"

# Memória do cache = CACHE x WORKERS x 15,7 MiB, e de novo para o loader de
# validação. 4 x 200 -> ~12 GiB (treino) + ~12 GiB (validação).
WORKERS="${WORKERS:-4}"
CACHE="${CACHE:-200}"

echo "[$(hostname)] job ${SLURM_JOB_ID} | GPUs ${CUDA_VISIBLE_DEVICES:-nenhuma} | workers ${WORKERS} | cache ${CACHE}"

# --- Ambiente ----------------------------------------------------------------
module purge
module load ${MODULOS}
source "${PROJETO}/venv/bin/activate"
cd "${PROJETO}"

# --- Verificações rápidas, antes de mover 15 GiB -----------------------------
python - <<'EOF'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("ERRO: torch não enxerga GPU (build CPU-only ou CUDA incompatível com o driver)")
print(f"torch {torch.__version__} | {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}")
EOF

TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
export TORCH_HOME
if [[ ! -f "${TORCH_HOME}/hub/checkpoints/vgg19-dcbb9e9d.pth" ]]; then
    echo "ERRO: pesos da VGG19 ausentes em ${TORCH_HOME}. No nó de login, rode:"
    echo "  venv/bin/python -c 'from torchvision.models import vgg19, VGG19_Weights; vgg19(weights=VGG19_Weights.DEFAULT)'"
    exit 1
fi

# --- Dados para o scratch do nó ----------------------------------------------
mkdir -p "${TRABALHO}/data"

# Recuperação: roda em qualquer saída (fim normal, erro ou walltime)
recuperar_resultados() {
    if [[ ! -d "${TRABALHO}/run" ]]; then
        echo "AVISO: nada a recuperar em ${TRABALHO}/run"
        return
    fi
    mkdir -p "${PROJETO}/runs"
    if cp -r "${TRABALHO}/run" "${PROJETO}/runs/${SLURM_JOB_ID}"; then
        echo "Resultados em ${PROJETO}/runs/${SLURM_JOB_ID}"
        rm -rf "${TRABALHO}"
    else
        echo "ERRO: falha ao copiar; resultados seguem em ${TRABALHO}/run"
    fi
}
trap recuperar_resultados EXIT

if [[ -f "${PROJETO}/data/raw.zip" ]] && command -v sbcast >/dev/null 2>&1; then
    echo "Enviando data/raw.zip via sbcast..."
    sbcast -f "${PROJETO}/data/raw.zip" "${TRABALHO}/raw.zip"
    unzip -q "${TRABALHO}/raw.zip" -d "${TRABALHO}/data/"
    rm -f "${TRABALHO}/raw.zip"
else
    echo "Copiando data/raw (sem raw.zip)..."
    cp -r "${PROJETO}/data/raw" "${TRABALHO}/data/"
fi
cp -r "${PROJETO}/data/step_edges" "${TRABALHO}/data/" 2>/dev/null || true

N_FATIAS=$(find "${TRABALHO}/data/raw" -maxdepth 1 -type f \( -name '*.tif*' -o -name '*.fits' \) | wc -l)
N_BORDAS=$(find "${TRABALHO}/data/step_edges" -maxdepth 1 -type f \( -name '*.tif*' -o -name '*.fits' -o -name '*.png' \) 2>/dev/null | wc -l)
echo "Fatias no scratch: ${N_FATIAS} | phantoms de borda: ${N_BORDAS}"
if (( N_FATIAS < 1000 )); then
    echo "AVISO: só ${N_FATIAS} fatias — com val/test 0,10 e gap 30 a divisão fica pobre ou impossível"
fi
if (( N_BORDAS == 0 )); then
    echo "AVISO: sem phantoms de borda — o treino roda, mas sem a validação de MTF"
fi

ARG_RESUME=""
if [[ -n "${RESUME:-}" ]]; then
    cp "${PROJETO}/${RESUME}" "${TRABALHO}/retomar.pth"
    ARG_RESUME="--resume ${TRABALHO}/retomar.pth"
    echo "Retomando de ${RESUME}"
fi

# --- Treinamento -------------------------------------------------------------
# Roda em segundo plano para que o bash consiga reagir ao USR1, que o SLURM
# envia 10 min antes do walltime (--signal=B:USR1@600). Em primeiro plano, o
# trap só rodaria depois que o python terminasse.
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
    --cache-images "${CACHE}" \
    --amp \
    --num-workers "${WORKERS}" \
    --seed 42 \
    ${ARG_RESUME} &
PID=$!

trap 'echo "Walltime próximo: encerrando o treino (o último checkpoint completo vale)"; kill -TERM ${PID} 2>/dev/null' USR1

set +e
wait "${PID}"; RC=$?
# Se o wait foi interrompido pelo USR1, espera o python de fato terminar
if (( RC > 128 )) && kill -0 "${PID}" 2>/dev/null; then
    wait "${PID}"; RC=$?
fi
set -e

echo "Treino terminou com código ${RC}"
exit ${RC}
