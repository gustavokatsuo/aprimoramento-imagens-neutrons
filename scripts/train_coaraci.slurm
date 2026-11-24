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
#   sbatch --export=ALL,WORKERS=2,CACHE=500 scripts/train_coaraci.slurm
#
# O job roda a RODADA DE VARIANTES (scripts/rodada_variantes.sh): SRResNet,
# três receitas de fase GAN e a avaliação final contra os métodos clássicos,
# com o tempo dividido pelo relógio. Tudo volta em runs/<job>/.
#
# Recursos alinhados ao job 1879476, que rodou com 1 A30 e 16 tarefas (a
# versão anterior deste script pedia 2 GPUs e 32 tarefas). A conta e a QOS
# vêm da linha de comando de quem submete, como no job 1879476.
#===============================================================================

#SBATCH --job-name=srgan-neutrons
#SBATCH --partition=gpu-x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus=1
#SBATCH --time=24:00:00
# Aviso 45 min antes do walltime: a rodada interrompe o treino em andamento e
# ainda tem tempo de fazer a avaliação final. Normalmente não dispara: as fases
# já terminam ~90 min antes do fim (RESERVA_MIN em rodada_variantes.sh).
#SBATCH --signal=B:USR1@2700
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
# TESTE_CPU=1 existe só para tests/test_rodada_cpu.py rodar este script inteiro
# numa máquina sem GPU. No cluster, não defina.
if [[ "${TESTE_CPU:-0}" == 1 ]]; then
    echo "TESTE_CPU=1: checagem de GPU pulada (teste local, não usar no cluster)"
else
python - <<'EOF'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("ERRO: torch não enxerga GPU (build CPU-only ou CUDA incompatível com o driver)")
print(f"torch {torch.__version__} | {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}")
EOF
fi

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
        # A saída do SLURM vai junto, para que a pasta runs/<job> baste
        cp "${PROJETO}/logs/slurm_${SLURM_JOB_ID}".{out,err} \
           "${PROJETO}/runs/${SLURM_JOB_ID}/" 2>/dev/null || true
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
# '|| true': sem a pasta, o find falha e, com pipefail + set -e, o job morreria
# aqui em silêncio (achado no teste em CPU)
N_BORDAS=$( (find "${TRABALHO}/data/step_edges" -maxdepth 1 -type f \( -name '*.tif*' -o -name '*.fits' -o -name '*.png' \) 2>/dev/null || true) | wc -l)
echo "Fatias no scratch: ${N_FATIAS} | phantoms de borda: ${N_BORDAS}"
if (( N_FATIAS < 1000 )); then
    echo "AVISO: só ${N_FATIAS} fatias — com val/test 0,10 e gap 30 a divisão fica pobre ou impossível"
fi

# Phantoms SINTÉTICOS gerados aqui quando não vieram phantoms com os dados: a
# MTF sai sempre. Eles medem quanto cada método inverte a degradação sintética
# do projeto, NÃO a resolução do equipamento (ver scripts/gerar_phantoms.py).
if (( N_BORDAS == 0 )); then
    echo "Sem phantoms de borda: gerando os sintéticos no scratch..."
    python scripts/gerar_phantoms.py --saida "${TRABALHO}/data/step_edges"
fi

# Faixa de valores da pilha completa (pendência P2 em docs/decisoes.md): o
# teto 0-8300 do treino veio de só 3 fatias. Só mede e registra; não muda a
# normalização. Uma falha aqui não impede o treino.
mkdir -p "${TRABALHO}/run/ambiente"
python scripts/medir_faixa.py "${TRABALHO}/data/raw" "${TRABALHO}/run/ambiente/faixa_pilha.json" \
    --teto 8300 || echo "AVISO: não foi possível medir a faixa da pilha (o treino segue)"

# --- Procedência do código -------------------------------------------------------
if git -C "${PROJETO}" rev-parse HEAD >/dev/null 2>&1; then
    echo "Código: clone git, commit $(git -C "${PROJETO}" rev-parse HEAD)"
elif grep -q '\$Format' "${PROJETO}/src/_versao.py" 2>/dev/null || [[ ! -f "${PROJETO}/src/_versao.py" ]]; then
    echo "AVISO: o código NÃO veio de um clone git nem de um 'git archive' — o"
    echo "       config.json não terá commit, só os hashes SHA-256 dos arquivos."
else
    echo "Código: git archive, $(grep COMMIT "${PROJETO}/src/_versao.py")"
fi

# --- Rodada ------------------------------------------------------------------
# Em segundo plano, para que este bash reaja ao USR1 do SLURM (em primeiro
# plano o trap só rodaria depois do fim). O sinal é repassado à rodada, que
# interrompe o treino atual e segue para a avaliação final.
set +e
scripts/rodada_variantes.sh "${TRABALHO}/data/raw" "${TRABALHO}/data/step_edges" \
    "${TRABALHO}/run" &
PID=$!
trap 'echo "Walltime próximo: avisando a rodada"; kill -USR1 ${PID} 2>/dev/null' USR1

wait "${PID}"; RC=$?
# Se o wait foi interrompido pelo USR1, espera a rodada de fato terminar
while (( RC > 128 )) && kill -0 "${PID}" 2>/dev/null; do
    wait "${PID}"; RC=$?
done
set -e

echo "Rodada terminou com código ${RC}"
exit ${RC}
