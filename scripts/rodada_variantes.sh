#!/bin/bash
#===============================================================================
# Rodada de variantes: SRResNet + três receitas de fase GAN + avaliação final,
# tudo num único job, com o tempo dividido pelo relógio.
#
# Motivação e escolha das variantes: docs/diagnostico_1879476.md §7.
#
#   V0_srresnet    só MSE (pré-treino estendido). É o baseline SRResNet e o
#                  ponto de partida comum de V1-V3.
#   V1_perceptual  parte de V0: MSE de pixel x1 + VGG relu3_4 x0,006, SEM
#                  discriminador. Referência para saber se o GAN acrescenta algo.
#   V2_adv_forte   V1 + discriminador 'compacto' com adversarial x1e-3. Com a
#                  VGG em 0,006, é a razão adversarial/conteúdo do artigo, ~170x
#                  o peso relativo que o adversarial teve no job 1879476.
#   V3_patch_sn    como V2, com discriminador local 'patch_sn' (PatchGAN +
#                  normalização espectral, sem BatchNorm) e peso adversarial
#                  ADAPTATIVO (--adv-razao 0.5): a norma espectral limita o
#                  gradiente do D, e o peso fixo do V2 o deixaria abafado.
#
# Todas com a mesma semente; V1-V3 partem do MESMO best.pth de V0 com
# otimizadores novos. Ao fim, src.classical_baselines compara todos os
# Geradores com bicúbica, Lanczos e unsharp no bloco de teste, com a mesma
# normalização e divisão do treino, e mede a MTF nos phantoms de borda.
#
# Divisão do tempo: o prazo final do job (SLURM_JOB_END_TIME) menos uma
# reserva para a avaliação é repartido assim — V0 recebe FRACAO_V0 do que
# resta; cada fase seguinte recebe o que resta dividido pelas fases que
# faltam (sobra de uma fase vai para as seguintes). Cada treino para sozinho
# com --max-horas, antes de começar uma época que não caiba.
#
# Uso:
#   scripts/rodada_variantes.sh <pasta_dados> <pasta_bordas> <pasta_saida>
#
# Variáveis de ambiente (todas opcionais):
#   PRAZO_FINAL   instante final, em segundos desde 1970 (padrão:
#                 SLURM_JOB_END_TIME; sem ele, agora + 24 h)
#   RESERVA_MIN   minutos reservados para a avaliação final (padrão 60)
#   FRACAO_V0     fração do tempo de treino para V0 (padrão 0.25)
#   WORKERS, CACHE  workers e cache do DataLoader (padrão 4 e 200)
#   PYTHON        interpretador (padrão: python)
#   EXTRA_TREINO  argumentos acrescentados a TODO treino; os últimos valem
#                 (usado pelos testes em CPU para encolher tudo)
#   EXTRA_AVAL    argumentos acrescentados à avaliação final
#   FASES         subconjunto das fases, ex.: "V0_srresnet V2_adv_forte"
#
# Sinais: SIGUSR1 ou SIGTERM interrompem o treino em andamento (o train.py
# termina o batch e sai limpo), pulam as fases restantes e seguem para a
# avaliação com o que houver.
#===============================================================================

set -o pipefail
# Números com ponto decimal, seja qual for a localidade do nó: com pt_BR o
# awk escreveria "0,0167" e o train.py recusaria --max-horas (achado no teste
# em CPU, tests/test_rodada_cpu.py).
export LC_ALL=C

if (( $# != 3 )); then
    echo "uso: $0 <pasta_dados> <pasta_bordas> <pasta_saida>" >&2
    exit 2
fi
DADOS="$1"; BORDAS="$2"; SAIDA="$3"
PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-4}"
CACHE="${CACHE:-200}"
RESERVA_MIN="${RESERVA_MIN:-60}"
FRACAO_V0="${FRACAO_V0:-0.25}"
FASES="${FASES:-V0_srresnet V1_perceptual V2_adv_forte V3_patch_sn}"
read -r -a EXTRA_T <<< "${EXTRA_TREINO:-}"
read -r -a EXTRA_A <<< "${EXTRA_AVAL:-}"

if [[ -n "${PRAZO_FINAL:-}" ]]; then
    :
elif [[ -n "${SLURM_JOB_END_TIME:-}" ]]; then
    PRAZO_FINAL="${SLURM_JOB_END_TIME}"
else
    PRAZO_FINAL=$(( $(date +%s) + 24 * 3600 ))
    echo "AVISO: sem SLURM_JOB_END_TIME nem PRAZO_FINAL; supondo 24 h a partir de agora."
fi

mkdir -p "${SAIDA}"
ESTADO="${SAIDA}/estado_fases.txt"
: > "${ESTADO}"
echo "Prazo final: $(date -d "@${PRAZO_FINAL}" '+%d/%m %H:%M:%S') | reserva para avaliação: ${RESERVA_MIN} min"

# --- Interrupção -------------------------------------------------------------
PARAR=0
PID_ATUAL=""
interromper() {
    echo "[$(date +%H:%M:%S)] Sinal recebido: interrompendo o treino atual e seguindo para a avaliação."
    PARAR=1
    if [[ -n "${PID_ATUAL}" ]] && kill -0 "${PID_ATUAL}" 2>/dev/null; then
        kill -USR1 "${PID_ATUAL}" 2>/dev/null
    fi
}
trap interromper USR1 TERM

# Espera o processo em segundo plano. Um sinal tira o 'wait' do bloqueio
# (código > 128) antes de o filho terminar; espera de novo até ele sair.
esperar() {
    local pid=$1 rc
    wait "${pid}"; rc=$?
    while (( rc > 128 )) && kill -0 "${pid}" 2>/dev/null; do
        wait "${pid}"; rc=$?
    done
    return ${rc}
}

# --- Ambiente: o que de fato rodou --------------------------------------------
mkdir -p "${SAIDA}/ambiente"
"${PYTHON}" -m pip freeze > "${SAIDA}/ambiente/pip_freeze.txt" 2>&1 || true
command -v nvidia-smi >/dev/null && nvidia-smi > "${SAIDA}/ambiente/nvidia_smi.txt" 2>&1
env | grep '^SLURM_' | sort > "${SAIDA}/ambiente/slurm_env.txt" || true

# --- Configuração comum e das variantes -----------------------------------------
# Época curta (20 recortes por fatia, ~1/5 da época do job 1879476) para que
# o best.pth e o corte pelo orçamento tenham granularidade fina. Sem
# decaimento de taxa: com o tempo dividido pelo relógio, as épocas de
# decaimento não são conhecidas de antemão.
COMUNS=(
    --data-dir "${DADOS}"
    --step-edge-dir "${BORDAS}"
    --epochs 100000
    --batch-size 16
    --lr 1e-4
    --patch-size 256
    --patches-per-image 20
    --tiff-normalization range --tiff-range 0 8300
    --min-nonzero 0.5
    --val-fraction 0.10 --test-fraction 0.10 --split-gap 30
    --cache-images "${CACHE}"
    --num-workers "${WORKERS}"
    --amp
    --seed 42
    --sample-interval 10
    --vgg-layer relu3_4
)
# Fase GAN comum a V1-V3: conteúdo = MSE de pixel x1 + VGG relu3_4 x0,006
GAN=(--pretrain-epochs 0 --pixel-weight 1.0 --content-weight 0.006)

args_da_fase() {
    case "$1" in
        V0_srresnet)   echo "--pretrain-epochs 100000" ;;
        V1_perceptual) echo "${GAN[*]} --adv-weight 0" ;;
        V2_adv_forte)  echo "${GAN[*]} --adv-weight 1e-3 --discriminator compacto" ;;
        # patch_sn: peso ADAPTATIVO. A norma espectral limita o gradiente do D
        # e um peso fixo deixaria o adversarial abafado (calibrar-adv mediu
        # razão adv/VGG de 0,003 com 1e-3). 0,5 é a razão que o D compacto
        # treinado do job 1879476 tinha com 1e-3, ou seja, o regime do V2.
        V3_patch_sn)   echo "${GAN[*]} --adv-weight 1e-3 --adv-razao 0.5 --discriminator patch_sn" ;;
        *) return 1 ;;
    esac
}

# --- Fases de treino ------------------------------------------------------------
read -r -a LISTA <<< "${FASES}"
BASE_V0="${SAIDA}/V0_srresnet/weights/best.pth"
n_fases=${#LISTA[@]}

for (( k = 0; k < n_fases; k++ )); do
    fase="${LISTA[k]}"
    if ! extra=$(args_da_fase "${fase}"); then
        echo "ERRO: fase desconhecida '${fase}'" | tee -a "${ESTADO}"
        continue
    fi
    if (( PARAR )); then
        echo "${fase}: pulada (interrupção)" | tee -a "${ESTADO}"
        continue
    fi

    # Orçamento desta fase, em horas
    agora=$(date +%s)
    disponivel=$(( PRAZO_FINAL - agora - RESERVA_MIN * 60 ))
    restam=$(( n_fases - k ))
    if [[ "${fase}" == V0_srresnet ]]; then
        horas=$(awk -v d="${disponivel}" -v f="${FRACAO_V0}" 'BEGIN { printf "%.4f", d * f / 3600 }')
    else
        horas=$(awk -v d="${disponivel}" -v n="${restam}" 'BEGIN { printf "%.4f", d / n / 3600 }')
    fi
    if awk -v h="${horas}" 'BEGIN { exit !(h < 0.01) }'; then
        echo "${fase}: pulada (sem tempo: ${horas} h)" | tee -a "${ESTADO}"
        continue
    fi

    init=()
    if [[ "${fase}" != V0_srresnet ]]; then
        if [[ ! -f "${BASE_V0}" ]]; then
            echo "${fase}: pulada (não há ${BASE_V0})" | tee -a "${ESTADO}"
            continue
        fi
        init=(--init-generator "${BASE_V0}")
    fi

    echo "=== [$(date +%H:%M:%S)] ${fase}: orçamento ${horas} h ==="
    read -r -a ARGS_FASE <<< "${extra}"
    "${PYTHON}" -u -m src.train "${COMUNS[@]}" "${ARGS_FASE[@]}" "${init[@]}" \
        --run-dir "${SAIDA}/${fase}" --max-horas "${horas}" "${EXTRA_T[@]}" &
    PID_ATUAL=$!
    esperar "${PID_ATUAL}"; rc=$?
    PID_ATUAL=""
    resumo="${SAIDA}/${fase}/logs/resumo.json"
    echo "${fase}: código ${rc} | $(tr -d '\n' < "${resumo}" 2>/dev/null || echo 'sem resumo.json')" \
        | tee -a "${ESTADO}"
done

# --- Avaliação final ------------------------------------------------------------
AVAL="${SAIDA}/avaliacao"
mkdir -p "${AVAL}"
PESOS=()
for fase in ${FASES}; do
    [[ -f "${SAIDA}/${fase}/weights/best.pth" ]] && PESOS+=("${fase}=${SAIDA}/${fase}/weights/best.pth")
    # Para as variantes GAN, o último Gerador também: o best por PSNR tende a
    # ser o menos adversarial (decisão A8 em docs/decisoes.md)
    if [[ "${fase}" != V0_srresnet && -f "${SAIDA}/${fase}/weights/gen_ultimo.pth" ]]; then
        PESOS+=("${fase}_ultimo=${SAIDA}/${fase}/weights/gen_ultimo.pth")
    fi
done

RELATORIO="${AVAL}/relatorio_final.txt"
{
    echo "RELATÓRIO FINAL — $(date '+%d/%m/%Y %H:%M:%S')"
    echo
    echo "Estado das fases:"
    cat "${ESTADO}"
    echo
} > "${RELATORIO}"

if (( ${#PESOS[@]} == 0 )); then
    echo "ERRO: nenhum Gerador treinado; avaliando só os métodos clássicos." | tee -a "${RELATORIO}"
    CONFIG_ARGS=(--config "${SAIDA}/V0_srresnet/logs/config.json")
    [[ -f "${CONFIG_ARGS[1]}" ]] || CONFIG_ARGS=(--tiff-normalization range --tiff-range 0 8300
                                                --val-fraction 0.10 --test-fraction 0.10 --split-gap 30)
    GEN_ARGS=()
else
    CONFIG_ARGS=()
    GEN_ARGS=(--gen-weights "${PESOS[@]}")
fi

echo "=== [$(date +%H:%M:%S)] Avaliação final no bloco de teste ==="
"${PYTHON}" -u -m src.classical_baselines \
    --data-dir "${DADOS}" \
    --step-edge-dir "${BORDAS}" \
    --split test \
    --patch-size 0 \
    --output "${AVAL}/teste.csv" \
    "${GEN_ARGS[@]}" "${CONFIG_ARGS[@]}" "${EXTRA_A[@]}" 2>&1 | tee -a "${RELATORIO}"
rc_aval=$?
echo "Avaliação terminou com código ${rc_aval}" | tee -a "${RELATORIO}"

# Código de saída: 0 só se a avaliação rodou e nenhuma fase falhou
if (( rc_aval != 0 )) || grep -q "código [1-9]\|ERRO" "${ESTADO}"; then
    exit 1
fi
exit 0
