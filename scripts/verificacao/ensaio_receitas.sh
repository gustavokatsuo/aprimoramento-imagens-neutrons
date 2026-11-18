#!/bin/bash
#===============================================================================
# Mini-ensaios da §10 de docs/diagnostico_1879476.md: as receitas da fase GAN
# em CPU, com as fatias reais 2028-2030, partindo do gerador da época 4 do job
# 1879476. ~300 passos por receita, ~25-50 min com tudo em paralelo.
#
# Uso (da raiz do repositório):
#   scripts/verificacao/ensaio_receitas.sh <gen_epoch_4.pth> <pasta_saida>
#
# A pasta data/raw precisa ter 2028.tiff, 2029.tiff e 2030.tiff (treino nas
# duas primeiras, validação na última). As receitas 'v3_original' e
# 'b_patch_sem_sn' usam os lançadores desta pasta (sem tocar em src/).
#===============================================================================
set -e
G4="$1"; SAIDA="$2"
PY="${PYTHON:-venv/bin/python}"
mkdir -p "${SAIDA}"
C=(--data-dir data/raw --step-edge-dir "${SAIDA}/sem_bordas" --patch-size 128 --batch-size 8
   --patches-per-image 50 --tiff-normalization range --tiff-range 0 8300 --min-nonzero 0.5
   --val-fraction 0.34 --test-fraction 0 --split-gap 0 --num-workers 0 --cache-images -1
   --seed 42 --sample-interval 5 --vgg-layer relu3_4 --pretrain-epochs 0 --epochs 25
   --init-generator "${G4}")
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
r() { local nome=$1; shift; "$@" "${C[@]}" "${EXTRA[@]}" --run-dir "${SAIDA}/${nome}" > "${SAIDA}/${nome}.txt" 2>&1; }
EXTRA=(--pixel-weight 0 --content-weight 1.0 --adv-weight 1e-3 --discriminator compacto)
r controle_1879476 "${PY}" -u -m src.train &
EXTRA=(--pixel-weight 1 --content-weight 0.006 --adv-weight 0)
r V1_perceptual "${PY}" -u -m src.train &
EXTRA=(--pixel-weight 1 --content-weight 0.006 --adv-weight 1e-3 --discriminator compacto)
r V2_adv_forte "${PY}" -u -m src.train &
EXTRA=(--pixel-weight 1 --content-weight 0.006 --adv-weight 1e-3 --adv-razao 0.5 --discriminator patch_sn)
r v3_original "${PY}" -u -m src.train &
EXTRA=(--pixel-weight 1 --content-weight 0 --adv-weight 1e-3 --discriminator compacto)
r a_sem_vgg "${PY}" -u -m src.train &
EXTRA=(--pixel-weight 1 --content-weight 0.006 --adv-weight 1e-3 --discriminator patch_sn)
r b_patch_sem_sn "${PY}" -u scripts/verificacao/sem_sn.py "${PWD}" &
wait
echo "Logs por época em ${SAIDA}/<receita>/logs/training_log.csv"
