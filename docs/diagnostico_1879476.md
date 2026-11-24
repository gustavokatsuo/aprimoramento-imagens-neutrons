# Diagnóstico do job 1879476 — colapso da fase GAN e padrão em grade

Primeira rodada completa no Coaraci: 100 épocas (5 de pré-treino MSE + 95
adversariais), 1 GPU A30, torch 2.6.0+cu124, semente 42, iniciada em
26/10/2026. Os resultados voltaram em `resultados_cluster.zip` (pasta
`1879476/`: `logs/`, `samples/`, `weights/`).

Todas as medidas abaixo foram refeitas em CPU, com
`scripts/diagnostico_1879476.py`, sobre os pesos que o job devolveu e as três
fatias locais (2028–2030). **Essas fatias ficam no bloco de teste** (topo da
pilha), então o gerador nunca as viu; em compensação são só 3 das 2031 fatias.
Os números são indicativos, não resultado do trabalho.

`[medido]` = reproduzível com o script; `[log]` = lido em
`logs/training_log.csv`; `[hipótese]` = interpretação ainda não verificada.

---

## 1. O que o log mostra

| fase | épocas | val_psnr | val_ssim | D_real / D_fake |
|---|---|---|---|---|
| pré-treino (só MSE) | 0–4 | 30,2 → **34,5 dB** | 0,819 → **0,878** | — |
| GAN, início | 5 | 32,0 dB | 0,806 | 0,85 / 0,049 |
| GAN, colapso | 7 em diante | ~32,5 → 33,1 dB | ~0,83–0,84 | 0,9000 / < 0,001 |

`[log]`

- `loss_D` estaciona em **0,1625**, que é exatamente BCE(0,9; 0,9)/2: o valor
  de um discriminador perfeito sob label smoothing 0,9 (acerta todo falso e
  dá 0,9 a todo real).
- `loss_GAN` sobe de 4,8 para 15,8, mas multiplicada por `--adv-weight 1e-3`
  responde por ~1,7% da `loss_G` no fim.
- O melhor gerador em PSNR é o da **época 4**, o último do pré-treino. Ele só
  existe porque os pesos são salvos a cada 5 épocas.

---

## 2. A queda de PSNR vem da troca MSE → VGG, não do adversarial

Na fase GAN, `src/train.py` **substitui** a perda MSE de pixel pela perda VGG
relu3_4 (não soma: a MSE sai da `loss_G`). Na época 5 o `val_psnr` já caiu
2,5 dB, quando D_fake ainda valia 0,05 e o termo adversarial pesava ~0,3% da
perda `[log]`. O adversarial não teve tempo nem peso para causar isso.

Teste causal `[medido]`: partir do gerador da época 4 e treinar ~100 passos
com UMA perda, **sem discriminador**, Adam novo, recortes de 128 px:

| perda no ajuste fino | passos | PSNR (8 recortes) | grade (RMS) |
|---|---|---|---|
| — (gerador da época 4, ponto de partida) | 0 | 37,96 dB | 1,04 ‰ |
| só MSE de pixel (controle) | 100 | 38,01 dB | 1,14 ‰ |
| só VGG relu2_2 | 100 | 33,66 dB | 2,94 ‰ |
| só VGG relu3_4 (a do job) | 100 | 32,37 dB | 5,51 ‰ |
| só VGG relu5_4 (a do artigo) | 100 | 24,31 dB | 24,46 ‰ |
| só VGG relu3_4, MaxPool → AvgPool | 100 | 31,24 dB | 5,75 ‰ |
| VGG relu3_4 ×0,006 + MSE de pixel ×1 | 200 | 37,21 dB | 4,12 ‰ |

Execuções repetidas em CPU variam ~±2 ‰ na grade e ~±1 dB no PSNR, porque a
ordem das somas em várias threads muda. Diferenças menores que isso não
significam nada.

- Só VGG, em qualquer camada: o PSNR cai 4–14 dB em 100 passos e a grade
  aparece. Só MSE: nada muda.
- relu5_4, a camada do artigo, é a pior sem âncora de pixel. Trocar de
  camada, sozinho, não resolve.
- VGG reescalada como no artigo (×0,006) **somada** à MSE de pixel mantém o
  PSNR perto do inicial. **Não** elimina a grade.

---

## 3. O padrão em grade

Visível em `samples/epoch_99.png` (painel do meio) e ausente em
`samples/epoch_4.png`. É uma trama regular de período 4 px sobre toda a
imagem, inclusive sobre as partículas.

Medida: fração da energia do resíduo SR−HR nas frequências múltiplas de
1/4 ciclo/px (períodos de 2 e 4 px), em 5 recortes fixos da fatia 2029
`[medido]`:

| época | 0 | 4 | 9 | 14 | 19 | … | 99 |
|---|---|---|---|---|---|---|---|
| PSNR | 30,38 | **34,97** | 33,43 | 33,58 | 33,62 | … | 33,84 |
| energia de grade | 0,4% | **0,2%** | 20,8% | 18,9% | 21,2% | ~20% | 20,4% |
| amplitude RMS | 1,4 ‰ | 0,8 ‰ | 9,6 ‰ | 8,8 ‰ | 9,5 ‰ | ~9 ‰ | 9,0 ‰ |

(‰ da escala cheia; 1 ‰ ≈ 8 níveis de cinza na faixa 0–8300. Sem estrutura periódica, a energia de grade fica em ~0,2% nestes recortes de 256 px; o piso depende do tamanho, 135/(H·W).)

**O que se conclui:**

1. **Não é a PixelShuffle sozinha.** O gerador da época 4 tem a mesma
   arquitetura e é limpo.
2. **Aparece inteira entre as épocas 4 e 9**, logo na troca de perda, e
   depois não cresce nem diminui.
3. **A perda VGG a cria** (seção 2): com VGG ela surge sem discriminador; com
   MSE, não. Acontece com relu2_2, relu3_4 e relu5_4.
4. **Mecanismo:** não fechado. A hipótese de que o retrocesso do max-pooling
   da VGG concentra o gradiente numa rede de passo 2/4 foi testada trocando
   MaxPool por AvgPool. A grade não muda além da variação entre execuções,
   então essa hipótese fica descartada. Hipótese restante `[hipótese]`: a VGG penaliza pouco
   as altas frequências, e as duas PixelShuffle 2× dão ao gerador controle
   independente sobre as 4 fases subpixel. Um padrão de período 4 fica quase
   "de graça" para a perda VGG.

---

## 4. Por que o discriminador "ganhou"

O discriminador vencer foi **consequência**, não causa:

- **A grade torna a tarefa trivial.** Nenhuma imagem real tem uma trama
  periódica de 4 px. D_fake cai para 0,005 já na época 6 `[log]`, junto com o
  surgimento da grade.
- **O gerador não tem como reagir.** Com Adam, a escala absoluta da perda não
  importa, só a razão entre os termos. A razão adversarial/conteúdo aqui é
  1e-3 / 1,0; no artigo é 1e-3 / 0,006, ou seja **~170× mais forte** que aqui.
  Medido no gradiente que chega à imagem na época 99, o termo VGG é **~350 a
  750× maior** que o adversarial, conforme o recorte `[medido]`.
- **O D apontava na direção certa.** O cosseno entre o passo de descida do
  termo adversarial e a componente de grade do resíduo é **negativo em todas
  as épocas salvas** (−0,46 a −0,78) `[medido]`. O gradiente do D empurra para
  *remover* a grade, e mais da metade dele está nas frequências da grade. O D
  detectava o defeito certo; só não tinha peso.
- **Fator secundário `[hipótese]`.** O mesmo Adam do gerador atravessa a troca
  pré-treino → GAN. A norma do gradiente salta ~2500× (MSE → VGG), e o segundo
  momento do Adam (β2 = 0,999) leva centenas de passos para se ajustar. Nesse
  intervalo, os passos ficam alguns múltiplos de `lr` (limite teórico ~3–6×).
  Isso amplifica o choque da troca, mas não é a causa: com Adam novo a grade
  também aparece (seção 2).
- **Contribuições plausíveis do discriminador `compacto`, não medidas
  `[hipótese]`.** A BatchNorm vê lotes só reais ou só falsos: as estatísticas
  do lote já denunciam a classe. O `AdaptiveAvgPool2d(1)` resume a imagem num
  único vetor.

**Consequência para a próxima rodada:** enfraquecer o D (taxa menor, pular
atualizações, ruído de instância) ataca o sintoma e enfraquece justamente o
sinal que se opõe à grade. A alavanca principal é o **peso relativo** do
termo adversarial, junto com um termo que mantenha a fidelidade de pixel.

---

## 5. A SRResNet contra a bicúbica no bloco de teste

24 recortes aleatórios de 256 px das fatias 2028–2030, mesmo critério do
treino (`--min-nonzero 0.5`), mesmo PSNR/SSIM de `src/utils.py` `[medido]`:

| método | PSNR | SSIM |
|---|---|---|
| bicúbica | **39,45 dB** | 0,9314 |
| época 4 (SRResNet, fim do pré-treino) | 39,10 dB | **0,9341** |
| época 99 (fim da fase GAN) | 37,84 dB | 0,9035 |

Por recorte, o ganho da época 4 sobre a bicúbica vai de **−1,3 a +0,7 dB**:

- **Recortes com fundo** (parte fora do círculo de reconstrução): a época 4
  perde ~1 dB em todos. A saída `Tanh` nunca chega exatamente a −1, e o
  gerador deixa de 36 a 120 níveis de cinza no fundo, onde o correto é 0. A
  bicúbica reproduz o zero exatamente.
- **Recortes inteiramente dentro:** a época 4 ganha nos recortes de PSNR mais
  baixo (mais estrutura) e perde nos mais lisos.

Com só 3 fatias isto **não é conclusivo**. Mas indica que, depois de 5 épocas,
a SRResNet ainda não supera a bicúbica com folga. E levanta duas questões de
método: a meta de +5 dB (decisões A5 e A6 em `decisoes.md`) e a região sobre
a qual o PSNR é calculado. A próxima rodada deve medir isso no bloco de teste
inteiro.

---

## 6. Problemas de processo encontrados

| problema | efeito | correção prevista |
|---|---|---|
| `git_commit: null` no `config.json` | o código foi copiado, não clonado; não há como provar qual versão rodou | job avisa de forma explícita e grava um resumo do código |
| `scripts/train_coaraci.slurm` pede `--gpus=2` e 32 tarefas, mas rodou com 1 GPU e 16 (`config.json`) | quem roda editou o script; a versão que rodou não está no repositório | alinhar o script a 1 GPU A30 |
| `data/step_edges/` existe localmente, mas não chegou ao cluster | MTF não foi medida | gerar os phantoms sintéticos dentro do job |
| `requirements.txt` fixa torch 2.13.0 (venv local de CPU); o cluster usou 2.6.0+cu124 | ambiente não reprodutível pelo arquivo | alinhar ao que funcionou no cluster |
| pesos só a cada 5 épocas; sem `best.pth` | o melhor modelo existe por acaso | `best.pth` por `val_psnr` a cada época |
| sem tempo por época nem pico de memória no log | impossível planejar a próxima rodada | registrar no CSV |
| **tempo real desconhecido** | só se sabe que 100 épocas couberam em 24 h (≤ ~14 min/época) | job divide o tempo pelo relógio, não por número de épocas |

---

## 7. Próxima rodada (confirmada em 16/11/2026, V3 trocado em 18/11/2026 após os mini-ensaios da §10; implementada em `scripts/rodada_variantes.sh`)

**Estado (24/11/2026):** rodada aprovada na reunião de orientação. Código pronto
e testado no commit `7c04d23`, que acrescenta a medição da faixa da pilha
(pendência P2), empacotado em `aprimoramento-imagens-neutrons_7c04d23.zip` via
`git archive`. **Envio ao cluster pendente.** Ao receber os resultados, confira
em `runs/<job>/V0_srresnet/logs/config.json` que `git_commit` é
`7c04d233e74fda4aa200ed134e14ecfa40cc9dd9-archive`.

Um único job de 24 h, em duas partes.

**Parte 1 — SRResNet (V0).** Treina o gerador só com MSE, como no pré-treino,
mas por mais tempo: no job 1879476 o pré-treino ainda ganhava +0,33 dB por
época quando parou. Este gerador é o **baseline SRResNet** e o ponto de
partida comum da parte 2.

**Parte 2 — três receitas de fase GAN (V1, V2, V3).** Cada uma recomeça do
mesmo gerador V0, com a mesma semente e a mesma fatia de tempo. Assim a única
diferença entre elas é a receita, e cada uma responde a uma pergunta:

| | o que muda | pergunta que responde |
|---|---|---|
| **V1 perceptual** | MSE de pixel + VGG; **sem** discriminador | O que a perda VGG sozinha faz, com âncora de pixel? É a referência para saber se o GAN acrescenta algo. |
| **V2 adv-forte** | V1 + discriminador atual (`compacto`), adversarial ×1e-3 fixo | A hipótese principal: dar ao D, que já empurra contra a grade, peso suficiente para agir. |
| **V3 SRGAN-MSE** | MSE de pixel + discriminador `compacto` ×1e-3, **sem** VGG (o "SRGAN-MSE" de Ledig et al., 2017) | O adversarial acrescenta nitidez sem a fonte comprovada da grade, que é a VGG? |

As quatro formam uma grade 2×2, com o mesmo D e o mesmo peso adversarial:

| | sem adversarial | com adversarial |
|---|---|---|
| **sem VGG** | V0 SRResNet | V3 SRGAN-MSE |
| **com VGG** | V1 perceptual | V2 adv-forte |

V1 × V2 isola o efeito do adversarial; V2 × V3, o da VGG; V0 × os demais, o
que cada perda acrescenta à SRResNet.

Comum a V1–V3 (correções, não hipóteses): Adam novo na transição; MSE de
pixel ×1 (VGG relu3_4 ×0,006 em V1 e V2); log por época de tempo, memória,
norma do gradiente de cada termo e energia de grade no val.

**Por que 1e-3 no V2.** Com a VGG reescalada para 0,006, o peso 1e-3 é a razão
adversarial/conteúdo do artigo, ~170× o peso relativo do job 1879476.
Calibração `[medido]` (`diagnostico_1879476.py calibrar-adv`: cada D treinado
200 passos contra o G da época 4 congelado): o D compacto chega a uma razão
adversarial/VGG de ~1,1. Com o D treinado do job 1879476 a razão era ~0,5
(seção 4). O peso fixo cai, portanto, numa faixa razoável.

**A V3 original (descartada).** Era um discriminador `patch_sn` (PatchGAN com
normalização espectral) com peso adaptativo. No mesmo teste de calibração, o `patch_sn` dá razão de só
**0,003** com 1e-3, e a escala ainda subia ~30× em 200 passos. A normalização
espectral limita o gradiente do D em relação à imagem por construção. Com
peso fixo, o V3 repetiria a falha do job 1879476 (adversarial abafado), e
nenhum número fixo escolhido agora seria confiável. O peso adaptativo (ideia
de Esser et al., 2021) reajusta o peso a cada 50 batches para que o
gradiente adversarial seja 0,5× o de conteúdo — a razão que o D compacto
treinado tinha, isto é, o regime do V2. O peso fica limitado a [1e-6, 1]
para que um D ainda sem sinal não receba peso ilimitado. O log registra o
peso efetivo por época (`adv_weight_efetivo`).

No mini-ensaio com dados reais (§10), porém, o `patch_sn` não aprendeu a
separar real de falso, e o peso preso no teto empurrou o Gerador com ruído.
Ele foi substituído pelo SRGAN-MSE. O código (`--discriminator patch_sn`,
`--adv-razao`) continua no repositório, sem uso na rodada.

**Detalhes da execução.** Época de 20 recortes por fatia (~1/5 da época do
job 1879476), para que o `best.pth` e o corte pelo orçamento tenham
granularidade fina. Taxa de aprendizado constante 1e-4: com o tempo dividido
pelo relógio, as épocas de decaimento não são conhecidas de antemão. Recortes
de validação fixos por índice, para que o `val_psnr` não oscile pelo sorteio.
Divisão do tempo: o prazo do job menos 90 min de reserva para a avaliação; V0
recebe 1/4, e cada fase seguinte recebe o que resta dividido pelas fases que
faltam. Avaliação final na fatia inteira (`--patch-size 0`) de cada imagem do
bloco de teste, com PSNR no recorte inteiro e dentro do círculo de
reconstrução, SSIM, energia de grade e MTF10 nos phantoms.

Ao fim, a avaliação automática compara V0–V3 com bicúbica, Lanczos e unsharp
masking no bloco de teste, incluindo a MTF nos phantoms sintéticos.

**Não incluídas:**

- **Taxa de aprendizado menor para o D:** o TTUR de Heusel et al. usa
  normalmente taxa *maior* no D, e aqui o problema não é o D aprender rápido.
- **Pular a atualização do D:** com o D no piso desde a época 7, ele ficaria
  congelado quase o tempo todo.
- **Ruído de instância:** para cegar o D à grade (~9 ‰), o ruído precisaria
  ser maior que isso, e apagaria a textura fina que se quer recuperar.

Pode entrar como opção desligada, ou como V4 se o tempo couber.

---

## 8. Como reproduzir

```bash
unzip resultados_cluster.zip '1879476/*' -d ../resultados_cluster
P=../resultados_cluster/1879476/weights
python scripts/diagnostico_1879476.py evolucao   --pesos $P   # tabela da seção 3
python scripts/diagnostico_1879476.py gradientes --pesos $P   # seção 4
python scripts/diagnostico_1879476.py bicubica   --pesos $P   # seção 5
python scripts/diagnostico_1879476.py ajuste-fino --pesos $P --perda pix          # seção 2
python scripts/diagnostico_1879476.py ajuste-fino --pesos $P --perda vgg --camada relu2_2
python scripts/diagnostico_1879476.py ajuste-fino --pesos $P --perda vgg --camada relu3_4
python scripts/diagnostico_1879476.py ajuste-fino --pesos $P --perda vgg --camada relu5_4
python scripts/diagnostico_1879476.py ajuste-fino --pesos $P --perda vgg --avgpool
python scripts/diagnostico_1879476.py ajuste-fino --pesos $P --perda vgg+pix --passos 200
python scripts/diagnostico_1879476.py calibrar-adv --pesos $P   # seção 7
```

Medido com torch 2.13.0 em CPU. O ajuste fino leva 3 a 10 min por
configuração.

---

## 9. Testes em CPU e o que NÃO foi testado fora do cluster

`tests/test_rodada_cpu.py` (14 testes) passa em CPU com dados sintéticos
(40 fatias de 160×160), tanto no torch 2.13 local quanto no **torch 2.6.0 +
torchvision 0.21.0**, as versões do cluster (build de CPU). Os testes acharam
e corrigiram dois bugs que teriam derrubado o job:

- **Vírgula decimal.** Com localidade pt_BR, o `awk` escrevia o orçamento como
  `0,0167` e o `train.py` recusava: nenhuma fase rodaria. Agora o script força
  `LC_ALL=C`.
- **`.slurm` sem `data/step_edges/`.** O `find` numa pasta inexistente, com
  `pipefail` + `set -e`, abortava o job em silêncio logo depois de copiar os
  dados.

**Não testado, e por quê:**

| item | por que não | risco e mitigação |
|---|---|---|
| Precisão mista (`--amp`) **na GPU**: autocast float16 e GradScaler em CUDA | não há GPU aqui | baixo. O `train.py` real rodou em CPU com float16 forçado, com os dois D, sem erro nem NaN (§10, verificação 1), e o caminho AMP já rodou 100 épocas no job 1879476. Os kernels de CUDA em si não foram exercitados. Se uma fase falhar, a rodada segue para as outras e para a avaliação |
| Tempo por época e pico de memória na A30 | sem GPU | baixo para memória: o `patch_sn` (2,8 M parâmetros) é menor que o `artigo`. O tempo é desconhecido; a divisão pelo relógio se adapta, e o log passa a registrar os dois |
| Primeira época mais longa que o orçamento da fase | depende da velocidade real | a primeira época de cada fase sempre roda (ainda não há estimativa). Com épocas ~1/5 das do job 1879476 (≤ ~3 min), é improvável. O aviso de walltime 45 min antes do fim cobre o pior caso |
| Avaliação final em fatias inteiras (~2028 px), 203 fatias × 7 geradores | sem as 203 fatias nem a GPU | rodada em 1 fatia real inteira em CPU (§10): pico de 3,3 GB de RAM; a parte em CPU custa ~7 s/fatia, ~25 min nas 203. Com a inferência na GPU, estimativa de 35–40 min, com 90 min reservados |
| Recortes de validação fixos com `--num-workers > 0` | **testado** depois: a verificação 2 da §10 rodou com 2 workers | — |
| Ambiente do Coaraci: `module load`, o venv de quem roda, `sbcast`, entrega do `USR1` pelo SLURM, localidade do nó | só existem lá | o `.slurm` foi testado inteiro com `module` simulado e sem `sbcast` (cai no `cp`). O venv do cluster precisa ter as versões do `requirements.txt`; o job grava `pip_freeze.txt` para conferir |
| Build CUDA do torch 2.6.0 | testado só o build de CPU da mesma versão | baixo: mesma API |
| `git archive` com o código real | o código novo ainda não foi commitado | testado num repositório temporário. O commit precisa incluir `.gitattributes` e `src/_versao.py` |
| **Resultado científico**: se V2/V3 evitam a grade, se o D fica em equilíbrio, se o peso adaptativo se estabiliza | é exatamente o que a rodada mede | o log por época (`val_grade_pct`, `D_*_prob`, `grad_*`, `adv_weight_efetivo`) permite diagnosticar qualquer desfecho sem nova rodada de investigação |

---

## 10. Verificações finais antes de enviar (17–18/11/2026)

Três verificações, todas na pilha exata do cluster (torch 2.6.0+cu124 e
torchvision 0.21.0+cu124, rodando em CPU).

**1. Precisão mista com os discriminadores.** O `train.py` real rodou com o
autocast em float16 forçado em CPU (trocando só `usar_amp` e o `dtype` na
memória). Épocas GAN completas com o D `compacto` e com o `patch_sn`, sem erro
nem NaN `[medido]`.

**2. Rodada inteira com dados reais.** `rodada_variantes.sh` nas 3 fatias
reais (2048×2007), com 2 workers e avaliação na fatia inteira (2004×2004): as
4 fases e a avaliação terminaram com código 0. Pico de RAM 3,3 GB. A parte da
avaliação que roda em CPU (métricas, Lanczos) custa ~7 s por fatia, ~25 min
nas 203 fatias de teste; por isso a reserva subiu de 60 para 90 min `[medido]`.

**3. Mini-ensaios do algoritmo.** O `train.py` real, partindo do gerador da
época 4 do job 1879476: treino nas fatias 2028–2029, validação na 2030,
recortes de 128 px, lote 8, 25 épocas de ~12 passos (~300 passos). O controle
reproduz a falha do job 1879476, então o ensaio tem poder de detectá-la
`[medido]`:

| receita | PSNR val | SSIM | grade (% / amplitude) | D_real / D_fake |
|---|---|---|---|---|
| controle (receita do job 1879476) | 32,4 dB | 0,878 | 15,9% / 9,6 ‰ | 0,89 / 0,01 |
| V1 perceptual | 35,8 dB | 0,897 | 15,0% / 6,4 ‰ | — |
| V2 adv-forte | 35,9 dB | 0,897 | 13,8% / 6,1 ‰ | 0,56 / 0,34 |
| V3 original (`patch_sn`, peso adaptativo) | 35,7 dB | 0,882 | 22,4% / 7,5 ‰ | 0,45 / 0,45 |
| candidata (a): sem VGG (SRGAN-MSE) | 34,8 dB | 0,905 | 8,2% / 4,9 ‰ | 0,46 / 0,45 |
| candidata (b): PatchGAN sem norma espectral | 36,5 dB | 0,902 | 11,1% / 4,9 ‰ | 0,46 / 0,44 |

Médias das 5 últimas épocas. Piso da grade em 128 px: 0,82%. A SRResNet (época
4) tem ~0,8 ‰.

**O que se conclui:**

- **O V2 resolve o desequilíbrio.** O D compacto aprende sem colapsar, e a força
  adversarial cresce até ~0,3 da força da VGG.
- **A V3 original não funciona.** O `patch_sn` não aprendeu a separar real de
  falso (D_real = D_fake o tempo todo). A diferença entre SR e HR é sutil, e a
  normalização espectral limita quanto a saída do D pode variar. O peso
  adaptativo ficou no teto de 1,0, empurrando o Gerador com ruído: a grade
  chegou a 56–72% nas primeiras épocas. Foi substituída.
- **A grade não some com nenhuma receita testada.** V1 e V2 ficam em ~6 ‰.
  Sem VGG (a), a grade fica no piso na maior parte das épocas, mas dá picos de
  ~20% (épocas 4 e 20). Então a VGG não é a única fonte. Hipótese
  `[hipótese]`: o gradiente do próprio D tem estrutura de grade (54% da energia
  nessas frequências no diagnóstico, §4), e mesmo com peso pequeno ele a
  induz de vez em quando.
- **Por que (a) e não (b).** Nenhuma venceu de forma clara em 300 passos, e em
  nenhuma das duas o D chegou a aprender. A escolha é pelo desenho: com (a)
  a rodada forma a grade 2×2 da §7, que isola o efeito de cada perda, e (a) é
  uma variante do próprio artigo de Ledig et al.

**Limitação:** 300 passos contra dezenas de milhares no job. O ensaio detecta
falhas rápidas (a do job 1879476 aparece em 50 passos), mas não prevê o
comportamento de longo prazo. **Expectativa para a rodada:** as variantes GAN
provavelmente ainda terão alguma grade, menor que a do job 1879476. O
resultado garantido é a SRResNet (V0) comparada com a bicúbica.

**Como reproduzir esta seção.**

```bash
# ambiente igual ao do cluster (do PyPI vêm os builds cu124, que também rodam em CPU; ~2,5 GB)
python3 -m venv venv26
venv26/bin/pip install torch==2.6.0 torchvision==0.21.0 torchmetrics==1.9.0 \
    numpy==2.4.4 pillow==12.2.0 tifffile==2026.3.3
venv26/bin/python -m unittest tests.test_rodada_cpu -v                 # testes (§9)

# 1. float16 forçado em CPU no train.py real (aqui com o D compacto)
venv26/bin/python scripts/verificacao/amp_cpu.py "$PWD" --amp --discriminator compacto ...

# 3. mini-ensaios (precisa do gen_epoch_4.pth de resultados_cluster.zip)
PYTHON=venv26/bin/python scripts/verificacao/ensaio_receitas.sh \
    ../resultados_cluster/1879476/weights/gen_epoch_4.pth ../ensaio_receitas
```

A verificação 2 é a própria `scripts/rodada_variantes.sh` apontada para as 3
fatias reais, com `EXTRA_TREINO="--patch-size 256 --batch-size 2
--patches-per-image 4 --val-fraction 0.34 --test-fraction 0.34 --split-gap 0"`.
