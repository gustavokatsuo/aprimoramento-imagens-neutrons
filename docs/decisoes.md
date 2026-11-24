# Decisões de projeto

Registro das escolhas que moldam o que o trabalho mede e defende. Cada entrada
traz o estado atual, a evidência disponível e **o que a mudaria** — de modo que
uma decisão possa ser revista sem refazer o raciocínio do zero.

> **Rascunho.** As decisões em aberto são de pesquisa, não de implementação:
> cabem a Gustavo e ao orientador. O que está aqui é o material para essa
> conversa, não a conclusão dela.

Convenções: `[medido]` verificado neste repositório; `[artigo]` vem de Ledig
et al. (2017) ou da literatura citada; `[herdado]` veio do desenvolvimento
inicial sem justificativa registrada.

---

## Reunião de orientação de 24/11/2026

| decisão | resultado |
|---|---|
| A5. Meta de +5 dB | vale para as radiografias de nêutrons, não para o proxy de raios X; referência e baseline em aberto, aguardando os dados |
| A6. Região do PSNR | calcular os dois; decidir o principal depois |
| A7. MSE na fase GAN | manter |
| A8. Checkpoint da GAN | relatar os dois |
| A3. Degradação sintética | manter nesta etapa; declarar como limitação |
| P2. Faixa 0–8300 | medir no job (`scripts/medir_faixa.py`) |
| Rodada V0–V3 | aprovada como está |

O orientador não questionou as conclusões do diagnóstico e pediu que a rodada
seja enviada ao cluster.

---

# Parte I — Decisões tomadas

## D1. Normalização por faixa fixa, não pelo teto do dtype

**Estado:** `--tiff-normalization range --tiff-range 0 8300`

Os dados ocupam 12,7% dos 16 bits `[medido]`. Dividir por 65535 comprimiria
tudo em [−1, −0,75] no domínio do modelo. A faixa precisa ser a mesma em todas
as fatias: `minmax` por imagem destrói a comparabilidade radiométrica entre
elas, e o tom de cinza codifica coeficiente de atenuação.

**O que mudaria:** o máximo global da pilha completa. As três fatias
examinadas são consecutivas e do fim da aquisição; 8300 pode não valer para o
volume inteiro. **Medir antes do primeiro treino que valha.**

## D2. Divisão em blocos contíguos, com margem

**Estado:** `--val-fraction 0.10 --test-fraction 0.10 --split-gap 30`

Fatias adjacentes têm SSIM de 0,969 `[medido]`. Divisão aleatória colocaria
quase-duplicatas em treino e teste. Ver `dataset.md` §6.

**O que mudaria:** a curva de SSIM contra distância entre fatias, quando a
pilha completa permitir traçá-la. A margem de 30 é estimativa física, não
medição.

## D3. Rejeição de recortes majoritariamente vazios

**Estado:** `--min-nonzero 0.5`

24,68% da imagem é zero, fora do círculo de reconstrução; 11,3% dos recortes
aleatórios caem majoritariamente fora `[medido]`.

**O que mudaria:** nada previsível. É consequência da geometria da
reconstrução.

## D4. Múltiplas GPUs por DataParallel, não DDP

**Estado:** ativa sozinho quando há mais de uma GPU.

Segue a prática dos scripts do grupo que já rodam no Coaraci. Não exige
`torchrun`, `DistributedSampler` nem process group. Custo: a agregação
concentra na GPU 0.

**O que mudaria:** se o modelo crescer a ponto de a GPU 0 virar gargalo, ou se
o treino passar a exigir mais de um nó — aí DDP é o caminho.

---

# Parte II — Decisões em aberto

## A1. Profundidade da VGG na content loss — **a mais consequente**

**Estado atual:** `relu3_4`, com `--content-weight 1.0` `[herdado]`

A perda do Gerador é `content_weight × loss_content + adv_weight × loss_GAN`.
A magnitude de `loss_content` depende fortemente da profundidade da VGG, de
modo que essa escolha determina o **peso efetivo** do termo adversarial:

| `--vgg-layer` | `content-weight 1.0` | `content-weight 0.006` |
|---|---|---|
| `relu2_2` | 0,005 % | 0,88 % |
| **`relu3_4`** *(atual)* | **0,013 %** | 2,06 % |
| `relu4_4` | 0,164 % | 21,5 % |
| `relu5_4` *(VGG54 do artigo)* | 1,100 % | **65,0 %** |

Na configuração atual, o termo adversarial responde por **0,013%** da perda.
Na prática o projeto treina uma SRResNet, não uma SRGAN — o Discriminador
consome computação e quase não influencia o Gerador.

**Origem da escolha:** `relu3_4` entrou no commit `3800439`, com o comentário
*"preserva melhor texturas e bordas estruturais de materiais, ignorando
semântica profunda"*. É justificativa plausível para radiografia — camadas
rasas respondem a textura e borda, que é o que interessa aqui, enquanto
`relu5_4` responde a estrutura semântica de objetos naturais, que não existe
numa tomografia. O efeito colateral sobre o equilíbrio da perda parece não ter
sido percebido.

**As duas posições são defensáveis:**

- *Manter `relu3_4`* e corrigir o desequilíbrio subindo `--adv-weight`, se o
  argumento é que o domínio pede features rasas.
- *Adotar `relu5_4` + `--content-weight 0.006`*, se o argumento é fidelidade ao
  protocolo do artigo, contra o qual o trabalho se compara.

**O que fecha:** rodar as duas, mesma semente, e comparar por MTF10 contra a
bicúbica. Está implementado e pronto:

```bash
python -m src.train --seed 42 --vgg-layer relu3_4 --content-weight 1.0    --run-dir runs/vgg-relu3_4
python -m src.train --seed 42 --vgg-layer relu5_4 --content-weight 0.006 --discriminator artigo --run-dir runs/vgg-relu5_4
```

> Os percentuais da tabela foram medidos com tensores aleatórios. A ordem de
> grandeza é confiável, o valor exato não — remedir com as radiografias.

**Atualização (job 1879476, 16/11/2026):** a primeira rodada completa usou a
configuração atual e confirmou o problema por outro caminho. O termo
adversarial terminou em ~1,7% da perda, e no gradiente que chega à imagem o
termo VGG é ~350–750× maior. Além disso, a perda VGG **sozinha** (sem a MSE de
pixel, que sai da perda na fase GAN) criou um padrão em grade de período 4 px
e custou 1–2,5 dB. Isso vale para relu2_2, relu3_4 e relu5_4. Detalhes e
evidência em `diagnostico_1879476.md`. Para a próxima rodada a proposta é
manter relu3_4, reescalar para 0,006 e somar a MSE de pixel (ver A7).

## A2. Arquitetura do Discriminador

**Estado atual:** `compacto` — 6 convoluções até 256 canais, 1,4 M parâmetros
`[herdado]`

O artigo usa 8 convoluções até 512 canais, 23,6 M parâmetros. A versão
compacta nunca correspondeu ao artigo; veio assim desde o primeiro commit
modular.

**Medido:** reverter custa pouco. A memória de ativação é praticamente a mesma
(712 MB contra 773 MB em batch 8 / patch 256); o que cresce são os pesos e os
estados do otimizador.

**Ressalva técnica:** a variante compacta usa `AdaptiveAvgPool2d(1)`, que
colapsa toda a informação espacial num valor por canal antes da cabeça densa.
Para um discriminador que deve julgar **textura**, isso descarta exatamente o
sinal de interesse.

**O que fecha:** acompanha A1 — se o termo adversarial passar a pesar, a
capacidade do Discriminador passa a importar.

## A3. Modelo de degradação — **a questão mais fundamental, e a menos examinada**

**Decisão (reunião de 24/11/2026):** manter a degradação sintética nesta etapa e declará-la como limitação no relatório. O orientador deixou a escolha conosco; trocar agora invalidaria os testes e os ensaios, e não há PSF medida para pôr no lugar.

**Estado atual:** desfoque gaussiano de `kernel_size=3`, seguido de redução
bicúbica 4× com antialiasing `[herdado]`

O σ não é passado: vem do padrão do torchvision, `0.3*((k-1)*0.5-1)+0.8`, que
para `k=3` dá **σ nominal 0,8 px**. O kernel de 3 taps trunca a gaussiana, e a
resposta ao impulso medida dá **σ efetivo 0,69 px** `[medido]` — o desfoque
aplicado é ainda mais estreito que o nominal.

**Isto define a tarefa.** O modelo aprende a inverter exatamente esta
degradação. Se ela não corresponde à perda de resolução real do sistema de
imageamento, o trabalho resolve um problema sintético, e o desempenho não
transfere para aquisições de baixa resolução reais — que é o objetivo declarado
para o IEA-R1 e o RMB.

**Perguntas em aberto:**

- A PSF real do sistema é gaussiana? Em imageamento com nêutrons ela vem do
  cintilador, da divergência do feixe (razão L/D) e do detector, e tipicamente
  **não** é gaussiana nem tão estreita quanto σ = 0,69 px.
- O σ vem do *padrão de uma biblioteca de visão computacional*, não de medição
  do sistema de imageamento. Nada o liga à física da aquisição.
- O fator 4× corresponde a alguma condição real de aquisição, ou é convenção
  herdada da literatura de super-resolução de imagens naturais?
- Haveria como obter **pares reais** — a mesma amostra adquirida em alta e em
  baixa resolução? Isso eliminaria a questão inteira e seria um resultado
  muito mais forte.
- Deveria haver ruído no modelo de degradação? Aquisições rápidas têm
  estatística de contagem pior, e o projeto cita explicitamente ruído como
  motivação.

**O que fecha:** medir a PSF real do sistema com phantom de borda (ver P1) e
ajustar o modelo de degradação a ela. Enquanto isso não acontece, vale declarar
a degradação sintética como limitação explícita no relatório.

## A4. Métrica primária do trabalho

**Estado atual:** PSNR, SSIM e MTF10 são calculadas; nenhuma foi eleita como a
métrica que o trabalho defende.

**Candidatas:**

| métrica | mede | limitação |
|---|---|---|
| PSNR | fidelidade global | imagem borrada pontua bem |
| SSIM | estrutura local | ainda é similaridade com a referência |
| **MTF10** | resolução física, em lp/mm | exige phantom de borda (P1); instável sob ruído |
| **distribuição de tamanhos** | recuperação de estrutura real | exige segmentação; ainda não implementada |

A última merece consideração. A professora mediu, por Avizo, a distribuição de
tamanhos de 356.078 partículas de ferro `[professora]`. Rodar a **mesma
segmentação** sobre HR, SR e bicúbico e comparar as três distribuições
responderia a pergunta que de fato importa: *a rede recupera a população de
partículas que a interpolação perde, ou inventa textura?*

Com voxel de 3,647937 µm, o pico dominante (4,5 µm, ~66.000 partículas) ocupa
1,2 fatias e **desaparece** na degradação 4×. Recuperá-lo é uma afirmação
verificável, que uma banca de física avalia diretamente — diferente de um ganho
em dB.

**O que fecha:** decisão de pesquisa, com o orientador. Depende também de
acesso ao Avizo ou de uma segmentação equivalente em código.

## A5. Meta de +5 dB: qual modelo, contra qual baseline — **cabe ao orientador**

**Decisão (reunião de 24/11/2026):** a meta de +5 dB vale para a etapa com **radiografias de nêutrons de baixa resolução**, não para o proxy de raios X atual (`dataset.md`). Nesta etapa, os resultados do proxy são reportados sem a meta, como validação do pipeline. **Ainda em aberto:** a referência para o PSNR nos nêutrons (pares reais de alta e baixa resolução ou degradação sintética) e o baseline. O orientador está buscando os dados de nêutrons.

**Estado atual:** o plano fala em "melhoria mínima de 5 dB" (`utils.calculate_psnr`),
sem dizer de qual modelo nem sobre o quê.

**Evidência (job 1879476, `diagnostico_1879476.md` §5):** em 24 recortes do
bloco de teste local, a SRResNet do fim do pré-treino empata com a bicúbica
(39,10 contra 39,45 dB; −1,3 a +0,7 dB por recorte) `[medido]`. São só 3
fatias. Mesmo assim, +5 dB sobre a bicúbica parece fora de alcance com a
degradação atual (σ efetivo 0,69 px + redução 4×, ver A3): a bicúbica já
recupera muito do que essa degradação tira.

**Opções:**

| modelo | baseline | comentário |
|---|---|---|
| SRResNet (só MSE) | bicúbica | o PSNR é o que a SRResNet otimiza; comparação justa e usual |
| SRGAN (fase GAN) | bicúbica | o GAN troca PSNR por textura; tende a ficar abaixo da SRResNet em dB |
| qualquer | entrada LR sem interpolação | não é comparável em PSNR (tamanhos diferentes) |
| SRGAN | SRResNet | mede o efeito do adversarial, não o ganho sobre o clássico |

**Recomendação:** afirmar o ganho de PSNR com a **SRResNet contra a bicúbica
no bloco de teste**, e avaliar o GAN por MTF10 e inspeção visual (A4). Se a
diferença medida no teste inteiro ficar longe de 5 dB, registrar isso como
resultado, com a degradação sintética (A3) como explicação provável, em vez de
trocar de critério depois de ver o número.

**O que fecha:** conversa com o orientador, de preferência **antes** de ver os
números da próxima rodada.

## A6. Região do cálculo de PSNR e SSIM

**Decisão (reunião de 24/11/2026):** calcular os dois (recorte inteiro e círculo de reconstrução) e decidir o principal depois de ver os resultados. A rodada já calcula os dois (`psnr` e `psnr_fov` no relatório final).

**Estado atual:** sobre o recorte inteiro, incluindo o fundo fora do círculo de
reconstrução (até 50% do recorte, por `--min-nonzero 0.5`).

**Evidência `[medido]`:** nos recortes com fundo, a SRResNet perde ~1 dB para a
bicúbica em todos. A `Tanh` nunca chega exatamente a −1, e o gerador deixa de
36 a 120 níveis de cinza no fundo, onde o correto é 0. A bicúbica reproduz o
zero exatamente. O fundo também infla o PSNR de todos os métodos, porque é
trivial.

**Opções:** (a) recorte inteiro, como hoje; (b) só pixels dentro do círculo de
reconstrução; (c) zerar a saída fora do círculo antes de medir, o que é
legítimo porque o fundo é zero por definição da reconstrução.

**Recomendação:** relatar (a) e (b) lado a lado, com (b) como principal.

## A7. Manter a MSE de pixel durante a fase GAN

**Decisão (reunião de 24/11/2026):** manter a MSE. Já é o que a rodada faz em V1–V3. O relatório declara o desvio em relação ao SRGAN original; o SRGAN puro (VGG + adversarial, sem MSE) tem como referência o job 1879476.

**Estado atual:** na fase GAN a perda de conteúdo é **só** VGG; a MSE sai.
É o protocolo de Ledig et al. (2017).

**Evidência `[medido]`:** sem a MSE, 100 passos de VGG relu3_4 derrubam ~5,6 dB e criam
a grade. Com VGG ×0,006 + MSE ×1, o PSNR fica perto do inicial (a grade
diminui, mas não some).

**Opções:** (a) SRGAN puro, como hoje; (b) somar a MSE de pixel, como fazem
ESRGAN (Wang et al., 2018) e a maior parte dos trabalhos posteriores.

**Recomendação:** (b). Afasta o trabalho do SRGAN original, e isso precisa ser
dito no relatório. Mas sem a âncora de pixel o PSNR cai abaixo da bicúbica, e o
trabalho perde a comparação que pretende fazer.

## A8. Critério de escolha do checkpoint da fase GAN

**Decisão (reunião de 24/11/2026):** relatar os dois (`best.pth` pelo PSNR de validação e o último). A rodada já avalia os dois.

**Estado atual:** não existe `best.pth`; a próxima rodada vai salvar o melhor
por `val_psnr`.

**Problema:** numa fase GAN, o melhor PSNR tende a ser o checkpoint *menos*
adversarial, o mais perto da SRResNet. Escolher por PSNR desfaz parcialmente o
que o GAN acrescenta.

**Recomendação:** salvar `best.pth` (por `val_psnr`) **e** o último checkpoint,
e relatar os dois. Se a MTF passar a ser medida por época, ela é candidata a
critério para os modelos GAN.

**Implementado:** a rodada salva os dois, e a avaliação final inclui
`<variante>` (best) e `<variante>_ultimo` para V1–V3.

## A9. Peso adversarial fixo ou adaptativo — **resolvida para esta rodada**

**Estado atual:** peso fixo 1e-3 com o D `compacto` em V2 e V3. O peso
adaptativo (`--adv-razao`) e o D `patch_sn` ficam no código, sem uso.

**Evidência `[medido]`** (`diagnostico_1879476.md` §7 e §10): com 1e-3, o D
`compacto` puxa o Gerador com ~0,3–1,1× a força da VGG e não colapsa num
ensaio de 300 passos. O `patch_sn` não aprendeu a separar real de falso; com
peso adaptativo, o peso ficou no teto e o Gerador foi empurrado por ruído
(grade de até 72%).

**Consequência:** a confusão anterior (V2 e V3 diferindo no D e no modo do
peso) deixou de existir. V2 e V3 agora diferem só na VGG.

**O que reabriria:** um D local que de fato aprenda (ex.: PatchGAN sem norma
espectral, com BatchNorm/InstanceNorm), testado num ensaio mais longo.

---

# Parte III — Pendências que bloqueiam decisões

## P0. Tempo por época desconhecido

O job 1879476 não registrou tempo, e a saída do SLURM não voltou. Só se sabe
que 100 épocas couberam em 24 h (≤ ~14 min/época, incluindo a validação).

**Bloqueia:** dividir a próxima rodada por número de épocas. Por isso a
proposta é dividir o tempo pelo relógio, e registrar tempo e memória no log
para as rodadas seguintes.

## P1. Phantoms de borda inexistentes

Sem eles a validação de MTF não roda, e a MTF medida sobre imagens de amostra
não vale: as três fatias deram 0,214, `sem_cruzamento` e 0,499 ciclos/px
`[medido]` — variação de 2,3× entre imagens 96,9% idênticas, porque
`mtf_from_edge` encontra um gradiente qualquer em vez de uma borda de degrau.

É a única métrica que mede **resolução** em vez de similaridade, e o
diferencial declarado do projeto. Requer aquisição: um objeto com interface
abrupta — lâmina de gadolínio ou cádmio para nêutrons; para o proxy de raios X,
qualquer borda reta de material denso — levemente inclinado em relação à grade
de pixels.

**Bloqueia:** A3 (medir a PSF real) e A4 (MTF como métrica primária).

## P2. Máximo global da pilha

**Decisão (reunião de 24/11/2026):** medir dentro do job, sem mudar a normalização. `scripts/medir_faixa.py` lê a pilha inteira logo depois da cópia dos dados e grava `runs/<job>/ambiente/faixa_pilha.json` (máximo global, percentis e fração de pixels acima de 8300). Nas 3 fatias locais o máximo é 8297, a 3 níveis do teto: é provável que a pilha completa o ultrapasse em algum ponto. Se ultrapassar, os números da rodada continuam comparáveis entre si, mas o corte entra como ressalva no relatório.

`--tiff-range 0 8300` vem de três fatias consecutivas do fim da aquisição.

**Bloqueia:** D1, e portanto qualquer treino cujo número vá para o relatório.

## P3. Curva de correlação contra distância entre fatias

`--split-gap 30` é estimativa física.

**Bloqueia:** o rigor de D2 — não o treino, mas a defesa da divisão no
relatório.
