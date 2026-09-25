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

---

# Parte III — Pendências que bloqueiam decisões

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

`--tiff-range 0 8300` vem de três fatias consecutivas do fim da aquisição.

**Bloqueia:** D1, e portanto qualquer treino cujo número vá para o relatório.

## P3. Curva de correlação contra distância entre fatias

`--split-gap 30` é estimativa física.

**Bloqueia:** o rigor de D2 — não o treino, mas a defesa da divisão no
relatório.
