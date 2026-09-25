# Caracterização do conjunto de dados

Documento de referência sobre os dados usados no projeto: o que são, como
foram adquiridos, o que foi medido sobre eles e quais decisões do pipeline
decorrem disso.

**Procedência de cada afirmação.** Marcações usadas ao longo do texto:
`[professora]` vem da documentação de quem disponibilizou as imagens;
`[medido]` foi verificado sobre os arquivos, com o comando indicado;
`[inferido]` é conclusão a partir dos dois anteriores, e está sujeita a
correção.

---

## 1. Identificação

**Conjunto atual: um proxy, não nêutrons.** As imagens são de microtomografia
de raios X, usadas para desenvolver e validar o pipeline enquanto os dados de
tomografia com nêutrons do IEA-R1 e do RMB não estão disponíveis.

| | |
|---|---|
| Identificação | `Recon-PH&HPM` `[professora]` |
| Material | micropartículas de ferro e óxidos de ferro `[professora]` |
| Equipamento | ZEISS Xradia Versa 610 (3D-XRM) `[professora]` |
| Porta-amostra | tubo de kapton, 5,0 mm de diâmetro interno, 7,4 mm de altura `[professora]` |
| Reconstrução | XMReconstructor (ZEISS) `[professora]` |
| Fases presentes | magnetita 51,2%, wustita 31,6%, ferro metálico 17,3% (por DRX) `[professora]` |

As imagens são **seções transversais reconstruídas**, não projeções. Isso
implica que a correção de transmitância — `(Exp − Black) / (Field − Black)` —
**já foi aplicada**, por ser parte do processo de reconstrução. O pipeline não
precisa fazê-la. `[inferido]`

## 2. Parâmetros de aquisição

Há **duas fontes da professora que divergem entre si**. Ambas são registro
legítimo; a conclusão é que descrevem aquisições diferentes.

| parâmetro | arquivo `.txt` | relatório | usado |
|---|---|---|---|
| radiografias | 2030 | 1601 | **2030** |
| largura | 2007 | 2008 | **2007** |
| tamanho de pixel | 3,647937 µm | 1,74 µm (voxel) | **3,647937 µm** |
| altura | 2048 | 2048 | 2048 |
| tensão | 140 kV | 140 kV | 140 kV |
| corrente | 150 µA | 150 µA | 150 µA |
| exposição | 3 s | 3 s | 3 s |
| fonte → eixo | 18,54 mm | 18,5 mm | 18,5 mm |
| detector → eixo | 154,73 mm | 154,7 mm | 154,7 mm |
| objetiva | 0,396× | 0,4× | 0,4× |

**Por que o `.txt` prevalece** `[medido]`:

1. Os arquivos baixados de `Recon-PH&HPM` são `2028.tiff`, `2029.tiff`,
   `2030.tiff` — a numeração vai até 2030, não 1601.
2. Cada arquivo tem 2048 × 2007 px, não 2008.
3. **Prova geométrica decisiva:** a reconstrução ocupa um círculo de raio
   1000 px (medido pelo perfil radial). Com voxel de 1,74 µm, o tubo de 5,0 mm
   ocuparia 2874 px de diâmetro — não caberia no círculo de 2000 px. Com
   3,647937 µm, ocupa 1371 px e cabe. A queda do perfil radial em raio ~690 px
   coincide com o raio esperado do tubo (685 px).

### Escala física derivada

| grandeza | valor |
|---|---|
| tamanho de voxel | 3,647937 µm |
| campo de visão | 7,32 × 7,47 mm |
| Nyquist nativo (0,5 ciclos/px) | **137,1 pares de linha/mm** |
| conversão | MTF10 [ciclos/px] ÷ 3,647937 µm × 1000 = **lp/mm** |

Essa conversão é o que permite reportar resolução em unidade física, como pede
`resolution_metrics.md`, em vez de em ciclos por pixel.

## 3. Estrutura dos arquivos `[medido]`

```
2048 × 2007 px, uint16, TIFF de página única
7,84 MiB por fatia  →  15,5 GiB para as 2030
```

Os arquivos são **uma fatia por arquivo**, não pilhas multi-página. A
numeração não tem zeros à esquerda (`0.tiff` … `2030.tiff`), o que exige
ordenação natural — ver seção 6.

## 4. Faixa dinâmica `[medido]`

O detector é de 16 bits, mas a aquisição ocupa uma fração pequena da faixa:

| fatia | máximo | ocupação de 65535 |
|---|---|---|
| 2028 | 8277 | 12,6% |
| 2029 | 8297 | 12,7% |
| 2030 | 8247 | 12,6% |

Percentis (fatia 2029): p1 = 0, mediana = 1421, p99 = 4035. Níveis distintos
presentes: 6.609.

**Consequência.** Normalizar dividindo pelo teto do dtype (65535) comprime
tudo em [0, 0.13], e depois de `Normalize(0.5, 0.5)` os dados ocupam
[−1, −0.75] — 12,7% da faixa de saída `Tanh` do Gerador. O restante da
capacidade de saída fica inutilizado.

**Decisão.** Usar faixa fixa explícita:

```bash
--tiff-normalization range --tiff-range 0 8300
```

A faixa precisa ser **a mesma em todas as fatias**. Normalizar cada uma por si
(`minmax`) destrói a comparabilidade radiométrica: medido, as três fatias saem
com máximo exatamente 1,0, apagando a diferença real entre elas
(0,9972 / 0,9996 / 0,9936 no modo `range`). Num volume onde o tom de cinza
codifica coeficiente de atenuação — e portanto distingue ferro, óxido e poro —
isso inutilizaria a interpretação física.

## 5. Geometria da reconstrução `[medido]`

O volume reconstruído é um cilindro inscrito no volume da imagem. Numa fatia,
isso aparece como um disco de sinal com os cantos exatamente em zero:

| raio (px) | média do sinal |
|---|---|
| 0–150 | 2975 |
| 300–450 | 2512 |
| 600–750 | 2004 |
| 900–975 | 1375 |
| 975–1000 | 970 |
| **acima de 1000** | **0** |

- Raio do círculo de reconstrução: **1000 px** (diâmetro 2000, inscrito na largura 2007)
- Pixels exatamente nulos: **24,68%** da imagem
- Quadrado inscrito no círculo: 1414 × 1414 px, com 0,01% de zeros

**Consequência.** Recortes aleatórios de 256 px sobre a imagem inteira caem
fora do círculo em **11,3%** dos casos (medido em 500 sorteios). Esses recortes
não são dado: são borda da reconstrução, e treinariam o Gerador a reproduzir
vazio.

**Decisão.** `--min-nonzero 0.5` resorteia o recorte até que ao menos metade
tenha conteúdo. Medido, leva os 11,3% a **0,0%**.

## 6. Correlação entre fatias e divisão dos conjuntos

Esta é a seção que sustenta a validade de qualquer métrica reportada.

### A medição `[medido]`

| par | distância | correlação de Pearson | PSNR | SSIM |
|---|---|---|---|---|
| 2028 vs 2029 | 1 fatia | 0,9962 | 39,1 dB | **0,9688** |
| 2029 vs 2030 | 1 fatia | 0,9962 | 39,2 dB | 0,9689 |
| 2028 vs 2030 | 2 fatias | 0,9875 | 34,0 dB | 0,9165 |
| par não relacionado (ruído) | — | −0,0003 | 7,8 dB | 0,0055 |

Fatias vizinhas de um mesmo volume são **96,9% idênticas por SSIM**. Não são
amostras independentes: são o mesmo objeto, um plano adiante.

### Por que divisão aleatória invalidaria o resultado

Sorteando as 2030 fatias entre treino, validação e teste, pares com SSIM de
0,97 cairiam em conjuntos diferentes. A métrica de teste mediria a capacidade
do modelo de reproduzir imagens que ele praticamente viu — não generalização.
O número resultante seria alto e sem significado, e **divisão aleatória seria
pior que divisão nenhuma**, por dar aparência de rigor ao que não tem.

Como `Recon-PH&HPM` é **uma amostra só** `[professora]`, não existe a opção de
separar por espécime. A única separação honesta é por **bloco contíguo em z**.

### O argumento físico para o tamanho da margem

Blocos contíguos ainda deixam um problema nas fronteiras: a última fatia do
treino e a primeira da validação continuam sendo vizinhas. Daí a margem
(`--split-gap`), que descarta fatias entre os blocos.

**O critério é físico: a margem precisa exceder a extensão em z das estruturas
que o modelo deve aprender.** Se uma partícula atravessa mais fatias do que a
margem descarta, o mesmo objeto aparece dos dois lados da fronteira, e o
conjunto de validação contém estrutura que o modelo já viu — o vazamento
persiste, apenas em escala menor.

Com voxel de 3,647937 µm, as populações de partículas medidas pela professora
ocupam: `[inferido]`

| população `[professora]` | comprimento | extensão em z |
|---|---|---|
| pico dominante (~66.000 partículas) | 4,5 µm | **1,2 fatias** |
| segundo pico (~34.000 partículas) | 35,7 µm | **9,8 fatias** |
| cauda superior (raras) | > 10³ µm | > 274 fatias |

A margem adotada é **`--split-gap 30`**: cerca de 3× a extensão do segundo
pico, o que descorrelaciona as duas populações que dominam numericamente a
amostra. As partículas da cauda superior atravessariam a margem, mas são
estatisticamente raras — mais de 80% das partículas têm volume abaixo de
10⁴ µm³ `[professora]`.

> **Limitação conhecida.** Este valor vem de argumento físico, **não de
> medição**. Com apenas 3 fatias disponíveis não foi possível traçar a curva de
> correlação contra distância e localizar onde ela descorrelaciona. Com a pilha
> completa, medir SSIM em função da distância entre fatias e escolher a margem
> onde a curva estabiliza é o procedimento correto, e substitui esta estimativa.

### Ordenação natural

A ordem alfabética coloca `1000.tiff` antes de `11.tiff`. Como a ordem da lista
**é** a ordem em z, isso tornaria qualquer bloco "contíguo" descontíguo no
volume, anulando em silêncio todo o cuidado acima. `list_supported_images` usa
ordenação natural (dígitos comparados como números).

### Divisão resultante

Com `--val-fraction 0.10 --test-fraction 0.10 --split-gap 30` sobre 2031 fatias:

```
treino  1565  (0    … 1564)
        gap 30
valid.   203  (1595 … 1797)
        gap 30
teste    203  (1828 … 2030)
```

As colunas `val_psnr` e `val_ssim` do CSV são as **únicas** que medem
generalização. `psnr` e `ssim_last_batch` são calculadas sobre os mesmos
patches que treinaram o Gerador e sobem mesmo quando o modelo apenas decora.

## 7. Configuração recomendada

Reunindo as decisões acima:

```bash
python -m src.train \
    --data-dir data/raw \
    --tiff-normalization range --tiff-range 0 8300 \
    --min-nonzero 0.5 \
    --val-fraction 0.10 --test-fraction 0.10 --split-gap 30 \
    --patches-per-image 100 \
    --cache-images 500 \
    --patch-size 256 --batch-size 16 --seed 42
```

Custo de carregamento: sem `--cache-images`, cada recorte relê e renormaliza a
fatia inteira (4,8 ms de leitura + 28,3 ms de normalização) para extrair
256 × 256 px. Com cache, o custo por recorte cai de 29,3 ms para 1,8 ms
`[medido]`. O cache existe em cada processo do DataLoader, então a memória é
`cache × num_workers`.

## 8. O que ainda não se sabe

- **Curva de correlação contra distância entre fatias.** Determina a margem
  correta; requer a pilha completa.
- **Faixa de valores ao longo de toda a aquisição.** As três fatias examinadas
  são consecutivas e do fim da pilha; o máximo de 8300 pode não valer para o
  volume inteiro. Convém medir o máximo global antes de fixar `--tiff-range`.
- **Phantoms de borda.** Não existem no conjunto. Sem eles a validação de MTF
  não roda, e a MTF medida sobre imagens de amostra não vale — as três fatias
  deram 0,214, `sem_cruzamento` e 0,499 ciclos/px, variação de 2,3× entre
  imagens quase idênticas, porque `mtf_from_edge` encontra um gradiente
  qualquer e não uma borda de degrau.
- **A que corresponde `PH&HPM`.** O relatório trata o conjunto como amostra
  única; o significado das siglas não está documentado.
