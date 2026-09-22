# Indicadores de melhoria real de resolução espacial

Este documento define as métricas usadas no projeto para demonstrar que a
super-resolução por SRGAN produz ganho **real** de resolução em radiografias
de nêutrons (IEA-R1/IPEN) — e não apenas imagens visualmente mais agradáveis.
A distinção importa: métodos que só realçam contraste (ex.: unsharp masking)
melhoram a aparência sem recuperar frequência espacial, e uma GAN mal
treinada pode "alucinar" texturas que não correspondem à física do objeto.

## 1. PSNR (Peak Signal-to-Noise Ratio)

Razão, em dB, entre o valor máximo possível do sinal e o erro quadrático
médio em relação à referência HR. Implementado em `src/utils.py`
(`calculate_psnr`): os tensores do modelo estão em [-1, 1], são
desnormalizados para [0, 1] e o cálculo usa MAX_I = 1.0. Isso é
matematicamente equivalente a calcular sobre o uint16 original com
`data_range = 65535`, pois o PSNR é invariante a reescala linear conjunta
do sinal e da faixa.

**Por que importa:** métrica de fidelidade global padrão na literatura de
super-resolução; a meta da IC é ganho mínimo de **5 dB** sobre a entrada
degradada. **Limitação:** mede erro médio pixel a pixel — uma imagem borrada
pode ter PSNR alto; não demonstra, sozinho, ganho de resolução.

## 2. SSIM (Structural Similarity Index)

Compara luminância, contraste e estrutura local em janelas deslizantes
(implementado via `torchmetrics` em `calculate_ssim`, com `data_range = 1.0`
após a mesma desnormalização do PSNR — a consistência do `data_range` entre
as duas métricas é essencial para comparações válidas).

**Por que importa:** mais sensível à preservação de estruturas (interfaces
entre materiais, trincas, canais de refrigeração) do que o PSNR. Para a
inspeção de componentes no reator, a integridade estrutural da imagem é o
que tem valor diagnóstico. **Limitação:** ainda é uma métrica de similaridade
com a referência, não uma medida direta de resolução.

## 3. ESF e LSF (Edge/Line Spread Function)

A **ESF** é o perfil de intensidade medido através de uma borda de degrau
física (phantom com interface abrupta, ex.: lâmina de gadolínio ou cádmio —
alta atenuação para nêutrons — sobre fundo transparente). A **LSF** é a sua
derivada: a resposta do sistema a uma "linha" ideal. Quanto mais estreita a
LSF, melhor a resolução.

**Por que importa:** é a medida física direta de borramento do sistema de
imageamento. Se a SRGAN realmente melhora a resolução, a ESF da imagem
super-resolvida deve ser mais íngreme (e a LSF mais estreita) que a da
imagem degradada — de forma mensurável, não apenas visual.

**No código:** os phantoms de borda ficam em `data/step_edges/` (carregados
por `StepEdgeDataset` em `src/data_loader.py`) e são usados **somente em
validação**, nunca na loss de treino, para não enviesar o gerador.

## 4. MTF (Modulation Transfer Function)

Módulo da transformada de Fourier da LSF, normalizado para MTF(0) = 1.
Descreve quanto contraste o sistema transfere em cada frequência espacial
(ciclos/pixel; com o pitch físico do detector, converte-se para pares de
linha/mm). Implementada em `src/utils.py` (`mtf_from_edge`) pelo método da
**borda inclinada** (slanted-edge, no espírito da ISO 12233): a leve
inclinação da borda em relação à grade de pixels permite superamostrar a ESF
(4 bins por pixel) e medir a MTF acima do limite de amostragem nativo.

**Por que importa:** é O indicador canônico de resolução em imageamento
físico. Comparar a curva MTF da saída da GAN com a da referência HR e a da
entrada LR interpolada mostra exatamente **em quais frequências** houve
recuperação de informação — e distingue recuperação real de mero realce de
contraste (que não desloca a curva para frequências mais altas).

**No código:** `validate_step_edges` (`src/train.py`) mede as **três** a cada
validação — `mtf10_lr_bicubic`, `mtf10_sr` e `mtf10_hr` — sobre o mesmo
phantom. A bicúbica é a linha de base que torna o número interpretável: é o
que se obtém sem nenhum aprendizado. MTF10 da SR **acima** da bicúbica é
evidência de recuperação; igual ou abaixo significa que a rede não entregou
resolução além da interpolação, por melhores que estejam PSNR e SSIM.

## 5. Frequência espacial de corte

Frequência em que a MTF cai abaixo de um limiar convencionado — usamos
**MTF10** (MTF = 0.10), implementada em `cutoff_frequency` em `src/utils.py`.
É o resumo escalar da curva MTF: o menor detalhe que o sistema ainda resolve
com contraste utilizável.

**Por que importa:** reduz a comparação "GAN vs. clássico vs. referência" a
um único número físico e auditável por época de treino (registrado em
`logs/step_edge_mtf.csv` pela validação de bordas em `src/train.py`). Um
aumento da frequência de corte da imagem super-resolvida em relação à
**bicúbica** é a evidência mais forte de ganho real de resolução.

**Quando o número não existe.** A coluna `status` distingue três desfechos
que de outro modo se confundiriam numa célula vazia:

| status | significado |
|---|---|
| `ok` | a MTF cruzou o limiar; o valor é utilizável |
| `sem_cruzamento` | a MTF nunca caiu abaixo de 0.1 no intervalo medido |
| `falha: <motivo>` | não foi possível medir (borda não detectável, LSF degenerada) |

`sem_cruzamento` **não** é resolução alta: é ausência de medida. Ocorre
tipicamente quando ruído de alta frequência sustenta a curva — o mesmo
mecanismo que pode fazer o MTF10 da saída da rede superar o da própria
referência HR. Por isso o CSV registra `psnr_lr_bicubic` e `psnr_sr` ao lado
das MTF: uma MTF10 alta acompanhada de PSNR baixo indica ruído ou alucinação,
não resolução.

## Protocolo de comparação

Todos os métodos (bicúbico, Lanczos, unsharp, SRGAN) são avaliados sobre o
**mesmo** pipeline degradação → reconstrução → métrica
(`src/classical_baselines.py`), com as mesmas funções de PSNR/SSIM e os
mesmos phantoms de borda, garantindo comparação justa. Cuidado
interpretativo: PSNR/SSIM altos com MTF baixa indicam suavização; MTF alta
com SSIM baixo pode indicar alucinação de detalhes pela GAN — as métricas
devem ser lidas em conjunto.
