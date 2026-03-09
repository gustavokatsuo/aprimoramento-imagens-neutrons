# Aprimoramento de Imagens para Tomografia com Nêutrons Utilizando Deep Learning

Este repositório contém o código-fonte, protótipos e documentação referentes ao projeto de Iniciação Científica focado na restauração e aprimoramento de imagens obtidas por tomografia com nêutrons (NT). 

O projeto visa superar as limitações de capacidade de generalização de algoritmos clássicos de filtragem (como o BM3D) perante ruídos e distorções complexas, estruturando uma metodologia baseada em Inteligência Artificial para o sistema de imageamento do Reator Multipropósito Brasileiro (RMB).

## Autores e Instituições
* **Pesquisdor:** Gustavo Katsuo Tsutsui (IME-USP)
* **Orientador:** Prof. Dr. Frederico A. Genezini (CERPq - IPEN/USP)
* **Instituições:** Universidade de São Paulo (USP) | Instituto de Pesquisas Energéticas e Nucleares (IPEN/CNEN)

---

## Arquitetura do Modelo
A abordagem principal deste projeto utiliza uma **SRGAN** (Super-Resolution Generative Adversarial Network). O modelo é composto por duas redes neurais que "interagem" entre si:
1. **Gerador:** Uma rede convolucional profunda focada em reconstruir imagens de alta resolução a partir de radiografias ruidosas ou de baixa qualidade.
2. **Discriminador:** Uma rede que tenta distinguir entre as imagens reais de alta qualidade e as imagens geradas (falsas).

A função de perda (*Loss*) combina a perda de conteúdo (garantindo que os detalhes físicos da amostra são mantidos) e a perda adversarial (garantindo que a textura e a nitidez pareçam reais).

---

## Estrutura do Repositório

```
aprimoramento-imagens-neutrons/
├── data/                   # Ficheiros de dados (não versionados)
│   ├── raw/                # Radiografias originais
│   └── processed/          # Dados após pré-processamento
├── docs/                   # Documentação e relatórios
│   └── plano_ic_gustavo.pdf
├── notebooks/              # Exploração de dados e prototipagem
│   └── arquitetura_ic.ipynb # Notebook principal de treino da SRGAN
├── src/                    # Código-fonte modularizado
│   ├── data_loader.py      
│   ├── model.py            
│   └── train.py            
├── requirements.txt        # Dependências do projeto
└── README.md               # Este arquivo
```

---

## Pré-requisitos e Instalação

Para executar este projeto localmente, é recomendável a utilização de um ambiente virtual (como `venv` ou `conda`) com suporte para aceleração por GPU (CUDA).

1. **Clonar o repositório:**
   ```bash
   git clone https://github.com/gustavokatsuo/aprimoramento-imagens-neutrons.git
   cd aprimoramento-imagens-neutrons
   ```

2. **Instalar as dependências:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Certifique-se de que as bibliotecas `torch` e `torchvision` estão configuradas corretamente para a sua versão do CUDA).*

---

## Como Executar

### Treino do Modelo
O treino e a experimentação inicial da SRGAN podem ser reproduzidos através do Jupyter Notebook disponibilizado:
* Abre e executa o arquivo `notebooks/arquitetura_ic.ipynb`.
* No final do treino, os pesos do modelo serão guardados automaticamente num arquivo chamado `srgan_generator.pth`.

## Resultados (Em progresso)
*(Esta secção será atualizada com métricas de desempenho - ex: PSNR, SSIM - e exemplos visuais das reconstruções tomográficas à medida que os treinos avançarem).*

---

## Referências Principais
* ALMAU, O.; ALARCÓN, T. E. *A residual dense u-net neural network for image denoising*. IEEE Access, 2021.
* MA, Y. et al. *Enhancing the spatial resolution of neutron radiography with generative adversarial networks*. Journal of Instrumentation, 2025.
* SCHOUERI, R. M. et al. *The new facility for neutron tomography of ipen-cnen/sp and its potential to investigate hydrogenous substances*. Applied Radiation and Isotopes, 2014.

---
*Projeto desenvolvido no âmbito do programa de Iniciação Científica (IC).*