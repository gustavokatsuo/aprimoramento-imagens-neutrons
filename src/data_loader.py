import os
import glob
import random
import re
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import tifffile
import numpy as np

# astropy é opcional: só é exigida se houver arquivos .fits no dataset
try:
    from astropy.io import fits as astropy_fits
except ImportError:
    astropy_fits = None

# Extensões suportadas (a ordem não importa; a lista final é ordenada)
SUPPORTED_EXTENSIONS = ("*.tiff", "*.tif", "*.png", "*.jpg", "*.fits")

def _chave_natural(caminho):
    """
    Chave de ordenação que trata sequências de dígitos como números.

    Necessária porque a ordem alfabética coloca '1000.tiff' antes de '11.tiff'.
    Numa pilha tomográfica a ordem da lista É a ordem em z, e a divisão em
    blocos contíguos (ver dividir_pilha) depende disso: com a ordem alfabética,
    um bloco contíguo na lista seria descontíguo no volume.
    """
    nome = os.path.basename(caminho)
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", nome)]

def list_supported_images(root_dir):
    """
    Lista ordenada dos arquivos de imagem suportados em root_dir, em ordem
    NATURAL. A ordenação mantém a indexação do Dataset estável entre execuções
    (reprodutibilidade), e ter a listagem isolada permite checar o dataset antes
    de montar o loader.
    """
    return sorted(
        (f for ext in SUPPORTED_EXTENSIONS
         for f in glob.glob(os.path.join(root_dir, ext))),
        key=_chave_natural,
    )

def dividir_pilha(arquivos, val_fraction=0.0, test_fraction=0.0, gap=0):
    """
    Divide uma pilha de imagens em blocos CONTÍGUOS de treino, validação e teste.

    Divisão aleatória não serve aqui. Fatias vizinhas de um mesmo volume são
    quase a mesma imagem — medido nestas reconstruções, SSIM de 0,969 entre
    fatias adjacentes, contra 0,006 para um par não relacionado. Sorteadas, o
    conjunto de teste conteria quase-duplicatas do treino e a métrica final não
    mediria generalização nenhuma.

    O layout é [treino] gap [validação] gap [teste], com 'gap' fatias
    descartadas nas fronteiras. A margem precisa ser maior que a extensão em z
    das estruturas de interesse, senão a mesma partícula aparece dos dois lados:
    com voxel de 3,65 um, as partículas de 35,7 um do segundo pico da amostra
    atravessam cerca de 10 fatias.

    Retorna (treino, validacao, teste) como listas de caminhos.
    """
    n = len(arquivos)
    n_val = int(n * val_fraction)
    n_test = int(n * test_fraction)
    n_gaps = (gap if n_val else 0) + (gap if n_test else 0)
    n_treino = n - n_val - n_test - n_gaps

    if n_treino <= 0:
        raise ValueError(
            f"Divisão impossível: {n} imagem(ns) para val={val_fraction}, "
            f"test={test_fraction} e gap={gap} não deixam nada para treino."
        )

    i = n_treino
    treino = arquivos[:i]
    if n_val:
        i += gap
        validacao = arquivos[i:i + n_val]
        i += n_val
    else:
        validacao = []
    if n_test:
        i += gap
        teste = arquivos[i:i + n_test]
    else:
        teste = []
    return treino, validacao, teste

def load_image_as_array(img_path, fits_normalization="minmax", fits_range=None,
                        tiff_normalization="dtype", tiff_range=None):
    """
    Carrega uma imagem científica como numpy float32 2D no intervalo [0, 1].

    Formatos e normalizações:
      - TIFF (inteiro): leitura via tifffile. A normalização é configurável,
        porque dividir pelo teto do dtype só é adequado quando a aquisição
        ocupa a faixa toda — e frequentemente não ocupa. As reconstruções
        deste projeto chegam a no máximo ~8300 de 65535 (12,7%), de modo que
        'dtype' as comprime em [0, 0.13] e desperdiça 87% da faixa de saída
        Tanh do Gerador. Modos:
          * tiff_normalization="dtype" (padrão): divide por np.iinfo(dtype).max.
          * tiff_normalization="range": usa tiff_range=(lo, hi) explícito, com
            clipping. É o modo correto para uma PILHA tomográfica: a mesma
            faixa em todas as fatias preserva a comparabilidade radiométrica
            entre elas, que é o que dá sentido físico aos tons de cinza.
          * tiff_normalization="minmax": (img - min)/(max - min) por imagem.
            CUIDADO: normaliza cada fatia por si, então o mesmo material recebe
            valores diferentes em fatias diferentes. Use apenas para inspeção
            de uma imagem isolada, nunca para treinar sobre uma pilha.
      - FITS (float32): leitura via astropy.io.fits. IMPORTANTE: diferente do
        uint16, dados float de FITS NÃO têm faixa fixa — ela varia por aquisição
        (contagens, transmitância, etc.). Por isso a normalização é configurável:
          * fits_normalization="minmax": (img - min) / (max - min) por imagem.
            Robusto, mas descarta a escala absoluta (ex.: transmitância física).
          * fits_normalization="range": usa fits_range=(lo, hi) explícito, com
            clipping. Preferível quando a faixa da aquisição é conhecida
            (ex.: (0.0, 1.0) para dados já corrigidos por transmitância).
          * fits_normalization="none": assume dados já em [0, 1], apenas clipa.
      - PNG/JPG (uint8): fallback via PIL, divisão por 255.
    """
    ext = os.path.splitext(img_path)[1].lower()

    if ext in (".tiff", ".tif"):
        # Leitura científica do TIFF de 16-bits
        img_array = tifffile.imread(img_path)

        # Um TIFF com mais de 2 eixos pode ser duas coisas MUITO diferentes:
        #   (H, W, C) — imagem colorida/RGBA  -> usa-se o primeiro canal
        #   (N, H, W) — pilha de páginas      -> usa-se a primeira fatia
        # Projeções de tomografia costumam vir como pilha. O antigo
        # img_array[:, :, 0] tratava os dois casos como se fossem canais e,
        # numa pilha (N, H, W), devolvia (N, H): a coluna 0 de cada fatia,
        # misturando as fatias num array 2D sem significado físico.
        # A distinção é pelo último eixo: canais são poucos (<= 4), colunas não.
        if img_array.ndim > 2:
            if img_array.shape[-1] <= 4:
                img_array = img_array[..., 0]   # (H, W, C) -> (H, W)
            else:
                img_array = img_array[0]        # (N, H, W) -> (H, W)
            while img_array.ndim > 2:           # ex.: (N, H, W, C)
                img_array = img_array[0]

        if not np.issubdtype(img_array.dtype, np.integer):
            raise ValueError(
                f"TIFF com dtype {img_array.dtype} em '{img_path}': só TIFF de "
                f"inteiros (uint8/uint16) é suportado. Para dados em ponto "
                f"flutuante use FITS, que tem normalização explícita "
                f"(ver --fits-normalization)."
            )

        # O dtype precisa ser lido ANTES da conversão para float32
        teto_dtype = float(np.iinfo(img_array.dtype).max)
        img_array = img_array.astype(np.float32)

        if tiff_normalization == "dtype":
            # A faixa vem do dtype: um TIFF uint8 dividido por 65535 sairia
            # quase preto (~0.004) silenciosamente.
            lo, hi = 0.0, teto_dtype
        elif tiff_normalization == "range":
            if tiff_range is None:
                raise ValueError("tiff_normalization='range' exige tiff_range=(lo, hi).")
            lo, hi = float(tiff_range[0]), float(tiff_range[1])
        elif tiff_normalization == "minmax":
            lo, hi = float(img_array.min()), float(img_array.max())
            if hi <= lo:
                raise ValueError(f"Imagem TIFF constante, min-max indefinido: {img_path}")
        else:
            raise ValueError(f"tiff_normalization desconhecida: {tiff_normalization}")

        img_array = np.clip((img_array - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    elif ext == ".fits":
        if astropy_fits is None:
            raise ImportError(
                "astropy é necessária para ler arquivos .fits "
                "(pip install astropy)."
            )
        with astropy_fits.open(img_path) as hdul:
            # Usa o primeiro HDU que contém dados de imagem
            img_array = None
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim >= 2:
                    img_array = np.asarray(hdu.data, dtype=np.float32)
                    break
            if img_array is None:
                raise ValueError(f"Nenhum HDU de imagem encontrado em {img_path}")

        if img_array.ndim > 2:
            img_array = img_array[0]

        # Normalização configurável para a faixa variável do float32 (ver docstring)
        if fits_normalization == "minmax":
            lo, hi = float(img_array.min()), float(img_array.max())
            if hi <= lo:
                raise ValueError(f"Imagem FITS constante, min-max indefinido: {img_path}")
        elif fits_normalization == "range":
            if fits_range is None:
                raise ValueError("fits_normalization='range' exige fits_range=(lo, hi).")
            lo, hi = float(fits_range[0]), float(fits_range[1])
        elif fits_normalization == "none":
            lo, hi = 0.0, 1.0
        else:
            raise ValueError(f"fits_normalization desconhecida: {fits_normalization}")

        img_array = np.clip((img_array - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    else:
        # Fallback para imagens comuns caso você coloque um .jpg de teste
        img = Image.open(img_path).convert("L")
        img_array = np.asarray(img, dtype=np.float32) / 255.0

    return img_array

class NeutronDataset(Dataset):
    def __init__(self, root_dir, patch_size=256, lr_scale=4, arquivos=None,
                 patches_per_image=1, min_nonzero=0.0, max_tentativas=10,
                 cache_images=0,
                 fits_normalization="minmax", fits_range=None,
                 tiff_normalization="dtype", tiff_range=None):
        """
        root_dir: Caminho para a pasta com as radiografias.
        patch_size: Tamanho do recorte perfeito (sem distorção) para treino.
        lr_scale: Fator de redução para a imagem de baixa resolução.
        patches_per_image: Quantos recortes aleatórios cada radiografia rende
            por época. Com 1, uma época tem tantas amostras quanto imagens —
            com algumas dezenas de radiografias isso são algumas dezenas de
            patches, ordens de magnitude abaixo do necessário para treinar uma
            GAN. O recorte e o espelhamento são sorteados a cada acesso, então
            índices diferentes da mesma imagem produzem patches diferentes.
        min_nonzero: fração mínima de pixels não-nulos exigida de um recorte.
            Reconstruções tomográficas têm um círculo útil inscrito na imagem e
            zeros nos cantos — medido nestes dados, 24,7% da área e 10,8% dos
            recortes aleatórios ficam majoritariamente fora. Com 0 (padrão) nada
            é rejeitado; com 0.5 o recorte é resorteado até passar.
        max_tentativas: quantas vezes resortear antes de aceitar o último.
        cache_images: quantas imagens JÁ NORMALIZADAS manter em memória (0 = sem
            cache, -1 = todas). Sem cache, cada recorte relê e renormaliza a
            imagem inteira: medido nestes dados, 4,8 ms de leitura (servida pelo
            cache de página do sistema) e 28,3 ms de normalização dos 4,1 M de
            pixels, para extrair um patch de 256. Com patches_per_image alto, a
            normalização repetida é o custo dominante do carregamento.
            ATENÇÃO: com num_workers > 0 cada processo tem a SUA cópia do cache,
            então a memória total é cache x num_workers. Como o cache remove o
            gargalo, poucos workers com cache grande costumam render mais que
            muitos workers sem cache.
        fits_normalization / fits_range: ver load_image_as_array (só afetam .fits).
        tiff_normalization / tiff_range: idem, para .tif/.tiff.
        """
        # 'arquivos' permite passar um subconjunto já dividido (ver dividir_pilha)
        self.files = list_supported_images(root_dir) if arquivos is None else list(arquivos)

        self.patch_size = patch_size
        self.lr_scale = lr_scale
        self.patches_per_image = max(1, int(patches_per_image))
        self.min_nonzero = float(min_nonzero)
        self.max_tentativas = max(1, int(max_tentativas))
        self.cache_images = int(cache_images)
        self._cache = {}
        self.fits_normalization = fits_normalization
        self.fits_range = fits_range
        self.tiff_normalization = tiff_normalization
        self.tiff_range = tiff_range

        # Normalize transforma [0, 1] em [-1, 1] (domínio do Tanh do Gerador)
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def _imagem(self, img_path):
        """Carrega a imagem normalizada, servindo do cache quando disponível."""
        if self.cache_images != 0 and img_path in self._cache:
            return self._cache[img_path]

        img_array = load_image_as_array(
            img_path, self.fits_normalization, self.fits_range,
            self.tiff_normalization, self.tiff_range
        )
        if self.cache_images < 0 or len(self._cache) < self.cache_images:
            self._cache[img_path] = img_array
        return img_array

    def __len__(self):
        return len(self.files) * self.patches_per_image

    def __getitem__(self, idx):
            # Os índices percorrem as imagens de forma intercalada, de modo que
            # qualquer prefixo da época cubra o conjunto todo em vez de esgotar
            # uma imagem antes de passar para a seguinte.
            img_path = self.files[idx % len(self.files)]

            img_array = self._imagem(img_path)

            # Conversão DIRETA numpy -> tensor (1, H, W), preservando float32.
            # NÃO usar TF.to_pil_image aqui: para arrays float ele converte para
            # PIL modo 'L' (8 bits), quantizando os 65536 níveis do uint16 em
            # apenas 256 — destruindo a precisão radiométrica do detector.
            img = torch.from_numpy(img_array).unsqueeze(0)

            # --- 1. Conferência de Tamanho ---
            # O ajuste anterior redimensionava com max(h, patch) por eixo, o que
            # escala SÓ o eixo deficiente: um recorte 200x300 virava 256x300,
            # esticando 28% na vertical e deformando a geometria da amostra.
            # Num pipeline cuja finalidade é MEDIR resolução, interpolar a
            # entrada para cima antes de degradá-la de novo inventa informação
            # que as métricas depois contabilizam como ganho. Melhor recusar e
            # deixar a escolha explícita.
            _, h, w = img.shape
            if h < self.patch_size or w < self.patch_size:
                raise ValueError(
                    f"'{os.path.basename(img_path)}' tem {h}x{w}, menor que "
                    f"patch_size={self.patch_size} em ao menos um eixo. Use "
                    f"--patch-size {min(h, w)} ou menor, ou tire esta imagem "
                    f"do conjunto de treino."
                )

            # --- 2. Extração de Patch Aleatório ---
            # Um recorte quase todo nulo não é dado: é a borda do círculo de
            # reconstrução. Treinar sobre ele ensina o Gerador a reproduzir
            # vazio. Resorteia até passar no critério, ou aceita o último.
            for _ in range(self.max_tentativas):
                i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
                    img, output_size=(self.patch_size, self.patch_size)
                )
                img_hr = TF.crop(img, i, j, h_crop, w_crop)
                if self.min_nonzero <= 0.0:
                    break
                if (img_hr > 0).to(torch.float32).mean().item() >= self.min_nonzero:
                    break

            # --- 3. Data Augmentation Científico ---
            if random.random() > 0.5:
                img_hr = TF.hflip(img_hr)
            if random.random() > 0.5:
                img_hr = TF.vflip(img_hr)

            # --- 4. Degradação Realista ---
            lr_size = self.patch_size // self.lr_scale
            img_lr = TF.gaussian_blur(img_hr, kernel_size=3)
            img_lr = TF.resize(
                img_lr, (lr_size, lr_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ).clamp(0.0, 1.0)  # bicúbico pode ter overshoot fora de [0, 1]

            # --- 5. Normalização para o domínio do modelo ---
            # Os tensores já estão em [0, 1]; Normalize leva para [-1, 1]
            tensor_hr = self.normalize(img_hr)
            tensor_lr = self.normalize(img_lr)

            return {"lr": tensor_lr, "hr": tensor_hr}

class StepEdgeDataset(Dataset):
    """
    Dataset dos phantoms de borda de degrau (step-edge), usados APENAS em
    validação para medir ESF/LSF/MTF (ver utils.mtf_from_edge) — não entram
    na loss de treino. Convenção: subdiretório próprio, ex. data/step_edges/.

    Diferente do NeutronDataset, o recorte é CENTRAL e determinístico e não há
    augmentation: a geometria da borda precisa ser estável entre épocas para
    que a MTF seja comparável ao longo do treino.
    """
    def __init__(self, root_dir="data/step_edges", patch_size=512, lr_scale=4,
                 fits_normalization="minmax", fits_range=None,
                 tiff_normalization="dtype", tiff_range=None):
        self.files = list_supported_images(root_dir)
        self.patch_size = patch_size
        self.lr_scale = lr_scale
        self.fits_normalization = fits_normalization
        self.fits_range = fits_range
        self.tiff_normalization = tiff_normalization
        self.tiff_range = tiff_range
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img_array = load_image_as_array(
            img_path, self.fits_normalization, self.fits_range,
            self.tiff_normalization, self.tiff_range
        )
        img = torch.from_numpy(img_array).unsqueeze(0)

        # Recorte central, limitado ao tamanho da imagem e múltiplo de lr_scale
        _, h, w = img.shape
        crop = min(self.patch_size, h, w)
        crop -= crop % self.lr_scale
        img_hr = TF.center_crop(img, (crop, crop))

        # Mesma degradação do treino, para avaliar a MTF na mesma condição
        lr_size = crop // self.lr_scale
        img_lr = TF.gaussian_blur(img_hr, kernel_size=3)
        img_lr = TF.resize(
            img_lr, (lr_size, lr_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ).clamp(0.0, 1.0)

        return {
            "lr": self.normalize(img_lr),
            "hr": self.normalize(img_hr),
            "name": os.path.basename(img_path),
        }

def get_dataloader(root_dir, batch_size=8, shuffle=True, num_workers=4,
                   patch_size=256, lr_scale=4, patches_per_image=1,
                   min_nonzero=0.0, cache_images=0, pin_memory=False,
                   arquivos=None, drop_last=True,
                   fits_normalization="minmax", fits_range=None,
                   tiff_normalization="dtype", tiff_range=None):
    dataset = NeutronDataset(root_dir, patch_size=patch_size, lr_scale=lr_scale,
                             arquivos=arquivos,
                             patches_per_image=patches_per_image,
                             min_nonzero=min_nonzero,
                             cache_images=cache_images,
                             fits_normalization=fits_normalization,
                             fits_range=fits_range,
                             tiff_normalization=tiff_normalization,
                             tiff_range=tiff_range)
    # pin_memory acelera a transferência para a GPU; persistent_workers evita
    # recriar os processos de leitura a cada época, o que pesa quando a época é
    # curta e há muitos workers.
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=drop_last,
                      pin_memory=pin_memory,
                      persistent_workers=num_workers > 0)

def get_step_edge_loader(root_dir="data/step_edges", patch_size=512, lr_scale=4,
                         num_workers=0, fits_normalization="minmax", fits_range=None,
                         tiff_normalization="dtype", tiff_range=None):
    """
    Loader de validação dos phantoms de borda. Retorna None se o diretório não
    existir ou estiver vazio (o treino segue normalmente sem a validação MTF).
    """
    dataset = StepEdgeDataset(root_dir, patch_size=patch_size, lr_scale=lr_scale,
                              fits_normalization=fits_normalization,
                              fits_range=fits_range,
                              tiff_normalization=tiff_normalization,
                              tiff_range=tiff_range)
    if len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
