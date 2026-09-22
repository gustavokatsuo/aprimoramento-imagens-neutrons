import os
import glob
import random
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

def list_supported_images(root_dir):
    """
    Lista ordenada dos arquivos de imagem suportados em root_dir. A ordenação
    mantém a indexação do Dataset estável entre execuções (reprodutibilidade),
    e ter a listagem isolada permite checar o dataset antes de montar o loader.
    """
    return sorted(
        f for ext in SUPPORTED_EXTENSIONS
        for f in glob.glob(os.path.join(root_dir, ext))
    )

def load_image_as_array(img_path, fits_normalization="minmax", fits_range=None):
    """
    Carrega uma imagem científica como numpy float32 2D no intervalo [0, 1].

    Formatos e normalizações:
      - TIFF (uint16): leitura via tifffile, divisão por 65535 (faixa fixa do
        detector ZEISS de 16 bits).
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

        # A faixa de normalização vem do dtype, não de uma constante: um TIFF
        # uint8 dividido por 65535 sairia quase preto (~0.004) silenciosamente.
        if np.issubdtype(img_array.dtype, np.integer):
            img_array = img_array.astype(np.float32) / np.iinfo(img_array.dtype).max
        else:
            raise ValueError(
                f"TIFF com dtype {img_array.dtype} em '{img_path}': só TIFF de "
                f"inteiros (uint8/uint16) tem faixa fixa conhecida. Para dados "
                f"em ponto flutuante use FITS, que tem normalização explícita "
                f"(ver --fits-normalization)."
            )

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
    def __init__(self, root_dir, patch_size=256, lr_scale=4,
                 fits_normalization="minmax", fits_range=None):
        """
        root_dir: Caminho para a pasta com as radiografias.
        patch_size: Tamanho do recorte perfeito (sem distorção) para treino.
        lr_scale: Fator de redução para a imagem de baixa resolução.
        fits_normalization / fits_range: ver load_image_as_array (só afetam .fits).
        """
        self.files = list_supported_images(root_dir)

        self.patch_size = patch_size
        self.lr_scale = lr_scale
        self.fits_normalization = fits_normalization
        self.fits_range = fits_range

        # Normalize transforma [0, 1] em [-1, 1] (domínio do Tanh do Gerador)
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
            img_path = self.files[idx]

            img_array = load_image_as_array(
                img_path, self.fits_normalization, self.fits_range
            )

            # Conversão DIRETA numpy -> tensor (1, H, W), preservando float32.
            # NÃO usar TF.to_pil_image aqui: para arrays float ele converte para
            # PIL modo 'L' (8 bits), quantizando os 65536 níveis do uint16 em
            # apenas 256 — destruindo a precisão radiométrica do detector.
            img = torch.from_numpy(img_array).unsqueeze(0)

            # --- 1. Ajuste de Tamanho ---
            _, h, w = img.shape
            if w < self.patch_size or h < self.patch_size:
                img = TF.resize(
                    img,
                    (max(h, self.patch_size), max(w, self.patch_size)),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ).clamp(0.0, 1.0)

            # --- 2. Extração de Patch Aleatório ---
            i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
                img, output_size=(self.patch_size, self.patch_size)
            )
            img_hr = TF.crop(img, i, j, h_crop, w_crop)

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
                 fits_normalization="minmax", fits_range=None):
        self.files = list_supported_images(root_dir)
        self.patch_size = patch_size
        self.lr_scale = lr_scale
        self.fits_normalization = fits_normalization
        self.fits_range = fits_range
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img_array = load_image_as_array(
            img_path, self.fits_normalization, self.fits_range
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
                   patch_size=256, lr_scale=4,
                   fits_normalization="minmax", fits_range=None):
    dataset = NeutronDataset(root_dir, patch_size=patch_size, lr_scale=lr_scale,
                             fits_normalization=fits_normalization,
                             fits_range=fits_range)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=True)

def get_step_edge_loader(root_dir="data/step_edges", patch_size=512, lr_scale=4,
                         num_workers=0, fits_normalization="minmax", fits_range=None):
    """
    Loader de validação dos phantoms de borda. Retorna None se o diretório não
    existir ou estiver vazio (o treino segue normalmente sem a validação MTF).
    """
    dataset = StepEdgeDataset(root_dir, patch_size=patch_size, lr_scale=lr_scale,
                              fits_normalization=fits_normalization,
                              fits_range=fits_range)
    if len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
