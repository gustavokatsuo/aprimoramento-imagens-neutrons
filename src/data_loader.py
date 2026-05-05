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

class NeutronDataset(Dataset):
    def __init__(self, root_dir, patch_size=256, lr_scale=4):
        """
        root_dir: Caminho para a pasta com as radiografias.
        patch_size: Tamanho do recorte perfeito (sem distorção) para treino.
        lr_scale: Fator de redução para a imagem de baixa resolução.
        """
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.tiff")) + 
                           glob.glob(os.path.join(root_dir, "*.png")) +
                           glob.glob(os.path.join(root_dir, "*.jpg")))
        
        self.patch_size = patch_size
        self.lr_scale = lr_scale

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
            img_path = self.files[idx]
            
            # Leitura científica do TIFF de 16-bits
            if img_path.endswith('.tiff') or img_path.endswith('.tif'):
                img_array = tifffile.imread(img_path)
                
                # Se a imagem tiver mais de 2 dimensões (ex: RGBA), pega só o canal de cinza
                if len(img_array.shape) > 2:
                    img_array = img_array[:, :, 0]
                    
                # Converte de 16-bit (0 a 65535) para float32 no intervalo [0, 1]
                img_array = img_array.astype(np.float32) / 65535.0
                
                # Converte Numpy para PIL Image para poder usar o torchvision transforms
                img = TF.to_pil_image(img_array)
            else:
                # Fallback para imagens comuns caso você coloque um .jpg de teste
                img = Image.open(img_path).convert("L")
            
            # --- 1. Ajuste de Tamanho ---
            w, h = img.size
            if w < self.patch_size or h < self.patch_size:
                img = TF.resize(img, (max(h, self.patch_size), max(w, self.patch_size)))

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
            img_lr = TF.resize(img_lr, (lr_size, lr_size), interpolation=transforms.InterpolationMode.BICUBIC)

            # --- 5. Transformação em Tensores ---
            # Como as imagens de 16-bit já estão no intervalo [0, 1] internamente pelo to_pil_image
            to_tensor = transforms.ToTensor()
            normalize = transforms.Normalize(mean=[0.5], std=[0.5]) # Transforma [0, 1] para [-1, 1]

            tensor_hr = normalize(to_tensor(img_hr))
            tensor_lr = normalize(to_tensor(img_lr))

            return {"lr": tensor_lr, "hr": tensor_hr}

def get_dataloader(root_dir, batch_size=8, shuffle=True, num_workers=4):
    dataset = NeutronDataset(root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)