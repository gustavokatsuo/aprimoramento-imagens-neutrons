import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class NeutronDataset(Dataset):
    def __init__(self, root_dir, patch_size=256, lr_scale=4):
        """
        root_dir: Caminho para a pasta com as radiografias.
        patch_size: Tamanho do recorte perfeito (sem distorção) para treino.
        lr_scale: Fator de redução para a imagem de baixa resolução.
        """
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.tif")) + 
                           glob.glob(os.path.join(root_dir, "*.png")) +
                           glob.glob(os.path.join(root_dir, "*.jpg")))
        
        self.patch_size = patch_size
        self.lr_scale = lr_scale

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        # Converte para escala de cinza preservando detalhes
        img = Image.open(img_path).convert("L")
        
        # --- 1. Ajuste de Tamanho (Proteção contra imagens pequenas) ---
        w, h = img.size
        if w < self.patch_size or h < self.patch_size:
            img = TF.resize(img, (max(h, self.patch_size), max(w, self.patch_size)))

        # --- 2. Extração de Patch Aleatório (Sem Esmagamento!) ---
        i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
            img, output_size=(self.patch_size, self.patch_size)
        )
        img_hr = TF.crop(img, i, j, h_crop, w_crop)

        # --- 3. Data Augmentation Científico ---
        if random.random() > 0.5:
            img_hr = TF.hflip(img_hr) # Espelhamento Horizontal
        if random.random() > 0.5:
            img_hr = TF.vflip(img_hr) # Espelhamento Vertical

        # --- 4. Degradação Realista (Simulando o Detector) ---
        lr_size = self.patch_size // self.lr_scale
        
        # Leve desfoque antes de reduzir a escala (simula o Point Spread Function do cintilador)
        img_lr = TF.gaussian_blur(img_hr, kernel_size=3)
        # Redução bicúbica para gerar a versão corrompida
        img_lr = TF.resize(img_lr, (lr_size, lr_size), interpolation=transforms.InterpolationMode.BICUBIC)

        # --- 5. Transformação em Tensores e Normalização [-1, 1] ---
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])

        tensor_hr = normalize(to_tensor(img_hr))
        tensor_lr = normalize(to_tensor(img_lr))

        return {"lr": tensor_lr, "hr": tensor_hr}

def get_dataloader(root_dir, batch_size=8, shuffle=True, num_workers=4):
    dataset = NeutronDataset(root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)