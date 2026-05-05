import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class NeutronDataset(Dataset):
    def __init__(self, root_dir, hr_size=(256, 256), lr_scale=4, transform=None):
        """
        root_dir: Caminho para a pasta com as radiografias.
        hr_size: Tamanho da imagem de alta resolução (High Res).
        lr_scale: Fator de redução para criar a baixa resolução (Low Res).
        """
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.tif")) + 
                           glob.glob(os.path.join(root_dir, "*.png")) +
                           glob.glob(os.path.join(root_dir, "*.jpg")))
        
        self.hr_transform = transforms.Compose([
            transforms.Resize(hr_size, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) # Normaliza para [-1, 1] (compatível com Tanh)
        ])
        
        self.lr_transform = transforms.Compose([
            transforms.Resize((hr_size[0] // lr_scale, hr_size[1] // lr_scale), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("L") # Garante escala de cinza para radiografia

        img_hr = self.hr_transform(img)
        img_lr = self.lr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

def get_dataloader(root_dir, batch_size=8, shuffle=True, num_workers=4):
    dataset = NeutronDataset(root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)