from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
import os
from tqdm import tqdm
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
def expand_greyscale(t):
    return t.expand(3, -1, -1)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []

        for path in tqdm(Path(f'{folder}').glob('**/*')):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)