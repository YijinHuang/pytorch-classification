import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset


class FusionModel(nn.Module):
    def __init__(self, encoder):
        super(FusionModel, self).__init__()

        self.encoder = encoder
        self.encoder.fc = nn.Identity()

        self.fusion = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.MaxPool1d(2, 2),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.MaxPool1d(2, 2),
            nn.ReLU(True),
            nn.Linear(256, 1, bias=False)
        )

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x_1, x_2):
        with torch.no_grad():
            x_1 = self.encoder(x_1)
            x_2 = self.encoder(x_2)

        x = torch.cat((x_1, x_2), 1)
        x = x.view(x.size(0), 1, -1)
        x = self.fusion(x)
        x = x.squeeze()
        return x


class PairedEyeDataset(Dataset):
    def __init__(self, dataset, transform=None):
        super(PairedEyeDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.classes = [0, 1, 2, 3, 4]

    def __getitem__(self, index):
        x, x_aux, y = self.dataset[index]
        x = self.pil_loader(x)
        x_aux = self.pil_loader(x_aux)

        if self.transform:
            x = self.transform(x)
            x_aux = self.transform(x_aux)
        return x, x_aux, y

    def __len__(self):
        return len(self.dataset)
    
    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
