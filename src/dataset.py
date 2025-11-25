import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd

class FoodDataset(Dataset):
    """Dataset that returns (pixel_tensor, class_label, nutrition_targets).

    Expects a CSV with columns: image,class_id,calories,carbs,protein,fat
    and an image directory containing images.
    """

    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        else:
            # default transform: convert to tensor in caller
            pass

        class_id = int(row['class_id'])
        nutrition = torch.tensor([
            float(row['calories']),
            float(row['carbs']),
            float(row['protein']),
            float(row['fat'])
        ], dtype=torch.float)

        return image, class_id, nutrition
