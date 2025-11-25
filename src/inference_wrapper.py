from PIL import Image
import torch
from torchvision import transforms
from transformers import ViTFeatureExtractor

from model import NutritionModel
from utils import load_classes

class FoodPredictor:
    def __init__(self, model_path, labels_file, vit_model_name='google/vit-base-patch16-224', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = load_classes(labels_file)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(vit_model_name)
        self.transform = transforms.Compose([
            transforms.Resize((self.feature_extractor.size, self.feature_extractor.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.feature_extractor.image_mean, std=self.feature_extractor.image_std)
        ])

        self.model = NutritionModel(vit_model_name=vit_model_name, num_classes=len(self.classes), regression_outputs=4)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, pil_image_or_path):
        if isinstance(pil_image_or_path, str):
            img = Image.open(pil_image_or_path).convert('RGB')
        else:
            img = pil_image_or_path.convert('RGB')

        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            cls_logits, reg_out = self.model(x)
            idx = int(torch.argmax(cls_logits, dim=1).item())
            name = self.classes[idx]
            calories, carbs, protein, fat = reg_out.squeeze().cpu().numpy().tolist()

        return {
            'food_name': name,
            'calories': float(round(calories, 2)),
            'carbs_g': float(round(carbs, 2)),
            'protein_g': float(round(protein, 2)),
            'fat_g': float(round(fat, 2))
        }
