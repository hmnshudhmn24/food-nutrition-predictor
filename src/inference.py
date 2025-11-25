import argparse
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTFeatureExtractor

from model import NutritionModel
from utils import load_classes

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--labels_file', type=str, required=True)
    p.add_argument('--image', type=str, required=True)
    p.add_argument('--vit_model_name', type=str, default='google/vit-base-patch16-224')
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    classes = load_classes(args.labels_file)

    feature_extractor = ViTFeatureExtractor.from_pretrained(args.vit_model_name)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])

    model = NutritionModel(vit_model_name=args.vit_model_name, num_classes=len(classes), regression_outputs=4)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    img = Image.open(args.image).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        cls_logits, reg_out = model(x)
        pred_class_idx = int(torch.argmax(cls_logits, dim=1).item())
        pred_name = classes[pred_class_idx]
        calories, carbs, protein, fat = reg_out.squeeze().cpu().numpy().tolist()

    output = {
        'food_name': pred_name,
        'calories': float(round(calories, 2)),
        'carbs_g': float(round(carbs, 2)),
        'protein_g': float(round(protein, 2)),
        'fat_g': float(round(fat, 2))
    }

    import json
    print(json.dumps(output, indent=2))

if __name__ == '__main__':
    main()
