import argparse
import os
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import ViTFeatureExtractor, get_cosine_schedule_with_warmup

from dataset import FoodDataset
from model import NutritionModel
from utils import set_seed, save_checkpoint

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--data_csv', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--labels_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()

def main():
    args = get_args()

    import json
    cfg = json.load(open(args.config))

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(42)

    # Feature extractor from transformers to get normalization params
    fe = ViTFeatureExtractor.from_pretrained(cfg['model_name'])

    transform = transforms.Compose([
        transforms.Resize((cfg['image_size'], cfg['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=fe.image_mean, std=fe.image_std)
    ])

    dataset = FoodDataset(args.data_csv, args.img_dir, transform=transform)

    # Train/val split
    val_ratio = 0.1
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)

    model = NutritionModel(vit_model_name=cfg['model_name'], num_classes=cfg['num_classes'], regression_outputs=cfg['regression_outputs'])
    model.to(device)

    # Losses
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg.get('weight_decay', 0.0))

    total_steps = len(train_loader) * cfg['epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    best_val_loss = float('inf')
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(cfg['epochs']):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        for images, labels, nutric in pbar:
            images = images.to(device)
            labels = labels.to(device)
            nutric = nutric.to(device)

            optimizer.zero_grad()
            cls_logits, reg_out = model(images)

            loss_cls = criterion_cls(cls_logits, labels)
            loss_reg = criterion_reg(reg_out, nutric)
            loss = loss_cls + loss_reg

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels, nutric in tqdm(val_loader, desc='Validation'):
                images = images.to(device)
                labels = labels.to(device)
                nutric = nutric.to(device)
                cls_logits, reg_out = model(images)
                loss_cls = criterion_cls(cls_logits, labels)
                loss_reg = criterion_reg(reg_out, nutric)
                val_loss += (loss_cls + loss_reg).item()

        val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} VAL loss: {val_loss:.4f}")

        # Save best
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        save_checkpoint(model, checkpoint_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
            save_checkpoint(model, best_path)

    print('Training finished.')

if __name__ == '__main__':
    main()
