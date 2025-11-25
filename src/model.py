import torch
import torch.nn as nn
from transformers import ViTModel

class NutritionModel(nn.Module):
    def __init__(self, vit_model_name='google/vit-base-patch16-224', num_classes=50, regression_outputs=4):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_model_name)
        hidden = self.vit.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden//2, num_classes)
        )

        # Regression head (calories, carbs, protein, fat)
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden//2, regression_outputs)
        )

    def forward(self, pixel_values):
        # ViTModel returns: last_hidden_state, pooler_output (if present)
        outputs = self.vit(pixel_values=pixel_values)
        # Use pooled output (CLS token projection)
        features = outputs.pooler_output

        class_logits = self.classifier(features)
        nutrition_preds = self.regressor(features)

        return class_logits, nutrition_preds

    @classmethod
    def from_pretrained_checkpoint(cls, checkpoint_path, vit_model_name, num_classes, regression_outputs, device='cpu'):
        model = cls(vit_model_name=vit_model_name, num_classes=num_classes, regression_outputs=regression_outputs)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model
