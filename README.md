# 🍽️ Food Nutrition Predictor

A ViT-based model that predicts:

- **Food name (classification)**
- **Estimated calories (regression)**
- **Macronutrient breakdown: carbs, protein, fat (regression)**

This repo contains training, inference, and Hugging Face–ready export using `google/vit-base-patch16-224` as the backbone.

## Features
- Multi-head model: classification + regression
- Trainer script with validation and checkpointing
- Inference wrapper returning JSON output
- Hugging Face model card and sample inference
- Example `labels.csv` format and dataset loader

## Quick start

1. Create a virtualenv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare `data/labels.csv` and `data/images/` (see `data/labels.csv` example below).

3. Train:

```bash
python src/train.py --config config.json --data_csv data/labels.csv --img_dir data/images --output_dir outputs
```

4. Run inference:

```bash
python src/inference.py --model_path outputs/checkpoint_best.pth --labels_file data/classes.txt --image sample.jpg
```

## Data format
`data/labels.csv` should have columns:

```
image,class_id,calories,carbs,protein,fat
burger_001.jpg,0,550,45,25,30
pizza_01.jpg,1,285,36,12,10
```

Create `data/classes.txt` listing class names in order (one per line).

## File structure

```
food-nutrition-predictor/
│
├── README.md
├── requirements.txt
├── config.json
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── utils.py
│
├── data/
│   ├── images/
│   └── labels.csv
│
├── notebooks/
│   └── EDA_and_Training.ipynb
│
└── huggingface/
    ├── model_card.md
    └── sample_inference.py
```

## Notes & tips
- Use mixed precision (AMP) for faster training on modern GPUs.
- If you have few classes or small dataset, apply heavy augmentation.
- Calorie/macros targets can be normalized (optional) — this repo expects raw grams/calorie values.


