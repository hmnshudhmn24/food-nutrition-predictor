# 🍽️ Food Nutrition Predictor

**Model:** ViT-based multi-task network (classification + regression)

**Paper / Backbone:** google/vit-base-patch16-224

## What it does
Given an image of a meal, the model outputs:
- Predicted food class
- Estimated calories (kcal)
- Estimated carbs (g)
- Estimated protein (g)
- Estimated fat (g)

## Limitations
- Accuracy depends on dataset quality and diversity.
- Portion size estimation is approximate — better results when images include a reference object (ruler, hand).
- Works best for single-item plates; mixed dishes may be less accurate.

## Recommended usage
- Use on clear photos with single-dish plates.
- Add calibration for portion-size if precision is needed.

## License
Choose an appropriate license before publishing (MIT recommended for research/demo).
