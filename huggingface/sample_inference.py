from inference_wrapper import FoodPredictor

pred = FoodPredictor(model_path='outputs/checkpoint_best.pth', labels_file='data/classes.txt')
print(pred.predict('data/images/pizza_01.jpg'))
