# Seat Occupancy Detection System

## Model Performance
- Validation Accuracy: 84.62%
- Confusion Matrix: [[6, 1], [1, 5]]
- Overfitting starts at epoch 16 (improved from epoch 11)

## Files
- models.py - CNN architecture
- train.py - Training script with data augmentation
- best_model.pth - Trained model weights
- training_history.json - Training metrics for graphing

## How to Use
1. Upload an image of a chair
2. Model predicts VACANT or OCCUPIED

## Deployment
https://huggingface.co/spaces/sabr111111/seat-optimizer

## Latest Improvements
- Stronger data augmentation (rotation, color jitter, affine, perspective)
- Overfitting delayed from epoch 11 to epoch 16
- More balanced predictions between vacant and occupied classes
