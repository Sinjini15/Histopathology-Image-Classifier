import torch
import os
from dataset import get_dataloaders
from model import get_model, get_device
from utils.eval_utils import calculate_metrics, plot_roc_curve, plot_confusion_matrix
from utils.path_utils import generate_path

def main():

    train_x_path, train_y_path, test_x_path, test_y_path, val_x_path, val_y_path = generate_path()

    batch_size = 64

    device = get_device()
    model = get_model().to(device)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.eval()

    _, _, test_loader = get_dataloaders(
        train_x_path, train_y_path,
        val_x_path, val_y_path,
        test_x_path, test_y_path,
        batch_size=batch_size
    )

    os.makedirs('outputs/metrics', exist_ok=True)

    labels, preds, probs = calculate_metrics(model, test_loader, device)
    
    test_accuracy = (labels == preds).mean()
    print(f"Test Accuracy: {test_accuracy:.4f}")


    plot_roc_curve(labels, probs, 'outputs/metrics/roc_curve.png')
    plot_confusion_matrix(labels, preds, 'outputs/metrics/confusion_matrix.png')

if __name__ == "__main__":
    main()
