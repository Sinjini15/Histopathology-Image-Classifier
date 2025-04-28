import torch
import os
import cv2
import numpy as np
from dataset import get_dataloaders
from model import get_model, get_device
from utils.gradcam_utils import GradCAM
from torchvision import transforms
from utils.path_utils import generate_path

def overlay_heatmap(heatmap, image):
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image = image.permute(1, 2, 0).cpu().numpy()
    image = np.uint8(255 * image)
    overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
    return overlay

def main():
    
    # Get the paths
    
    train_x_path, train_y_path, test_x_path, test_y_path, val_x_path, val_y_path = generate_path()

    batch_size = 1

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

    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    os.makedirs('outputs/gradcam', exist_ok=True)

    for idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        heatmap = gradcam.generate(images)

        overlay = overlay_heatmap(heatmap, images[0])

        save_path = f"outputs/gradcam/gradcam_{idx}.png"
        cv2.imwrite(save_path, overlay[:, :, ::-1])  # Convert RGB to BGR for cv2

        if idx == 9:
            break  # generate for first 10 samples only

if __name__ == "__main__":
    main()
