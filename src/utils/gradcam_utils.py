# src/utils/gradcam_utils.py

import torch
import numpy as np
import cv2
import os
from torchvision import transforms

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        loss = output[:, class_idx]
        loss.backward()

        gradients = self.gradients
        activations = self.activations

        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        alpha = alpha.view(b, k, 1, 1)

        weighted = (alpha * activations).sum(1, keepdim=True)
        heatmap = torch.relu(weighted)
        heatmap = heatmap.squeeze().detach().cpu().numpy()

        heatmap -= heatmap.min()
        heatmap /= heatmap.max()

        return heatmap
