

import torch
import torch.nn as nn
import torchvision.models as models

class TILBinaryCNN(nn.Module):
    def __init__(self, pretrained=True):
        super(TILBinaryCNN, self).__init__()
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        # Modify the last fully connected layer for binary classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 1)  # Output single logit
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.backbone(x)
        probs = self.sigmoid(logits)
        return probs


# GradCAM-compatible hook registration function
def register_hooks(model):
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        output.retain_grad()
        activations['value'] = output

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    # Register hooks on the final convolutional layer
    target_layer = model.backbone.layer4[1].conv2
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    return activations, gradients