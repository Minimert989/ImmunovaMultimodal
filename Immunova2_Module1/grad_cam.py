import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import os
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            output.retain_grad()
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)

    def forward(self, x):
        return self.model(x)

    def generate_cam(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.forward(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        target = output[0, class_idx]
        target.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu()
        heatmap = heatmap.detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        else:
            heatmap.fill(0)

        return heatmap

def save_gradcam_overlay(input_tensor, heatmap, output_path):
    input_img = input_tensor.squeeze().cpu()
    input_img = input_img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    input_img = input_img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    input_img = torch.clamp(input_img, 0, 1)

    img = to_pil_image(input_img)
    img_np = np.array(img)
    heatmap_uint8 = np.uint8(255 * heatmap)
    # Resize heatmap to image size
    heatmap_uint8 = cv2.resize(heatmap_uint8, (img_np.shape[1], img_np.shape[0]))
    # Apply JET colormap for better visibility
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    # Blend images
    superimposed_img = cv2.addWeighted(img_np, 0.5, heatmap_color, 0.5, 0)
    # Optionally add prediction label (example: TIL positive)
    predicted_label = "Prediction: TIL Positive"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(superimposed_img, predicted_label, (10, 25), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    Image.fromarray(superimposed_img).save(output_path)

def get_gradcam_visualization(model, input_tensor, target_layer, output_path=None):
    cam_generator = GradCAM(model, target_layer)
    heatmap = cam_generator.generate_cam(input_tensor)

    if output_path:
        save_gradcam_overlay(input_tensor, heatmap, output_path)

    return heatmap

if __name__ == "__main__":
    import torchvision.models as models
    from torchvision import transforms
    from PIL import Image

    class TILModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = models.resnet18(pretrained=False)
            self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, 1)

        def forward(self, x):
            return self.backbone(x)

    # Load model
    model_path = "til_model.pth"
    model = TILModel()
    model.load_state_dict(torch.load(model_path, map_location="mps"))
    model.eval()
    model.to("mps")

    # Load and preprocess image
    image_path = "sample_input2.jpg"  # Replace this with your real image file path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to("mps")

    # Generate and save GradCAM
    output_path = "gradcam_output2.png"
    get_gradcam_visualization(model, input_tensor, target_layer="backbone.layer4.1.conv2", output_path=output_path)
    print(f"GradCAM visualization saved to {output_path}")