import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import ChestXRayModel

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.device = next(model.parameters()).device
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class):
        self.model.eval()
        input_tensor = input_tensor.to(self.device)

        output = self.model(input_tensor)

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot.to(self.device))

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured. Hook failed!")

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, output

    @staticmethod
    def visualize_and_save(cam, original_image, filename):
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        overlay = heatmap + np.float32(original_image) / 255
        overlay = overlay / np.max(overlay)
        combined = np.hstack([
            np.float32(original_image) / 255,
            heatmap,
            overlay
        ])
        combined = (combined * 255).astype(np.uint8)
        cv2.imwrite(filename, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


def demonstrate_gradcam(model, image_path, diseases):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    original_image = image.copy()

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    input_tensor = transform(image=image)['image'].unsqueeze(0).to(device)

    target_layer = model.backbone.layer4[-1].conv3
    gradcam = GradCAM(model, target_layer)


    output = model(input_tensor)
    probs = torch.sigmoid(output)[0]
    top3_conf, top3_idx = torch.topk(probs, 3)

    for i, (conf, idx) in enumerate(zip(top3_conf, top3_idx)):
        cam, _ = gradcam.generate_cam(input_tensor, idx.item())
        filename = f'gradcam_top_{i+1}_{diseases[idx.item()]}.png'
        gradcam.visualize_and_save(cam, original_image, filename)
        print(f"Saved {filename} with confidence {conf.item():.2f}")


diseases = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
    'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
    'No Finding'
]

model = ChestXRayModel()
model.load_model('../outputs/models/best_model.pth')
demonstrate_gradcam(model, '../data/images/00000011_006.png', diseases)
