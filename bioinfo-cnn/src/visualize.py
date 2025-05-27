import torch
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import LayerGradCam
from pathlib import Path

from models.cnn import BacterialColonyCNN
from data.dataset import BacterialColonyDataset, get_transforms

def visualize_attention(model, image, label, device):
    """Visualize attention maps using Grad-CAM"""
    model.eval()
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    
    # Get the target layer (third conv block)
    target_layer = model.features[8]  # Third conv block's ReLU
    
    # Create Grad-CAM
    grad_cam = LayerGradCam(model, target_layer)
    
    # Get attribution
    attribution = grad_cam.attribute(image, target=label)
    
    # Convert tensors to numpy arrays for visualization
    image_np = image.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
    attribution_np = attribution.squeeze(0).cpu().detach().numpy()
    
    # Resize attribution map to match image dimensions
    attribution_np = np.mean(attribution_np, axis=0)  # Average across channels
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)
    
    # Normalize attribution map
    if np.max(np.abs(attribution_np)) > 0:
        attribution_np = attribution_np / np.max(np.abs(attribution_np))
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image
    ax1.imshow(image_np)
    ax1.set_title(f'Original Image - Class {label}')
    ax1.axis('off')
    
    # Overlay attribution
    ax2.imshow(image_np)
    ax2.imshow(attribution_np, alpha=0.5, cmap='jet')
    ax2.set_title(f'Grad-CAM Overlay - Class {label}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'visualizations/attention_map_class_{label}.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataset for visualization
    dataset = BacterialColonyDataset(
        root_dir='datasets',
        transform=get_transforms(subset='test')  # Use test transforms
    )
    
    # Create model and load trained weights
    model = BacterialColonyCNN(num_classes=len(dataset.classes)).to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Create output directory for visualizations
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # Generate visualizations for a few samples from each class
    samples_per_class = 2
    for class_idx in range(len(dataset.classes)):
        class_samples = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
        selected_indices = class_samples[:samples_per_class]
        
        for idx in selected_indices:
            image, label = dataset[idx]
            visualize_attention(model, image, label, device)
            print(f'Generated visualization for class {dataset.classes[label]}')

if __name__ == '__main__':
    main() 