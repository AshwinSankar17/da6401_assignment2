import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_predictions(model, test_dataloader, class_names):
    """
    Plot 10x3 grid of images with prediction results.
    Shows 3 samples from each class with labels for predicted and true classes.
    
    Args:
        model: The trained neural network model
        test_dataloader: DataLoader containing test images
        class_names: List of class names for the dataset
    """
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dictionary to store samples by class
    samples_by_class = {i: [] for i in range(len(class_names))}
    
    # Normalization parameters used in the dataset
    mean = torch.tensor([0.4749, 0.4611, 0.3898])
    std = torch.tensor([0.1947, 0.1887, 0.1850])
    
    # Collect samples from test dataloader
    device = next(model.parameters()).device
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            # Store images by true class
            for img, true_label, pred_label in zip(images, labels, preds):
                true_class = true_label.item()
                pred_class = pred_label.item()
                
                # Store (image, true_class, pred_class) if we need more samples for this class
                if len(samples_by_class[true_class]) < 3:
                    # Denormalize the image
                    img_denorm = denormalize_image(img.cpu(), mean, std)
                    
                    samples_by_class[true_class].append((img_denorm, true_class, pred_class))
            
            # Check if we have enough samples
            if all(len(samples) >= 3 for samples in samples_by_class.values()):
                break
    
    # Create figure for plotting
    fig, axes = plt.subplots(10, 3, figsize=(12, 30))
    fig.tight_layout(pad=3.0)
    
    # Plot images
    for class_idx, class_samples in samples_by_class.items():
        for sample_idx, (img, true_class, pred_class) in enumerate(class_samples):
            ax = axes[class_idx, sample_idx]
            
            # Handle grayscale images
            if img.shape[0] == 1:
                ax.imshow(img.squeeze(), cmap='gray')
            else:
                # Convert from CxHxW to HxWxC for matplotlib
                img_np = img.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 1)  # Ensure values are in valid range
                ax.imshow(img_np)
            
            # Set title with true and predicted class
            is_correct = true_class == pred_class
            color = 'green' if is_correct else 'red'
            
            title = f"True: {class_names[true_class]}\nPred: {class_names[pred_class]}"
            ax.set_title(title, color=color)
            ax.axis('off')
    
    plt.suptitle("Model Predictions on Test Data", fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.savefig('predictions_grid.png', dpi=300, bbox_inches='tight')
    plt.show()

def denormalize_image(img, mean, std):
    """
    Denormalize an image by reversing the normalization process.
    
    Args:
        img: Normalized image tensor [C, H, W]
        mean: Mean values used for normalization
        std: Std values used for normalization
        
    Returns:
        Denormalized image tensor with values in [0, 1] range
    """
    # Create copies to avoid modifying the original tensors
    img = img.clone()
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    
    # Reverse the normalization: img = (normalized * std) + mean
    img = img * std + mean
    
    # Clamp values to [0, 1] range for displaying
    img = torch.clamp(img, 0, 1)
    
    return img