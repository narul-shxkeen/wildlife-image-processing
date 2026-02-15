#!/usr/bin/env python
# coding: utf-8

"""
Wildlife Three-Class Classification with ResNet
HPC-optimized version with proper logging

Label Mapping:
- Class 0: Empty (no animals/humans)
- Class 1: Humans
- Class 2: Wildlife
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import json
import os
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from pathlib import Path
import random
from tqdm.auto import tqdm
import shutil
import logging
import argparse
from datetime import datetime
import sys

# Configure logging
def setup_logging(log_dir='logs'):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Create logger
    logger = logging.getLogger('wildlife_classifier')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear any existing handlers
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class WildlifeDataset(Dataset):
    """Custom Dataset for wildlife image classification"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, image_path


def load_and_validate_data(dataset_path, labels_path, logger):
    """Load labels and validate dataset"""
    logger.info(f"Loading labels from {labels_path}")
    with open(labels_path, 'r') as f:
        labels_dict = json.load(f)
    
    # Dataset statistics
    total_images = len(labels_dict)
    empty_count = sum(1 for label in labels_dict.values() if label == 0)
    human_count = sum(1 for label in labels_dict.values() if label == 1)
    wildlife_count = sum(1 for label in labels_dict.values() if label == 2)
    
    logger.info("Dataset Statistics:")
    logger.info(f"  Total images: {total_images}")
    logger.info(f"  Empty images (label 0): {empty_count} ({empty_count/total_images*100:.1f}%)")
    logger.info(f"  Human images (label 1): {human_count} ({human_count/total_images*100:.1f}%)")
    logger.info(f"  Wildlife images (label 2): {wildlife_count} ({wildlife_count/total_images*100:.1f}%)")
    
    # Validate image existence
    dataset_path = Path(dataset_path)
    existing_images = []
    missing_images = []
    
    for image_name in labels_dict.keys():
        image_path = dataset_path / image_name
        if image_path.exists():
            existing_images.append(image_name)
        else:
            missing_images.append(image_name)
    
    logger.info(f"Image availability:")
    logger.info(f"  Images found: {len(existing_images)}")
    logger.info(f"  Images missing: {len(missing_images)}")
    
    if missing_images:
        logger.warning(f"  First few missing images: {missing_images[:5]}")
    
    return labels_dict, existing_images


def prepare_data_loaders(dataset_path, labels_dict, existing_images, 
                         batch_size=32, num_workers=4, train_prefix='T', logger=None):
    """Prepare training and validation data loaders"""
    
    # Data transformations
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Filter images for training
    image_paths = []
    labels = []
    dataset_path = Path(dataset_path)
    
    for image_name in existing_images:
            image_path = dataset_path / image_name
            image_paths.append(str(image_path))
            labels.append(labels_dict[image_name])
    
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    logger.info(f"Total labeled images found: {len(existing_images)}")
    logger.info(f"Images starting with '{train_prefix}' used for training/testing: {len(image_paths)}")
    logger.info(f"Images excluded (for inference only): {len(existing_images) - len(image_paths)}")
    
    # Create dataset
    full_dataset = WildlifeDataset(image_paths, labels, transform=transform_train)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update validation dataset transforms
    val_indices = val_dataset.indices
    val_paths = image_paths[val_indices]
    val_labels = labels[val_indices]
    val_dataset = WildlifeDataset(val_paths, val_labels, transform=transform_val)
    
    logger.info(f"Train set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    
    logger.info(f"Data loaders created with batch size: {batch_size}, num_workers: {num_workers}")
    
    return train_loader, val_loader


def create_model(num_classes=3, freeze_backbone=True, device='cuda', logger=None):
    """Create and configure ResNet model"""
    logger.info("Loading pre-trained ResNet18 model")
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze early layers if specified
    if freeze_backbone:
        logger.info("Freezing backbone layers")
        for param in model.parameters():
            param.requires_grad = False
    
    # Modify final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Unfreeze final layer
    for param in model.fc.parameters():
        param.requires_grad = True
    
    model = model.to(device)
    
    logger.info("Model architecture:")
    logger.info(f"  ResNet18 with modified final layer")
    logger.info(f"  Input features to FC layer: {num_ftrs}")
    logger.info(f"  Output classes: {num_classes} (Empty, Humans, Wildlife)")
    logger.info(f"  Model moved to: {device}")
    
    return model


def save_false_negatives(img_paths, true_labels, predictions, output_dirs, logger):
    """Save false negative images to respective directories
    
    False negatives: Humans (label 1) or Wildlife (label 2) predicted as Empty (label 0)
    """
    saved_counts = {'human': 0, 'wildlife': 0}
    
    for img_path, true_label, pred_label in zip(img_paths, true_labels, predictions):
        # Human marked as empty (false negative)
        if true_label == 1 and pred_label == 0:
            dest_path = Path(output_dirs['human']) / Path(img_path).name
            shutil.copy2(img_path, dest_path)
            saved_counts['human'] += 1
        
        # Wildlife marked as empty (false negative)
        elif true_label == 2 and pred_label == 0:
            dest_path = Path(output_dirs['wildlife']) / Path(img_path).name
            shutil.copy2(img_path, dest_path)
            saved_counts['wildlife'] += 1
    
    if saved_counts['human'] > 0 or saved_counts['wildlife'] > 0:
        logger.info(f"Saved false negatives - Human: {saved_counts['human']}, Wildlife: {saved_counts['wildlife']}")
    
    return saved_counts


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs, logger):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                      disable=not logger.isEnabledFor(logging.INFO))
    
    for batch_data in train_pbar:
        images, batch_labels = batch_data[0], batch_data[1]
        images, batch_labels = images.to(device, non_blocking=True), batch_labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += batch_labels.size(0)
        correct_predictions += (predicted == batch_labels).sum().item()
        
        train_pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100 * correct_predictions / total_predictions:.2f}%'
        })
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct_predictions / total_predictions
    
    logger.info(f'Epoch [{epoch+1}/{num_epochs}] Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
    
    return train_loss, train_acc


def validate_epoch(model, val_loader, criterion, device, epoch, num_epochs, 
                   output_dirs, save_false_negs, logger):
    """Validate for one epoch"""
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    all_img_paths = []
    all_true_labels = []
    all_predictions = []
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]',
                       disable=not logger.isEnabledFor(logging.INFO))
        
        for batch_data in val_pbar:
            images, batch_labels, img_paths = batch_data
            images, batch_labels = images.to(device, non_blocking=True), batch_labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, batch_labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += batch_labels.size(0)
            val_correct += (predicted == batch_labels).sum().item()
            
            # Store for false negative analysis
            all_img_paths.extend(img_paths)
            all_true_labels.extend(batch_labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            val_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * val_correct / val_total:.2f}%'
            })
    
    val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    
    logger.info(f'Epoch [{epoch+1}/{num_epochs}] Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
    
    # Save false negatives if requested
    if save_false_negs:
        save_false_negatives(all_img_paths, all_true_labels, all_predictions, output_dirs, logger)
    
    return val_loss, val_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, output_dir, logger):
    """Train the model"""
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, 'best_wildlife_model_3class.pth')
    
    # Create directories for false negatives
    output_dirs = {
        'human': os.path.join(output_dir, 'falseEmptyHumans'),
        'wildlife': os.path.join(output_dir, 'falseEmptyWildlife')
    }
    os.makedirs(output_dirs['human'], exist_ok=True)
    os.makedirs(output_dirs['wildlife'], exist_ok=True)
    logger.info(f"Created directories: {output_dirs['human']}, {output_dirs['wildlife']}")
    
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs, logger
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase (save false negatives only on final epoch)
        save_false_negs = (epoch == num_epochs - 1)
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, epoch, num_epochs,
            output_dirs, save_false_negs, logger
        )
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f'New best model saved with validation accuracy: {best_val_acc:.4f}')
        
        logger.info('-' * 70)
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, val_loader, device, output_dir, logger):
    """Evaluate model and save false negatives"""
    logger.info("Starting model evaluation")
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_img_paths = []
    
    # Create directories for false negatives
    output_dirs = {
        'human': os.path.join(output_dir, 'falseEmptyHumans'),
        'wildlife': os.path.join(output_dir, 'falseEmptyWildlife')
    }
    os.makedirs(output_dirs['human'], exist_ok=True)
    os.makedirs(output_dirs['wildlife'], exist_ok=True)
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Evaluating", 
                              disable=not logger.isEnabledFor(logging.INFO)):
            images, batch_labels, img_paths = batch_data
            images, batch_labels = images.to(device, non_blocking=True), batch_labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_img_paths.extend(img_paths)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Save false negatives
    saved_counts = save_false_negatives(all_img_paths, all_labels, all_predictions, output_dirs, logger)
    logger.info(f"False negatives saved - Human: {saved_counts['human']}, Wildlife: {saved_counts['wildlife']}")
    
    return all_predictions, all_labels


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, output_dir, logger):
    """Plot and save training history"""
    logger.info("Generating training history plots")
    
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'bo-', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'ro-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, 'ro-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot final metrics
    plt.subplot(1, 3, 3)
    final_metrics = ['Train Acc', 'Val Acc']
    final_values = [train_accuracies[-1], val_accuracies[-1]]
    bars = plt.bar(final_metrics, final_values, color=['blue', 'red'], alpha=0.7)
    plt.ylabel('Accuracy')
    plt.title('Final Model Performance')
    plt.ylim(0, 1)
    
    for bar, value in zip(bars, final_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history_3class.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Training history plot saved to {plot_path}")


def plot_confusion_matrix(true_labels, predictions, output_dir, logger):
    """Plot and save confusion matrix"""
    logger.info("Generating confusion matrix")
    
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Empty (0)', 'Humans (1)', 'Wildlife (2)'],
               yticklabels=['Empty (0)', 'Humans (1)', 'Wildlife (2)'])
    plt.title('Confusion Matrix - Three Class Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(output_dir, 'confusionMatrix_3class.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    # Log confusion matrix details
    logger.info("Confusion Matrix Details:")
    logger.info(f"  True Empty → Empty: {cm[0,0]}")
    logger.info(f"  True Empty → Humans: {cm[0,1]}")
    logger.info(f"  True Empty → Wildlife: {cm[0,2]}")
    logger.info(f"  True Humans → Empty: {cm[1,0]} (false negatives)")
    logger.info(f"  True Humans → Humans: {cm[1,1]}")
    logger.info(f"  True Humans → Wildlife: {cm[1,2]}")
    logger.info(f"  True Wildlife → Empty: {cm[2,0]} (false negatives)")
    logger.info(f"  True Wildlife → Humans: {cm[2,1]}")
    logger.info(f"  True Wildlife → Wildlife: {cm[2,2]}")
    
    return cm


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Wildlife Three-Class Classification')
    parser.add_argument('--dataset_path', type=str, default='cleanData', help='Path to dataset directory')
    parser.add_argument('--labels_path', type=str, default='wildlife_data.json', help='Path to labels JSON file')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--num_epochs', type=int, default=70, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_prefix', type=str, default='T', help='Prefix for training images')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze ResNet backbone')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(os.path.join(args.output_dir, 'logs'))
    set_seed(args.seed)
    
    logger.info("="*70)
    logger.info("Wildlife Three-Class Classification - Training Pipeline")
    logger.info("="*70)
    logger.info(f"Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Torchvision version: {torchvision.__version__}")
    
    # Load and prepare data
    labels_dict, existing_images = load_and_validate_data(
        args.dataset_path, args.labels_path, logger
    )
    
    train_loader, val_loader = prepare_data_loaders(
        args.dataset_path, labels_dict, existing_images,
        args.batch_size, args.num_workers, args.train_prefix, logger
    )
    
    # Create model
    model = create_model(num_classes=3, freeze_backbone=args.freeze_backbone, 
                        device=device, logger=logger)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    logger.info(f"Loss function: CrossEntropyLoss")
    logger.info(f"Optimizer: Adam (lr={args.learning_rate})")
    
    # Train model
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        args.num_epochs, device, args.output_dir, logger
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies,
                         args.output_dir, logger)
    
    # Load best model and evaluate
    best_model_path = os.path.join(args.output_dir, 'best_wildlife_model_3class.pth')
    logger.info(f"Loading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    
    predictions, true_labels = evaluate_model(model, val_loader, device, args.output_dir, logger)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    
    logger.info("="*70)
    logger.info("Model Evaluation Results")
    logger.info("="*70)
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions, args.output_dir, logger)
    
    logger.info("="*70)
    logger.info("Training Complete")
    logger.info("="*70)
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"False negatives saved in: {os.path.join(args.output_dir, 'falseEmptyWildlife')} "
               f"and {os.path.join(args.output_dir, 'falseEmptyHumans')}")


if __name__ == '__main__':
    main()
