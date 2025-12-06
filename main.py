import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import os
import time
import numpy as np
import wandb
from sklearn.model_selection import StratifiedKFold
from cnn_model import CNN_Model_Trainer
from cnn_DatasetHandler import ImageDatasetHandler
from collections import Counter
from MetricsPlotter import plot_roc_and_prc_curve

def main():
    # Initialize W&B
    wandb.init()
    
    # Configurations
    dataset_dir = 'C:/Users/Zsombor/Documents/Egyetem/Szakdoga/SzakdogaPythonEnvironment/DataBase/CRC_DX/train'
    config = {
        "num_epochs": 25,
        "patience": 5,
        "batch_size": 32,
        "layers_to_unfreeze": 2,
        "dropout_rate": 0.0,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "train_split": 0.8,
        "seed": 42,
        "optimizer": "adam"
    }

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset handling
    dataset_handler = ImageDatasetHandler()
    train_dataset, val_dataset = dataset_handler.split(dataset_dir, config["train_split"], config["seed"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    # Model initialization
    print("Initializing model...")
    model_trainer = CNN_Model_Trainer(device,
                                    config["layers_to_unfreeze"],
                                    config["optimizer"],
                                    wandb.config["dropout_rate"],
                                    wandb.config["learning_rate"],
                                    wandb.config["weight_decay"])
    # Training the model with early stopping
    history = model_trainer.train(
        train_loader,
        val_loader,
        num_epochs=config["num_epochs"],
        patience=config["patience"]
    )

def k_fold_cross_validation():
    # Configurations
    dataset_dir_1 = 'C:/Users/Zsombor/Documents/Egyetem/Szakdoga/SzakdogaPythonEnvironment/DataBase/CRC_DX/train'
    dataset_dir_2 = 'C:/Users/Zsombor/Documents/Egyetem/Szakdoga/SzakdogaPythonEnvironment/DataBase/CRC_KR/train'
    dataset_dir_3 = 'C:/Users/Zsombor/Documents/Egyetem/Szakdoga/SzakdogaPythonEnvironment/DataBase/STAD/train'
    print(f"Dataset directory: {dataset_dir_1} used for training")
    config = {
        "num_epochs": 30,
        "patience": 5,
        "batch_size": 32,
        "layers_to_unfreeze": 2,
        "dropout_rate": 0.316,
        "learning_rate": 1.732e-04,
        "weight_decay": 9.083e-06,
        "n_splits": 5,
        "seed": 42,
        "optimizer": "adam",
        "k_folds": 5
    }

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset handling
    dataset_handler = ImageDatasetHandler()
    full_dataset = dataset_handler.get_full_dataset(dataset_dir_1, dataset_dir_2, dataset_dir_3)

    all_targets = []
    for ds in full_dataset.datasets:
        all_targets.extend(ds.targets)

    kfold = StratifiedKFold(n_splits=config["k_folds"], shuffle=True, random_state=config["seed"])
    fold_best_accuracys = []

    print(f"\nStarting {config['k_folds']}-Fold Cross-Validation...")

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset, all_targets)):
        print(f"\n===== FOLD {fold + 1}/{config['k_folds']} =====")
        # Start a new W&B run for each fold
        wandb.init(project="cnn-msi-classification", name=f"combined_dataset__FOLD_{fold+1}", reinit=True, config=config)

        # Create subsets
        train_subset = Subset(full_dataset, train_ids)
        val_subset = Subset(full_dataset, val_ids)
        for ds in train_subset.dataset.datasets:
            ds.transform = dataset_handler.train_transforms
        for ds in val_subset.dataset.datasets:
            ds.transform = dataset_handler.validation_transforms

        train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

        print("Initializing new model for this fold...")
        trainer = CNN_Model_Trainer(
            device=device,
            layers_to_unfreeze=config["layers_to_unfreeze"],
            optimizer_str=config["optimizer"],
            dropout_rate=config["dropout_rate"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )

        # Train and get history
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config["num_epochs"],
            patience=config["patience"]
        )

        # Find the best epoch by accuracy
        fold_best_accuracys.append(trainer.best_accuracy)
        # Finish the W&B run for this fold
        wandb.finish()

    # Aggregate results
    mean_accuracy = np.mean(fold_best_accuracys)
    std_accuracy = np.std(fold_best_accuracys)
    print(f"\n{config['k_folds']}-Fold Cross-Validation Patient-Level Results:\nMean Accuracy: {mean_accuracy:.4f}, Std Dev: ±{std_accuracy:.4f}")

def evaluate_on_test_dataset(dataset_dir, model_path):
    config = {
        "num_epochs": 30,
        "patience": 5,
        "batch_size": 32,
        "layers_to_unfreeze": 2,
        "dropout_rate": 0.0,
        "learning_rate": 9.85645792276934e-05,
        "weight_decay": 7.3365537284867425e-06,
        "n_splits": 5,
        "seed": 42,
        "optimizer": "adam",
        "k_folds": 5
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_trainer = CNN_Model_Trainer(
            device=device,
            layers_to_unfreeze=config["layers_to_unfreeze"],
            optimizer_str=config["optimizer"],
            dropout_rate=config["dropout_rate"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )

    print("Setting up test dataset and loader...")
    dataset_handler = ImageDatasetHandler()
    test_dataset = dataset_handler._init_test_dataset(dataset_dir)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    print("Load the best model for testing...")
    model_trainer.load(path=model_path)
    test_metrics = model_trainer.detailed_evaluate_one_epoch(test_loader)
    tile_metrics = test_metrics['tile_metrics']
    patient_metrics = test_metrics['patient_metrics']
    print("Validation Metrics:")
    model_trainer.print_metrics(tile_metrics, level="Tile")
    model_trainer.print_metrics(patient_metrics, level="Patient")
    print("\n--------------------------------\n")

    # Plotting ROC and PRC curves
    # Tile Level
    plot_roc_and_prc_curve(tile_metrics['fpr_curve'],
                        tile_metrics['tpr_curve'],
                        tile_metrics['precision_curve'],
                        tile_metrics['recall_curve'],
                        tile_metrics['roc_auc_score'],
                        tile_metrics['prc_auc_score'],
                        tile_metrics['tp'],
                        tile_metrics['fp'],
                        tile_metrics['tn'],
                        tile_metrics['fn'],
                        title='Test',
                        level='Tile')
    # Patient Level
    plot_roc_and_prc_curve(patient_metrics['fpr_curve'],
                        patient_metrics['tpr_curve'],
                        patient_metrics['precision_curve'],
                        patient_metrics['recall_curve'],
                        patient_metrics['roc_auc_score'],
                        patient_metrics['prc_auc_score'],
                        patient_metrics['tp'],
                        patient_metrics['fp'],
                        patient_metrics['tn'],
                        patient_metrics['fn'],
                        title='Test',
                        level='Patient')


if __name__ == "__main__":
    main()