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
from MetricsPlotter import plot_roc_and_prc_curve, plot_roc_and_prc_curves_together
from focal_loss import FocalLoss


def main():
    # Initialize W&B
    wandb.init()
    
    # Configurations
    dataset_dir = 'DataBase/CRC_DX/train'
    config = {
        "num_epochs": 15,
        "patience": 3,
        "batch_size": 32,
        "layers_to_unfreeze": 2,
        "dropout_rate": 0.316,
        "learning_rate": 1.732e-04,
        "weight_decay": 9.083e-06,
        "k_folds": 5,
        "seed": 42,
        "optimizer": "adam"
    }

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset handling
    split = 1-1/config["k_folds"] # 0,8 for 5-Fold
    dataset_handler = ImageDatasetHandler()
    train_dataset, val_dataset = dataset_handler.split(dataset_dir, split, config["seed"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    # Model initialization
    print("Initializing model...")
    model_trainer = CNN_Model_Trainer(device,
                                    config["layers_to_unfreeze"],
                                    config["optimizer"],
                                    config["dropout_rate"],
                                    config["learning_rate"],
                                    config["weight_decay"],
                                    criterion=FocalLoss(wandb.config["gamma"], wandb.config["alpha"]))
    # Training the model with early stopping
    history = model_trainer.train(
        train_loader,
        val_loader,
        num_epochs=config["num_epochs"],
        patience=config["patience"]
    )

def train_cnn_model_k_fold(config, training_name, dataset_dir_1, dataset_dir_2 = None, dataset_dir_3 = None):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Configurations
    print(f"Dataset directory: {dataset_dir_1}, {dataset_dir_2 if dataset_dir_2 is not None else ''}, {dataset_dir_3 if dataset_dir_3 is not None else ''} used for training")
    # Dataset handling
    dataset_handler = ImageDatasetHandler()
    full_dataset = dataset_handler.get_full_dataset(dataset_dir_1, dataset_dir_2, dataset_dir_3)

    all_targets = []
    for ds in full_dataset.datasets:
        all_targets.extend(ds.targets)

    kfold = StratifiedKFold(n_splits=config["k_folds"], shuffle=True, random_state=config["seed"])
    fold_best_accuracys = []

    print(f"\nStarting {config['k_folds']}-Fold Training...")

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset, all_targets)):
        print(f"\n===== FOLD {fold + 1}/{config['k_folds']} =====")
        # Start a new W&B run for each fold
        wandb.init(project="cnn-msi-classification", name=f"{training_name}_{fold+1}", reinit=True, config=config)

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
            weight_decay=config["weight_decay"],
            criterion=config["criterion"]
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

def evaluate_cnn_model_on_dataset(dataset_dir, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Setting up test dataset and loader...")
    dataset_handler = ImageDatasetHandler()
    test_dataset = dataset_handler._init_test_dataset(dataset_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print("Loading selected model...")
    model_trainer = CNN_Model_Trainer(
            device=device
        )
    model_trainer.load(path=model_path)

    print("Started evaluation on dataset...")
    test_metrics = model_trainer.detailed_evaluate_one_epoch(test_loader)
    tile_metrics = test_metrics['tile_metrics']
    patient_metrics = test_metrics['patient_metrics']

    print("Validation Metrics:")
    model_trainer.print_metrics(tile_metrics, level="Tile")
    model_trainer.print_metrics(patient_metrics, level="Patient")
    print("\n--------------------------------\n")
    
    # Tile Level
    plot_roc_and_prc_curve(tile_metrics,
                        title='Test',
                        level='Tile')
    # Patient Level
    plot_roc_and_prc_curve(patient_metrics,
                        title='Test',
                        level='Patient')
    
def evaluate_models_together(data_loader, model_paths):
    # setting upt model paths
    model_1_path, model_2_path, model_3_path, model_4_path, model_5_path = model_paths

    # evaluating models
    tile_metrics_1, patient_metrics_1 = evaluate_model(data_loader, model_1_path)
    tile_metrics_2, patient_metrics_2 = evaluate_model(data_loader, model_2_path)
    tile_metrics_3, patient_metrics_3 = evaluate_model(data_loader, model_3_path)
    tile_metrics_4, patient_metrics_4 = evaluate_model(data_loader, model_4_path)
    tile_metrics_5, patient_metrics_5 = evaluate_model(data_loader, model_5_path)

    # Print results and get results
    print("--- Model 1 Results: ---")
    print(" Tile Level Metrics:")
    t_roc_1, t_prc_1 = metrics_helper(tile_metrics_1)
    print(" Patient Level Metrics:")
    p_roc_1, p_prc_1 = metrics_helper(patient_metrics_1, True)
    print("--- Model 2 Results: ---")
    print(" Tile Level Metrics:")
    t_roc_2, t_prc_2 = metrics_helper(tile_metrics_2)
    print(" Patient Level Metrics:")
    p_roc_2, p_prc_2 = metrics_helper(patient_metrics_2, True)
    print("--- Model 3 Results: ---")
    print(" Tile Level Metrics:")
    t_roc_3, t_prc_3 = metrics_helper(tile_metrics_3)
    print(" Patient Level Metrics:")
    p_roc_3, p_prc_3 = metrics_helper(patient_metrics_3, True)
    print("--- Model 4 Results: ---")
    print(" Tile Level Metrics:")
    t_roc_4, t_prc_4 = metrics_helper(tile_metrics_4)
    print(" Patient Level Metrics:")
    p_roc_4, p_prc_4 = metrics_helper(patient_metrics_4, True)
    print("--- Model 5 Results: ---")
    print(" Tile Level Metrics:")
    t_roc_5, t_prc_5 = metrics_helper(tile_metrics_5)
    print(" Patient Level Metrics:")
    p_roc_5, p_prc_5 = metrics_helper(patient_metrics_5, True)

    # calculate averages and deviations
    p_roc = np.mean([p_roc_1, p_roc_2, p_roc_3, p_roc_4, p_roc_5])
    p_roc_dev = np.std([p_roc_1, p_roc_2, p_roc_3, p_roc_4, p_roc_5])
    p_prc = np.mean([p_prc_1, p_prc_2, p_prc_3, p_prc_4, p_prc_5])
    p_prc_dev = np.std([p_prc_1, p_prc_2, p_prc_3, p_prc_4, p_prc_5])

    t_roc = np.mean([t_roc_1, t_roc_2, t_roc_3, t_roc_4, t_roc_5])
    t_roc_dev = np.std([t_roc_1, t_roc_2, t_roc_3, t_roc_4, t_roc_5])
    t_prc = np.mean([t_prc_1, t_prc_2, t_prc_3, t_prc_4, t_prc_5])
    t_prc_dev = np.std([t_prc_1, t_prc_2, t_prc_3, t_prc_4, t_prc_5])

    # final results
    print("\n=== Summary of Tile-level results ===")
    print(f"Average ROC AUC: {t_roc:.4f} ± {t_roc_dev:.4f}")
    print(f"Average PRC AUC: {t_prc:.4f} ± {t_prc_dev:.4f}")
    plot_roc_and_prc_curves_together(
        [tile_metrics_1, tile_metrics_2, tile_metrics_3, tile_metrics_4, tile_metrics_5],
        title="Test",
        level="Tile"
    )
    print("\n=== Summary of Patient-level results ===")
    print(f"Average ROC AUC: {p_roc:.4f} ± {p_roc_dev:.4f}")
    print(f"Average PRC AUC: {p_prc:.4f} ± {p_prc_dev:.4f}")
    plot_roc_and_prc_curves_together(
        [patient_metrics_1, patient_metrics_2, patient_metrics_3, patient_metrics_4, patient_metrics_5],
        title="Test",
        level="Patient"
    )
    
def evaluate_model(data_loader, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_trainer = CNN_Model_Trainer(
            device=device
        )
    model_trainer.load(path=model_path)
    test_metrics = model_trainer.detailed_evaluate_one_epoch(data_loader)
    tile_metrics = test_metrics['tile_metrics']
    patient_metrics = test_metrics['patient_metrics']
    return tile_metrics, patient_metrics

def metrics_helper(metrics, use_sensitivity_specificity_thresholds=False):
    roc = metrics.get('roc_auc_score', 0.0)
    prc = metrics.get('prc_auc_score', 0.0)
    print(f" ROC AUC: {roc:.4f}")
    print(f" PRC AUC: {prc:.4f}")
    if use_sensitivity_specificity_thresholds and 'sensitivity_specificity_thresholds' in metrics:
        print(" Sensitivity/Specificity/Threshold pairs:")
        for sens, spec, thresh in metrics['sensitivity_specificity_thresholds']:
            print(f"  Sensitivity: {sens:.4f}, Specificity: {spec:.4f}, Threshold: {thresh}")
    return roc, prc
    
if __name__ == "__main__":
    main()