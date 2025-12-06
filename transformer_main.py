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
from transformer_model import Transformer_Model_Trainer
from transformer_DatasetHandler import FeatureDatasetHandler, PrecomputedPatientFeatureDataset, build_patient_tile_map, PrecomputedPatientCombinedFeatureDataset
from collections import Counter
from MetricsPlotter import plot_roc_and_prc_curve, plot_roc_and_prc_curves_together
from focal_loss import FocalLoss


def main():
    # Initialize W&B
    wandb.init()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset configurations
    dataset_dir = 'DataBase/CRC_DX/train'
    train_ids, val_ids = FeatureDatasetHandler.get_patient_split_ids(dataset_dir)
    train_ds, val_ds = FeatureDatasetHandler.build_feature_datasets('SavedData\\Features\\CRC-DX\\Train\\CTransPath', train_ids, val_ids, max_tiles=500, seed=123)
    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    # Model initialization
    model_trainer = Transformer_Model_Trainer(device,
                                optimizer_str=wandb.config["optimizer_str"],
                                learning_rate=wandb.config["learning_rate"],
                                weight_decay=wandb.config["weight_decay"],
                                criterion=FocalLoss(wandb.config["gamma"], wandb.config["alpha"]),
                                dropout=wandb.config["dropout"]
                                )
    # Training the model with early stopping
    history = model_trainer.train(
    train_loader,
    val_loader,
    num_epochs=20,
    patience=5
    )

def train_transformer_model_k_fold(config, training_name, dataset_dir_1, features_dir_1, dataset_dir_2 = None, features_dir_2 = None, dataset_dir_3 = None, features_dir_3 = None):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Configurations
    print(f"Dataset directory: {dataset_dir_1}, {dataset_dir_2 if dataset_dir_2 is not None else ''}, {dataset_dir_3 if dataset_dir_3 is not None else ''} used for training")
    if "max_tiles" in config:
        max_tiles = config["max_tiles"]
    else:
        max_tiles = None
    # Dataset handling
    full_patient_map_1 = build_patient_tile_map(dataset_dir_1)
    dataSet_1 = PrecomputedPatientFeatureDataset(
                                            features_dir=features_dir_1,
                                            patient_ids=list(full_patient_map_1.keys()),
                                            max_tiles=max_tiles,
                                            seed=config["seed"])
    full_dataset = dataSet_1
    all_targets = [full_patient_map_1[pid]['label'] for pid in full_patient_map_1.keys()]
    if dataset_dir_2 is not None and features_dir_2 is not None:
        full_patient_map_2 = build_patient_tile_map(dataset_dir_2)
        dataSet_2 = PrecomputedPatientFeatureDataset(
                                                features_dir=features_dir_2,
                                                patient_ids=list(full_patient_map_2.keys()),
                                                max_tiles=max_tiles,
                                                seed=config["seed"])
        full_dataset = torch.utils.data.ConcatDataset([full_dataset, dataSet_2])
        all_targets += [full_patient_map_2[pid]['label'] for pid in full_patient_map_2.keys()]
    if dataset_dir_3 is not None and features_dir_3 is not None:
        full_patient_map_3 = build_patient_tile_map(dataset_dir_3)
        dataSet_3 = PrecomputedPatientFeatureDataset(
                                                features_dir=features_dir_3,
                                                patient_ids=list(full_patient_map_3.keys()),
                                                max_tiles=max_tiles,
                                                seed=config["seed"])
        full_dataset = torch.utils.data.ConcatDataset([full_dataset, dataSet_3])
        all_targets += [full_patient_map_3[pid]['label'] for pid in full_patient_map_3.keys()]
    
    # Create Stratified K-Fold splits
    skf = StratifiedKFold(n_splits=config["k_folds"], shuffle=True, random_state=config["seed"])
    fold_best_aucs = []

    print(f"Starting {config['k_folds']}-Fold Cross-Validation...")
    for fold, (train_ids, val_ids) in enumerate(skf.split(full_dataset, all_targets)):
        print(f"Fold {fold + 1}/{config['k_folds']}")
        wandb.init(project="transformer-msi-classification", name=f"{training_name}_{fold+1}", reinit=True, config=config)

        train_subset = Subset(full_dataset, train_ids)
        val_subset = Subset(full_dataset, val_ids)

        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=4)

        # Model initialization
        model_trainer = Transformer_Model_Trainer(device,
                                    optimizer_str=config["optimizer_str"],
                                    learning_rate=config["learning_rate"],
                                    weight_decay=config["weight_decay"],
                                    criterion=FocalLoss(config["gamma"], config["alpha"]),
                                    dropout=config["dropout"]
                                    )
        # Training the model with early stopping
        history = model_trainer.train(
            train_loader,
            val_loader,
            num_epochs=config["num_epochs"],
            patience=config["patience"]
        )

        # Evaluate and store best AUC for this fold
        fold_best_aucs.append(model_trainer.best_auc)
        print(f"Best AUC for fold {fold + 1}: {model_trainer.best_auc:.4f}")
        wandb.finish()

    print(f"All folds' best AUCs: {fold_best_aucs}")
    print(f"Mean best AUC: {np.mean(fold_best_aucs):.4f} ± {np.std(fold_best_aucs):.4f}")

def train_transformer_model_with_textures_k_fold(config, training_name,
                                                 dataset_dir_1, features_dir_1, textures_dir_1,
                                                 dataset_dir_2 = None, features_dir_2 = None, textures_dir_2 = None,
                                                 dataset_dir_3 = None, features_dir_3 = None, textures_dir_3 = None):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Configurations
    print(f"Dataset directory: {dataset_dir_1}, {dataset_dir_2 if dataset_dir_2 is not None else ''}, {dataset_dir_3 if dataset_dir_3 is not None else ''} used for training")
    
    # Dataset handling
    full_patient_map_1 = build_patient_tile_map(dataset_dir_1)
    dataSet_1 = PrecomputedPatientCombinedFeatureDataset(
                                            features_dir=features_dir_1,
                                            texture_features_dir=textures_dir_1,
                                            patient_ids=list(full_patient_map_1.keys()),
                                            max_tiles=config["max_tiles"],
                                            seed=config["seed"])
    full_dataset = dataSet_1
    all_targets = [full_patient_map_1[pid]['label'] for pid in full_patient_map_1.keys()]
    if dataset_dir_2 is not None and features_dir_2 is not None:
        full_patient_map_2 = build_patient_tile_map(dataset_dir_2)
        dataSet_2 = PrecomputedPatientCombinedFeatureDataset(
                                                features_dir=features_dir_2,
                                                texture_features_dir=textures_dir_2,
                                                patient_ids=list(full_patient_map_2.keys()),
                                                max_tiles=config["max_tiles"],
                                                seed=config["seed"])
        full_dataset = torch.utils.data.ConcatDataset([full_dataset, dataSet_2])
        all_targets += [full_patient_map_2[pid]['label'] for pid in full_patient_map_2.keys()]
    if dataset_dir_3 is not None and features_dir_3 is not None:
        full_patient_map_3 = build_patient_tile_map(dataset_dir_3)
        dataSet_3 = PrecomputedPatientCombinedFeatureDataset(
                                                features_dir=features_dir_3,
                                                texture_features_dir=textures_dir_3,
                                                patient_ids=list(full_patient_map_3.keys()),
                                                max_tiles=config["max_tiles"],
                                                seed=config["seed"])
        full_dataset = torch.utils.data.ConcatDataset([full_dataset, dataSet_3])
        all_targets += [full_patient_map_3[pid]['label'] for pid in full_patient_map_3.keys()]

    # Create Stratified K-Fold splits
    skf = StratifiedKFold(n_splits=config["k_folds"], shuffle=True, random_state=config["seed"])
    fold_best_aucs = []
    
    print(f"Starting {config['k_folds']}-Fold Cross-Validation...")
    for fold, (train_ids, val_ids) in enumerate(skf.split(full_dataset, all_targets)):
        print(f"Fold {fold + 1}/{config['k_folds']}")
        wandb.init(project="transformer-msi-classification", name=f"{training_name}_{fold+1}", reinit=True, config=config)

        train_subset = Subset(full_dataset, train_ids)
        val_subset = Subset(full_dataset, val_ids)

        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=4)

        # Model initialization
        model_trainer = Transformer_Model_Trainer(device,
                                    optimizer_str=config["optimizer_str"],
                                    learning_rate=config["learning_rate"],
                                    weight_decay=config["weight_decay"],
                                    criterion=FocalLoss(config["gamma"], config["alpha"]),
                                    dropout=config["dropout"],
                                    use_texture_features=True,
                                    texture_optimizer_str=config["texture_optimizer_str"],
                                    texture_lr=config["texture_lr"],
                                    texture_weight_decay=config["texture_weight_decay"]
                                    )
        # Training the model with early stopping
        history = model_trainer.train(
            train_loader,
            val_loader,
            num_epochs=config["num_epochs"],
            patience=config["patience"]
        )

        # Evaluate and store best AUC for this fold
        fold_best_aucs.append(model_trainer.best_auc)
        print(f"Best AUC for fold {fold + 1}: {model_trainer.best_auc:.4f}")
        wandb.finish()

    print(f"All folds' best AUCs: {fold_best_aucs}")
    print(f"Mean best AUC: {np.mean(fold_best_aucs):.4f} ± {np.std(fold_best_aucs):.4f}")

def evaluate_transformer_model_on_dataset(aggregator_path, texture_path, data_loader):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model initialization
    model_trainer = Transformer_Model_Trainer(device,
                                              use_texture_features=True)
    model_trainer.load(aggregator_path)
    model_trainer.load_texture(texture_path)

    # Evaluation
    model_trainer.detailed_evaluate_one_epoch(data_loader)
    patient_metrics = model_trainer.history['patient_metrics'][-1]
    texture_metrics = model_trainer.history['texture_metrics'][-1]
    combined_metrics = model_trainer.history['combined_metrics'][-1]
    return patient_metrics, texture_metrics, combined_metrics

def metrics_helper(metrics):
    roc = metrics.get('roc_auc_score', 0.0)
    prc = metrics.get('prc_auc_score', 0.0)
    print(f" ROC AUC: {roc:.4f}")
    print(f" PRC AUC: {prc:.4f}")
    print(" Sensitivity/Specificity/Threshold pairs:")
    for sens, spec, thresh in metrics['sensitivity_specificity_thresholds']:
        print(f"  Sensitivity: {sens:.4f}, Specificity: {spec:.4f}, Threshold: {thresh}")
    return roc, prc

def evalutate_models_together(aggregator_paths, texture_paths, data_loader):
    # Assign paths
    aggregator_1_path = aggregator_paths[0]
    texture_1_path = texture_paths[0]
    aggregator_2_path = aggregator_paths[1]
    texture_2_path = texture_paths[1]
    aggregator_3_path = aggregator_paths[2]
    texture_3_path = texture_paths[2]
    aggregator_4_path = aggregator_paths[3]
    texture_4_path = texture_paths[3]
    aggregator_5_path = aggregator_paths[4]
    texture_5_path = texture_paths[4]

    # Evaluate each model
    patient_metrics_1, texture_metrics_1, combined_metrics_1 = evaluate_transformer_model_on_dataset(aggregator_1_path, texture_1_path, data_loader)
    patient_metrics_2, texture_metrics_2, combined_metrics_2 = evaluate_transformer_model_on_dataset(aggregator_2_path, texture_2_path, data_loader)
    patient_metrics_3, texture_metrics_3, combined_metrics_3 = evaluate_transformer_model_on_dataset(aggregator_3_path, texture_3_path, data_loader)
    patient_metrics_4, texture_metrics_4, combined_metrics_4 = evaluate_transformer_model_on_dataset(aggregator_4_path, texture_4_path, data_loader)
    patient_metrics_5, texture_metrics_5, combined_metrics_5 = evaluate_transformer_model_on_dataset(aggregator_5_path, texture_5_path, data_loader)

    # Print results and get results
    print("--- Model 1 Results: ---")
    print("Patient-level Metrics:")
    p_roc_1, p_prc_1 = metrics_helper(patient_metrics_1)
    print("Texture-level Metrics:")
    t_roc_1, t_prc_1 = metrics_helper(texture_metrics_1)
    print("Combined-level Metrics:")
    c_roc_1, c_prc_1 = metrics_helper(combined_metrics_1)
    print("\n--- Model 2 Results: ---")
    print("Patient-level Metrics:")
    p_roc_2, p_prc_2 = metrics_helper(patient_metrics_2)
    print("Texture-level Metrics:") 
    t_roc_2, t_prc_2 = metrics_helper(texture_metrics_2)
    print("Combined-level Metrics:")
    c_roc_2, c_prc_2 = metrics_helper(combined_metrics_2)
    print("--- Model 3 Results: ---")
    print("Patient-level Metrics:")
    p_roc_3, p_prc_3 = metrics_helper(patient_metrics_3)
    print("Texture-level Metrics:")
    t_roc_3, t_prc_3 = metrics_helper(texture_metrics_3)
    print("Combined-level Metrics:")
    c_roc_3, c_prc_3 = metrics_helper(combined_metrics_3)
    print("\n--- Model 4 Results: ---")
    print("Patient-level Metrics:")
    p_roc_4, p_prc_4 = metrics_helper(patient_metrics_4)
    print("Texture-level Metrics:") 
    t_roc_4, t_prc_4 = metrics_helper(texture_metrics_4)
    print("Combined-level Metrics:")
    c_roc_4, c_prc_4 = metrics_helper(combined_metrics_4)
    print("\n--- Model 5 Results: ---")
    print("Patient-level Metrics:")
    p_roc_5, p_prc_5 = metrics_helper(patient_metrics_5)
    print("Texture-level Metrics:")
    t_roc_5, t_prc_5 = metrics_helper(texture_metrics_5)
    print("Combined-level Metrics:")
    c_roc_5, c_prc_5 = metrics_helper(combined_metrics_5)

    # calculate average metrics
    p_roc = np.mean([p_roc_1, p_roc_2, p_roc_3, p_roc_4, p_roc_5])
    p_roc_dev = np.std([p_roc_1, p_roc_2, p_roc_3, p_roc_4, p_roc_5])
    p_prc = np.mean([p_prc_1, p_prc_2, p_prc_3, p_prc_4, p_prc_5])
    p_prc_dev = np.std([p_prc_1, p_prc_2, p_prc_3, p_prc_4, p_prc_5])
    
    t_roc = np.mean([t_roc_1, t_roc_2, t_roc_3, t_roc_4, t_roc_5])
    t_roc_dev = np.std([t_roc_1, t_roc_2, t_roc_3, t_roc_4, t_roc_5])
    t_prc = np.mean([t_prc_1, t_prc_2, t_prc_3, t_prc_4, t_prc_5])
    t_prc_dev = np.std([t_prc_1, t_prc_2, t_prc_3, t_prc_4, t_prc_5])
    
    c_roc = np.mean([c_roc_1, c_roc_2, c_roc_3, c_roc_4, c_roc_5])
    c_roc_dev = np.std([c_roc_1, c_roc_2, c_roc_3, c_roc_4, c_roc_5])
    c_prc = np.mean([c_prc_1, c_prc_2, c_prc_3, c_prc_4, c_prc_5])
    c_prc_dev = np.std([c_prc_1, c_prc_2, c_prc_3, c_prc_4, c_prc_5])

    # Final results
    print("\n=== Summary of Patient-level results ===")
    print(f"Average ROC AUC: {p_roc:.4f} ± {p_roc_dev:.4f}")
    print(f"Average PRC AUC: {p_prc:.4f} ± {p_prc_dev:.4f}")
    plot_roc_and_prc_curves_together(
        [patient_metrics_1, patient_metrics_2, patient_metrics_3, patient_metrics_4, patient_metrics_5],
        title="Test",
        level="Patient"
    )
    print("=== Summary of Texture-level results ===")
    print(f"Average ROC AUC: {t_roc:.4f} ± {t_roc_dev:.4f}")
    print(f"Average PRC AUC: {t_prc:.4f} ± {t_prc_dev:.4f}")
    plot_roc_and_prc_curves_together(
        [texture_metrics_1, texture_metrics_2, texture_metrics_3, texture_metrics_4, texture_metrics_5],
        title="Test",
        level="Texture"
    )
    print("=== Summary of Combined-level results ===")
    print(f"Average ROC AUC: {c_roc:.4f} ± {c_roc_dev:.4f}")
    print(f"Average PRC AUC: {c_prc:.4f} ± {c_prc_dev:.4f}")
    plot_roc_and_prc_curves_together(
        [combined_metrics_1, combined_metrics_2, combined_metrics_3, combined_metrics_4, combined_metrics_5],
        title="Test",
        level="Combined"
    )


if __name__ == '__main__':
    main()