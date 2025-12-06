import gc
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
import wandb

class CNN_Model_Trainer:
    def __init__(
            self,
            device,
            layers_to_unfreeze=0,
            optimizer_str='adam',
            dropout_rate=0.0,
            learning_rate=1e-4,
            weight_decay=1e-5,
            criterion=None
        ):
        """
        Initializes a CNN model (ResNet50) for binary classification of MSI.
        Sets up the optimizer of the model.
        Furthermore, it unfreezes the last `param_number_to_unfreeze` parameters
        for the model to be traineable.

        Args:
            device (torch.device): Device to run the model on (CPU or GPU).
            layers_to_unfreeze (int): Number of last layers to unfreeze for training.
            dropout_rate (float): Dropout rate for the final layer.
            optimizer (torch.optim.Optimizer, optional): Custom optimizer. If None, Adam optimizer is used.
            criterion (torch.nn.Module): Loss function to use.
        """
        # Set device and criterion
        self.device = device
        if criterion is None:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = criterion
        
        # Initialize the model
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, 1)
        )

        # The final layer is always trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True
        # Unfreeze the last `layers_to_unfreeze` layers
        if layers_to_unfreeze > 0:
            layer_blocks = [self.model.layer4, self.model.layer3, self.model.layer2, self.model.layer1]
            for i in range(min(layers_to_unfreeze, len(layer_blocks))):
                for param in layer_blocks[i].parameters():
                    param.requires_grad = True
        '''
        params_to_unfreeze = list(self.model.parameters())[-param_number_to_unfreeze:]
        for param in params_to_unfreeze:
            param.requires_grad = True
        '''

        # Set up the optimizer
        optimizers = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "sgd": optim.SGD}
        optimizer = optimizers[optimizer_str.lower()]
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optimizer(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # State tracking
        self.best_accuracy = 0.0
        self.epochs_no_improve = 0
        self.history = {
            'train_loss': [],
            'tile_metrics': [],
            'patient_metrics': []
        }
        self.model.to(self.device)

    def set_feature_extractor(self):
        """
        Modify the underlying ResNet so it returns features instead of logits.

        This replaces the classification head (Dropout+Linear) with nn.Identity().
        The original head is preserved and can be restored via restore_classifier().
        After calling this, forward(x) will return (B, D) features, where D=2048 for ResNet50.
        """
        # Preserve original head only once
        if not hasattr(self, "_original_fc") or self._original_fc is None:
            self._original_fc = self.model.fc
            # Best-effort inference of feature dimension from original head
            try:
                if isinstance(self._original_fc, nn.Sequential) and isinstance(self._original_fc[1], nn.Linear):
                    self.feature_dim = int(self._original_fc[1].in_features)
                elif hasattr(self.model, "fc") and hasattr(self.model.fc, "in_features"):
                    self.feature_dim = int(self.model.fc.in_features)  # type: ignore[attr-defined]
                else:
                    self.feature_dim = 0
            except Exception:
                self.feature_dim = 0
        # Swap to identity head for feature outputs
        self.model.fc = nn.Identity()
        # Ensure no training side-effects during extraction
        self.model.eval()

    def restore_classifier(self):
        """
        Restore the original classification head if it was previously replaced.
        """
        if hasattr(self, "_original_fc") and self._original_fc is not None:
            self.model.fc = self._original_fc

    @torch.no_grad()
    def extract_features(self, tiles: torch.Tensor, feature_batch_size: int = 32, to_cpu: bool = True) -> torch.Tensor:
        """
        Extract per-tile CNN features using the fine-tuned ResNet backbone.

        Accepts raw tiles and runs them through the network with the classifier removed.

        Args:
            tiles (torch.Tensor): Tile tensor with shape (N, C, H, W) or (1, N, C, H, W).
            feature_batch_size (int): Mini-batch size used during forward passes to control memory.
            to_cpu (bool): If True, returns features on CPU memory.

        Returns:
            torch.Tensor: Feature tensor with shape (N, D), where D≈2048 for ResNet50.
        """
        # Ensure feature extractor mode is enabled
        if not isinstance(self.model.fc, nn.Identity):
            self.set_feature_extractor()

        # Normalize input shape to (N, C, H, W)
        if tiles.dim() == 5:
            # (1, N, C, H, W)
            if tiles.size(0) != 1:
                raise ValueError(f"Expected tiles shape (1, N, C, H, W) for 5D input, got {tuple(tiles.shape)}")
            tiles = tiles.squeeze(0)
        elif tiles.dim() != 4:
            raise ValueError(f"Expected tiles shape (N, C, H, W) or (1, N, C, H, W), got {tuple(tiles.shape)}")

        N = tiles.size(0)
        device = next(self.model.parameters()).device

        self.model.eval()

        features = []
        for start in range(0, N, feature_batch_size):
            end = min(start + feature_batch_size, N)
            batch = tiles[start:end].to(device, non_blocking=True)
            out = self.model(batch)  # (b, D)
            # Ensure (b, D) shape (ResNet returns flattened vec after avgpool when fc=Identity)
            if out.dim() > 2:
                out = torch.flatten(out, 1)
            features.append(out.detach())

        feats = torch.cat(features, dim=0)
        if to_cpu:
            feats = feats.cpu()
        return feats

    @torch.no_grad()
    def extract_and_save_features(
        self,
        dataloader,
        out_dir: str,
        feature_batch_size: int = 32,
        overwrite: bool = False,
    ) -> None:
        """
        Iterate over a patient-level DataLoader, extract CNN features for all tiles, and save them.

        Each batch is expected to be (tiles, label) or (tiles, label, patient_id). If patient_id is not
        provided, patients will be saved with index-based names.

        Files are saved as '<out_dir>/<patient_id>.pt' containing a dict with keys:
            - 'features': FloatTensor of shape (N, D)
            - 'label': int
        """
        os.makedirs(out_dir, exist_ok=True)
        self.set_feature_extractor()
        self.model.eval()

        device = next(self.model.parameters()).device

        for i, batch in enumerate(dataloader):
            # Unpack batch supporting multiple dataset formats
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                tiles, label, patient_id = batch
            elif len(batch) == 2:
                tiles, label = batch
                patient_id = f"patient_{i:05d}"
            else:
                raise ValueError("Expected batch of (tiles, label) or (tiles, label, patient_id)")

            # Normalize patient_id to a clean string
            if isinstance(patient_id, (list, tuple)):
                patient_id = patient_id[0]
            
            if torch.is_tensor(label):
                    try:
                        label_int = int(label.item())
                    except Exception:
                        label_int = int(label.view(-1)[0].item())
            else:
                label_int = int(label)

            out_path = os.path.join(out_dir, f"{patient_id}.pt")
            if (not overwrite) and os.path.exists(out_path):
                continue

            # Extract features in manageable batches
            feats = self.extract_features(tiles, feature_batch_size=feature_batch_size, to_cpu=True)
            torch.save({
                'features': feats.float(),
                'label': label_int,
            }, out_path)

    def train(self, train_loader, val_loader, num_epochs, patience=5):
        """
        Trains the model for a specified number of epochs, evaluating on the validation set after each epoch.

        Args:
            train_loader (DataLoader): DataLoader for the training data.
            val_loader (DataLoader): DataLoader for the validation data.
            num_epochs (int): Number of epochs to train the model.
            patience (int): Number of epochs to wait for improvement before early stopping is triggered.
        Returns:
            history (dict): Dictionary containing training and evaluation metrics history.
        """
        print("\nStarting training...\n")
        self.patience = patience
        start_time = time.time()
        self.model_start_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())

        for epoch in range(num_epochs):
            # Train and evaluate
            train_loss = self.train_one_epoch(train_loader)
            eval_metrics = self.detailed_evaluate_one_epoch(val_loader)
            tile_metrics = eval_metrics['tile_metrics']
            patient_metrics = eval_metrics['patient_metrics']
            
            # Store history & log to wandb
            self.history['train_loss'].append(train_loss)
            self.history['tile_metrics'].append(tile_metrics)
            self.history['patient_metrics'].append(patient_metrics)
            self.log_metrics(epoch+1, train_loss, tile_metrics, patient_metrics)

            # Print Epoch Summary
            print(f"--- Epoch [{epoch+1}/{num_epochs}] Summary ---")
            print(f" Training Loss: {train_loss:.4f}%")
            self.print_metrics(tile_metrics, level="Tile")
            self.print_metrics(patient_metrics, level="Patient")
            print("\n----------------------------------------\n")

            # Early Stopping Check
            if self.early_stopping_check(patient_metrics['roc_auc_score']):
                break

            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()
        
        end_time = time.time()
        print(f"\nTraining finished in {(end_time - start_time)/60:.2f} minutes.")
        print(f"Best Patient-Level Accuracy achieved: {self.best_accuracy:.2f}%")
        return self.history

    def train_one_epoch(self, train_loader):
        """
        Trains the model for one epoch.

        Training steps:
            1. Set the model to training mode.
            2. Iterate over the training data.
            3. For each batch:

                a. Move data to the specified device.
                b. Zero the gradients.
                c. Perform a forward pass.
                d. Compute the loss.
                e. Perform a backward pass.
                f. Update the model parameters.
        Args:
            train_loader (DataLoader): DataLoader for the training data.
        Returns:
            epoch_loss (float): Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        for images, labels, filenames in train_loader:
            images, labels = images.to(self.device), labels.to(self.device).float()
            self.optimizer.zero_grad()
            outputs = self.model(images).squeeze(1)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def evaluate_one_epoch(self, val_loader):
        """
        Evaluates the model on the provided test dataset.

        Args:
            val_loader (DataLoader): DataLoader for the test/validation data.

        Returns:
            accuracy (float): Classification accuracy (%) on the test dataset.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, filenames in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy
    
    def detailed_evaluate_one_epoch(self, val_loader):
        """
        Performs a detailed evaluation of the model for one epoch using the provided DataLoader, calculates both tile-level and patient-level metrics.
        Args:
            val_loader (DataLoader): DataLoader for the validation or test dataset. Each batch should yield (images, labels, filenames).
        Returns:
            dict: Dictionary containing:
                - 'tile_metrics': dict of tile-level metrics (accuracy, precision, recall, F1, ROC AUC, PRC AUC, TP, TN, FP, FN).
                - 'patient_metrics': dict of patient-level metrics (accuracy, precision, recall, F1, ROC AUC, PRC AUC, TP, TN, FP, FN).
        """
        self.model.eval()
        patient_data = {}

        # Initialize counters for tile-level metrics
        tile_true_labels = []
        tile_predicted_labels = []
        tile_scores = []
        TILE_THRESHOLD = 0.5  # Default threshold for binary classification

        # Evaluation loop
        with torch.no_grad():
            for images, labels, filenames in val_loader:
                images, labels = images.to(self.device), labels.to(self.device).float()
                outputs = self.model(images)
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > TILE_THRESHOLD).long()

                tile_true_labels.extend(labels.cpu().numpy())
                tile_predicted_labels.extend(predicted.cpu().numpy())
                tile_scores.extend(probabilities.cpu().numpy())

                for i in range(len(filenames)):
                    patient_id = '-'.join(filenames[i].split('-')[2:5])
                    true_label = labels[i].item()
                    predicted_label = predicted[i].item()
                    score_for_auc = probabilities[i].item()

                    if patient_id not in patient_data:
                        patient_data[patient_id] = {
                            'true_label': true_label,
                            'predicted_labels': [],
                            'scores_for_auc': []
                        }
                    patient_data[patient_id]['predicted_labels'].append(predicted_label)
                    patient_data[patient_id]['scores_for_auc'].append(score_for_auc)

        # --- Tile-level metrics calculation ---
        # Confusion matrix based metrics
        tn, fp, fn, tp = confusion_matrix(tile_true_labels, tile_predicted_labels).ravel()
        tile_metrics = self.calculate_classification_metrics(tp, tn, fp, fn)
        tile_metrics['tp'], tile_metrics['tn'], tile_metrics['fp'], tile_metrics['fn'] = tp, tn, fp, fn

        # AUC based metrics
        if len(np.unique(tile_true_labels)) > 1:
            tile_metrics['roc_auc_score'] = roc_auc_score(tile_true_labels, tile_scores)
            tile_metrics['prc_auc_score'] = average_precision_score(tile_true_labels, tile_scores)
            tile_metrics['fpr_curve'], tile_metrics['tpr_curve'], _ = roc_curve(tile_true_labels, tile_scores)
            tile_metrics['precision_curve'], tile_metrics['recall_curve'], _ = precision_recall_curve(tile_true_labels, tile_scores)            
        else:
            tile_metrics['roc_auc'], tile_metrics['prc_auc'] = 0.0, 0.0

        # --- Patient-level metrics calculation ---
        patient_true_labels = []
        patient_predictions = []
        patient_aggregated_scores = []

        for _, data in patient_data.items():
            true_label = data['true_label']

            # Aggregate scores using mean
            aggregated_score = np.mean(data['scores_for_auc'])
            patient_true_labels.append(true_label)
            patient_aggregated_scores.append(aggregated_score)

            # Final prediction based on mean score thresholding at 0.5
            MSI_THRESHOLD = 0.5
            final_prediction = 1 if aggregated_score > MSI_THRESHOLD else 0
            patient_predictions.append(final_prediction)

        # Confusion matrix based metrics
        ptn, pfp, pfn, ptp = confusion_matrix(patient_true_labels, patient_predictions).ravel()
        patient_metrics = self.calculate_classification_metrics(ptp, ptn, pfp, pfn)
        patient_metrics['tp'], patient_metrics['tn'], patient_metrics['fp'], patient_metrics['fn'] = ptp, ptn, pfp, pfn

        # AUC based metrics
        if len(np.unique(patient_true_labels)) > 1:
            patient_metrics['roc_auc_score'] = roc_auc_score(patient_true_labels, patient_aggregated_scores)
            patient_metrics['prc_auc_score'] = average_precision_score(patient_true_labels, patient_aggregated_scores)
            patient_metrics['fpr_curve'], patient_metrics['tpr_curve'], fpr_tpr_thresholds = roc_curve(patient_true_labels, patient_aggregated_scores)
            patient_metrics['precision_curve'], patient_metrics['recall_curve'], pr_thresholds = precision_recall_curve(patient_true_labels, patient_aggregated_scores)
        else:
            patient_metrics['roc_auc'], patient_metrics['prc_auc'] = 0.0, 0.0

        f1_scores = 2 * (patient_metrics['precision_curve'] * patient_metrics['recall_curve']) / (patient_metrics['precision_curve'] + patient_metrics['recall_curve'])
        best_f1_index = np.argmax(f1_scores)
        best_threshold = pr_thresholds[best_f1_index]

        print(f"Best Patient-Level F1 Score: {f1_scores[best_f1_index]:.4f} at threshold: {best_threshold:.4f}")

        spec_at_sens = self.get_specificity_at_sensitivity(
            patient_metrics['fpr_curve'],
            patient_metrics['tpr_curve'],
            fpr_tpr_thresholds
        )
        patient_metrics['sensitivity_specificity_thresholds'] = spec_at_sens

        return {
            'tile_metrics': tile_metrics,
            'patient_metrics': patient_metrics,
        }
    
    def early_stopping_check(self, current_accuracy):
        """
        Checks if early stopping criteria are met based on patient-level accuracy.

        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
        Returns:
            bool: True if training should stop, False otherwise.
        """
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.epochs_no_improve = 0
            self.save('best_model')
            return False
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping triggered after {self.patience} epochs with no improvement.")
                return True
            return False

    def get_specificity_at_sensitivity(self, fpr_curve, tpr_curve, fpr_tpr_thresholds):
        """
        Returns the last 3 sensitivity values from the sensitivity array,
        along with their corresponding specificity and threshold values from the ROC curve data.

        Args:
            fpr_curve (array-like): False positive rates from the ROC curve.
            tpr_curve (array-like): True positive rates from the ROC curve.
            fpr_tpr_thresholds (array-like): Thresholds from the ROC curve.
            sensitivity_level (float): Desired sensitivity level (between 0 and 1).
        """
        sensitivity_array = np.array(tpr_curve)
        specificity_array = 1 - np.array(fpr_curve)

        # Get the last 10 indices in the sensitivity array
        last_idxs = np.arange(len(sensitivity_array))[-10:]

        results = []
        for idx in last_idxs:
            sensitivity = sensitivity_array[idx]
            specificity = specificity_array[idx]
            threshold = fpr_tpr_thresholds[idx]
            results.append((sensitivity, specificity, threshold))
            print(f"Sensitivity at index {idx}: {sensitivity:.4f}, Specificity: {specificity:.4f}, Threshold: {threshold:.4f}")
        return results


    def calculate_classification_metrics(self, tp, tn, fp, fn):
        """
        Calculates classification metrics (accuracy, precision, recall, F1-score)
        from confusion matrix components.

        Args:
            tp (int): True Positives.
            tn (int): True Negatives.
            fp (int): False Positives.
            fn (int): False Negatives.

        Returns:
            dict: Dictionary containing accuracy (%), precision, recall, and F1-score.
        """
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # Recall/Sensitivity
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        # F1-Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        return {
            "accuracy": accuracy * 100,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1
        }

    def print_metrics(self, metrics, level):
        """
        Prints the classification metrics in a formatted manner.

        Args:
            metrics (dict): Dictionary containing classification metrics.
            level (str): Level of metrics ('Tile' or 'Patient').
        """
        print(f"\n{level}-level Metrics:")
        print(f" Accuracy: {metrics['accuracy']:.2f}%")
        print(f" Precision: {metrics['precision']:.4f}")
        print(f" Recall: {metrics['recall']:.4f}")
        print(f" Specificity: {metrics['specificity']:.4f}")
        print(f" F1-Score: {metrics['f1_score']:.4f}")
        print(f" ROC AUC: {metrics.get('roc_auc_score', 0.0):.4f}")
        print(f" PRC AUC: {metrics.get('prc_auc_score', 0.0):.4f}")
        print(f" TP: {metrics['tp']}, TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}\n")

    def log_metrics(self, epoch, train_loss, tile_metrics, patient_metrics):
        """
        Logs the classification metrics to Weights & Biases (wandb).

        Args:
            epoch (int): Current epoch number.
            train_loss (float): Training loss for the epoch.
            tile_metrics (dict): Dictionary containing tile-level metrics.
            patient_metrics (dict): Dictionary containing patient-level metrics.
        """
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'tile_accuracy': tile_metrics['accuracy'],
            'tile_precision': tile_metrics['precision'],
            'tile_recall': tile_metrics['recall'],
            'tile_specificity': tile_metrics['specificity'],
            'tile_f1_score': tile_metrics['f1_score'],
            'tile_roc_auc': tile_metrics.get('roc_auc_score', 0.0),
            'tile_prc_auc': tile_metrics.get('prc_auc_score', 0.0),
            'tile_TP': tile_metrics['tp'],
            'tile_TN': tile_metrics['tn'],
            'tile_FP': tile_metrics['fp'],
            'tile_FN': tile_metrics['fn'],
            'patient_accuracy': patient_metrics['accuracy'],
            'patient_precision': patient_metrics['precision'],
            'patient_recall': patient_metrics['recall'],
            'patient_specificity': patient_metrics['specificity'],
            'patient_f1_score': patient_metrics['f1_score'],
            'patient_roc_auc': patient_metrics.get('roc_auc_score', 0.0),
            'patient_prc_auc': patient_metrics.get('prc_auc_score', 0.0),
            'patient_TP': patient_metrics['tp'],
            'patient_TN': patient_metrics['tn'],
            'patient_FP': patient_metrics['fp'],
            'patient_FN': patient_metrics['fn'],
        })

    def save(self, name):
        """
        Saves the model's state dictionary to the specified file path.

        Args:
            path (str): File path to save the model weights.
        """
        path = f"{self.model_start_time}_{name}.pth"
        print(f"Saving best model to {path} with accuracy: {self.best_accuracy:.2f}%")
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """
        Loads the model's state dictionary from the specified file path.

        Args:
            path (str): File path from which to load the model weights.
        """
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.to(self.device)
        print(f"Model loaded from {path}")
