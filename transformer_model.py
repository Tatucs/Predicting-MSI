from sklearn import metrics
import torch
import torch.nn as nn
import timm
import time
import gc
from ctran import ctranspath
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc
import os
from MetricsPlotter import plot_roc_and_prc_curve

class Feature_Extraction_Module(nn.Module):
    """
    A feature extractor using the pre-trained CTransPath model.
    The model's classification head is removed to output feature embeddings.
    """
    def __init__(self):
        """
        Initializes the CTransPath feature extractor and freezes its weights.
        """
        super().__init__()
        self.model = ctranspath()
        self.model.head = nn.Identity()  # Remove the classification head
        td = torch.load(r'SavedData/SavedModels/ctranspath.pth')
        self.model.load_state_dict(td['model'], strict=True)

        # Always freeze the feature extractor parameters
        for param in self.model.parameters():
            param.requires_grad = False
        

    def forward(self, x):
        """
        Performs a forward pass to extract features from an input tensor.

        Args:
            x (torch.Tensor): A batch of image tiles (B, C, H, W) belonging to a patient.

        Returns:
            torch.Tensor: A batch of feature embeddings (B, embedding_dim).
        """
        x = self.model(x)
        return x
    
class Transformer_Aggregator_Module(nn.Module):
    """
    A module that aggregates tile-level features to a patient-level prediction using Transformer Encoder layers.
    """
    def __init__(self,
                 embedding_dim = 768,
                 num_heads = 4,
                 num_layers = 2,
                 dropout = 0.1):
        """
        Initializes the Transformer-based aggregator.

        Args:
            embedding_dim (int): The dimension of the input feature embeddings.
            num_heads (int): The number of attention heads in the Transformer Encoder.
            num_layers (int): The number of layers in the Transformer Encoder.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, tile_features: torch.Tensor):
        """
        Performs a forward pass to aggregate tile-level features to a patient-level prediction.

        Args:
            tile_features (torch.Tensor): Tile feature embeddings for a single patient.
                Accepts either shape (N, D) or (1, N, D), where N is the number of tiles
                and D is the embedding dimension.

        Returns:
            torch.Tensor: Patient-level prediction logits with shape (1, 1).
        """
        # Normalize input to (B=1, N, D)
        if tile_features.dim() == 2:
            # (N, D) -> (1, N, D)
            tile_features = tile_features.unsqueeze(0)
        elif tile_features.dim() == 3:
            # (B, N, D) – keep as is, but this module is intended for B == 1
            pass
        else:
            raise ValueError(f"tile_features must be 2D or 3D, got shape {tuple(tile_features.shape)}")

        B, N, _ = tile_features.shape

        # Prepare and concatenate CLS token with tile features
        cls_tokens = self.cls_token.expand(B, 1, self.embedding_dim)  # (B, 1, D)
        x = torch.cat((cls_tokens, tile_features), dim=1)  # (B, N+1, D)

        # Pass through the Transformer Encoder
        x = self.transformer_encoder(x)  # (B, N+1, D)

        # Classification using CLS token
        cls_output = x[:, 0, :]  # (B, D)
        logits = self.classifier(cls_output)  # (B, 1)

        return logits
    
class Transformer_Model(nn.Module):
    """
    A class that combines the CTransPath based feature extraction with the transformer based aggregation module.
    """
    def __init__(self, **aggregator_kwargs):
        """
        Initializes the combined model.

        Args:
            **aggregator_kwargs: Additional arguments for the Transformer_Aggregator_Module.
        """
        super().__init__()
        self.feature_extractor = Feature_Extraction_Module()
        self.aggregator = Transformer_Aggregator_Module(**aggregator_kwargs)
        self.texture_aggregator = Transformer_Aggregator_Module(embedding_dim=252)

    def forward(self, inputs: torch.Tensor, feature_batch_size: int = 32) -> torch.Tensor:
        """
        The main forward pass for a single patient. Accepts either raw tiles or precomputed features.

        Args:
            inputs (torch.Tensor):
                - Raw tiles tensor of shape (N, C, H, W) or (1, N, C, H, W), or
                - Precomputed features tensor of shape (N, D) or (1, N, D), where D == aggregator.embedding_dim.
            feature_batch_size (int): Number of tiles to process at once through the feature extractor (when inputs are tiles).

        Returns:
            torch.Tensor: Patient-level prediction logits with shape (1, 1).
        """
        # If inputs look like precomputed features (2D or 3D ending with D == embedding_dim), pass directly to aggregator
        if inputs.dim() in (2, 3) and inputs.size(-1) == getattr(self.aggregator, 'embedding_dim', inputs.size(-1)):
            return self.aggregator(inputs)

        tiles = inputs
        # Normalize input shape for tile images
        if tiles.dim() == 5:
            if tiles.size(0) != 1:
                raise ValueError(f"Expected batch_size==1 for patient data, got batch size {tiles.size(0)}")
            tiles = tiles.squeeze(0)  # (N, C, H, W)

        # Extract features in batches
        N = tiles.size(0)
        device = tiles.device
        features = []
        with torch.no_grad():
            for start in range(0, N, feature_batch_size):
                end = min(start + feature_batch_size, N)
                batch_tiles = tiles[start:end].to(device)
                batch_features = self.feature_extractor(batch_tiles)
                features.append(batch_features.cpu())
        # Concatenate all features and process through aggregator
        tile_features = torch.cat(features, dim=0).to(device)  # (N, D)
        logits = self.aggregator(tile_features)
        return logits
    
    def forward_texture_aggregator(self, texture_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the texture feature aggregator.

        Args:
            texture_features (torch.Tensor): Texture features tensor of shape (N, 252) or (1, N, 252).

        Returns:
            torch.Tensor: Patient-level prediction logits with shape (1, 1).
        """
        return self.texture_aggregator(texture_features)

    def extract_features(self, tiles: torch.Tensor, feature_batch_size: int = 32, to_cpu: bool = True) -> torch.Tensor:
        """
        Extracts CTransPath features for all tiles of a single patient without running the aggregator.

        Args:
            tiles (torch.Tensor): Tile tensor with shape (N, C, H, W) or (1, N, C, H, W).
            feature_batch_size (int): Batch size for feature extraction.
            to_cpu (bool): If True, returns features on CPU.

        Returns:
            torch.Tensor: Feature tensor with shape (N, D).
        """
        self.feature_extractor.eval()
        # Normalize input shape
        if tiles.dim() == 5:
            if tiles.size(0) != 1:
                raise ValueError(f"Expected batch_size==1 for patient data, got batch size {tiles.size(0)}")
            tiles = tiles.squeeze(0)

        N = tiles.size(0)
        device = next(self.feature_extractor.parameters()).device
        out_chunks = []
        with torch.no_grad():
            for start in range(0, N, feature_batch_size):
                end = min(start + feature_batch_size, N)
                batch_tiles = tiles[start:end].to(device)
                feats = self.feature_extractor(batch_tiles)
                out_chunks.append(feats.cpu() if to_cpu else feats)
        features = torch.cat(out_chunks, dim=0)
        if not to_cpu:
            features = features.to(device)
        return features

    def extract_and_save_features(
        self,
        dataloader,
        out_dir: str,
        feature_batch_size: int = 32,
        overwrite: bool = False,
    ) -> None:
        """
        Iterates over a patient-level DataLoader, extracts features for each patient, and saves them.

        Each batch is expected to be (tiles, label) or (tiles, label, patient_id). If patient_id is not
        provided, patients will be saved with index-based names.
        
        Args:
            dataloader: DataLoader yielding per-patient batches.
            out_dir (str): Directory to save feature files.
            feature_batch_size (int): Batch size for feature extraction.
            overwrite (bool): If False and file exists, skip extraction for that patient.
        
        Files are saved as '<out_dir>/<patient_id>.pt' containing a dict with keys:
            - 'features': FloatTensor of shape (N, D)
            - 'label': int
        """
        os.makedirs(out_dir, exist_ok=True)
        self.eval()
        self.feature_extractor.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                # Unpack batch supporting 2 or 3 items
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    tiles, label, patient_id = batch
                else:
                    tiles, label = batch
                    patient_id = f"patient_{idx:05d}"

                # Move label to CPU scalar int for saving
                if torch.is_tensor(label):
                    try:
                        label_int = int(label.item())
                    except Exception:
                        label_int = int(label.view(-1)[0].item())
                else:
                    label_int = int(label)
                    
                if isinstance(patient_id, tuple):
                    patient_id = patient_id[0]
                save_path = os.path.join(out_dir, f"{patient_id}.pt")
                if (not overwrite) and os.path.exists(save_path):
                    continue

                feats = self.extract_features(tiles, feature_batch_size=feature_batch_size, to_cpu=True)  # (N, D) on CPU
                torch.save({
                    'features': feats,  # CPU tensor
                    'label': label_int,
                }, save_path)
        
    
class Transformer_Model_Trainer:
    """
    A trainer class for the Transformer_Model.
    """
    def __init__(
            self,
            device,
            optimizer_str='adam',
            learning_rate=1e-4,
            weight_decay=1e-5,
            criterion=None,
            use_texture_features: bool = False,
            train_together: bool = False,
            texture_optimizer_str='adam',
            texture_lr=1e-4,
            texture_weight_decay=1e-5,
            texture_criterion=nn.BCEWithLogitsLoss(),
            **aggregator_kwargs,
        ):
        """
        Initialize the trainer and its components for the Transformer_Model. Constructs the combined model as self.model = Transformer_Model(**aggregator_kwargs).

        Args:
            device (torch.device or str): The target device on which the model and tensors will be placed (e.g. "cpu" or "cuda").
            optimizer_str (str, optional): Name of the optimizer to use for training. Supported values (case-insensitive):
                'adam' (default), 'adamw', 'sgd'. If an unknown name is provided, defaults to 'adam'.
            learning_rate (float, optional): Learning rate passed to the optimizer (default: 1e-4).
            weight_decay (float, optional): Weight decay (L2 regularization) passed to the optimizer (default: 1e-5).
            criterion (callable or torch.nn.Module, optional): Loss function to use. If None, nn.BCEWithLogitsLoss() is used by default.
            **aggregator_kwargs: Additional keyword arguments forwarded to Transformer_Model (and its internal Transformer_Aggregator_Module) when constructing self.model.
        """
        # Initialize device and loss criterion
        self.device = device
        if criterion is None:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = criterion
        self.texture_criterion = texture_criterion
        
        # Configure aggregator input size depending on whether we expect concatenated features
        self.use_texture_features = bool(use_texture_features)
        self.train_together = bool(train_together)

        # Initialize the combined model
        self.model = Transformer_Model(**aggregator_kwargs)

        # Set up the optimizer
        optimizers = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD}
        optimizer = optimizers.get(optimizer_str.lower(), torch.optim.Adam)
        texture_optimizer = optimizers.get(texture_optimizer_str.lower(), torch.optim.Adam)
        trainable_params = self.model.aggregator.parameters()
        if self.train_together:
            trainable_params = list(self.model.aggregator.parameters()) + list(self.model.texture_aggregator.parameters())
        self.optimizer = optimizer(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.texture_optimizer = texture_optimizer(
            self.model.texture_aggregator.parameters(),
            lr=texture_lr,
            weight_decay=texture_weight_decay
        )

        # State tracking
        # - Patient -
        self.val_loss = float('inf')
        self.best_accuracy = 0.0
        self.best_recall = 0.0
        self.best_auc = 0.0
        # - Texture -
        self.best_texture_accuracy = 0.0
        self.best_texture_recall = 0.0
        self.best_texture_auc = 0.0
        self.epochs_no_improve = 0
        # History
        self.history = {
            'train_loss': [],
            'patient_metrics': [],
            'texture_loss': [],
            'texture_metrics': [],
            'combined_loss': [],
            'combined_metrics': [],
        }

        # Move model to device
        self.model.to(self.device)

    @staticmethod
    def _set_epoch_on_loader(loader, epoch):
        """If the underlying dataset supports set_epoch, call it. Handles common wrappers like Subset."""
        if loader is None:
            return
        ds = getattr(loader, 'dataset', None)
        if ds is None:
            return
        if hasattr(ds, 'set_epoch'):
            try:
                ds.set_epoch(epoch)
                return
            except Exception:
                print(f"Warning: failed to set_epoch on dataset {type(ds)}")
        inner = getattr(ds, 'dataset', None)
        if inner is not None and hasattr(inner, 'set_epoch'):
            try:
                inner.set_epoch(epoch)
                return
            except Exception:
                print(f"Warning: failed to set_epoch on inner dataset {type(inner)}")

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
        self.model_start_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        start_time = time.time()

        for epoch in range(num_epochs):
            # Make epoch visible to datasets that implement epoch-aware sampling
            self._set_epoch_on_loader(train_loader, epoch)
            self._set_epoch_on_loader(val_loader, epoch)
            print("Successfully set epoch on loaders.")

            train_loss = self.train_one_epoch(train_loader)
            patient_metrics = self.detailed_evaluate_one_epoch(val_loader)

            self.history['train_loss'].append(train_loss)
            self.log_metrics(epoch+1, train_loss, patient_metrics)

            print(f"--- Epoch [{epoch+1}/{num_epochs}] Summary ---")
            print(f" Training Loss: {train_loss:.4f}")
            print(f" Validation Loss: {patient_metrics['val_loss']:.4f}")
            self.print_metrics(patient_metrics)
            print("\n----------------------------------------\n")

            # Texture feature aggregator evaluation, if used
            if self.use_texture_features and 'texture_metrics' in self.history:
                print(" Texture Feature Aggregator Metrics:")
                texture_metrics = self.history['texture_metrics'][-1]
                self.print_metrics(texture_metrics)
                print("\n----------------------------------------\n")
                print(" Combined Metrics: ")
                self.print_metrics(self.history['combined_metrics'][-1])
                print("\n----------------------------------------\n")
                # Track best texture AUC
                if texture_metrics['roc_auc_score'] >= self.best_texture_auc:
                    self.best_texture_auc = texture_metrics['roc_auc_score']
                    self.save_texture('best_texture_model')

            if self.early_stopping_check(patient_metrics['roc_auc_score']):
                break

            gc.collect()
            torch.cuda.empty_cache()

        end_time = time.time()
        print(f"\nTraining finished in {(end_time - start_time)/60:.2f} minutes.")
        print(f"Best Patient-Level Accuracy achieved: {self.best_accuracy:.2f}%")
        return self.history

    def train_one_epoch(self, train_loader):
        """
        Trains the model for one epoch.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
        Returns:
            epoch_loss (float): The average training loss for the epoch.
        """
        self.model.train()
        self.model.feature_extractor.eval()  # Ensure feature extractor is in eval mode

        running_loss = 0.0
        total_patients = 0
        running_texture_loss = 0.0
        texture_patients = 0

        for batch in train_loader:
            texture_features = None  # ensure defined for this iteration
            # Support three modes:
            #  - (tiles_or_features, label) or (tiles_or_features, label, patient_id)
            #  - (features, texture_features, label, patient_id) when using combined dataset
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                x, texture_features, label, _pid = batch
                x = x.to(self.device)
                texture_features = texture_features.to(self.device)
            else:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x, label, _pid = batch
                else:
                    x, label = batch
                x = x.to(self.device)

            label = label.to(self.device).view(-1, 1).float()  # (1, 1)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass for a single patient
            logits = self.model(x)  # (1, 1)
            loss = self.criterion(logits, label)

            # Backward and optimize
            loss.backward()
            self.optimizer.step()

            # Stats
            running_loss += loss.item()
            total_patients += 1

            # Forward pass with texture features if applicable
            if (self.use_texture_features and texture_features is not None):
                self.texture_optimizer.zero_grad()
                texture_logits = self.model.forward_texture_aggregator(texture_features)  # (1, 1)
                texture_loss = self.texture_criterion(texture_logits, label)
                texture_loss.backward()
                self.texture_optimizer.step()
                running_texture_loss += texture_loss.item()
                texture_patients += 1

        epoch_loss = running_loss / max(total_patients, 1)
        if self.use_texture_features and texture_patients > 0:
            epoch_texture_loss = running_texture_loss / texture_patients
            self.history['texture_loss'].append(epoch_texture_loss)
        return epoch_loss

    def detailed_evaluate_one_epoch(self, val_loader):
        """
        Performs a detailed evaluation of the model for one epoch using the provided DataLoader, calculates patient-level metrics.
        Args:
            val_loader (DataLoader): DataLoader for the validation or test dataset. Each batch yields (tiles, label).
        Returns:
            dict: Dictionary containing 'patient_metrics': dict of patient-level metrics.
        """
        self.model.eval()
        patient_true_labels = []
        patient_predicted_labels = []
        patient_scores = []
        MSI_THRESHOLD = 0.5

        texture_predicted_labels = []
        texture_scores = []
        combined_predicted_labels = []
        combined_scores = []

        val_loss = 0.0
        total_patients = 0

        with torch.no_grad():
            for batch in val_loader:
                texture_features = None  # ensure defined for this iteration
                if isinstance(batch, (list, tuple)) and len(batch) == 4:
                    x, texture_features, label, _pid = batch
                    x = x.to(self.device)
                    texture_features = texture_features.to(self.device)
                else:
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        x, label, _pid = batch
                    else:
                        x, label = batch
                    x = x.to(self.device)

                label = label.to(self.device).item()
                logits = self.model(x)
                score = torch.sigmoid(logits).item()
                patient_true_labels.append(label)
                patient_scores.append(score)
                pred = 1 if score > MSI_THRESHOLD else 0
                patient_predicted_labels.append(pred)
                
                loss = self.criterion(logits, torch.tensor([[label]], device=self.device).float())
                val_loss += loss.item()
                total_patients += 1

                if (self.use_texture_features and texture_features is not None):
                    texture_logits = self.model.forward_texture_aggregator(texture_features)
                    texture_score = torch.sigmoid(texture_logits).item()
                    texture_scores.append(texture_score)
                    texture_pred = 1 if texture_score > MSI_THRESHOLD else 0
                    texture_predicted_labels.append(texture_pred)

                    # Also consider combined prediction
                    combined_logits = 0.5 * logits.squeeze() + 0.5 * texture_logits.squeeze()
                    combined_score = torch.sigmoid(combined_logits).item()
                    combined_predicted_labels.append(1 if combined_score > MSI_THRESHOLD else 0)
                    combined_scores.append(combined_score)

        patient_metrics = self.perform_evaluation_metrics(patient_true_labels, patient_predicted_labels, patient_scores)
        patient_metrics['val_loss'] = val_loss / max(total_patients, 1)    
        self.history['patient_metrics'].append(patient_metrics)

        if self.use_texture_features and len(texture_scores) > 0:
            texture_metrics = self.perform_evaluation_metrics(
                patient_true_labels,
                texture_predicted_labels,
                texture_scores
            )
            combined_metrics = self.perform_evaluation_metrics(
                patient_true_labels,
                combined_predicted_labels,
                combined_scores
            )
            self.history['texture_metrics'].append(texture_metrics)
            self.history['combined_metrics'].append(combined_metrics)

        return patient_metrics
    
    def evaluate_on_test_set(self, test_loader):
        """
        Evaluates the model on a test dataset using the provided DataLoader, calculating patient-level metrics.

        Args:
            test_loader (DataLoader): DataLoader for the test dataset. Each batch yields (tiles, label).
        """
        print("\nStarting evaluation on test set...\n")
        patient_metrics = self.detailed_evaluate_one_epoch(test_loader)
        print("Test Set Patient-Level Metrics:")
        self.print_metrics(patient_metrics)
        plot_roc_and_prc_curve(patient_metrics, title='Test', level='Feature')
        if self.use_texture_features and 'texture_metrics' in self.history:
            print("\nTexture Feature Aggregator Metrics:")
            texture_metrics = self.history['texture_metrics'][-1]
            self.print_metrics(texture_metrics)
            plot_roc_and_prc_curve(texture_metrics, title='Test', level='Texture')
            print("\nCombined Metrics:")
            combined_metrics = self.history['combined_metrics'][-1]
            self.print_metrics(combined_metrics)
            plot_roc_and_prc_curve(combined_metrics, title='Test', level='Combined')

    def early_stopping_check(self, current_auc):
        """
        Checks if early stopping criteria are met based on patient-level validation AUC.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if current_auc > self.best_auc:
            self.best_auc = current_auc
            self.epochs_no_improve = 0
            self.save('best_model')
            return False
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping triggered. No improvement for {self.patience} epochs.")
                return True
            return False
        
    def perform_evaluation_metrics(self, true_labels, predicted_labels, scores):
        """
        Calculate evaluation metrics given true labels, predicted labels, and scores.

        Args:
            true_labels (list or np.array): True binary labels.
            predicted_labels (list or np.array): Predicted binary labels.
            scores (list or np.array): Predicted scores/probabilities.
        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        # Confusion matrix based metrics
        tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
        patient_metrics = self.calculate_classification_metrics(tp, tn, fp, fn)
        patient_metrics['tp'], patient_metrics['tn'], patient_metrics['fp'], patient_metrics['fn'] = tp, tn, fp, fn

        # AUC based metrics
        if len(np.unique(true_labels)) > 1:
            patient_metrics['roc_auc_score'] = roc_auc_score(true_labels, scores)
            patient_metrics['prc_auc_score'] = average_precision_score(true_labels, scores)
            patient_metrics['fpr_curve'], patient_metrics['tpr_curve'], fpr_tpr_thresholds = roc_curve(true_labels, scores)
            patient_metrics['precision_curve'], patient_metrics['recall_curve'], pr_thresholds = precision_recall_curve(true_labels, scores)
        else:
            patient_metrics['roc_auc_score'], patient_metrics['prc_auc_score'] = 0.0, 0.0
            patient_metrics['fpr_curve'], patient_metrics['tpr_curve'], fpr_tpr_thresholds = [], [], []
            patient_metrics['precision_curve'], patient_metrics['recall_curve'], pr_thresholds = [], [], []

        # Best F1 Score and Sensitivity/Specificity pairs
        if len(patient_metrics['precision_curve']) > 0 and len(patient_metrics['recall_curve']) > 0:
            precision_arr = np.array(patient_metrics['precision_curve'])
            recall_arr = np.array(patient_metrics['recall_curve'])
            f1_scores = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-8)
            best_f1_index = int(np.argmax(f1_scores))
            best_threshold = pr_thresholds[best_f1_index] if len(pr_thresholds) > best_f1_index else None
            # Store best F1 and corresponding threshold
            patient_metrics['best_f1_score'] = float(f1_scores[best_f1_index])
            patient_metrics['best_f1_threshold'] = best_threshold
            # Get sensitivity/specificity pairs and store them
            spec_at_sens = self.get_specificity_at_sensitivity(
                patient_metrics['fpr_curve'],
                patient_metrics['tpr_curve'],
                fpr_tpr_thresholds
            )
            patient_metrics['sensitivity_specificity_thresholds'] = spec_at_sens

        return patient_metrics

    def calculate_classification_metrics(self, tp, tn, fp, fn):
        """
        Calculate common classification metrics from confusion matrix components.

        Args:
            tp (int): Number of true positives.
            tn (int): Number of true negatives.
            fp (int): Number of false positives.
            fn (int): Number of false negatives.

        Returns:
            dict: A dictionary containing the following keys and values:
                - "accuracy" (float): Overall accuracy expressed as a percentage (0.0 - 100.0).
                - "precision" (float): Precision / positive predictive value (0.0 - 1.0).
                - "recall" (float): Recall / sensitivity / true positive rate (0.0 - 1.0).
                - "specificity" (float): True negative rate (0.0 - 1.0).
                - "f1_score" (float): Harmonic mean of precision and recall (0.0 - 1.0).
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        return {
            "accuracy": accuracy * 100.0,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1,
        }
    
    def get_specificity_at_sensitivity(self, fpr_curve, tpr_curve, fpr_tpr_thresholds):
        """
        Returns up to three (sensitivity, specificity, threshold) tuples from ROC data.

        Args:
            fpr_curve, tpr_curve, fpr_tpr_thresholds: sequences for the ROC curve.

        Returns:
            list of (sensitivity, specificity, threshold) tuples for the last up-to-three points.
        """
        sensitivity_array = np.array(tpr_curve)
        specificity_array = 1 - np.array(fpr_curve)
        if sensitivity_array.size == 0:
            return []
        last_idxs = np.arange(len(sensitivity_array))[-10:]
        results = []
        for idx in last_idxs:
            sensitivity = float(sensitivity_array[idx])
            specificity = float(specificity_array[idx])
            threshold = fpr_tpr_thresholds[idx] if len(fpr_tpr_thresholds) > idx else None
            results.append((sensitivity, specificity, threshold))
        return results

    def print_metrics(self, metrics):
        """
        Prints the classification metrics in a formatted manner.

        Args:
            metrics (dict): Dictionary containing classification metrics.
        """
        print("\nPatient-level Metrics:")
        print(f" Accuracy: {metrics['accuracy']:.2f}%")
        print(f" Precision: {metrics['precision']:.4f}")
        print(f" Recall: {metrics['recall']:.4f}")
        print(f" Specificity: {metrics['specificity']:.4f}")
        print(f" F1-Score: {metrics['f1_score']:.4f}")
        print(f" ROC AUC: {metrics.get('roc_auc_score', 0.0):.4f}")
        print(f" PRC AUC: {metrics.get('prc_auc_score', 0.0):.4f}")
        print(f" TP: {metrics['tp']}, TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}\n")
        print(f" Best F1 Score: {metrics.get('best_f1_score', 0.0):.4f} at Threshold: {metrics.get('best_f1_threshold', None)}")
        if 'sensitivity_specificity_thresholds' in metrics:
            print(" Sensitivity/Specificity/Threshold pairs:")
            for sens, spec, thresh in metrics['sensitivity_specificity_thresholds']:
                print(f"  Sensitivity: {sens:.4f}, Specificity: {spec:.4f}, Threshold: {thresh}")
        

    def log_metrics(self, epoch, train_loss, patient_metrics):
        """
        Logs the classification metrics to Weights & Biases (wandb), if available.

        Args:
            epoch (int): Current epoch number.
            train_loss (float): Training loss for the epoch.
            patient_metrics (dict): Dictionary containing patient-level metrics.
        """
        try:
            import wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
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
        except ImportError:
            pass

    def save(self, name):
        """
        Saves the current aggregator state to a file.
            Args:
                name (str): The name to save the model as.
        """
        path = f"{self.model_start_time}_{name}_aggregator.pth"
        print(f"Saving aggregator weights to {path} with AUC: {self.best_auc:.2f}%")
        torch.save(self.model.aggregator.state_dict(), path)

    def save_texture(self, name):
        """
        Saves the current texture aggregator state to a file (if present).
        Args:
            name (str): The name to save the texture model as.
        """
        path = f"{self.model_start_time}_{name}_texture_aggregator.pth"
        print(f"Saving texture aggregator weights to {path} with accuracy: {self.best_texture_accuracy:.2f}%")
        torch.save(self.model.texture_aggregator.state_dict(), path)

    def load(self, path):
        """
        Loads the aggregator state from a file.
        """
        state_dict = torch.load(path, weights_only=True)
        self.model.aggregator.load_state_dict(state_dict)
        self.model.to(self.device)
        print(f"Aggregator weights loaded from {path}")

    def load_texture(self, path):
        """
        Loads the texture aggregator state from a file.
        """
        state_dict = torch.load(path, weights_only=True)
        self.model.texture_aggregator.load_state_dict(state_dict)
        self.model.to(self.device)
        print(f"Texture aggregator weights loaded from {path}")


