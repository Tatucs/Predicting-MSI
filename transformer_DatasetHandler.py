import os
import torch
import numpy
from collections import defaultdict
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, Dataset
from glob import glob
from PIL import Image
from sklearn.model_selection import StratifiedKFold



def build_patient_tile_map(root_dir: str) -> dict:
    """
    Build a mapping patient_id -> {'paths': [...], 'label': int}.

    Scans the two label subfolders '1_MSI' and '0_MSS' under root_dir for PNG files.
    Patient ID is extracted from each filename by joining the filename parts
    at indices 2..4 (i.e. "-".join(filename.split("-")[2:5])). Image paths are
    grouped per patient and the patient is assigned label 1 for files in
    '1_MSI' and 0 for files in '0_MSS'.

    Args:
        root_dir (str): Path to the dataset root containing '1_MSI' and '0_MSS' subfolders.

    Returns:
        dict: Mapping patient_id (str) -> {'paths': list[str], 'label': int}.
    """
    patient_to_tiles = defaultdict(lambda: {'paths': [], 'label': None})
    label_map = {'1_MSI': 1, '0_MSS': 0}

    for label_name, label_value in label_map.items():
        folder = os.path.join(root_dir, label_name)
        for img_path in glob(os.path.join(folder, "*.png")):
            filename = os.path.basename(img_path)
            # Extract patient ID (e.g. blk-TGINPMKGNHYS-TCGA-CK-5912-01Z-00-DX1)
            patient_id = "-".join(filename.split("-")[2:5])

            patient_to_tiles[patient_id]['paths'].append(img_path)
            patient_to_tiles[patient_id]['label'] = label_value

    return dict(patient_to_tiles)

def get_max_tiles(dataset_dir):
        """
        Calculates the maximum number of tiles across all patients in the dataset.
        
        Args:
            dataset_dir (str): Path to the dataset directory.
            
        Returns:
            int: Maximum number of tiles for any single patient.
        """
        full_patient_map = build_patient_tile_map(dataset_dir)
        max_tiles = max(len(data['paths']) for data in full_patient_map.values())
        return max_tiles

def get_patient_num(dataset_dir):
        """
        Calculates the total number of unique patients in the dataset.
        
        Args:
            dataset_dir (str): Path to the dataset directory.
            
        Returns:
            int: Total number of unique patients.
        """
        full_patient_map = build_patient_tile_map(dataset_dir)
        num_patients = len(full_patient_map)
        return num_patients


class PatientDataset(Dataset):
    def __init__(self, patient_to_tiles, transform=None):
        self.patient_ids = list(patient_to_tiles.keys())
        self.patient_to_tiles = patient_to_tiles
        self.transform = transform

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        data = self.patient_to_tiles[patient_id]

        images = []
        for path in data['paths']:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # Stack all tiles into a single tensor: (num_tiles, C, H, W)
        images = torch.stack(images)
        label = torch.tensor(data['label'], dtype=torch.long)
        # Also return patient_id so downstream feature extraction can name files deterministically
        return images, label, patient_id

class PrecomputedPatientFeatureDataset(Dataset):
    """
    Loads precomputed per-patient feature tensors saved as '<features_dir>/<patient_id>.pt'.
    Each file must be a torch-saved dict with keys:
      - 'features': FloatTensor (N, D)
      - 'label': int
    """
    def __init__(self, features_dir: str, patient_ids: list[str], max_tiles: int = None, seed: int = None):
        """
        Args:
            features_dir: directory with per-patient .pt files (each a dict with 'features' and 'label')
            patient_ids: list of patient ids to expose
            max_tiles: if provided, randomly sample up to this many tile-features per patient on each __getitem__ call
            seed: optional integer seed to make sampling reproducible in combination with per-epoch control
        """
        self.features_dir = features_dir
        self.patient_ids = list(patient_ids)
        self.max_tiles = max_tiles
        self.seed = seed
        # current_epoch can be set from training loop to vary sampling per epoch deterministically
        self.current_epoch = 0

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        path = os.path.join(self.features_dir, f"{patient_id}.pt")
        data = torch.load(path, map_location='cpu',weights_only=True)
        feats = data['features']  # (N, D)
        label = torch.tensor(int(data['label']), dtype=torch.long)
        # If requested, randomly sample up to max_tiles features for this patient.
        # Sampling is without replacement. If seed is provided, sampling will be deterministic
        # across epochs by combining seed and current_epoch.
        if self.max_tiles is not None and feats.size(0) > self.max_tiles:
            # Build RNG with optional seed + current_epoch for reproducible epoch-wise sampling
            if self.seed is not None:
                rng = numpy.random.default_rng(self.seed + int(self.current_epoch))
            else:
                rng = numpy.random.default_rng()
            indices = rng.choice(feats.size(0), size=self.max_tiles, replace=False)
            # Convert to torch index
            indices = torch.as_tensor(indices, dtype=torch.long)
            feats = feats[indices]

        return feats, label, patient_id

    def set_epoch(self, epoch: int):
        """Set the current epoch (used to vary sampling each epoch when seed is set)."""
        self.current_epoch = int(epoch)

    def set_seed(self, seed: int):
        """Set base seed for deterministic sampling across runs."""
        self.seed = int(seed)

class PrecomputedPatientTextureFeatureDataset(Dataset):
    """
    Loads precomputed per-patient texture feature tensors saved as `<features_dir>/<patient_id>.pt`.
    Each file must be a torch-saved dict with keys:
      - 'texture_features': FloatTensor (N, 252)
      - 'label': int

    Supports optional max_tiles sampling per __getitem__ and deterministic epoch-wise control.
    """

    def __init__(
        self,
        features_dir: str,
        patient_ids: list[str],
        max_tiles: int = None,
        seed: int = None,
    ):
        self.features_dir = features_dir
        self.patient_ids = list(patient_ids)
        self.max_tiles = max_tiles
        self.seed = seed
        self.current_epoch = 0

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int):
        patient_id = self.patient_ids[idx]
        # Normalize patient_id to string if wrapped in list/tuple
        if isinstance(patient_id, (list, tuple)):
            if len(patient_id) == 0:
                raise ValueError("Empty patient_id container encountered")
            patient_id = patient_id[0]
        path = os.path.join(self.features_dir, f"{patient_id}.pt")
        data = torch.load(path, map_location='cpu')
        feats: torch.Tensor = data['texture_features']  # (N, 252)
        label = torch.tensor(int(data['label']), dtype=torch.long)

        if self.max_tiles is not None and feats.size(0) > self.max_tiles:
            # Deterministic sampling if seed provided
            if self.seed is not None:
                rng = numpy.random.default_rng(self.seed + int(self.current_epoch))
            else:
                rng = numpy.random.default_rng()
            indices = rng.choice(feats.size(0), size=self.max_tiles, replace=False)
            feats = feats[torch.as_tensor(indices, dtype=torch.long)]

        return feats, label, patient_id

    # Epoch/seed controls to align with training loops
    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def set_seed(self, seed: int):
        self.seed = int(seed)


class PrecomputedPatientCombinedFeatureDataset(Dataset):
    """
    Loads combined per-patient CTransPath features and texture features from two directories.

    For each patient_id, it loads:
      - '<features_dir>/<patient_id>.pt' with keys: 'features' (N, Df), 'label' (int)
      - '<texture_features_dir>/<patient_id>.pt' with keys: 'texture_features' (M, Dt), 'label' (int)

    It aligns tiles by index (assumes both extractions iterated tiles in the same order),
    and trims to the minimum of N and M to guarantee paired rows refer to the same tile.

    Optional max_tiles sampling is applied using the same indices to BOTH modalities so pairing is preserved.
    """

    def __init__(
        self,
        features_dir: str,
        texture_features_dir: str,
        patient_ids: list[str],
        max_tiles: int = None,
        seed: int = None,
    ):
        self.features_dir = features_dir
        self.texture_features_dir = texture_features_dir
        self.patient_ids = list(patient_ids)
        self.max_tiles = max_tiles
        self.seed = seed
        self.current_epoch = 0

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int):
        patient_id = self.patient_ids[idx]
        # Normalize patient_id to string if wrapped in list/tuple
        if isinstance(patient_id, (list, tuple)):
            if len(patient_id) == 0:
                raise ValueError("Empty patient_id container encountered")
            patient_id = patient_id[0]

        # Load CTransPath features
        fpath = os.path.join(self.features_dir, f"{patient_id}.pt")
        fdata = torch.load(fpath, map_location='cpu')
        feats: torch.Tensor = fdata['features']  # (N, Df)
        label_f = int(fdata['label'])

        # Load texture features
        tpath = os.path.join(self.texture_features_dir, f"{patient_id}.pt")
        tdata = torch.load(tpath, map_location='cpu')
        tex: torch.Tensor = tdata['texture_features']  # (M, Dt)
        label_t = int(tdata['label'])

        # Sanity: labels should match
        if label_f != label_t:
            raise ValueError(
                f"Mismatched labels for patient_id '{patient_id}': features label={label_f}, texture label={label_t}"
            )
        label = torch.tensor(label_f, dtype=torch.long)

        # Align by index: trim to min length so row i matches across modalities
        n = min(feats.size(0), tex.size(0))
        if feats.size(0) != tex.size(0):
            if n == 0:
                raise ValueError(
                    f"No tiles to align for patient_id '{patient_id}' (N={feats.size(0)}, M={tex.size(0)})"
                )
            feats = feats[:n]
            tex = tex[:n]
            print(
                f"Warning: trimmed patient_id '{patient_id}' to {n} tiles for alignment (original N={fdata['features'].size(0)}, M={tdata['texture_features'].size(0)})"
            )

        # Optional sampling with shared indices to keep pairing intact
        if self.max_tiles is not None and n > self.max_tiles:
            if self.seed is not None:
                rng = numpy.random.default_rng(self.seed + int(self.current_epoch))
            else:
                rng = numpy.random.default_rng()
            indices = rng.choice(n, size=self.max_tiles, replace=False)
            indices_t = torch.as_tensor(indices, dtype=torch.long)
            feats = feats[indices_t]
            tex = tex[indices_t]

        return feats, tex, label, patient_id

    # Epoch/seed controls to align with training loops
    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def set_seed(self, seed: int):
        self.seed = int(seed)

class PatientDatasetHandler:
    """
    Handles loading and splitting of the Patient-level dataset.
    """
    def __init__(self):
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.validation_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_patient_split(self, dataset_dir, train_split=0.8, seed=42):
        """
        Builds the patient map and splits it into train/validation sets
        at the PATIENT level, ensuring stratification.
        """
        # Build the map for all patients
        full_patient_map = build_patient_tile_map(dataset_dir)
        
        # Get all patient IDs and their corresponding labels
        patient_ids = list(full_patient_map.keys())
        patient_labels = [full_patient_map[id]['label'] for id in patient_ids]
        
        # Use StratifiedKFold to get train/val indices
        n_splits = int(1 / (1 - train_split))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        # Get the indices for the first fold
        train_indices, val_indices = next(skf.split(patient_ids, patient_labels))
        
        # Get the patient IDs for each set
        train_patient_ids = [patient_ids[i] for i in train_indices]
        val_patient_ids = [patient_ids[i] for i in val_indices]
        
        # Create new maps for each split
        train_map = {id: full_patient_map[id] for id in train_patient_ids}
        val_map = {id: full_patient_map[id] for id in val_patient_ids}
        
        # Create the final Dataset objects
        train_dataset = PatientDataset(train_map, transform=self.train_transforms)
        val_dataset = PatientDataset(val_map, transform=self.validation_transforms)
        
        return train_dataset, val_dataset

class FeatureDatasetHandler:
    """
    Helper to create train/val feature datasets from an existing split of patient IDs.
    Use PatientDatasetHandler.get_patient_split_ids to build the split lists, then call build_feature_datasets.
    """
    @staticmethod
    def get_patient_split_ids(dataset_dir, train_split=0.8, seed=42):
        full_patient_map = build_patient_tile_map(dataset_dir)
        patient_ids = list(full_patient_map.keys())
        patient_labels = [full_patient_map[id]['label'] for id in patient_ids]

        n_splits = int(1 / (1 - train_split))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_indices, val_indices = next(skf.split(patient_ids, patient_labels))
        train_patient_ids = [patient_ids[i] for i in train_indices]
        val_patient_ids = [patient_ids[i] for i in val_indices]
        return train_patient_ids, val_patient_ids

    @staticmethod
    def build_feature_datasets(features_dir: str, train_ids: list[str], val_ids: list[str], max_tiles: int = None, seed: int = None):
        train_ds = PrecomputedPatientFeatureDataset(features_dir, train_ids, max_tiles=max_tiles, seed=seed)
        val_ds = PrecomputedPatientFeatureDataset(features_dir, val_ids, max_tiles=max_tiles, seed=seed)
        return train_ds, val_ds

    @staticmethod
    def build_combined_feature_datasets(
        features_dir: str,
        texture_features_dir: str,
        train_ids: list[str],
        val_ids: list[str],
        max_tiles: int = None,
        seed: int = None,
    ):
        train_ds = PrecomputedPatientCombinedFeatureDataset(
            features_dir, texture_features_dir, train_ids, max_tiles=max_tiles, seed=seed
        )
        val_ds = PrecomputedPatientCombinedFeatureDataset(
            features_dir, texture_features_dir, val_ids, max_tiles=max_tiles, seed=seed
        )
        return train_ds, val_ds

    @staticmethod
    def combine_and_save_features(
        features_dir: str,
        texture_features_dir: str,
        out_dir: str,
        patient_ids: list[str],
        overwrite: bool = False,
    ) -> None:
        """
        Create a combined on-disk dataset with both modalities for each patient.

        Saves files as '<out_dir>/<patient_id>.pt' with keys:
          - 'features': FloatTensor (N, Df)
          - 'texture_features': FloatTensor (N, Dt)
          - 'label': int
        Trims to the min tile count between the two modalities to ensure alignment.
        """
        os.makedirs(out_dir, exist_ok=True)
        for pid in patient_ids:
            out_path = os.path.join(out_dir, f"{pid}.pt")
            if (not overwrite) and os.path.exists(out_path):
                continue

            fpath = os.path.join(features_dir, f"{pid}.pt")
            tpath = os.path.join(texture_features_dir, f"{pid}.pt")
            if not (os.path.exists(fpath) and os.path.exists(tpath)):
                raise FileNotFoundError(f"Missing feature files for patient_id '{pid}': {fpath} or {tpath}")

            fdata = torch.load(fpath, map_location='cpu')
            tdata = torch.load(tpath, map_location='cpu')
            feats: torch.Tensor = fdata['features']
            tex: torch.Tensor = tdata['texture_features']
            label_f = int(fdata['label'])
            label_t = int(tdata['label'])
            if label_f != label_t:
                raise ValueError(
                    f"Mismatched labels for patient_id '{pid}': features label={label_f}, texture label={label_t}"
                )
            if (feats.size(0) != tex.size(0)):
                n = min(feats.size(0), tex.size(0))
                if n == 0:
                    raise ValueError(f"No tiles to align for patient_id '{pid}'")
                feats = feats[:n]
                tex = tex[:n]
                print(f"Warning: trimmed patient_id '{pid}' to {n} tiles for alignment (original N={fdata['features'].size(0)}, M={tdata['texture_features'].size(0)})")
            torch.save({'features': feats, 'texture_features': tex, 'label': label_f}, out_path)