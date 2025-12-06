import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

import GLCMFeatures


class TextureFeatureExtractor:
    """
    Computes per-tile texture features using GLCMs over 9 image channels:
    - RGB: R, G, B
    - HSV: H, S, V (via PIL)
    - LAB: L, A, B (via PIL "LAB")

    For each channel, 4 GLCMs are computed using angles: 0°, 45°, 90°, 135° at distance=1.
    From each GLCM, 7 texture features are extracted (in this output order):
      contrast, correlation, energy, homogeneity, entropy, max probability, autocorrelation

    Total features per tile: 9 channels x 4 angles x 7 features = 252.
    """

    def __init__(
        self,
        distances: Tuple[int, ...] = (1,),
        angles: Tuple[float, ...] = (0.0, math.pi/4, math.pi/2, 3*math.pi/4),
        levels: int = 256,
    ):
        self.distances = tuple(distances)
        self.angles = tuple(angles)
        # number of gray levels to use for GLCM (1..256). Higher = more detail, slower.
        self.levels = int(levels)

    def extract_from_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        """
        Compute GLCM texture features for a batch of tiles.

        Args:
            tiles: Tensor of shape (N, 3, H, W) or (1, N, 3, H, W).

        Returns:
            Tensor of shape (N, 252) with float32 features on CPU.
        """
        if tiles.dim() == 5:
            if tiles.size(0) != 1:
                raise ValueError(f"Expected batch_size==1 for patient tiles, got {tiles.size(0)}")
            tiles = tiles.squeeze(0)
        if tiles.dim() != 4 or tiles.size(1) != 3:
            raise ValueError(f"tiles must have shape (N,3,H,W), got {tuple(tiles.shape)}")

        tiles_cpu = tiles.detach().cpu()
        features = []
        for tile in tiles_cpu:  # (3,H,W)
            current_features = self._features_for_single_tile(tile)
            features.append(current_features)
        feat_mat = np.stack(features, axis=0).astype(np.float32)  # (N, 252)
        return torch.from_numpy(feat_mat)

    def extract_and_save_from_loader(self, dataloader, out_dir: str, overwrite: bool = False) -> None:
        """
        Iterate over a patient-level DataLoader that yields (tiles, label) or (tiles, label, patient_id),
        compute per-tile texture features, and save to `<out_dir>/<patient_id>.pt`.

        Args:
            dataloader (DataLoader): yielding (tiles, label) or (tiles, label, patient_id)
            out_dir (str): directory to save per-patient .pt files
            overwrite (bool): if False, skip patients whose files already exist

        Saved file content:
            - 'texture_features': FloatTensor (N, 252) on CPU,
            - 'label': int
        """
        os.makedirs(out_dir, exist_ok=True)
        for idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                tiles, label, patient_id = batch
            else:
                tiles, label = batch
                patient_id = f"patient_{idx:05d}"

            if isinstance(patient_id, (tuple, list)):
                patient_id = patient_id[0]

            if torch.is_tensor(label):
                try:
                    label_int = int(label.item())
                except Exception:
                    label_int = int(label.view(-1)[0].item())
            else:
                label_int = int(label)

            save_path = os.path.join(out_dir, f"{patient_id}.pt")
            if (not overwrite) and os.path.exists(save_path):
                continue

            feats = self.extract_from_tiles(tiles)  # (N, 252)
            torch.save({'texture_features': feats.cpu(), 'label': label_int}, save_path)

    def _features_for_single_tile(self, tile: torch.Tensor) -> np.ndarray:
        """
        Compute a 252-d texture feature vector for a single RGB tile.

        Parameters
        ----------
        tile : torch.Tensor
            3xHxW (CHW) RGB image. Supported input formats:
            - uint8: used as-is.
            - float in [0,1]: scaled to 0-255.
            - ImageNet-normalized floats (approx. -5..5): denormalized with
              mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] then scaled to 0-255.
            - floats in 0-255: rounded/clipped to uint8.
            Other ranges are clipped to [0,255] and cast to uint8.

        Returns
        -------
        np.ndarray
            1-D float32 array of length 252. Features are computed from nine uint8 channels
            (R,G,B; H,S,V; L,A,B) via gray-level co-occurrence matrices
            (levels=256, normed=True, symmetric=False) using self.distances, self.angles
            and the helper _calculate_glcm_features.
        """
        # Convert CHW tensor to HWC uint8 RGB
        arr = tile.detach().cpu().numpy()
        if arr.ndim != 3 or arr.shape[0] != 3:
            raise ValueError(f"Expected tile with shape (3,H,W), got {arr.shape}")
        rgb = np.transpose(arr, (1, 2, 0))

        if rgb.dtype != np.uint8:
            vmin, vmax = float(np.nanmin(rgb)), float(np.nanmax(rgb))
            if vmin >= -1e-3 and vmax <= 1.0 + 1e-3:
                rgb_uint8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
            elif vmin > -5.0 and vmax < 5.0:
                # Assume ImageNet-normalized floats; invert using standard stats inline
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]
                rgb_denorm = (rgb * std) + mean
                rgb_uint8 = np.clip(np.round(rgb_denorm * 255.0), 0, 255).astype(np.uint8)
            elif vmin >= 0.0 and vmax <= 255.0:
                rgb_uint8 = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
            else:
                rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb

        # Derive HSV and LAB
        pil_img = Image.fromarray(rgb_uint8, mode='RGB')
        hsv = np.array(pil_img.convert('HSV'))
        lab = np.array(pil_img.convert('LAB'))

        # 9 uint8 channels
        R = rgb_uint8[:, :, 0]
        G = rgb_uint8[:, :, 1]
        B = rgb_uint8[:, :, 2]
        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]
        Ls = lab[:, :, 0]
        As = lab[:, :, 1]
        Bs = lab[:, :, 2]
        channels = [R, G, B, H, S, V, Ls, As, Bs]

        all_feats: List[float] = []
        for ch in channels:
            # Quantize channel values to 0..levels-1 as required by graycomatrix
            if self.levels == 256:
                ch_q = ch
            else:
                # ch is uint8 in 0..255; map to 0..levels-1 evenly
                ch_q = ((ch.astype(np.uint16) * int(self.levels)) // 256).astype(np.uint8)

            P4 = graycomatrix(
                ch_q,
                distances=self.distances,
                angles=self.angles,
                levels=self.levels,
                symmetric=False,
                normed=True,
            )

            # Compute features per angle, shape (7, A)
            feats_mat = self._calculate_glcm_features(P4)
            feats_mat = np.asarray(feats_mat)
            if feats_mat.ndim != 2 or feats_mat.shape[0] != 7:
                raise RuntimeError("_calculate_glcm_features must return array of shape (7, num_angles)")

            # Append features
            cols = np.nan_to_num(feats_mat, nan=0.0, posinf=0.0, neginf=0.0)  # shape (7, A)
            all_feats.extend(cols.T.ravel().astype(float).tolist())

        if len(all_feats) != 252:
            raise RuntimeError(f"Expected 252 features, got {len(all_feats)}")
        return np.asarray(all_feats, dtype=np.float32)

    def _calculate_glcm_features(self, P: np.ndarray) -> np.ndarray:
        """
        Wrap GLCMFeatures graycoprops with scikit's graycoprops to compute 7 features.
        Returns a (7, A) array when num_dist=1 (distance=1) ordered as:
        [contrast, correlation, energy, homogeneity, entropy, max probability, autocorrelation].
        """
        contrast = graycoprops(P, prop='contrast')
        correlation = graycoprops(P, prop='correlation')
        energy = graycoprops(P, prop='energy')
        homogeneity = graycoprops(P, prop='homogeneity')

        entropy = GLCMFeatures.graycoprops(P, prop='entropy')
        max_probability = GLCMFeatures.graycoprops(P, prop='max probability')
        autocorrelation = GLCMFeatures.graycoprops(P, prop='autocorrelation')

        feature_matrix = np.concatenate(
            (
                contrast,
                correlation,
                energy,
                homogeneity,
                entropy,
                max_probability,
                autocorrelation,
            ),
            axis=0,
        )  # (7,A) when D=1
        return feature_matrix



