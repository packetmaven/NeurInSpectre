"""
Real dataset loading for NeurInSpectre Table 1 reproduction.

Implements proper dataset handling for:
- CIFAR-10 (Content Moderation domain)
- ImageNet-100 subset (Content Moderation domain)
- EMBER (Malware Detection domain)
- nuScenes (AV Perception domain)
"""

from __future__ import annotations

import json
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image


def _safe_extract_tar(tar: tarfile.TarFile, dest_dir: Path) -> None:
    """
    Safely extract a tar archive into `dest_dir`.

    We explicitly reject absolute paths, parent traversal, and symlinks/hardlinks.
    This avoids common tar extraction vulnerabilities.
    """

    dest_dir = Path(dest_dir).resolve()
    for member in tar.getmembers():
        name = str(member.name)
        if not name or name.startswith("/"):
            raise RuntimeError(f"Refusing to extract unsafe tar member: {name!r}")
        parts = Path(name).parts
        if any(p == ".." for p in parts):
            raise RuntimeError(f"Refusing to extract tar member with '..': {name!r}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"Refusing to extract symlink/hardlink from tar: {name!r}")
        target = (dest_dir / name).resolve()
        if dest_dir not in target.parents and target != dest_dir:
            raise RuntimeError(f"Refusing to extract tar member outside destination: {name!r}")

    tar.extractall(path=str(dest_dir))


class DatasetConfig:
    """Configuration for dataset loading."""

    CIFAR10_PATH = "./data/cifar10"
    IMAGENET_PATH = "./data/imagenet"
    EMBER_PATH = "./data/ember"
    NUSCENES_PATH = "./data/nuscenes"

    IMAGENET_100_CLASSES = list(range(100))

    EVAL_SAMPLES = {
        "cifar10": 1000,
        "imagenet100": 1000,
        "ember": 1000,
        "nuscenes": 500,
    }


class CIFAR10Dataset:
    """CIFAR-10 dataset for Content Moderation evaluations."""

    @staticmethod
    def load(
        root: str = DatasetConfig.CIFAR10_PATH,
        n_samples: int = DatasetConfig.EVAL_SAMPLES["cifar10"],
        seed: int = 42,
        batch_size: int = 100,
        num_workers: int = 4,
        split: str = "test",
        download: bool = True,
        pin_memory: bool = True,
    ) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        train = str(split).lower() in {"train", "training"}
        testset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=bool(download),
            transform=transform,
        )

        torch.manual_seed(seed)
        np.random.seed(seed)
        if n_samples <= 0 or n_samples >= len(testset):
            indices = np.arange(len(testset))
        else:
            indices = np.random.choice(len(testset), n_samples, replace=False)
        subset = Subset(testset, indices)

        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(pin_memory),
        )

        x_test = torch.stack([testset[i][0] for i in indices])
        y_test = torch.tensor([testset[i][1] for i in indices])
        return loader, x_test, y_test


class ImageNet100Dataset:
    """ImageNet-100 subset for Content Moderation."""

    @staticmethod
    def load(
        root: str = DatasetConfig.IMAGENET_PATH,
        n_samples: int = DatasetConfig.EVAL_SAMPLES["imagenet100"],
        seed: int = 42,
        batch_size: int = 50,
        num_workers: int = 4,
        split: str = "val",
        pin_memory: bool = True,
    ) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
        # Respect the user-provided root. We accept either:
        #   1) root=/path/to/imagenet   (contains train/ + val/)
        #   2) root=/path/to/imagenet/val (points directly at the split dir)
        root_path = Path(os.path.expandvars(os.path.expanduser(str(root)))).resolve()
        if not root_path.exists():
            raise FileNotFoundError(
                f"ImageNet-100 not found at {root_path}. "
                "Download from https://www.image-net.org/download.php"
            )

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

        # ImageNet conventions are (train, val). Treat "test" as "val" for UX.
        split_key = str(split).lower().strip()
        split_dir = "val" if split_key in {"val", "valid", "validation", "test"} else "train"

        # First try the canonical layout: <root>/{train,val}
        split_root = root_path / split_dir
        if not split_root.exists():
            # If the caller passed <root>/<train|val>, treat parent as base.
            if root_path.name in {"train", "val"}:
                parent = root_path.parent
                candidate = parent / split_dir
                if candidate.exists():
                    split_root = candidate
            # Otherwise, fail with a targeted message.
        if not split_root.exists():
            raise FileNotFoundError(
                f"ImageNet-100 split directory not found. "
                f"Expected '{(root_path / split_dir)}' to exist (split={split_key!r}). "
                f"If you passed a split directory directly, use root='{(root_path.parent / split_dir)}' "
                f"or set split accordingly."
            )

        valset = torchvision.datasets.ImageFolder(
            root=str(split_root),
            transform=transform,
        )

        valid_indices = [
            i for i, (_, label) in enumerate(valset.samples) if label < 100
        ]

        torch.manual_seed(seed)
        np.random.seed(seed)
        if n_samples <= 0 or n_samples >= len(valid_indices):
            sampled_indices = np.array(valid_indices)
        else:
            sampled_indices = np.random.choice(
                valid_indices,
                min(n_samples, len(valid_indices)),
                replace=False,
            )

        subset = Subset(valset, sampled_indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(pin_memory),
        )

        x_test = torch.stack([valset[i][0] for i in sampled_indices])
        y_test = torch.tensor([valset[i][1] for i in sampled_indices])
        return loader, x_test, y_test


class EMBERDataset:
    """EMBER malware detection dataset."""

    @staticmethod
    def _read_vectorized_memmaps(data_dir: str, *, subset: str) -> Tuple[np.memmap, np.memmap]:
        """
        Read EMBER vectorized features from memmap-backed `.dat` files.

        This avoids importing the upstream `ember` package (which transitively
        imports heavyweight optional deps like `lightgbm`) and lets us sample
        without materializing the full dataset in RAM.
        """
        subset = str(subset).lower().strip()
        if subset not in {"train", "test"}:
            raise ValueError(f"subset must be 'train' or 'test', got {subset!r}")

        x_path = os.path.join(data_dir, f"X_{subset}.dat")
        y_path = os.path.join(data_dir, f"y_{subset}.dat")
        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            raise FileNotFoundError(
                "Missing vectorized EMBER files. Expected:\n"
                f"  - {x_path}\n"
                f"  - {y_path}\n"
                "From scratch:\n"
                "  1) Download/extract raw shards: python scripts/download_ember2018.py\n"
                "  2) Vectorize (macOS-safe): python scripts/vectorize_ember_safe.py\n"
                "Raw shards should end up under `data/ember/ember2018/`."
            )

        # `scripts/vectorize_ember_safe.py` writes float32 for both X and y.
        y_bytes = int(os.path.getsize(y_path))
        x_bytes = int(os.path.getsize(x_path))
        if y_bytes % 4 != 0 or x_bytes % 4 != 0:
            raise ValueError("EMBER .dat files must be float32-aligned (size % 4 == 0)")

        n = y_bytes // 4
        if n <= 0:
            raise ValueError("EMBER label file is empty")

        n_floats = x_bytes // 4
        if n_floats % n != 0:
            raise ValueError(
                "EMBER feature file size is not divisible by label count. "
                f"floats={n_floats} labels={n}"
            )
        d = n_floats // n
        if d <= 0:
            raise ValueError("EMBER feature dimension inferred as <= 0")

        x_mm = np.memmap(x_path, dtype=np.float32, mode="r", shape=(n, d))
        y_mm = np.memmap(y_path, dtype=np.float32, mode="r", shape=(n,))
        return x_mm, y_mm

    @staticmethod
    def download_ember(root: str, year: str = "2018") -> None:
        os.makedirs(root, exist_ok=True)
        base_url = "https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"
        tar_path = os.path.join(root, f"ember_{year}.tar.bz2")

        if not os.path.exists(tar_path):
            print("[EMBER] Downloading EMBER2018 dataset...")
            urllib.request.urlretrieve(base_url, tar_path)
            print("[EMBER] Download complete.")

        extract_path = os.path.join(root, f"ember_{year}")
        if not os.path.exists(extract_path):
            print("[EMBER] Extracting...")
            with tarfile.open(tar_path, "r:bz2") as tar:
                _safe_extract_tar(tar, Path(root))
            print("[EMBER] Extraction complete.")

    @staticmethod
    def load(
        root: str = DatasetConfig.EMBER_PATH,
        n_samples: int = DatasetConfig.EVAL_SAMPLES["ember"],
        seed: int = 42,
        batch_size: int = 100,
        num_workers: int = 2,
        split: str = "test",
        pin_memory: bool = True,
        filter_label: Optional[int] = None,
    ) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
        subset = "train" if str(split).lower() in {"train", "training"} else "test"
        data_dir = os.path.join(root, "ember_2018")

        # Prefer the memmap path (no `ember` import required). This is the artifact
        # format we ship for AE; it also avoids loading ~GBs of data into RAM.
        x_mm, y_mm = EMBERDataset._read_vectorized_memmaps(data_dir, subset=subset)

        # Filter out unlabeled samples (-1). We only materialize the chosen subset.
        y_arr = np.asarray(y_mm, dtype=np.float32)
        labeled_idx = np.nonzero(y_arr >= 0)[0]
        if filter_label is not None:
            labeled_idx = labeled_idx[y_arr[labeled_idx] == int(filter_label)]
        if labeled_idx.size == 0:
            raise ValueError("EMBER vectorized labels contain no labeled samples (y >= 0).")

        rng = np.random.default_rng(int(seed))
        if n_samples <= 0 or int(n_samples) >= int(labeled_idx.size):
            chosen = labeled_idx
        else:
            chosen = rng.choice(labeled_idx, size=int(n_samples), replace=False)

        x_sel = np.array(x_mm[chosen], copy=True)
        y_sel = np.array(y_arr[chosen].astype(np.int64, copy=False), copy=True)

        x_test = torch.from_numpy(x_sel).float()
        y_test = torch.from_numpy(y_sel).long()

        dataset = torch.utils.data.TensorDataset(x_test, y_test)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(pin_memory),
        )
        return loader, x_test, y_test


class NuScenesImageDataset(Dataset):
    """Pickle-safe dataset for nuScenes images."""

    def __init__(self, items: List[Dict], transform):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        x = self.transform(img)
        y = int(item["label"])
        return x, y


class nuScenesDataset:
    """nuScenes perception dataset for AV domain."""

    @staticmethod
    def load(
        root: str = DatasetConfig.NUSCENES_PATH,
        n_samples: int = DatasetConfig.EVAL_SAMPLES["nuscenes"],
        seed: int = 42,
        version: str = "v1.0-mini",
        labels_path: str | None = None,
        image_size: tuple[int, int] = (224, 224),
        batch_size: int = 16,
        num_workers: int = 2,
        pin_memory: bool = True,
        split: str = "val",
    ) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
        if not os.path.exists(root):
            raise FileNotFoundError(
                f"nuScenes not found at {root}. "
                "Download from https://www.nuscenes.org/nuscenes#download"
            )

        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError as exc:
            raise ImportError(
                "Please install nuscenes-devkit: pip install nuscenes-devkit"
            ) from exc

        if labels_path is None:
            raise ValueError(
                "nuScenes loader requires labels_path mapping sample_token -> class label."
            )

        with open(labels_path, "r", encoding="utf-8") as handle:
            label_map = json.load(handle)

        nusc = NuScenes(version=version, dataroot=root, verbose=False)
        camera_channel = "CAM_FRONT"
        samples: List[Dict] = []

        # Respect nuScenes official scene splits (mini_train/mini_val for v1.0-mini,
        # train/val/test otherwise). This avoids mixing train/val and makes the
        # "split" knob in YAML configs meaningful.
        split_norm = str(split).lower()
        is_mini = "mini" in str(version).lower()
        scene_names = None
        if split_norm not in {"all", "any", "full", ""}:
            # nuScenes v1.0-mini provides mini_train/mini_val only. For CLI
            # ergonomics, treat "test" as an alias of "val".
            if is_mini and split_norm in {"test", "testing"}:
                split_norm = "val"
            try:
                from nuscenes.utils.splits import create_splits_scenes

                splits = create_splits_scenes()
                if split_norm in {"train", "training"}:
                    key = "mini_train" if is_mini else "train"
                elif split_norm in {"val", "valid", "validation"}:
                    key = "mini_val" if is_mini else "val"
                elif split_norm in {"test", "testing"}:
                    key = "test"
                else:
                    key = None
                if key and key in splits:
                    scene_names = set(splits[key])
            except Exception:
                scene_names = None

        scenes = list(nusc.scene)
        if scene_names is not None:
            scenes = [s for s in scenes if s.get("name") in scene_names]
        if not scenes:
            raise ValueError(f"nuScenes split produced 0 scenes (split={split}, version={version})")

        for scene in scenes:
            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = nusc.get("sample", sample_token)
                camera_data = nusc.get("sample_data", sample["data"][camera_channel])
                img_path = os.path.join(root, camera_data["filename"])
                if not os.path.exists(img_path):
                    sample_token = sample["next"]
                    continue
                if sample_token in label_map:
                    samples.append(
                        {
                            "image_path": img_path,
                            "sample_token": sample_token,
                            "label": int(label_map[sample_token]),
                        }
                    )
                sample_token = sample["next"]

        np.random.seed(seed)
        if n_samples <= 0 or n_samples >= len(samples):
            sampled_indices = list(range(len(samples)))
        else:
            sampled_indices = np.random.choice(
                len(samples),
                min(n_samples, len(samples)),
                replace=False,
            )
        samples = [samples[i] for i in sampled_indices]

        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )
        dataset = NuScenesImageDataset(samples, transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(pin_memory),
        )

        x_list = []
        y_list = []
        for x_batch, y_batch in loader:
            x_list.append(x_batch)
            y_list.append(y_batch)
        x_tensor = torch.cat(x_list, dim=0)
        y_tensor = torch.cat(y_list, dim=0)

        return loader, x_tensor, y_tensor


class DatasetFactory:
    """Factory for loading datasets based on defense domain."""

    DEFENSE_TO_DATASET = {
        "jpeg_compression": "cifar10",
        "bit_depth_reduction": "cifar10",
        "randomized_smoothing": "cifar10",
        "ensemble_diversity": "cifar10",
        "feature_squeezing": "ember",
        "gradient_regularization": "ember",
        "defensive_distillation": "ember",
        "adversarial_training_transform": "ember",
        "spatial_smoothing": "nuscenes",
        "random_pad_crop": "nuscenes",
        "thermometer_encoding": "nuscenes",
        "certified_defense": "nuscenes",
    }

    @staticmethod
    def get_dataset(dataset_name: str, **kwargs):
        if dataset_name == "cifar10":
            return CIFAR10Dataset.load(**kwargs)
        if dataset_name == "imagenet100":
            return ImageNet100Dataset.load(**kwargs)
        if dataset_name == "ember":
            return EMBERDataset.load(**kwargs)
        if dataset_name == "nuscenes":
            return nuScenesDataset.load(**kwargs)
        raise ValueError(f"Unknown dataset: {dataset_name}")
