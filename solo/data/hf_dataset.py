from pathlib import Path

import datasets as hf_datasets
from torch.utils.data import Dataset

_SUBSET_DIR = Path(__file__).parent / "dataset_subset"


def _imagenet100_synsets():
    """Returns alphabetically sorted synset IDs, matching ImageFolder's label assignment."""
    return sorted((_SUBSET_DIR / "imagenet100_classes.txt").read_text().split())


class HFImageNetDataset(Dataset):
    """Wraps a HuggingFace ImageNet dataset split with a torchvision transform.

    label_map: optional dict {hf_label_int: new_label_int} for subset remapping.
    """

    def __init__(self, hf_dataset, transform=None, label_map=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        img = item["image"].convert("RGB")
        label = item["label"]
        if self.label_map is not None:
            label = self.label_map[label]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def load_hf_imagenet(data_path, hf_split: str, dataset: str = "imagenet"):
    """Load an ImageNet HuggingFace dataset.

    Tries (1) load_from_disk on local path, (2) load_dataset on local path
    (imagefolder), then (3) HF Hub / cache.
    For imagenet100, filters to the 100-class subset and remaps labels to 0-99.

    Returns:
        (hf_dataset, label_map)  — label_map is None for full imagenet.
    """
    def _load_hub(split):
        return hf_datasets.load_dataset("imagenet-1k", split=split, trust_remote_code=True)

    hf_ds = None
    if data_path is not None:
        # Arrow dataset saved with save_to_disk.
        try:
            ds = hf_datasets.load_from_disk(str(data_path))
            hf_ds = ds[hf_split] if isinstance(ds, hf_datasets.DatasetDict) else ds
        except Exception:
            pass
        # Imagefolder-style directory.
        if hf_ds is None:
            try:
                hf_ds = hf_datasets.load_dataset(
                    str(data_path), split=hf_split, trust_remote_code=True
                )
            except Exception:
                pass
    if hf_ds is None:
        hf_ds = _load_hub(hf_split)

    if dataset != "imagenet100":
        return hf_ds, None

    import numpy as np

    # Ensure the label feature has named synset IDs (nXXXXXXXX format).
    # If not, fall back to Hub which always has the canonical synset names.
    label_feature = hf_ds.features.get("label")
    names = getattr(label_feature, "names", None)
    if not names or not names[0].startswith("n"):
        hf_ds = _load_hub(hf_split)
        names = hf_ds.features["label"].names

    synsets = _imagenet100_synsets()  # sorted → indices 0..99 match ImageFolder
    synset_to_hf = {name: i for i, name in enumerate(names)}
    label_map = {synset_to_hf[s]: new_i for new_i, s in enumerate(synsets) if s in synset_to_hf}

    if not label_map:
        raise RuntimeError(
            f"No imagenet100 synsets found in HF label names. "
            f"First few label names: {names[:5]}"
        )

    # Read only the integer label column (no image decoding) then select by index.
    labels = np.array(hf_ds["label"])
    indices = np.where(np.isin(labels, list(label_map.keys())))[0].tolist()
    hf_ds = hf_ds.select(indices)
    return hf_ds, label_map
