import datasets as hf_datasets
from torch.utils.data import Dataset


class HFImageNetDataset(Dataset):
    """Wraps a HuggingFace ImageNet dataset split with a torchvision transform."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        img = item["image"].convert("RGB")
        label = item["label"]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def load_hf_imagenet(data_path, hf_split: str):
    """Load an ImageNet HuggingFace dataset.

    Tries (1) local path via load_dataset, then (2) HF Hub / cache.
    """
    if data_path is not None:
        try:
            return hf_datasets.load_dataset(str(data_path), split=hf_split, trust_remote_code=True)
        except Exception:
            pass
    return hf_datasets.load_dataset("imagenet-1k", split=hf_split, trust_remote_code=True)
