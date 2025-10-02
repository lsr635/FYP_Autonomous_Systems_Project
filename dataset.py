from pathlib import Path
from typing import Callable, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset


def readTxt(file_path: Path) -> List[List[str]]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Index file not found: {file_path}")

    img_list: List[List[str]] = []
    with open(file_path, 'r') as file_to_read:
        for line in file_to_read:
            stripped = line.strip()
            if not stripped:
                continue
            img_list.append(stripped.split())
    return img_list

class RoadSequenceDataset(Dataset):
    def __init__(
        self,
        file_path: Path,
        transforms: Callable,
        sequence_length: Optional[int] = None,
        include_label: bool = True,
        label_transforms: Optional[Callable] = None,
    ) -> None:
        if transforms is None:
            raise ValueError("`transforms` must be provided for RoadSequenceDataset.")

        self.file_path = Path(file_path)
        self.transforms = transforms
        self.label_transforms = label_transforms or transforms
        self.include_label = include_label
        self.sequence_length = sequence_length

        self.img_list = readTxt(self.file_path)
        if not self.img_list:
            raise ValueError(f"Index file {self.file_path} does not contain any entries.")

        min_expected = 2 if self.include_label else 1
        for line in self.img_list:
            if len(line) < min_expected:
                raise ValueError(
                    f"Index entry must contain at least {min_expected} paths, received: {line}"
                )

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, idx: int):
        img_path_list = self.img_list[idx]
        frame_paths: Sequence[str]
        label_path: Optional[str] = None

        if self.include_label:
            frame_paths = img_path_list[:-1]
            label_path = img_path_list[-1]
        else:
            frame_paths = img_path_list

        if self.sequence_length is not None:
            if len(frame_paths) < self.sequence_length:
                raise ValueError(
                    f"Requested sequence length {self.sequence_length} but only "
                    f"found {len(frame_paths)} frames for index {idx}."
                )
            frame_paths = frame_paths[-self.sequence_length:]

        frames = [
            self.transforms(Image.open(Path(path)).convert("RGB"))
            for path in frame_paths
        ]
        data = torch.stack(frames, dim=0)

        sample = {'data': data}

        if self.include_label and label_path is not None:
            label_img = Image.open(Path(label_path))
            label_tensor = self.label_transforms(label_img)
            if label_tensor.ndim > 2:
                label_tensor = torch.squeeze(label_tensor, dim=0)
            sample['label'] = label_tensor.to(dtype=torch.long)

        return sample


class RoadSequenceDatasetList(RoadSequenceDataset):
    """Dataset variant for inference-only workflows (no labels)."""

    def __init__(self, file_path: Path, transforms: Callable, sequence_length: Optional[int] = None) -> None:
        super().__init__(
            file_path=file_path,
            transforms=transforms,
            sequence_length=sequence_length,
            include_label=False,
        )