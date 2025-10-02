import importlib
import time
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from config import args_setting
from dataset import RoadSequenceDataset
from model import generate_model


def prepare_input(batch: torch.Tensor, expects_sequence: bool) -> torch.Tensor:
    """Ensure the batch matches the model's expected input shape."""
    if expects_sequence:
        return batch
    if batch.ndim == 5:
        # (batch, seq, C, H, W) -> (batch, C, H, W) using the most recent frame
        return batch[:, -1, :, :, :]
    return batch


def forward_pass(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    return outputs


def build_dataloaders(args, transform) -> Tuple[Iterable, Iterable]:
    common_kwargs = {
        'transforms': transform,
        'sequence_length': args.sequence_length,
    }

    train_dataset = RoadSequenceDataset(
        file_path=args.train_index,
        include_label=True,
        **common_kwargs,
    )
    val_dataset = RoadSequenceDataset(
        file_path=args.val_index,
        include_label=True,
        **common_kwargs,
    )

    pin_memory = args.cuda and torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def train_epoch(args, epoch: int, model: torch.nn.Module, train_loader, device, optimizer, criterion, expects_sequence: bool) -> float:
    since = time.time()
    model.train()
    running_loss = 0.0

    for batch_idx, sample in enumerate(train_loader):
        inputs = prepare_input(sample['data'].to(device), expects_sequence)
        targets = sample['label'].to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = forward_pass(model, inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        if batch_idx % args.log_interval == 0:
            progress = 100.0 * batch_idx / max(len(train_loader) - 1, 1)
            print(
                f"Epoch {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} "
                f"({progress:.0f}%)]\tLoss: {loss.item():.6f}"
            )

    epoch_loss = running_loss / len(train_loader.dataset)
    elapsed = time.time() - since
    print(f"Epoch {epoch} completed in {elapsed // 60:.0f}m {elapsed % 60:.0f}s | loss: {epoch_loss:.6f}")
    return epoch_loss


def evaluate(model: torch.nn.Module, val_loader, device, criterion, expects_sequence: bool) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for sample in val_loader:
            inputs = prepare_input(sample['data'].to(device), expects_sequence)
            targets = sample['label'].to(device, dtype=torch.long)

            outputs = forward_pass(model, inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == targets).sum().item()
            total_pixels += targets.numel()

    avg_loss = total_loss / len(val_loader.dataset)
    accuracy = 100.0 * total_correct / total_pixels if total_pixels else 0.0

    print(
        f"Validation | loss: {avg_loss:.4f}, accuracy: {total_correct}/{total_pixels} "
        f"({accuracy:.5f}%)"
    )
    return avg_loss, accuracy


def save_checkpoint(model: torch.nn.Module, accuracy: float, checkpoint_dir: Path) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{accuracy:.5f}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def load_pretrained(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if checkpoint_path and checkpoint_path.exists():
        print(f"Loading pretrained weights from {checkpoint_path}")
        pretrained_dict = torch.load(checkpoint_path, map_location=device)
        model_dict = model.state_dict()
        filtered_weights = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(filtered_weights)
        model.load_state_dict(model_dict)
    else:
        print("No pretrained checkpoint found. Training from scratch.")


def main():
    args = args_setting()
    torch.manual_seed(args.seed)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    expects_sequence = args.model in config.SEQUENCE_MODELS

    transforms_module = importlib.import_module('torchvision.transforms')
    transform = transforms_module.Compose([transforms_module.ToTensor()])
    train_loader, val_loader = build_dataloaders(args, transform)

    model = generate_model(args)
    load_pretrained(model, args.pretrained, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    class_weight = torch.tensor(config.class_weight, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)

    best_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        train_epoch(args, epoch, model, train_loader, device, optimizer, criterion, expects_sequence)
        val_loss, val_accuracy = evaluate(model, val_loader, device, criterion, expects_sequence)
        scheduler.step(val_loss)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(model, val_accuracy, args.checkpoint_dir)


if __name__ == '__main__':
    main()
