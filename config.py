import argparse
import os
from pathlib import Path

# project directories
PROJECT_ROOT = Path(__file__).resolve().parent


def _path_from_env(variable: str, default: Path) -> Path:
    """Resolve a filesystem path from an environment variable with a sensible default."""
    value = os.getenv(variable)
    if value:
        return Path(value).expanduser()
    return Path(default).expanduser()

# globel param
# dataset setting
img_width = 1280
img_height = 720
img_channel = 3
label_width = 256
label_height = 128
label_channel = 1
data_loader_numworkers = 8
class_num = 2

# path
train_path = _path_from_env('FYP_TRAIN_INDEX', PROJECT_ROOT / 'data' / 'train_index.txt')
val_path = _path_from_env('FYP_VAL_INDEX', PROJECT_ROOT / 'data' / 'val_index.txt')
test_path = _path_from_env('FYP_TEST_INDEX', PROJECT_ROOT / 'data' / 'test_index.txt')
save_path = _path_from_env('FYP_SAVE_PATH', PROJECT_ROOT / 'save' / 'media2')
pretrained_path = _path_from_env('FYP_PRETRAINED', PROJECT_ROOT / 'pretrained' / 'unetlstm.pth')
checkpoint_dir = _path_from_env('FYP_CHECKPOINT_DIR', PROJECT_ROOT / 'runs' / 'checkpoints')
# pretrained_path='/root/autodl-fs/RLD_O/code/model/momentum0.9/last_98.06154165694963.pth'
# pretrained_path='/root/autodl-fs/RLD_O/code/model/momentum0.9/last_98.06154165694963.pth'

# weight
class_weight = [0.02, 1.02]

SEQUENCE_MODELS = {"UNet-ConvLSTM", "SegNet-ConvLSTM"}
DEFAULT_SEQUENCE_LENGTH = int(os.getenv('FYP_SEQUENCE_LENGTH', 5))

def args_setting():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch UNet-ConvLSTM')
    parser.add_argument('--model', type=str, default='UNet-ConvLSTM',
                        choices=['UNet-ConvLSTM', 'SegNet-ConvLSTM', 'UNet', 'SegNet'],
                        help='Segmentation architecture to train/evaluate.')
    parser.add_argument('--train-index', type=str, default=str(train_path),
                        help='Path to the training index file.')
    parser.add_argument('--val-index', type=str, default=str(val_path),
                        help='Path to the validation index file.')
    parser.add_argument('--test-index', type=str, default=str(test_path),
                        help='Path to the testing index file.')
    parser.add_argument('--save-dir', type=str, default=str(save_path),
                        help='Directory to store visualisations and intermediate outputs.')
    parser.add_argument('--checkpoint-dir', type=str, default=str(checkpoint_dir),
                        help='Directory to store trained model checkpoints.')
    parser.add_argument('--pretrained', type=str, default=str(pretrained_path),
                        help='Path to a pretrained checkpoint for weight initialisation.')
    parser.add_argument('--sequence-length', type=int, default=DEFAULT_SEQUENCE_LENGTH,
                        help='Number of consecutive frames fed into sequence models.')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num-workers', type=int, default=data_loader_numworkers,
                        help='Number of worker processes for the dataloaders.')
    parser.add_argument('--output-video', type=str, default=None,
                        help='Optional override for the rendered video output path (test pipeline).')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    # Normalise path-like arguments for downstream modules
    args.train_index = Path(args.train_index).expanduser()
    args.val_index = Path(args.val_index).expanduser()
    args.test_index = Path(args.test_index).expanduser()
    args.save_dir = Path(args.save_dir).expanduser()
    args.checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    args.pretrained = Path(args.pretrained).expanduser()
    if args.output_video is not None:
        args.output_video = Path(args.output_video).expanduser()
    return args
    return args
