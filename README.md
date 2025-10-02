# Lane Segmentation & Distance Estimation 

This project is about lane segmentation on first-person driving videos. It uses UNet or SegNet as the main models, and can also add ConvLSTM to handle video sequences. I included YOLOv8 for extra analysis. 

## Repository Map

| Path | Purpose |
| --- | --- |
| `config.py` | Global constants, CLI parsing, path/environment handling. |
| `dataset.py` | Configurable sequence datasets with/without labels. |
| `model.py`, `utils.py` | Segmentation architectures and ConvLSTM blocks. |
| `train.py` | Training loop with best-checkpoint saving and LR scheduling. |
| `test.py` | Validation + overlay renderer with MP4 export. |
| `video.py` | CLI utilities for frame extraction and index generation. |
| `YOLO_pre.py` | YOLOv8 overlay generator (CLI). |
| `YOLO_distance.py` | Distance estimation video pipeline (CLI). |
| `tools.py` | Misc helpers (augmentation, index splitting, folder reset). |

## 1. Environment Setup

### 1.1 Prerequisites

- Python ≥ 3.8 (Python ≥ 3.10 recommended) with CUDA-enabled PyTorch. 

### 1.2 Install dependencies

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pillow scikit-learn ultralytics tqdm
```

## 2. Data Preparation Workflow

Segmentation training expects **`sequence_length` RGB frames + 1 label path per line** (defaults to 6 entries: 5 frames + 1 mask). Paths can be relative but absolute paths avoid ambiguity when moving between machines.

### 2.1 Extract frames from raw video

```powershell
python video.py extract --video D:\data\raw\drive.mp4 --output D:\data\frames\drive1
```

Frames are saved as `frame_0000.jpg`, `frame_0001.jpg`, ... inside the chosen output directory.

### 2.2 Create segmentation masks

- Make lane masks that match each frame’s pixels (using annotation tools or some simple rules).  
- Save them together with the RGB frames, or in a separate folder with the same file names.  

### 2.3 Build rolling index files

```powershell
python video.py index --images D:\data\frames\drive1 --output D:\data\indices\drive1_train.txt --window-size 6 --stride 1
```

Each line will contain `window_size` consecutive frames. Append the mask path manually or post-process the file if the mask lives elsewhere.

Alternative flat index (one path per line):

```powershell
python video.py index-line --images D:\data\frames\drive1 --output D:\data\indices\drive1_all.txt
```

The helper defaults to an identity transform when `op` is not supplied.

## 3. Configuration Reference

`config.py` exposes sane defaults relative to the repo root. Override via:

| Environment variable | CLI flag (from `args_setting`) | Default |
| --- | --- | --- |
| `FYP_TRAIN_INDEX` | `--train-index` | `./data/train_index.txt` |
| `FYP_VAL_INDEX` | `--val-index` | `./data/val_index.txt` |
| `FYP_TEST_INDEX` | `--test-index` | `./data/test_index.txt` |
| `FYP_SAVE_PATH` | `--save-dir` | `./save/media2` |
| `FYP_CHECKPOINT_DIR` | `--checkpoint-dir` | `./runs/checkpoints` |
| `FYP_PRETRAINED` | `--pretrained` | `./pretrained/unetlstm.pth` |
| `FYP_SEQUENCE_LENGTH` | `--sequence-length` | `5` |


## 4. Training

Choose your architecture (`UNet`, `SegNet`, `UNet-ConvLSTM`, `SegNet-ConvLSTM`) and launch training:

```powershell
python train.py `
  --model UNet-ConvLSTM `
  --train-index D:\data\indices\train.txt `
  --val-index D:\data\indices\val.txt `
  --pretrained .\pretrained\unetlstm.pth `
  --checkpoint-dir .\runs\checkpoints `
  --batch-size 4 --epochs 30 --lr 1e-4 --cuda
```

What happens under the hood: 

- The dataloaders follow the `--sequence-length` setting when using sequence models, and switch to the last frame if it’s just a single-frame model.  
- By default, I use the Adam optimizer, and the learning rate is adjusted with `ReduceLROnPlateau`.  


## 5. Validation & Visualisation

Render overlays and optional video outputs:

```powershell
python test.py `
  --model UNet-ConvLSTM `
  --test-index D:\data\indices\test.txt `
  --pretrained .\runs\checkpoints\best.pth `
  --save-dir .\save\run1 `
  --output-video .\save\run1\lane.mp4 `
  --cuda
```

Outputs include per-frame overlays (`NNNN_frame.jpg` + `NNNN_overlay.jpg`) and an MP4 (if `--output-video` is supplied). The overlay colour defaults to `(234, 53, 57)` on lane pixels.

## 6. YOLO Post-processing 

### 6.1 Annotated object overlays

```powershell
python YOLO_pre.py --input .\save\run1\lane.mp4 --output .\save\run1\lane_yolo.mp4 --weights .\yolov8n.pt --fps 30
```

### 6.2 Distance estimation

```powershell
python YOLO_distance.py --input .\save\run1\lane.mp4 --output .\save\run1\lane_distance.mp4 --weights .\yolov8n.pt --focal-length 200
```

- Distances are colour-coded: green (>3 m), yellow (1–3 m), red (≤1 m).
- Adjust `REAL_HEIGHTS_INCHES` or the `--focal-length` value for your camera calibration.

## 7. Timeline 

1. Install dependencies and configure environment variables if needed.
2. Extract frames from raw driving videos.
3. Produce masks and build index files (train/val/test).
4. Launch training and monitor logs (loss, accuracy, LR adjustments).
5. Run `test.py` to inspect qualitative overlays and metrics.
