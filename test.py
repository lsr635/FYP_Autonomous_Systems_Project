import importlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

import config
from config import args_setting
from dataset import RoadSequenceDataset
from model import generate_model


def prepare_input(batch: torch.Tensor, expects_sequence: bool) -> torch.Tensor:
	if expects_sequence:
		return batch
	if batch.ndim == 5:
		return batch[:, -1, :, :, :]
	return batch


def forward_pass(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
	outputs = model(inputs)
	if isinstance(outputs, tuple):
		outputs = outputs[0]
	return outputs


def tensor_to_rgb(frame_tensor: torch.Tensor) -> np.ndarray:
	array = frame_tensor.detach().cpu().numpy()
	if array.ndim == 3:
		array = np.transpose(array, (1, 2, 0))
	array = np.clip(array * 255.0, 0, 255)
	return array.astype(np.uint8)


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
	overlay = frame.copy()
	overlay[mask] = (234, 53, 57)
	return overlay


def save_image(path: Path, image: np.ndarray) -> None:
	Image.fromarray(image).save(path)


def output_result(
	model: torch.nn.Module,
	data_loader,
	device: torch.device,
	save_dir: Path,
	video_path: Path,
	expects_sequence: bool,
	fps: int = 30,
) -> Optional[Path]:
	model.eval()
	save_dir.mkdir(parents=True, exist_ok=True)
	video_path.parent.mkdir(parents=True, exist_ok=True)

	cv2_module = importlib.import_module('cv2')
	writer = None
	fourcc = cv2_module.VideoWriter_fourcc(*'mp4v')

	with torch.no_grad():
		for index, sample in enumerate(data_loader, start=1):
			sequence_batch = sample['data']
			inputs = prepare_input(sequence_batch.to(device), expects_sequence)

			outputs = forward_pass(model, inputs)
			predictions = outputs.argmax(dim=1)
			mask = predictions[0].cpu().numpy().astype(bool)

			frames = sequence_batch[0]
			frame_tensor = frames[-1] if frames.ndim == 4 else frames
			frame_rgb = tensor_to_rgb(frame_tensor)
			blended = overlay_mask(frame_rgb, mask)

			if writer is None:
				height, width = blended.shape[:2]
				writer = cv2_module.VideoWriter(str(video_path), fourcc, fps, (width, height))

			save_image(save_dir / f"{index:04d}_frame.jpg", frame_rgb)
			save_image(save_dir / f"{index:04d}_overlay.jpg", blended)

			writer.write(cv2_module.cvtColor(blended, cv2_module.COLOR_RGB2BGR))

	if writer is not None:
		writer.release()
		print(f"Video saved to: {video_path}")
		return video_path

	print("No frames processed; video writer was not initialised.")
	return None


def load_pretrained(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
	if checkpoint_path and checkpoint_path.exists():
		print(f"Loading pretrained weights from {checkpoint_path}")
		weights = torch.load(checkpoint_path, map_location=device)
		model_dict = model.state_dict()
		filtered_weights = {k: v for k, v in weights.items() if k in model_dict}
		model_dict.update(filtered_weights)
		model.load_state_dict(model_dict)
	else:
		print("No pretrained checkpoint found. Inference will run with randomly initialised weights.")


def main():
	args = args_setting()
	torch.manual_seed(args.seed)

	use_cuda = args.cuda and torch.cuda.is_available()
	device = torch.device('cuda' if use_cuda else 'cpu')
	expects_sequence = args.model in config.SEQUENCE_MODELS

	transforms_module = importlib.import_module('torchvision.transforms')
	transform = transforms_module.Compose([transforms_module.ToTensor()])

	dataset = RoadSequenceDataset(
		file_path=args.test_index,
		transforms=transform,
		sequence_length=args.sequence_length,
		include_label=False,
	)
	test_loader = torch.utils.data.DataLoader(
		dataset,
		batch_size=args.test_batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=use_cuda,
	)

	model = generate_model(args)
	load_pretrained(model, args.pretrained, device)

	video_path = args.output_video or (args.save_dir / 'lane.mp4')
	output_result(model, test_loader, device, args.save_dir, Path(video_path), expects_sequence)


if __name__ == '__main__':
	main()
import config

from config import args_setting

