import argparse
import importlib
import os
from pathlib import Path
from typing import List


def extract_frames(video_path: Path, output_folder: Path) -> int:
    cv2_module = importlib.import_module('cv2')

    output_folder = output_folder.expanduser()
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2_module.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = output_folder / f"frame_{frame_count:04d}.jpg"
        cv2_module.imwrite(str(frame_filename), frame)
        print(f'Saved: {frame_filename}')
        frame_count += 1

    cap.release()
    print(f'Total frames extracted: {frame_count}')
    return frame_count


def _gather_image_paths(image_folder: Path, extension: str) -> List[Path]:
    image_folder = image_folder.expanduser()
    collected: List[Path] = []
    for subdir, _, files in os.walk(image_folder):
        jpg_files = sorted(
            f for f in files if f.lower().endswith(extension.lower())
        )
        if jpg_files:
            collected.extend(Path(subdir) / f for f in jpg_files)
    return collected


def generate_index(image_folder: Path, output_file: Path, window_size: int = 6, stride: int = 1, extension: str = '.jpg') -> int:
    image_paths = _gather_image_paths(image_folder, extension)
    total_written = 0

    output_file = output_file.expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as handle:
        for start in range(0, len(image_paths) - window_size + 1, stride):
            group = image_paths[start:start + window_size]
            if len(group) < window_size:
                continue
            line = " ".join(str(path) for path in group)
            handle.write(line + "\n")
            total_written += 1

    print(f'Index saved to {output_file} ({total_written} sequences).')
    return total_written


def generate_index_line(image_folder: Path, output_file: Path, extension: str = '.jpg') -> int:
    image_paths = _gather_image_paths(image_folder, extension)

    output_file = output_file.expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as handle:
        for path in image_paths:
            handle.write(f"{path}\n")

    print(f'Index saved to {output_file} ({len(image_paths)} entries).')
    return len(image_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Video utilities for frame extraction and index generation.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    extract_parser = subparsers.add_parser('extract', help='Extract frames from a video file.')
    extract_parser.add_argument('--video', type=Path, required=True, help='Input video file path.')
    extract_parser.add_argument('--output', type=Path, required=True, help='Directory where extracted frames will be stored.')

    index_parser = subparsers.add_parser('index', help='Generate rolling window index entries from image frames.')
    index_parser.add_argument('--images', type=Path, required=True, help='Folder containing extracted frames.')
    index_parser.add_argument('--output', type=Path, required=True, help='Destination index file path.')
    index_parser.add_argument('--window-size', type=int, default=6, help='Number of frames per index line.')
    index_parser.add_argument('--stride', type=int, default=1, help='Stride between consecutive sequences.')
    index_parser.add_argument('--extension', type=str, default='.jpg', help='Image file extension filter (e.g. .jpg, .png).')

    line_parser = subparsers.add_parser('index-line', help='Generate a flat index file with one path per line.')
    line_parser.add_argument('--images', type=Path, required=True, help='Folder containing extracted frames.')
    line_parser.add_argument('--output', type=Path, required=True, help='Destination index file path.')
    line_parser.add_argument('--extension', type=str, default='.jpg', help='Image file extension filter.')

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == 'extract':
        extract_frames(args.video, args.output)
    elif args.command == 'index':
        generate_index(args.images, args.output, args.window_size, args.stride, args.extension)
    elif args.command == 'index-line':
        generate_index_line(args.images, args.output, args.extension)


if __name__ == '__main__':
    main()