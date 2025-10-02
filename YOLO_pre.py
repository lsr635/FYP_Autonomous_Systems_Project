import argparse
import importlib
from pathlib import Path


def yolo_pre(video_input_path: Path, video_output_path: Path, weights_path: Path = Path('./yolov8n.pt'), fps: float = 30.0) -> Path:
    cv2_module = importlib.import_module('cv2')
    yolo_cls = getattr(importlib.import_module('ultralytics'), 'YOLO')

    model = yolo_cls(str(weights_path))

    cap = cv2_module.VideoCapture(str(video_input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_input_path}")

    frame_width = int(cap.get(cv2_module.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2_module.CAP_PROP_FRAME_HEIGHT))

    video_output_path = video_output_path.expanduser()
    video_output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2_module.VideoWriter_fourcc(*'mp4v')
    writer = cv2_module.VideoWriter(str(video_output_path), fourcc, fps, (frame_width, frame_height))

    try:
        while cap.isOpened():
            status, frame = cap.read()
            if not status:
                break

            result = model.predict(source=frame, save=False)[0]
            annotated = result.plot()
            writer.write(annotated)
    finally:
        cap.release()
        writer.release()
        cv2_module.destroyAllWindows()

    print(f'YOLO inference finished. Output saved to {video_output_path}')
    return video_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run YOLOv8 inference and save annotated video output.')
    parser.add_argument('--input', type=Path, required=True, help='Path to the input video file.')
    parser.add_argument('--output', type=Path, required=True, help='Path to the output annotated video file.')
    parser.add_argument('--weights', type=Path, default=Path('./yolov8n.pt'), help='YOLOv8 weights file to use.')
    parser.add_argument('--fps', type=float, default=30.0, help='Frames per second for the output video.')
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    yolo_pre(arguments.input, arguments.output, arguments.weights, arguments.fps)