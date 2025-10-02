import argparse
import importlib
from pathlib import Path
from typing import Dict


REAL_HEIGHTS_INCHES: Dict[str, float] = {
    'bicycle': 26.04,
    'car': 59.08,
    'motorcycle': 47.24,
    'bus': 125.98,
    'truck': 137.79,
}

DEFAULT_FOCAL_LENGTH_MM = 200.0


def detect_distance(real_height_inches: float, box_height: float, focal_length_mm: float) -> float:
    real_height_cm = real_height_inches * 2.54
    box_height = max(box_height, 1.0)
    return (real_height_cm * focal_length_mm) / box_height / 100.0


def measure_distance(frame, results, focal_length_mm: float, class_heights: Dict[str, float], cv2_module) -> None:
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            real_height = class_heights.get(class_name)
            if real_height is None:
                continue

            box_height = max((y2 - y1), 1.0)
            distance_m = detect_distance(real_height, box_height, focal_length_mm)

            if distance_m > 3:
                color = (0, 255, 0)
            elif 1 < distance_m <= 3:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            cv2_module.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2_module.putText(
                frame,
                f'{class_name}: {distance_m:.2f} m',
                (int(x1), int(y1 - 10)),
                cv2_module.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )


def distance(video_input_path: Path, video_output_path: Path, weights_path: Path = Path('./yolov8n.pt'), focal_length_mm: float = DEFAULT_FOCAL_LENGTH_MM) -> Path:
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
    writer = cv2_module.VideoWriter(str(video_output_path), fourcc, 30.0, (frame_width, frame_height))

    try:
        while cap.isOpened():
            status, frame = cap.read()
            if not status:
                break

            results = model.predict(source=frame, save=False)
            measure_distance(frame, results, focal_length_mm, REAL_HEIGHTS_INCHES, cv2_module)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
        cv2_module.destroyAllWindows()

    print(f'Distance estimation video saved to {video_output_path}')
    return video_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Estimate object distances from a video using YOLO detections.')
    parser.add_argument('--input', type=Path, required=True, help='Path to the input video file.')
    parser.add_argument('--output', type=Path, required=True, help='Path to the output video file.')
    parser.add_argument('--weights', type=Path, default=Path('./yolov8n.pt'), help='YOLO weights file to use for detection.')
    parser.add_argument('--focal-length', type=float, default=DEFAULT_FOCAL_LENGTH_MM, help='Camera focal length in millimetres.')
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    distance(arguments.input, arguments.output, arguments.weights, arguments.focal_length)

