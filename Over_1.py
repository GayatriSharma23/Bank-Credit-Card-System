import argparse
import logging
import os
import json

import cv2
import torch
from mivolo.data.data_reader import InputType, get_all_files, get_input_type
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging

_logger = logging.getLogger("inference")


def get_local_video_info(vid_uri):
    cap = cv2.VideoCapture(vid_uri)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source {vid_uri}")
    res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return res, fps


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Inference")
    parser.add_argument("--input", type=str, required=True, help="Video file or image folder/file")
    parser.add_argument("--output", type=str, required=True, help="Folder for output results")
    parser.add_argument("--detector-weights", type=str, required=True, help="Detector weights (YOLOv8).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MiVOLO checkpoint")
    parser.add_argument("--draw", action="store_true", default=False, help="Draw results on output video/images")
    parser.add_argument("--device", default="cuda", type=str, help="Device to use (cpu/cuda)")
    return parser


def main():
    parser = get_parser()
    setup_default_logging()
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.output, exist_ok=True)

    predictor = Predictor(args, verbose=True)
    input_type = get_input_type(args.input)

    # JSON results
    results = []

    if input_type == InputType.Video or input_type == InputType.VideoStream:
        if not args.draw:
            raise ValueError("Video processing only supported with --draw flag")

        bname = os.path.splitext(os.path.basename(args.input))[0]
        outfilename = os.path.join(args.output, f"out_{bname}.avi")
        res, fps = get_local_video_info(args.input)

        if args.draw:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(outfilename, fourcc, fps, res)
            _logger.info(f"Saving result to {outfilename}..")

        frame_idx = 0
        for (detected_objects_history, frame) in predictor.recognize_video(args.input):
            # Extract age directly from detected_objects_history (matches overlay)
            age_for_frame = None
            if isinstance(detected_objects_history, dict) and len(detected_objects_history) > 0:
                # Take first face's age
                for face_id, face_data in detected_objects_history.items():
                    if face_data and len(face_data) > 0:
                        age_for_frame = face_data[0][0]  # tuple (age, gender)
                        break

            results.append({
                "frame_idx": frame_idx,
                "age": age_for_frame
            })

            if args.draw:
                out.write(frame)

            frame_idx += 1

        # Save JSON
        json_path = os.path.join(args.output, f"{bname}_predictions.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        _logger.info(f"Saved per-frame predictions to {json_path}")

    elif input_type == InputType.Image:
        image_files = get_all_files(args.input) if os.path.isdir(args.input) else [args.input]

        for img_p in image_files:
            img = cv2.imread(img_p)
            detected_objects, out_im = predictor.recognize(img)

            if args.draw:
                bname = os.path.splitext(os.path.basename(img_p))[0]
                filename = os.path.join(args.output, f"out_{bname}.jpg")
                cv2.imwrite(filename, out_im)
                _logger.info(f"Saved result to {filename}")


if __name__ == "__main__":
    main()
