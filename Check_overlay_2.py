
import cv2
import json
import numpy as np
import os

def draw_overlay(frame, data, frame_idx, text_color=(255, 255, 255)):
    """
    Draws a professional translucent health overlay on the given video frame.
    """
    frame_copy = frame.copy()
    overlay_data = data.get(str(frame_idx))
    if overlay_data is None:
        return frame_copy

    # Extract parameters
    bpm = overlay_data.get("BPM")
    hrv = overlay_data.get("HRV_SDNN")
    stress = overlay_data.get("Stress", "")
    bp = overlay_data.get("BP", {})
    bmi = overlay_data.get("BMI")
    emotion = overlay_data.get("Emotion", "")
    systolic = bp.get("Systolic")
    diastolic = bp.get("Diastolic")

    # Age & Gender
    person_data = overlay_data.get("overlay", [])
    if person_data:
        age = person_data[0].get("age", "")
        gender = person_data[0].get("gender", "").capitalize()
    else:
        age, gender = "", ""

    # Create translucent background box
    overlay = frame.copy()
    box_x, box_y, box_w, box_h = 20, 20, 310, 170
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    frame_copy = cv2.addWeighted(overlay, 0.45, frame_copy, 0.55, 0)

    # Text lines with proper units
    lines = [
        f"Age: {age:.1f} Y | {gender}" if age else "",
        f"BPM: {bpm:.1f} beats/min" if bpm else "",
        f"HRV SDNN: {hrv:.1f} ms" if hrv else "",
        f"BP: {systolic:.1f}/{diastolic:.1f} mmHg" if systolic and diastolic else "",
        f"BMI: {bmi:.1f} kg/m²" if bmi else "",
        f"Stress: {stress}",
        f"Emotion: {emotion}"
    ]

    # Draw text
    y_offset = box_y + 25
    for line in lines:
        if line.strip():
            cv2.putText(frame_copy, line, (box_x + 12, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)
            y_offset += 23

    return frame_copy


def main(video_path, json_path, output_path):
    """
    Main function to overlay health info on a video based on consolidated JSON.
    """
    # Load JSON
    with open(json_path, "r") as f:
        json_data = json.load(f)

    # If list, convert to dict by frame index
    if isinstance(json_data, list):
        data_dict = {str(item["frame_idx"]): item for item in json_data}
    else:
        data_dict = json_data

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {video_path}")
    print(f"Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply overlay
        frame_with_overlay = draw_overlay(frame, data_dict, frame_idx)
        out.write(frame_with_overlay)

        # Show progress
        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}")

        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Output saved to: {output_path}")


if __name__ == "__main__":
    # ---- USER INPUT SECTION ----
    video_path = "input_video.mp4"           # your original video
    json_path = "consolidated_output.json"   # your JSON with frame-wise data
    output_path = "video_with_overlay.mp4"   # output video with overlay

    main(video_path, json_path, output_path)
