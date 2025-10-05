
import cv2
import json
import numpy as np

def draw_health_box(frame, data, frame_idx, text_color=(255, 255, 255)):
    """
    Draw fixed professional health overlay box
    """
    frame_copy = frame.copy()
    
    # Align frame index (adjust if JSON is 1-indexed)
    overlay_data = data.get(str(frame_idx))
    if overlay_data is None:
        return frame_copy

    # Extract metrics
    bpm = overlay_data.get("BPM")
    hrv = overlay_data.get("HRV_SDNN")
    stress = overlay_data.get("Stress", "")
    bp = overlay_data.get("BP", {})
    bmi = overlay_data.get("BMI")
    emotion = overlay_data.get("Emotion", "")
    systolic = bp.get("Systolic")
    diastolic = bp.get("Diastolic")

    person_data = overlay_data.get("overlay", [])
    if person_data:
        age = person_data[0].get("age", "")
        gender = person_data[0].get("gender", "").capitalize()
    else:
        age, gender = "", ""

    # Translucent box
    box_x, box_y, box_w, box_h = 20, 20, 350, 190
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    frame_copy = cv2.addWeighted(overlay, 0.45, frame_copy, 0.55, 0)

    # Prepare lines
    lines = [
        f"Frame: {frame_idx}",
        f"Age: {age} Y | {gender}" if age else "",
        f"BPM: {bpm:.1f} beats/min" if bpm else "",
        f"HRV SDNN: {hrv:.1f} ms" if hrv else "",
        f"BP: {systolic}/{diastolic} mmHg" if systolic and diastolic else "",
        f"BMI: {bmi:.1f} kg/m²" if bmi else "",
        f"Stress: {stress}" if stress else "",
        f"Emotion: {emotion}" if emotion else ""
    ]

    # Draw text
    y_offset = box_y + 25
    for line in lines:
        if line.strip():
            cv2.putText(frame_copy, line, (box_x + 12, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)
            y_offset += 23

    return frame_copy


def draw_face_box(frame, face_coords, age=None, gender=None, text_color=(0, 0, 0)):
    """
    Draw dynamic face box with age & gender
    """
    x, y, w, h = face_coords
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    text = ""
    if age:
        text += f"Age: {age} "
    if gender:
        text += f"| {gender}"

    if text:
        # Text background
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - 20), (x + text_width + 10, y), (0, 255, 0), -1)
        cv2.putText(frame, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def main(video_path, json_path, output_path):
    # Load JSON
    with open(json_path, "r") as f:
        json_data = json.load(f)

    # Convert JSON to dict if it’s a list
    if isinstance(json_data, list):
        data_dict = {str(item["frame_idx"]): item for item in json_data}
    else:
        # Adjust keys if needed (1-indexed → 0-indexed)
        data_dict = {str(int(k)-1): v for k, v in json_data.items()}

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total_frames} frames...")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Overlay health metrics
        frame = draw_health_box(frame, data_dict, frame_idx)

        # Overlay face boxes if present
        face_data = data_dict.get(str(frame_idx), {}).get("face", [])
        for face in face_data:
            x, y, w, h = face.get("bbox", (0, 0, 0, 0))
            age = face.get("age")
            gender = face.get("gender")
            frame = draw_face_box(frame, (x, y, w, h), age, gender)

        out.write(frame)

        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}/{total_frames}")

        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Output saved to: {output_path}")


if __name__ == "__main__":
    video_path = "input_video.mp4"           # Original video
    json_path = "consolidated_output.json"   # JSON with per-frame health & face data
    output_path = "video_with_overlay.mp4"   # Output video
    main(video_path, json_path, output_path)
