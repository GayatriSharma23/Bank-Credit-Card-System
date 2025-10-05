
import cv2
import json
import numpy as np

# Load JSON
with open("consolidated.json", "r") as f:
    data = json.load(f)

# Convert to dict by frame_idx
overlay_dict = {item['frame_idx']: item for item in data}

# Input/output video
video_path = "input_video.mp4"
output_path = "output_overlay.mp4"

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Overlay function with semi-transparent panel
def draw_overlay(frame, overlay):
    # Panel size
    panel_width = 300
    panel_height = 250
    x, y = 10, 10  # Top-left corner

    # Draw semi-transparent rectangle
    overlay_panel = frame.copy()
    cv2.rectangle(overlay_panel, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay_panel, alpha, frame, 1 - alpha, 0)

    # Start text position inside panel
    text_x, text_y = x + 10, y + 30
    line_height = 28

    # People info
    for person in overlay.get("overlay", []):
        text = f"Person {person['id']} | Age: {person['age']:.1f} | Gender: {person['gender'].capitalize()}"
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        text_y += line_height

    # Top-level metrics
    bpm = overlay.get("BPM", None)
    hrv = overlay.get("HRV_SDNN", None)
    stress = overlay.get("Stress", None)

    if bpm is not None:
        cv2.putText(frame, f"BPM: {bpm:.1f}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        text_y += line_height
    if hrv is not None:
        cv2.putText(frame, f"HRV SDNN: {hrv:.1f}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
        text_y += line_height
    if stress:
        cv2.putText(frame, f"Stress: {stress}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        text_y += line_height

    # BP / BMI / Emotion
    bp = overlay.get("BP", {})
    systolic = bp.get("Systolic", None)
    diastolic = bp.get("Diastolic", None)
    bmi = bp.get("BMI", None)
    emotion = bp.get("Emotion", None)

    if systolic and diastolic:
        cv2.putText(frame, f"BP: {systolic:.1f}/{diastolic:.1f}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        text_y += line_height
    if bmi:
        cv2.putText(frame, f"BMI: {bmi:.1f}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2, cv2.LINE_AA)
        text_y += line_height
    if emotion:
        cv2.putText(frame, f"Emotion: {emotion}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2, cv2.LINE_AA)
        text_y += line_height

    return frame

# Process video frame by frame
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx in overlay_dict:
        frame = draw_overlay(frame, overlay_dict[frame_idx])

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print("Overlay video saved successfully!")
