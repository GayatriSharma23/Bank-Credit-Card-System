

import cv2
import json
import numpy as np

# Global variable for age stabilization
AGE_BUFFER = []
BUFFER_SIZE = 5  # number of frames to smooth age

def smooth_age(new_age):
    """
    Smooth age prediction using running average
    """
    AGE_BUFFER.append(new_age)
    if len(AGE_BUFFER) > BUFFER_SIZE:
        AGE_BUFFER.pop(0)
    return sum(AGE_BUFFER) / len(AGE_BUFFER)

def draw_health_box(frame, data, frame_idx, text_color=(255,255,255)):
    """
    Draw fixed professional health overlay box
    """
    frame_copy = frame.copy()
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

    # Age & Gender (from JSON overlay or first face if multiple)
    person_data = overlay_data.get("overlay", [])
    if person_
        age = person_data[0].get("age", "")
        gender = person_data[0].get("gender", "").capitalize()
    else:
        age, gender = "", ""

    # Optional: stabilize age
    if age:
        age = smooth_age(age)

    # Translucent box for metrics
    box_x, box_y, box_w, box_h = 20, 20, 350, 190
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x+box_w, box_y+box_h), (0,0,0), -1)
    frame_copy = cv2.addWeighted(overlay, 0.45, frame_copy, 0.55, 0)

    # Text lines
    lines = [
        f"Frame: {frame_idx}",
        f"Age: {int(age)} Y | {gender}" if age else "",
        f"BPM: {bpm:.1f} beats/min" if bpm else "",
        f"HRV SDNN: {hrv:.1f} ms" if hrv else "",
        f"BP: {systolic}/{diastolic} mmHg" if systolic and diastolic else "",
        f"BMI: {bmi:.1f} kg/m²" if bmi else "",
        f"Stress: {stress}" if stress else "",
        f"Emotion: {emotion}" if emotion else ""
    ]

    y_offset = box_y + 25
    for line in lines:
        if line.strip():
            cv2.putText(frame_copy, line, (box_x+12, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)
            y_offset += 23

    return frame_copy

def draw_face_box_with_label(frame, bbox, age=None, gender=None, box_color=(0,255,0), text_color=(0,0,0)):
    """
    Draw a bounding box around the face with age and gender label above it
    bbox is a tuple (x, y, w, h)
    """
    x, y, w, h = bbox
    # Draw rectangle box around face
    cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

    text = ""
    if age is not None:
        text += f"Age: {int(age)} "
    if gender:
        text += f"| {gender}"

    if text:
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Draw filled rectangle for text background
        cv2.rectangle(frame, (x, y - 20), (x + text_width + 10, y), box_color, -1)
        cv2.putText(frame, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    return frame

def main(video_path, json_path, output_path):
    # Load JSON
    with open(json_path, "r") as f:
        json_data = json.load(f)

    # Convert list to dict if needed
    if isinstance(json_data, list):
        data_dict = {str(item["frame_idx"]): item for item in json_data}
    else:
        # If keys are strings of numbers
        data_dict = {str(int(k)-1): v for k,v in json_data.items()}

    # Initialize face detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

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

        # Detect faces in grayscale frame for bbox
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

        # Get JSON data for this frame
        frame_data = data_dict.get(str(frame_idx), {})

        # Draw the professional health overlay box on the frame
        frame = draw_health_box(frame, data_dict, frame_idx)

        # Draw bounding boxes with age and gender for each detected face
        # If your JSON 'overlay' contains multiple entries per frame, associate by order or heuristic
        overlay_faces = frame_data.get("overlay", [])
        for i, (x, y, w, h) in enumerate(faces):
            age = None
            gender = None
            if i < len(overlay_faces):
                age = overlay_faces[i].get("age")
                gender = overlay_faces[i].get("gender")
            frame = draw_face_box_with_label(frame, (x, y, w, h), age, gender)

        out.write(frame)

        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}/{total_frames}")

        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Output saved to: {output_path}")

if __name__ == "__main__":
    video_path = "input_video.mp4"            # Path to your video
    json_path = "consolidated_output.json"    # JSON with age, gender, health metrics etc.
    output_path = "video_with_overlay.mp4"    # Output video path
    main(video_path, json_path, output_path)
