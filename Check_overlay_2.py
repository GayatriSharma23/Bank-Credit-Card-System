
import cv2
import json
import numpy as np

# Load consolidated JSON
with open("consolidated.json", "r") as f:
    data = json.load(f)

# Sort by frame index
chunks = sorted(data, key=lambda x: x['frame_idx'])

# Interpolate numeric values across chunks
def interpolate_chunks(chunks):
    frame_values = {}
    for i in range(len(chunks) - 1):
        c1, c2 = chunks[i], chunks[i + 1]
        start, end = c1["frame_idx"], c2["frame_idx"]
        for f in range(start, end):
            ratio = (f - start) / (end - start)
            frame_values[f] = {
                "BPM": c1["BPM"] + ratio * (c2["BPM"] - c1["BPM"]),
                "HRV_SDNN": c1["HRV_SDNN"] + ratio * (c2["HRV_SDNN"] - c1["HRV_SDNN"]),
                "BP_Syst": c1["BP"]["Systolic"] + ratio * (c2["BP"]["Systolic"] - c1["BP"]["Systolic"]),
                "BP_Dia": c1["BP"]["Diastolic"] + ratio * (c2["BP"]["Diastolic"] - c1["BP"]["Diastolic"]),
                "BMI": c1.get("BMI", c1["BP"].get("BMI", 0)),
                "Stress": c1.get("Stress", "Unknown"),
                "Emotion": c1.get("Emotion", c1["BP"].get("Emotion", "Neutral"))
            }
    # Last chunk — repeat last values till end_frame (if present)
    last = chunks[-1]
    for f in range(last["frame_idx"], last.get("end_frame", last["frame_idx"] + 1)):
        frame_values[f] = {
            "BPM": last["BPM"],
            "HRV_SDNN": last["HRV_SDNN"],
            "BP_Syst": last["BP"]["Systolic"],
            "BP_Dia": last["BP"]["Diastolic"],
            "BMI": last.get("BMI", last["BP"].get("BMI", 0)),
            "Stress": last.get("Stress", "Unknown"),
            "Emotion": last.get("Emotion", last["BP"].get("Emotion", "Neutral"))
        }
    return frame_values

frame_values = interpolate_chunks(chunks)

# Input/Output video paths
video_path = "age_gender_output.mp4"
output_path = "final_video_with_health.mp4"

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Function to draw translucent health panel
def draw_health_panel(frame, health_data):
    x, y = 10, 10
    panel_width, panel_height = 300, 230
    alpha = 0.6

    overlay_panel = frame.copy()
    cv2.rectangle(overlay_panel, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay_panel, alpha, frame, 1 - alpha, 0)

    text_x, text_y = x + 10, y + 30
    lh = 28

    cv2.putText(frame, f"BPM: {health_data['BPM']:.1f}", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    text_y += lh
    cv2.putText(frame, f"HRV SDNN: {health_data['HRV_SDNN']:.1f}", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    text_y += lh
    cv2.putText(frame, f"BP: {health_data['BP_Syst']:.1f}/{health_data['BP_Dia']:.1f}", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    text_y += lh
    cv2.putText(frame, f"BMI: {health_data['BMI']:.1f}", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
    text_y += lh
    cv2.putText(frame, f"Stress: {health_data['Stress']}", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    text_y += lh
    cv2.putText(frame, f"Emotion: {health_data['Emotion']}", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)

    return frame

# --- AGE DISPLAY CONTROL (update once every 3 frames) ---
age_display_interval = 3
last_age_values = {}

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Find current chunk overlay
    overlay_data = None
    for chunk in chunks:
        if frame_idx >= chunk["frame_idx"]:
            overlay_data = chunk
        else:
            break

    # Age/Gender box display control
    if overlay_data:
        persons = overlay_data.get("overlay", [])
        for person in persons:
            pid = person["id"]
            # Update only every 3 frames
            if frame_idx % age_display_interval == 0 or pid not in last_age_values:
                last_age_values[pid] = (person["age"], person["gender"])
            age, gender = last_age_values[pid]

            # Draw age/gender text (top of face box)
            cv2.putText(frame, f"ID {pid}: {int(age)}y, {gender}",
                        (50, 50 + 40 * pid),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Health data panel (interpolated)
    health_data = frame_values.get(frame_idx)
    if health_data:
        frame = draw_health_panel(frame, health_data)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Final video with dynamic health + stable age overlay saved!")
