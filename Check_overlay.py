
import json

# ------------------------
# 1. Load JSONs
# ------------------------
with open("age_gender.json") as f:
    age_json = json.load(f)

with open("pyvhr_chunk1.json") as f:
    pyvhr_chunk1 = json.load(f)
with open("pyvhr_chunk2.json") as f:
    pyvhr_chunk2 = json.load(f)
pyvhr_chunks = pyvhr_chunk1 + pyvhr_chunk2

with open("bp_chunk1.json") as f:
    bp_chunk1 = json.load(f)
with open("bp_chunk2.json") as f:
    bp_chunk2 = json.load(f)
bp_chunks = bp_chunk1 + bp_chunk2

# Static values
BMI = 27.27
Emotion = "Neutral"

# ------------------------
# 2. Determine total frames
# ------------------------
total_frames = len(age_json)  # assuming age_json covers all frames
consolidated = []

# ------------------------
# 3. Map chunk values to each frame
# ------------------------
def get_chunk_value(frame_idx, chunks):
    """Return chunk values for the given frame index"""
    for chunk in chunks:
        start, end = chunk["Frame_range"]
        if start <= frame_idx < end:
            return chunk
    return None

# ------------------------
# 4. Build consolidated JSON frame by frame
# ------------------------
for frame_idx in range(total_frames):
    frame_data = {}
    frame_data["frame_idx"] = frame_idx
    frame_data["Overlay"] = age_json[frame_idx]["Overlay"]  # list of persons

    # pyVHR
    pyvhr = get_chunk_value(frame_idx, pyvhr_chunks)
    if pyvhr:
        frame_data["BPM"] = pyvhr.get("Bpm", None)
        frame_data["HRV_SDNN"] = pyvhr.get("Hrv_sdd", None)
        frame_data["SPO2"] = pyvhr.get("spo2", None)
        frame_data["Stress"] = pyvhr.get("stress_level", None)
    else:
        frame_data["BPM"] = None
        frame_data["HRV_SDNN"] = None
        frame_data["SPO2"] = None
        frame_data["Stress"] = None

    # BP
    bp = get_chunk_value(frame_idx, bp_chunks)
    if bp:
        frame_data["BP"] = {"Systolic": bp.get("sys_bp"), "Diastolic": bp.get("Dia_bp")}
    else:
        frame_data["BP"] = {"Systolic": None, "Diastolic": None}

    # Static values
    frame_data["BMI"] = BMI
    frame_data["Emotion"] = Emotion

    consolidated.append(frame_data)

# ------------------------
# 5. Save consolidated JSON
# ------------------------
with open("consolidated.json", "w") as f:
    json.dump(consolidated, f, indent=2)

print(f"✅ Consolidated JSON saved with {total_frames} frames.")
