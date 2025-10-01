
import cv2
import numpy as np
from pyVHR.signals.video import Video
from pyVHR.methods.pos import POS
import heartpy as hp

def overlay_text(frame, text, pos=(30, 50)):
    return cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2, cv2.LINE_AA)

def process_chunk(video_path, start_frame, end_frame):
    """Run pyVHR on a video sub-segment and return HR."""
    video = Video(video_path)               # init with file
    video.getCroppedFaces(detector='mtcnn', extractor='skvideo')

    # Restrict to chunk frames only
    video.faces = video.faces[start_frame:end_frame]

    video.setMask(typeROI='skin_fix', skinThresh_fix=[30, 50])
    params = {
        "video": video,
        "verb": 0,
        "ROImask": "skin_adapt",
        "skinAdapt": 0.2,
        "detrMethod": "tarvainen",
        "zeroMeanSTDnorm": 1
    }
    m = POS(**params)
    bpmES, timesES, bvpEs, fs, red, green, blue = m.runOffline(**params)

    arr_BVP = np.concatenate(bvpEs, axis=1).flatten()
    wd, measures = hp.process(arr_BVP, sample_rate=fs, high_precision=True, clean_rr=True)

    return measures.get("bpm", np.nan)

def analyze_video_chunks(input_path, output_path, chunk_seconds=10):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    chunk_size = fps * chunk_seconds
    frame_idx = 0
    last_hr = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Every chunk boundary → compute HR
        if frame_idx % chunk_size == 0 and frame_idx + chunk_size <= total_frames:
            try:
                last_hr = process_chunk(input_path, frame_idx, frame_idx + chunk_size)
                print(f"[INFO] HR at {frame_idx/fps:.1f}s: {last_hr:.1f} bpm")
            except Exception as e:
                print("[WARN] pyVHR failed:", e)
                last_hr = None

        # Overlay last HR
        if last_hr is not None:
            frame = overlay_text(frame, f"HR: {last_hr:.1f} bpm", pos=(30, 50))
        else:
            frame = overlay_text(frame, "HR: Processing...", pos=(30, 50))

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
