import cv2
import numpy as np
from pyVHR.signals.video import Video
from pyVHR.methods.pos import POS
import heartpy as hp

def overlay_text(frame, text, pos=(30, 50)):
    """Overlay text on video frame."""
    return cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2, cv2.LINE_AA)

def process_chunk(frames, fps):
    """Run pyVHR on 10s chunk of frames and return HR."""
    video = Video()
    video.faces = np.array(frames)   # cropped faces assumed
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
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    chunk_size = fps * chunk_seconds
    buffer = []
    last_hr = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        buffer.append(frame)
        frame_idx += 1

        # Every 10s → process chunk with pyVHR
        if frame_idx % chunk_size == 0:
            try:
                last_hr = process_chunk(buffer, fps)
            except Exception as e:
                print("[WARN] pyVHR failed:", e)
                last_hr = None
            buffer = []  # reset buffer

        # Overlay HR (last computed value) on video
        if last_hr is not None:
            text = f"HR: {last_hr:.1f} bpm"
            frame = overlay_text(frame, text, pos=(30, 50))
        else:
            frame = overlay_text(frame, "HR: Processing...", pos=(30, 50))

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_hr_overlay.avi"
    analyze_video_chunks(input_video, output_video, chunk_seconds=10)
