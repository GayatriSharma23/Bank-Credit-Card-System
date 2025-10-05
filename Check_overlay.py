
from pyVHR.signals.video import Video
from pyVHR.methods.pos import POS
import numpy as np
import heartpy as hp
import os
import json
from pyVHR.methods.spo2_utils import spo2_estimation


def flatten_signal(signal_list):
    return np.concatenate([s for s in signal_list if s is not None and len(s) > 0], axis=0).flatten()


def analyze_video_chunks(video_path, chunks=2, output_dir="chunk_results"):
    results = []
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Load full video and detect faces once
    video = Video(video_path)
    video.getCroppedFaces(detector='mtcnn', extractor='skvideo')

    if video.faces is None or video.faces.size == 0:
        print("❌ Error: No faces detected in the video.")
        return None

    print("Cropped faces shape:", video.faces.shape)
    video.setMask(typeROI='skin_fix', skinThresh_fix=[30, 50])

    total_frames = len(video.faces)
    frames_per_chunk = total_frames // chunks

    for i in range(chunks):
        start = i * frames_per_chunk
        end = (i + 1) * frames_per_chunk if i < chunks - 1 else total_frames

        print(f"\n=== Processing chunk {i+1}/{chunks} | Frames {start}:{end} ===")

        chunk_faces = video.faces[start:end]
        if chunk_faces is None or chunk_faces.size == 0:
            print(f"[WARN] No faces found in chunk {i+1}, skipping.")
            continue

        # Create a temporary chunk video object
        chunk_video = Video(video_path)
        chunk_video.faces = chunk_faces
        chunk_video.frameRate = 30  # <-- explicitly set fps
        chunk_video.setMask(typeROI='skin_fix', skinThresh_fix=[30, 50])

        chunk_result = {
            "chunk_id": i + 1,
            "frame_range": [int(start), int(end)],
            "bpm": None,
            "hrv_sdnn": None,
            "brpm": None,
            "stress_level": None,
            "spo2": None
        }

        try:
            params = {
                "video": chunk_video,
                "verb": 0,
                "ROImask": "skin_adapt",
                "skinAdapt": 0.2,
                "detrMethod": "tarvainen",
                "zeroMeanSTDnorm": 1
            }

            m = POS(**params)
            bpmES, timesES, bvpEs, fs, red, green, blue = m.runOffline(**params)

            arr_BVP = np.concatenate(bvpEs, axis=1).flatten()

            # Save BVP & FS
            bvp_file = os.path.join(output_dir, f"{base_name}_bvp_chunk_{i+1}.npy")
            fs_file = os.path.join(output_dir, f"{base_name}_fs_chunk_{i+1}.npy")
            np.save(bvp_file, arr_BVP)
            np.save(fs_file, np.array([fs]))

            # Process with heartpy
            wd, measures = hp.process(arr_BVP, sample_rate=fs, high_precision=True, clean_rr=True, high_precision_fs=50.0)

            # Simple stress estimation
            def estimate_stress_basic_hrv(measures):
                rmssd = measures.get("rmssd", 0)
                sdnn = measures.get("sdnn", 0)
                pnn50 = measures.get("pnn50", 0)
                score = 0
                if rmssd < 20: score += 2
                elif rmssd < 40: score += 1
                if sdnn < 30: score += 2
                elif sdnn < 50: score += 1
                if pnn50 < 0.1: score += 2
                elif pnn50 < 0.3: score += 1
                return "Low" if score <= 2 else "Moderate" if score <= 4 else "High"

            stress_score = estimate_stress_basic_hrv(measures)

            chunk_result.update({
                "bpm": float(measures.get("bpm", 0)),
                "hrv_sdnn": float(measures.get("sdnn", 0)),
                "brpm": float(measures.get("breathingrate", 0)) * 60,
                "stress_level": stress_score,
                "bvp_file": bvp_file,
                "fs_file": fs_file
            })

            # SPO2 estimation
            green_signal = flatten_signal(green)
            red_signal = flatten_signal(red)
            timestamps = np.arange(len(green_signal)) / fs

            try:
                spo2_values, hr_spo2 = spo2_estimation(
                    ppg_green_940=green_signal,
                    ppg_red_600=red_signal,
                    timestamps=timestamps,
                    fps=fs
                )
                spo2_mean = np.nanmean(spo2_values) * 100
                chunk_result["spo2"] = spo2_mean
            except Exception as e:
                print(f"[WARN] SPO2 estimation failed in chunk {i+1}: {e}")
                chunk_result["spo2"] = None

        except Exception as e:
            print(f"[ERROR] Chunk {i+1} failed: {e}")

        # Always save JSON per chunk — even if something failed
        json_file = os.path.join(output_dir, f"{base_name}_chunk_{i+1}.json")
        with open(json_file, "w") as f:
            json.dump(chunk_result, f, indent=2)

        print(f"✅ Saved results for chunk {i+1} → {json_file}")
        results.append(chunk_result)

    # Save combined file too
    combined_json = os.path.join(output_dir, f"{base_name}_all_chunks.json")
    with open(combined_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Combined results saved → {combined_json}")

    return results
