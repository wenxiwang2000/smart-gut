import json
import numpy as np

# Load the training data JSON produced by neurobuilder
with open("training_data.json", "r") as f:
    data = json.load(f)

videos = data.get("videos", [])
if not videos:
    print("No training data found in JSON.")
    exit()

# Detection parameters (matching defaults from detection.py or training assumptions)
base_z = 2.0             # base z-threshold for event detection
min_sep = 0.6            # minimum separation between events (seconds)

# Function to compute events from motion time series of a video
def detect_events_from_series(motion_series: np.ndarray, fps: float):
    # Compute mean and std for z-score thresholding
    mu = motion_series.mean()
    sigma = motion_series.std(ddof=0)
    thresh_value = mu + base_z * sigma
    # Identify frames above threshold
    above_thresh = motion_series > thresh_value
    events = []
    i = 0
    while i < len(above_thresh):
        if above_thresh[i]:
            start_idx = i
            # group consecutive above-threshold frames as one event
            while i < len(above_thresh) and above_thresh[i]:
                i += 1
            end_idx = i - 1
            events.append((start_idx, end_idx))
        i += 1
    # Merge events that are too close together (closer than min_sep)
    merged_events = []
    for (start, end) in events:
        if not merged_events:
            merged_events.append([start, end])
        else:
            prev_start, prev_end = merged_events[-1]
            prev_end_time = prev_end / fps
            curr_start_time = start / fps
            if curr_start_time - prev_end_time < min_sep:
                # Merge with previous event
                merged_events[-1][1] = end
            else:
                merged_events.append([start, end])
    return merged_events, thresh_value

# Use training data to compute average event rate (for potential threshold calibration)
total_events = 0
total_duration_minutes = 0.0
for vid in videos:
    fps = vid.get("fps", 30.0)
    motion_series = np.array(vid.get("motion", []), dtype=float)
    events, thresh_val = detect_events_from_series(motion_series, fps)
    event_count = len(events)
    duration_sec = len(motion_series) / float(fps) if fps > 0 else 0
    duration_min = duration_sec / 60.0
    rate = (event_count / duration_min) if duration_min > 0 else 0.0
    total_events += event_count
    total_duration_minutes += duration_min
    print(f"Video: {vid.get('name')} (Label: {vid.get('label')})")
    print(f" - Detected {event_count} events in {duration_sec:.1f}s (rate: {rate:.2f} events/min) using z-threshold={base_z:.1f}")
    if event_count > 0:
        # Print event start times (approximate) for verification
        event_times = [start_idx/float(fps) for start_idx, end_idx in events]
        print(f" - Event start times (sec): " + ", ".join(f"{t:.2f}" for t in event_times))
    else:
        print(" - No events detected.")
    print("")

# Compute average event rate across all videos
if total_duration_minutes > 0:
    avg_rate = total_events / total_duration_minutes
    print(f"Average event rate across training videos: {avg_rate:.2f} events per minute")
else:
    print("No valid video duration to compute average event rate.")

# Note: In a real scenario, one could adjust the detection threshold (z-value) to calibrate
# so that the event rate in new videos matches the training data rate. This simple script
# uses a base threshold of 2.0 for demonstration and prints the detected events for verification.