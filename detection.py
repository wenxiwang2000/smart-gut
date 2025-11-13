import cv2
import numpy as np
import pandas as pd
import math
try:
    import streamlit as st
except ImportError:
    st = None

# Default BIO detection profile (original thresholds, unchanged)
DEFAULT_BIO_PROFILE = {
    "window_start_s": 0,
    "window_end_s": 8,
    "thresh_min": 5,
    "thresh_max": 30,
    "area_min": 1000,
    "area_max": 8000,
    "dilate_iter": 2,
    "blur_ksize": 7,
    "min_box_w": 20,
    "max_box_w": 500,
    "min_box_h": 20,
    "max_box_h": 500
}

# --- Helper Functions for Gut Tracking (HSV Mode) ---

def _resize_keep_w(gray, target_w):
    """Resize grayscale image to target width, preserving aspect ratio."""
    H, W = gray.shape[:2]
    if W == target_w:
        return gray
    scale = target_w / float(W)
    new_h = max(1, int(round(H * scale)))
    return cv2.resize(gray, (target_w, new_h), interpolation=cv2.INTER_AREA)

def video_to_motion_stack(video_path, target_w=128):
    """Convert video to a stack of frame-to-frame motion differences (grayscale). 
    Returns (motion_stack, fps, orig_size, stack_size)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    # Read and downsample frames
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = _resize_keep_w(gray, target_w)
        frames.append(gray)
    cap.release()
    if len(frames) < 2:
        raise RuntimeError("Need at least 2 frames to compute motion.")
    # Compute absolute difference between consecutive frames
    Hs, Ws = frames[0].shape[:2]
    diffs = []
    prev = frames[0]
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], prev)
        diffs.append(diff.astype(np.float32))
        prev = frames[i]
    motion_stack = np.stack(diffs, axis=0)  # shape: [T, Hs, Ws]
    return motion_stack, float(fps), (W0, H0), (Ws, Hs)

def stack_to_motility(stack, fps):
    """Compute the global motility signal (mean abs diff per frame) and time array."""
    T = stack.shape[0]
    mot = stack.reshape(T, -1).mean(axis=1)
    t = np.arange(T, dtype=np.float32) / float(max(fps, 1e-6))
    return t, mot

def density_from_stack(stack):
    """Compute per-pixel average motion over time (motion density map)."""
    return stack.mean(axis=0)

def hsv_color_mask(frame_bgr, h_min=90, h_max=140, s_min=40, v_min=40):
    """Apply HSV threshold to isolate blue regions in a BGR frame."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_max, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # Reduce noise with median blur and morphological opening (3x3 kernel)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    return mask

def largest_blue_boxes_per_frame(video_path, hsv=(90, 140, 40, 40)):
    """
    For each frame of the video, find the largest blue contour (if any).
    Returns a DataFrame with columns [time_s, x, y, w, h] in original video pixels.
    """
    h_min, h_max, s_min, v_min = hsv
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    boxes = []
    prev_box = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask = hsv_color_mask(frame, h_min, h_max, s_min, v_min)
        # Find contours of thresholded mask; use the largest contour if available
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)  # largest area contour:contentReference[oaicite:5]{index=5}
            x, y, w, h = cv2.boundingRect(c)
            prev_box = (x, y, w, h)
        else:
            # No contour found: fallback to last known box (to maintain continuity)
            if prev_box is not None:
                x, y, w, h = prev_box
            else:
                x, y, w, h = 0, 0, 32, 32  # default small box if nothing detected yet
        boxes.append((frame_idx / fps, x, y, w, h))
        frame_idx += 1
    cap.release()
    return pd.DataFrame(boxes, columns=["time_s", "x", "y", "w", "h"])

def write_traced_video(input_path, boxes_df, output_path, stack_size=None):
    """
    Create a video with bounding boxes drawn on the original video frames.
    If boxes_df is from downsampled stack, provide stack_size for coordinate scaling.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    # Scaling factors if coordinates are in a downsampled space
    sx = sy = 1.0
    if stack_size:
        Ws, Hs = stack_size
        sx = W / float(Ws)
        sy = H / float(Hs)
    # Write each frame with the drawn bounding box and inset ROI preview
    for _, row in boxes_df.iterrows():
        ret, frame = cap.read()
        if not ret:
            break
        # Compute scaled coordinates
        x = int(row["x"] * sx);    y = int(row["y"] * sy)
        w = int(row["w"] * sx);    h = int(row["h"] * sy)
        # Clamp values to frame dimensions
        x = max(0, min(W-1, x));   y = max(0, min(H-1, y))
        w = max(1, min(W - x, w)); h = max(1, min(H - y, h))
        # Draw green rectangle on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Draw inset preview in the top-left corner
        roi = frame[y:y+h, x:x+w]
        if roi.size > 0:
            thumb = cv2.resize(roi, (int(W*0.22), int(H*0.22)))
            frame[0:thumb.shape[0], 0:thumb.shape[1]] = thumb
        writer.write(frame)
    # If video has more frames than boxes (e.g., tracking stopped early), 
    # continue drawing the last known box until video ends
    if not boxes_df.empty:
        last = boxes_df.iloc[-1]
        lx = int(last["x"] * sx); ly = int(last["y"] * sy)
        lw = int(last["w"] * sx); lh = int(last["h"] * sy)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.rectangle(frame, (lx, ly), (lx+lw, ly+lh), (0, 255, 0), 2)
            writer.write(frame)
    cap.release()
    writer.release()
    return output_path

def contractions_from_stack_roi(stack, t, fps, roi_xywh, motion_thr=10.0, max_regions_per_frame=9, min_duration_s=0.5):
    """
    Detect contractions within the given ROI (defined in the motion stack space) using the original biological rule.
    Returns a DataFrame with columns ["contraction_time", "contraction_area"].
    """
    x, y, w, h = map(int, roi_xywh)
    T = stack.shape[0]
    moving_pixels = np.zeros(T, dtype=np.int32)
    region_counts = np.zeros(T, dtype=np.int32)
    valid = np.zeros(T, dtype=bool)
    for i in range(T):
        roi_frame = stack[i, y:y+h, x:x+w]
        if roi_frame.size == 0:
            moving_pixels[i] = 0
            region_counts[i] = 0
            valid[i] = False
            continue
        # Threshold motion intensity in ROI
        mask = (roi_frame > float(motion_thr)).astype(np.uint8)
        moving_pixels[i] = int(mask.sum())
        # Count connected regions of motion in ROI
        num_labels, _ = cv2.connectedComponents(mask, connectivity=8)
        regions = max(0, num_labels - 1)
        region_counts[i] = regions
        # A frame is valid if between 1 and max_regions moving regions (inclusive)
        valid[i] = (regions >= 1 and regions <= max_regions_per_frame)
    # Group contiguous valid frames that meet minimum duration into events
    min_frames = int(math.ceil(min_duration_s * fps))
    events = []
    current_start = None
    for i in range(T):
        if valid[i]:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None and (i - current_start) >= min_frames:
                events.append((current_start, i-1))
            current_start = None
    if current_start is not None and (T - current_start) >= min_frames:
        events.append((current_start, T-1))
    # Prepare output events with peak area in each event window
    contractions = []
    for (s_idx, e_idx) in events:
        if e_idx < s_idx:
            e_idx = s_idx
        # Peak moving pixel count in this event
        peak_area = int(moving_pixels[s_idx:e_idx+1].max()) if e_idx >= s_idx else int(moving_pixels[s_idx])
        contractions.append({"contraction_time": float(t[s_idx]), "contraction_area": peak_area})
    return pd.DataFrame(contractions, columns=["contraction_time", "contraction_area"])

def run_hsv_mode(video_path, output_dir):
    """
    High-level function to run the HSV blue-tracking pipeline on a given video.
    Saves outputs to <output_dir>/hsv_mode/ and returns (boxes_csv_path, contractions_csv_path, traced_video_path).
    """
    import os
    # Create a subfolder for HSV mode outputs
    hsv_out_dir = os.path.join(output_dir, "hsv_mode")
    os.makedirs(hsv_out_dir, exist_ok=True)
    # Base name for output files (use video filename without extension)
    vid_name = os.path.splitext(os.path.basename(video_path))[0]
    out_base = os.path.join(hsv_out_dir, vid_name)
    # 1. Build motion stack (for contraction analysis) using default downsample width
    stack, fps, orig_size, stack_size = video_to_motion_stack(video_path, target_w=128)
    W0, H0 = orig_size
    Ws, Hs = stack_size
    T = stack.shape[0]
    t = np.arange(T, dtype=np.float32) / float(max(fps, 1e-6))
    # 2. Track largest blue region in each frame
    df_boxes_video = largest_blue_boxes_per_frame(video_path, hsv=(90, 140, 40, 40))
    if df_boxes_video.empty:
        raise RuntimeError("No blue contour detected in any frame.")
    # 3. Convert the first frame's bounding box to stack coordinates for ROI
    df_boxes_stack = df_boxes_video.copy()
    df_boxes_stack[["x", "w"]] = (df_boxes_stack[["x", "w"]] * (Ws / float(W0))).round().astype(int)
    df_boxes_stack[["y", "h"]] = (df_boxes_stack[["y", "h"]] * (Hs / float(H0))).round().astype(int)
    roi_xywh = (
        int(df_boxes_stack.iloc[0]["x"]), int(df_boxes_stack.iloc[0]["y"]),
        int(df_boxes_stack.iloc[0]["w"]), int(df_boxes_stack.iloc[0]["h"])
    )
    # 4. Detect contraction events inside this ROI using the motion stack
    contr_df = contractions_from_stack_roi(stack, t, fps, roi_xywh=roi_xywh,
                                           motion_thr=10.0, max_regions_per_frame=9, min_duration_s=0.5)
    # 5. Save the traced video with bounding boxes and inset previews
    traced_video_path = f"{out_base}_gut_traced.mp4"
    write_traced_video(video_path, df_boxes_video, traced_video_path, stack_size=None)
    # 6. Save outputs to CSV files
    boxes_csv_path = f"{out_base}_gut_tracing_boxes.csv"
    contr_csv_path = f"{out_base}_contractions.csv"
    df_boxes_video.to_csv(boxes_csv_path, index=False)
    contr_df.to_csv(contr_csv_path, index=False)
    return boxes_csv_path, contr_csv_path, traced_video_path

# (Original detection functions like detect_events, etc., would remain here if any – not shown for brevity)

def _summarize_contractions(time_s, events_df, motion_stack):
    """
    Summarize contraction events from detected events (original method).
    Returns a dict with count and list of events (time_s and area_px_mean).
    """
    T = motion_stack.shape[0]
    active_px = (motion_stack > 0).reshape(T, -1).sum(axis=1) if motion_stack.ndim == 3 \
                else (motion_stack[..., 0] > 0).reshape(T, -1).sum(axis=1)
    events = []
    for _, row in events_df.iterrows():
        # Determine start and end indices for the event
        if "start_idx" in row and "end_idx" in row:
            s_idx = int(row["start_idx"]); e_idx = int(row["end_idx"])
        else:
            s = float(row.get("start_s", 0)); e = float(row.get("end_s", 0))
            s_idx = int(np.clip(np.searchsorted(time_s, s, side="left"), 0, T-1))
            e_idx = int(np.clip(np.searchsorted(time_s, e, side="right") - 1, 0, T-1))
        if e_idx < s_idx:
            e_idx = s_idx
        mid_idx = (s_idx + e_idx) // 2
        # Compute mean active pixel area during the event
        area_mean = float(active_px[s_idx:e_idx+1].mean()) if e_idx >= s_idx else float(active_px[mid_idx])
        events.append({"time_s": float(time_s[mid_idx]), "area_px_mean": area_mean})
    return {"count": len(events), "events": events}

def render():
    """Streamlit UI rendering for original detection (for legacy use)."""
    st.subheader("Detect — Video ➜ CSV")
    # ... original UI components ...
    st.write("Original detection panel UI (unchanged).")
