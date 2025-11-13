# adjusting.py
import os
import json
import cv2
import math
import tempfile
import pathlib
import numpy as np
import streamlit as st
from collections import namedtuple

Meta = namedtuple("Meta", "W H fps frames duration")

# ---------- Shared defaults (match Detect panel) ----------
DEFAULT_PARAMS = {
    "window_start_s": 0.0,    # preview-only; kept for consistency
    "window_end_s":   8.0,    # preview-only
    "thresh_min":     12,
    "thresh_max":     40,
    "area_min":       300,
    "area_max":       20000,
    "dilate_iter":    2,
    "blur_ksize":     7,
    "min_box_w":      0,
    "max_box_w":      2000,
    "min_box_h":      0,
    "max_box_h":      2000,
}

PROFILES_DIR = pathlib.Path.cwd() / "profiles"
ACTIVE_PATH  = PROFILES_DIR / "ACTIVE.json"


# ---------- Small helpers ----------
def _safe_load_json(p: pathlib.Path):
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            return d if isinstance(d, dict) else None
    except Exception:
        return None

def _load_active():
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    d = _safe_load_json(ACTIVE_PATH)
    if not isinstance(d, dict):
        d = DEFAULT_PARAMS.copy()
    # ensure keys exist
    for k, v in DEFAULT_PARAMS.items():
        d.setdefault(k, v)
    return d

def _save_active(d: dict):
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

def _get_meta(path: str) -> Meta:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = (float(N) / float(fps)) if fps > 0 else 0.0
    return Meta(W=W, H=H, fps=float(fps), frames=int(N), duration=float(duration))

def _frame_idx_from_time(t: float, fps: float, nframes: int) -> int:
    idx = int(round(max(0.0, min(t, 1e9)) * fps))
    return max(0, min(idx, max(0, nframes - 1)))

def _clamp_roi(x, y, w, h, W, H):
    x = int(max(0, min(x, W - 1)))
    y = int(max(0, min(y, H - 1)))
    w = int(max(1, min(w, W - x)))
    h = int(max(1, min(h, H - y)))
    return x, y, w, h

def _grab_frame(path: str, idx: int, resize_w: int | None = None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    # try seeking; if not, manual skip
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, idx):
        for _ in range(idx):
            ok, _ = cap.read()
            if not ok:
                cap.release()
                return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    if resize_w is not None and resize_w > 0:
        h = int(round(frame.shape[0] * (resize_w / frame.shape[1])))
        frame = cv2.resize(frame, (resize_w, h), interpolation=cv2.INTER_AREA)
    return frame

def _open_video_writer(path_no_ext: str, fps: float, size_wh: tuple[int, int]):
    w, h = size_wh
    attempts = [
        ("avc1", ".mp4", "H.264 (avc1)"),
        ("mp4v", ".mp4", "MPEG-4 (mp4v)"),
        ("XVID", ".avi", "XVID (AVI)"),
        ("MJPG", ".avi", "Motion-JPEG (AVI)"),
    ]
    base = os.path.splitext(path_no_ext)[0]
    for tag, ext, _name in attempts:
        try_path = base + ext
        fourcc = cv2.VideoWriter_fourcc(*tag)
        vw = cv2.VideoWriter(try_path, fourcc, float(fps), (int(w), int(h)))
        if vw is not None and vw.isOpened():
            return vw, try_path
    return None, None

def _export_trim_crop(src_path: str, out_path: str, t0: float, t1: float, roi_xywh: tuple[int, int, int, int]):
    meta = _get_meta(src_path)
    i0 = _frame_idx_from_time(max(0.0, min(t0, meta.duration)), meta.fps, meta.frames)
    i1 = _frame_idx_from_time(max(0.0, min(t1, meta.duration)), meta.fps, meta.frames)
    if i1 <= i0:
        i1 = min(meta.frames - 1, i0 + int(max(1, meta.fps * 0.25)))  # ensure at least a blip

    x, y, w, h = _clamp_roi(*roi_xywh, meta.W, meta.H)

    # Writer with fallbacks
    vw, final_path = _open_video_writer(out_path, meta.fps, (w, h))
    if vw is None:
        raise RuntimeError("Could not open any video writer (MP4/AVI).")

    cap = cv2.VideoCapture(src_path)
    # robust seek
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, i0):
        for _ in range(i0):
            ok, _ = cap.read()
            if not ok:
                break

    total = max(1, i1 - i0 + 1)
    prog = st.progress(0.0, text="Exporting trimmed clip…")

    wrote = 0
    for k in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        roi = frame[y:y+h, x:x+w]
        vw.write(roi)
        wrote += 1
        if (k + 1) % 10 == 0 or (k + 1) == total:
            prog.progress(min((k + 1) / float(total), 1.0))

    cap.release()
    vw.release()
    prog.progress(1.0, text="Done")

    if wrote == 0 or not os.path.exists(final_path):
        raise RuntimeError("No frames written. Try different time window or ROI.")
    return final_path

def _save_settings_json(path: str, t0: float, t1: float, roi_xywh: tuple[int, int, int, int], meta: Meta, det_params: dict):
    out = {
        "trim_seconds": {"start": float(t0), "end": float(t1)},
        "roi_xywh": {"x": int(roi_xywh[0]), "y": int(roi_xywh[1]), "w": int(roi_xywh[2]), "h": int(roi_xywh[3])},
        "video_meta": {"width": meta.W, "height": meta.H, "fps": meta.fps, "frames": meta.frames, "duration": meta.duration},
        "detection_params": {k: (int(v) if isinstance(v, (int, np.integer)) or k.endswith("_iter") else v)
                             for k, v in det_params.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


# ---------- UI ----------
def render():
    st.subheader("Adjust — Trim, Crop & Tune Detection")

    # 1) Load video
    st.markdown("**1) Load video**")
    up = st.file_uploader("Select video file", type=["mp4", "mov", "m4v", "avi"], key="adj_video")
    if up is None:
        st.info("Upload a video to begin.")
        return

    # Persist uploaded file to a temp path for OpenCV processing
    if "adj_tmp_video" not in st.session_state or st.session_state.get("adj_tmp_name") != up.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as tmp:
            tmp.write(up.read())
            st.session_state["adj_tmp_video"] = tmp.name
            st.session_state["adj_tmp_name"] = up.name
    vpath = st.session_state["adj_tmp_video"]

    # 2) Probe video metadata
    try:
        meta = _get_meta(vpath)
    except Exception as e:
        st.error(f"Could not open video: {e}")
        return
    st.caption(f"Resolution: **{meta.W}×{meta.H}**, FPS: **{meta.fps:.2f}**, Frames: **{meta.frames}**, Duration: **{meta.duration:.2f}s**")

    # 3) Select trim range
    st.markdown("**2) Select trim range (seconds)**")
    t0, t1 = st.slider(
        "Select time window",
        min_value=0.0, max_value=max(0.1, float(meta.duration)),
        value=(0.0, min(8.0, float(meta.duration))),
        step=0.1
    )

    # 4) Crop region (ROI) settings
    with st.expander("3) Crop region (optional)", expanded=False):
        colR1, colR2, colR3, colR4 = st.columns(4)
        rx = colR1.number_input("x (left)", min_value=0, max_value=max(1, meta.W) - 1, value=0, step=1)
        ry = colR2.number_input("y (top)",  min_value=0, max_value=max(1, meta.H) - 1, value=0, step=1)
        rw = colR3.number_input("w (width)",  min_value=1, max_value=max(1, meta.W), value=meta.W, step=1)
        rh = colR4.number_input("h (height)", min_value=1, max_value=max(1, meta.H), value=meta.H, step=1)
        x, y, w, h = _clamp_roi(rx, ry, rw, rh, meta.W, meta.H)

        frame_t = st.slider("Preview frame at time (s)", 0.0, float(meta.duration), value=min(t0, float(meta.duration)), step=0.1)
        idx = _frame_idx_from_time(frame_t, meta.fps, meta.frames)
        frame_preview = _grab_frame(vpath, idx, resize_w=min(960, meta.W))
        if frame_preview is not None:
            shown_w, shown_h = frame_preview.shape[1], frame_preview.shape[0]
            sx = int(round(x * (shown_w / meta.W)));  sy = int(round(y * (shown_h / meta.H)))
            sw = int(round(w * (shown_w / meta.W)));  sh = int(round(h * (shown_h / meta.H)))
            vis = frame_preview.copy()
            cv2.rectangle(vis, (sx, sy), (sx + sw, sy + sh), (30, 220, 30), 2)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"Frame at {frame_t:.2f}s (with ROI)", use_container_width=True)
        else:
            st.caption("Unable to fetch frame at that time.")

    # 5) Detection parameters (same schema as Detect panel)
    st.markdown("**4) Detection parameters**")
    active = _load_active()

    c1, c2, c3 = st.columns(3)
    with c1:
        active["thresh_min"] = st.number_input("thresh_min", 0, 255, int(active["thresh_min"]), 1)
        active["area_min"]   = st.number_input("area_min",   0, 1_000_000, int(active["area_min"]), 10)
        active["min_box_w"]  = st.number_input("min_box_w",  0, 5000, int(active["min_box_w"]), 1)
    with c2:
        active["thresh_max"] = st.number_input("thresh_max", 0, 255, int(active["thresh_max"]), 1)
        active["area_max"]   = st.number_input("area_max",   0, 1_000_000, int(active["area_max"]), 10)
        active["min_box_h"]  = st.number_input("min_box_h",  0, 5000, int(active["min_box_h"]), 1)
    with c3:
        active["blur_ksize"]  = st.number_input("blur_ksize (odd)", 1, 99, int(active["blur_ksize"]), 2)
        if active["blur_ksize"] % 2 == 0:
            active["blur_ksize"] += 1
        active["dilate_iter"] = st.number_input("dilate_iter", 0, 20, int(active["dilate_iter"]), 1)
        active["max_box_w"]   = st.number_input("max_box_w",  1, 5000, int(active["max_box_w"]), 1)
        active["max_box_h"]   = st.number_input("max_box_h",  1, 5000, int(active["max_box_h"]), 1)

    col_save1, col_save2 = st.columns([0.5, 0.5])
    with col_save1:
        if st.button("💾 Save as ACTIVE profile (used by Detect)"):
            try:
                _save_active(active)
                st.success("Saved to profiles/ACTIVE.json")
            except Exception as e:
                st.error(f"Could not save ACTIVE profile: {e}")
    with col_save2:
        if st.button("↩️ Reset to defaults"):
            try:
                _save_active(DEFAULT_PARAMS.copy())
                st.success("ACTIVE reset to defaults.")
            except Exception as e:
                st.error(f"Reset failed: {e}")

    # 6) Preview snapshot & trimmed preview
    st.markdown("**5) Preview**")
    # Snapshot in the middle of selected window, with ROI box
    mid_idx = _frame_idx_from_time((t0 + t1) * 0.5, meta.fps, meta.frames)
    frame_mid = _grab_frame(vpath, mid_idx, resize_w=min(960, meta.W))
    if frame_mid is not None:
        shown_w, shown_h = frame_mid.shape[1], frame_mid.shape[0]
        sx = int(round(x * (shown_w / meta.W)));  sy = int(round(y * (shown_h / meta.H)))
        sw = int(round(w * (shown_w / meta.W)));  sh = int(round(h * (shown_h / meta.H)))
        vis = frame_mid.copy()
        cv2.rectangle(vis, (sx, sy), (sx + sw, sy + sh), (30, 220, 30), 2)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Middle of selected time window", use_container_width=True)
    else:
        st.caption("Unable to render snapshot at this time index.")

    if st.button("▶ Preview trimmed segment"):
        try:
            preview_path_noext = "preview_segment"
            final_preview = _export_trim_crop(vpath, preview_path_noext, t0, t1, (x, y, w, h))
            st.video(final_preview)
        except Exception as e:
            st.error(f"Preview failed: {e}")

    # 7) Export
    st.markdown("**6) Export**")
    out_name = st.text_input("Output filename (no need to add extension)", value="trimmed_cropped")
    if st.button("💾 Export video + settings", type="primary"):
        try:
            out_base = out_name.strip() or "trimmed_cropped"
            video_out = _export_trim_crop(vpath, out_base, t0, t1, (x, y, w, h))
            settings_json = f"{os.path.splitext(out_base)[0]}_settings.json"
            _save_settings_json(settings_json, t0, t1, (x, y, w, h), meta, active)
            st.success(f"Saved video: **{os.path.basename(video_out)}** & **{os.path.basename(settings_json)}**")

            if os.path.exists(video_out):
                st.download_button("Download MP4/AVI", open(video_out, "rb").read(),
                                   file_name=os.path.basename(video_out),
                                   mime="video/mp4" if video_out.lower().endswith(".mp4") else "video/avi")
            if os.path.exists(settings_json):
                st.download_button("Download settings.json", open(settings_json, "rb").read(),
                                   file_name=os.path.basename(settings_json), mime="application/json")
        except Exception as e:
            st.error(f"Export failed: {e}")
