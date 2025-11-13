# training.py
import os
import json
import pathlib
import cv2
import numpy as np
import streamlit as st
import tempfile
# ------------------- constants / paths -------------------
PROFILES_DIR = pathlib.Path.cwd() / "profiles"
ACTIVE_PATH  = PROFILES_DIR / "ACTIVE.json"

DEFAULT_PARAMS = {
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

# ------------------- small helpers -------------------
def _safe_load_json(p: pathlib.Path):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _list_profiles():
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for f in PROFILES_DIR.glob("*.json"):
        if f.name.upper() == "ACTIVE.json":
            continue
        d = _safe_load_json(f)
        if isinstance(d, dict):
            items.append((f.stem, f))
    items.sort()
    return items

def _save_profile(name: str, data: dict) -> pathlib.Path:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = PROFILES_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path

def _set_active(data_or_path):
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    if isinstance(data_or_path, dict):
        payload = data_or_path
    else:
        payload = _safe_load_json(data_or_path) or DEFAULT_PARAMS
    with open(ACTIVE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def _probe_video_meta(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    dur = float(n) / float(fps) if fps > 0 else 0.0
    return dict(fps=fps, W=w, H=h, N=n, dur=dur)

def _open_video_writer(path: str, fps: float, size_wh: tuple[int, int]):
    """Try multiple codecs, return (writer, actual_path)."""
    w, h = size_wh
    base, _ = os.path.splitext(path)
    for tag, ext in [("avc1", ".mp4"), ("mp4v", ".mp4"), ("XVID", ".avi"), ("MJPG", ".avi")]:
        try_path = base + ext
        fourcc = cv2.VideoWriter_fourcc(*tag)
        vw = cv2.VideoWriter(try_path, fourcc, float(fps), (int(w), int(h)))
        if vw is not None and vw.isOpened():
            return vw, try_path
    return None, None

def _build_previews(src_path: str, color_out: str, gray_out: str, P: dict):
    """Write color overlay + grayscale/tracing videos using params P (dict)."""
    meta = _probe_video_meta(src_path)
    if meta is None:
        raise RuntimeError("Could not open video")

    fps, w, h, dur = meta["fps"], meta["W"], meta["H"], meta["dur"]

    ws = max(0.0, float(P["window_start_s"]))
    we = min(float(dur), float(P["window_end_s"]))
    if we <= ws:
        we = min(dur, ws + 8.0)

    s_frame = int(ws * fps)
    e_frame = int(we * fps)
    n_frames = max(0, e_frame - s_frame)
    if n_frames == 0:
        raise RuntimeError("No frames in the selected preview window.")

    vw_color, color_path = _open_video_writer(color_out, fps, (w, h))
    vw_gray,  gray_path  = _open_video_writer(gray_out,  fps, (w, h))
    if vw_color is None or vw_gray is None:
        raise RuntimeError("No video writer available (MP4/AVI).")

    cap = cv2.VideoCapture(src_path)
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame):
        for _ in range(s_frame):
            if not cap.read()[0]:
                break

    ok, prev = cap.read()
    if not ok:
        cap.release(); vw_color.release(); vw_gray.release()
        raise RuntimeError("Could not read first frame at window start.")

    blur = int(P["blur_ksize"])
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (blur, blur), 0)

    prog = st.progress(0.0, text="Rendering previews…")
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (blur, blur), 0)

        diff = cv2.absdiff(gray_blur, prev_gray)

        # range threshold (dtype-safe)
        lower = np.full_like(diff, int(P["thresh_min"]), dtype=diff.dtype)
        upper = np.full_like(diff, int(P["thresh_max"]), dtype=diff.dtype)
        mask  = cv2.inRange(diff, lower, upper)

        if int(P["dilate_iter"]) > 0:
            mask = cv2.dilate(mask, None, iterations=int(P["dilate_iter"]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color_vis = frame.copy()
        gray_vis  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        for c in contours:
            area = cv2.contourArea(c)
            if area < int(P["area_min"]) or area > int(P["area_max"]):
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            if not (int(P["min_box_w"]) <= ww <= int(P["max_box_w"]) and int(P["min_box_h"]) <= hh <= int(P["max_box_h"])):
                continue
            cv2.rectangle(color_vis, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
            cv2.rectangle(gray_vis,  (x, y), (x + ww, y + hh), (255, 255, 255), 2)

        overlay = color_vis.copy()
        overlay[mask > 0] = (0, 0, 255)
        color_vis = cv2.addWeighted(color_vis, 0.8, overlay, 0.2, 0)

        vw_color.write(color_vis)
        vw_gray.write(gray_vis)

        prev_gray = gray_blur
        if (i + 1) % 10 == 0 or (i + 1) == n_frames:
            prog.progress((i + 1) / float(n_frames))

    cap.release(); vw_color.release(); vw_gray.release()
    prog.progress(1.0, text="Done")
    return color_path, gray_path

# ------------------- main UI -------------------
def render():
    st.subheader("Training — Tune & Save Detection Profile")

    # 1) Upload a short reference video
    st.markdown("**1) Upload a sample video**")
    vid = st.file_uploader("Video (mp4/mov/m4v/avi)", type=["mp4", "mov", "m4v", "avi"])
    video_path = None
    if vid is not None:
        suffix = os.path.splitext(vid.name)[1]
        tmp = pathlib.Path(tempfile.gettempdir()) / f"_train_video{suffix}"

        with open(tmp, "wb") as f:
            f.write(vid.read())
        video_path = str(tmp)

    # 2) Choose or start from an existing profile
    st.markdown("**2) Start from a profile**")
    colp1, colp2 = st.columns([2, 1])
    profiles = _list_profiles()
    names = ["<Defaults>"] + [n for n, _ in profiles]
    pick = colp1.selectbox("Profile", names, index=0)

    if pick == "<Defaults>":
        P = DEFAULT_PARAMS.copy()
    else:
        P = _safe_load_json(next(p for n, p in profiles if n == pick)) or DEFAULT_PARAMS.copy()

    # 3) Adjust parameters
    st.markdown("**3) Adjust parameters**")
    meta = _probe_video_meta(video_path) if video_path else None
    dur = meta["dur"] if meta else 60.0  # arbitrary max if no meta yet
    c0, c1, c2 = st.columns([1.3, 1, 1])

    P["window_start_s"], P["window_end_s"] = c0.slider(
        "Preview window (seconds)",
        0.0, float(max(0.5, dur)), (float(P["window_start_s"]), float(P["window_end_s"])),
        step=0.5
    )

    P["thresh_min"], P["thresh_max"] = c1.slider(
        "Motion threshold range", 0, 255, (int(P["thresh_min"]), int(P["thresh_max"])), 1
    )

    P["dilate_iter"] = c2.slider("Dilate iterations", 0, 5, int(P["dilate_iter"]), 1)
    P["blur_ksize"]  = c2.selectbox("Blur kernel", [3,5,7,9,11], index=[3,5,7,9,11].index(int(P["blur_ksize"])))

    P["area_min"], P["area_max"] = c1.slider(
        "Area range (px²)", 10, 300000, (int(P["area_min"]), int(P["area_max"])), 10
    )

    with st.popover("Advanced box size limits (px)"):
        d1, d2 = st.columns(2)
        P["min_box_w"], P["max_box_w"] = d1.slider("Box width", 0, 2000, (int(P["min_box_w"]), int(P["max_box_w"])), 1)
        P["min_box_h"], P["max_box_h"] = d2.slider("Box height", 0, 2000, (int(P["min_box_h"]), int(P["max_box_h"])), 1)

    # 4) Preview with current parameters
    st.markdown("**4) Preview**")
    build = st.button("Build previews", disabled=(video_path is None))
    if build and video_path:
        out_dir = pathlib.Path.cwd()
        color_out = out_dir / "_train_preview_color"
        gray_out  = out_dir / "_train_preview_gray"
        try:
            color_path, gray_path = _build_previews(video_path, str(color_out), str(gray_out), P)
            st.success("Previews ready")
            st.markdown("**Movement overlay (full color)**")
            st.video(str(color_path))
            st.download_button("Download color preview",
                               data=open(color_path, "rb").read(),
                               file_name=os.path.basename(color_path),
                               mime="video/mp4" if str(color_path).endswith(".mp4") else "video/avi")
            st.markdown("**Grayscale + tracing**")
            st.video(str(gray_path))
            st.download_button("Download grayscale preview",
                               data=open(gray_path, "rb").read(),
                               file_name=os.path.basename(gray_path),
                               mime="video/mp4" if str(gray_path).endswith(".mp4") else "video/avi")
        except Exception as e:
            st.error(f"Preview failed: {e}")

    # 5) Save profile + set active
    st.markdown("**5) Save profile**")
    coln1, coln2, coln3 = st.columns([2, 1, 1])
    name = coln1.text_input("Profile name", value=pick if pick != "<Defaults>" else "")
    save_btn = coln2.button("Save profile", disabled=len(name.strip()) == 0)
    set_active_btn = coln3.button("Save & set as ACTIVE", disabled=len(name.strip()) == 0)

    if save_btn or set_active_btn:
        try:
            path = _save_profile(name.strip(), P)
            st.success(f"Saved: {path.name}")
            if set_active_btn:
                _set_active(path)
                st.toast(f"Set ACTIVE: {path.name}")
        except Exception as e:
            st.error(f"Save failed: {e}")

    # Helper: quick view of ACTIVE
    with st.expander("Current ACTIVE profile (used by Detect)", expanded=False):
        active = _safe_load_json(ACTIVE_PATH) or DEFAULT_PARAMS
        st.json(active)
