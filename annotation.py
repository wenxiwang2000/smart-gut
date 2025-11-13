# annotation.py
import os
import pathlib
import cv2
import json
import base64
import pandas as pd
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import tempfile
import time

import streamlit as st
# Minimal fallback: run JS via Streamlit's HTML component.
# NOTE: This does not support returning values back to Python.
try:
    from streamlit_javascript import st_javascript  # real component, returns values
except ModuleNotFoundError:
    from streamlit.components.v1 import html as _st_html

    def st_javascript(js_code: str, *, height: int = 0):
        """
        Fallback shim: injects and runs JS but does not return a value.
        Matches the call signature loosely enough for most 'fire-and-forget' uses.
        """
        _st_html(f"<script>{js_code}</script>", height=height)
        return None

# ------------------- constants / paths -------------------
ANNOTATIONS_DIR = pathlib.Path.cwd() / "annotations"
ANNOTATIONS_DIR.mkdir(exist_ok=True)

CLIP_ROOT = pathlib.Path.home() / "SmartGutClips"
CLIP_ROOT.mkdir(exist_ok=True)

# ------------------- helpers (unchanged from your code) -------------------
def _extract_frame(video_path: str, frame_idx: int) -> Optional[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

def _save_annotation(video_name: str, annotations: list, fmt: str = "csv") -> pathlib.Path:
    base = ANNOTATIONS_DIR / pathlib.Path(video_name).stem
    if fmt == "json":
        out = base.with_suffix(".json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2)
    else:
        out = base.with_suffix(".csv")
        pd.DataFrame(annotations).to_csv(out, index=False)
    return out

def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def _ensure_clip_dir(video_path: str) -> pathlib.Path:
    stem = pathlib.Path(video_path).stem
    outdir = CLIP_ROOT / stem
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def _video_meta(video_path: str) -> Tuple[int, float, Tuple[int, int]]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return total, fps, (w, h)

def _bbox_from_obj(obj: dict) -> Tuple[int, int, int, int]:
    """Return (x, y, w, h) for rect/circle/polygon/freedraw."""
    t = obj.get("type")
    left = int(round(obj.get("left", 0)))
    top = int(round(obj.get("top", 0)))
    width = int(round(obj.get("width", 0) or 0))
    height = int(round(obj.get("height", 0) or 0))
    if t == "circle":
        r = int(round(obj.get("radius", max(width, height) / 2)))
        return left, top, 2 * r, 2 * r
    return left, top, max(width, 1), max(height, 1)

def _mask_from_obj(obj: dict, frame_w: int, frame_h: int, bbox: Tuple[int,int,int,int]) -> np.ndarray:
    """Binary mask for the object (255 inside shape)."""
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    t = obj.get("type")
    x, y, w, h = bbox

    if t == "rect":
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)
        return mask

    if t == "circle":
        r = int(round(min(w, h) / 2))
        cx = x + r
        cy = y + r
        cv2.circle(mask, (cx, cy), r, 255, thickness=-1)
        return mask

    pts = obj.get("points") or obj.get("path")
    if isinstance(pts, list) and len(pts) >= 3:
        try:
            if isinstance(pts[0], dict):
                arr = np.array([[p.get("x", 0), p.get("y", 0)] for p in pts], dtype=np.int32)
            else:
                arr = np.array([[int(round(p[0])), int(round(p[1]))] for p in pts], dtype=np.int32)
            cv2.fillPoly(mask, [arr], 255)
            return mask
        except Exception:
            pass

    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)
    return mask

def _write_region_clip(
    video_path: str,
    obj: dict,
    frame_idx: int,
    fps: float,
    total_frames: int,
    pre_sec: float,
    dur_sec: float,
    downscale: float,
    mask_outside: bool,
    outdir: pathlib.Path,
    frame_size_wh: Tuple[int,int],
) -> pathlib.Path:
    """Cut a masked region across time and save as MP4."""
    start_frame = max(0, int(round(frame_idx - pre_sec * fps)))
    end_frame = min(total_frames, start_frame + int(round(dur_sec * fps)))
    if end_frame <= start_frame:
        end_frame = min(total_frames, start_frame + 1)

    fw, fh = frame_size_wh

    x, y, w, h = _bbox_from_obj(obj)
    x = max(0, x); y = max(0, y)
    w = max(1, min(w, fw - x)); h = max(1, min(h, fh - y))

    base = pathlib.Path(video_path).stem
    t = obj.get("type", "shape")
    uniq = int(time.time() * 1000)  # ms timestamp to ensure uniqueness
    out_name = f"{base}_f{frame_idx}_{t}_t{start_frame}-{end_frame}_{uniq}.mp4"
    out_path = outdir / out_name

    rw, rh = int(round(w * downscale)), int(round(h * downscale))
    rw = max(1, rw); rh = max(1, rh)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (rw, rh))

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    full_mask = _mask_from_obj(obj, fw, fh, (x, y, w, h)) if mask_outside else None

    for _ in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        if mask_outside and full_mask is not None:
            frame = cv2.bitwise_and(frame, frame, mask=full_mask)
        crop = frame[y:y + h, x:x + w]
        if rw != w or rh != h:
            crop = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_AREA)
        writer.write(crop)

    cap.release()
    writer.release()
    return out_path

# ------------------- main UI -------------------
def render():
    st.subheader("Annotation — Draw to Crop (Auto-save MP4)")

    # 1) Upload video
    st.markdown("**1) Upload a video**")
    vid = st.file_uploader("Video (mp4/mov/m4v/avi)", type=["mp4", "mov", "m4v", "avi"])
    if not vid:
        st.info("Upload a video to begin.")
        return

    # Save uploaded video to a temp file
    suffix = os.path.splitext(vid.name)[1]
    tmp_path = pathlib.Path(tempfile.gettempdir()) / f"_annot_video{suffix}"
    with open(tmp_path, "wb") as f:
        f.write(vid.read())
    video_path = str(tmp_path)

    # Video metadata
    total_frames, fps, (FW, FH) = _video_meta(video_path)

    # 2) Settings
    st.markdown("**2) Settings**")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        tool = st.selectbox("Shape", ["rect", "circle"], index=0)
    with c2:
        speed_label = st.selectbox("Speed", ["0.5x", "1x", "2x"], index=1)
        speed = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}[speed_label]
    with c3:
        downscale = st.slider("Downscale", min_value=0.1, max_value=1.0, value=1.0, step=0.05)

    mask_outside = st.checkbox("Mask outside shape", value=True)

    st.caption("**How it works:** Press & drag on the video to draw. "
               "Mouse-down = start time, Mouse-up = end time. We auto-save the cropped MP4.")

    # 3) Render video+canvas in the browser and capture one draw cycle
    #    We embed the video as a data URL; suitable for small/medium files.
    with open(video_path, "rb") as vf:
        b64 = base64.b64encode(vf.read()).decode("utf-8")

    js = f"""
    const WRAP_ID = 'gut_wrap';
    let wrap = document.getElementById(WRAP_ID);
    if (!wrap) {{
      wrap = document.createElement('div');
      wrap.id = WRAP_ID;
      wrap.style.position = 'relative';
      wrap.style.maxWidth = '100%';
      wrap.style.marginBottom = '8px';
      document.body.appendChild(wrap);

     wrap.innerHTML = `
  <div style="position:relative; display:inline-block;">
    <video id="gut_vid"
           playsinline autoplay muted
           style="max-width:100%; display:block; background:#000; position:relative; z-index:1;"></video>
    <canvas id="gut_can"
            style="position:absolute; inset:0; z-index:2; pointer-events:auto;"></canvas>
  </div>`;
      const vid0 = wrap.querySelector('#gut_vid');
      vid0.src = "data:video/mp4;base64,{b64}";
    }}

    const vid = wrap.querySelector('#gut_vid');
    const can = wrap.querySelector('#gut_can');
    vid.playbackRate = {speed};

    // Size canvas to displayed video size
    function sizeCanvas(){{
      const r = vid.getBoundingClientRect();
      can.width = r.width;
      can.height = r.height;
      can.style.width = r.width + 'px';
      can.style.height = r.height + 'px';
    }}
    sizeCanvas();
    window.addEventListener('resize', sizeCanvas);
    window.addEventListener('scroll', sizeCanvas, true);
vid.addEventListener('loadedmetadata', sizeCanvas);

    // Await one draw cycle (press→release), return times + shape
    const res = await new Promise((resolve) => {{
      const ctx = can.getContext('2d');
      let drawing = false, sx=0, sy=0, start=null;

      function rel(e){{
        const bb = can.getBoundingClientRect();
        return [e.clientX - bb.left, e.clientY - bb.top];
      }}
      function drawPreview(x,y){{
        ctx.clearRect(0,0,can.width,can.height);
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'red';
        ctx.fillStyle = 'rgba(255,0,0,0.25)';
        if ('{tool}' === 'rect'){{
          const w = x - sx, h = y - sy;
          ctx.strokeRect(sx, sy, w, h);
          ctx.fillRect(Math.min(sx,x), Math.min(sy,y), Math.abs(w), Math.abs(h));
        }} else {{
          const dx = x - sx, dy = y - sy;
          const r = Math.sqrt(dx*dx + dy*dy);
          ctx.beginPath();
          ctx.arc(sx, sy, r, 0, Math.PI*2);
          ctx.stroke();
          ctx.fill();
        }}
      }}

      const move = (e) => {{ if (!drawing) return; const [x,y]=rel(e); drawPreview(x,y); }};
      const up = (e) => {{
        if (!drawing) return;
        drawing = false;
        const [ex,ey] = rel(e);
        ctx.clearRect(0,0,can.width,can.height);
        const end = vid.currentTime;

        // Map to natural pixels
        const scaleX = (vid.videoWidth || {FW}) / can.width;
        const scaleY = (vid.videoHeight || {FH}) / can.height;

        let shape = {{}};
        if ('{tool}' === 'rect'){{
          const x = Math.min(sx, ex) * scaleX;
          const y = Math.min(sy, ey) * scaleY;
          const w = Math.abs(ex - sx) * scaleX;
          const h = Math.abs(ey - sy) * scaleY;
          shape = {{type:'rect', left:Math.round(x), top:Math.round(y),
                    width:Math.round(w), height:Math.round(h)}};
        }} else {{
          const dx = (ex - sx) * scaleX;
          const dy = (ey - sy) * scaleY;
          const r = Math.sqrt(dx*dx + dy*dy);
          const left = (sx*scaleX) - r;
          const top  = (sy*scaleY) - r;
          shape = {{type:'circle', left:Math.round(left), top:Math.round(top),
                    radius:Math.round(r)}};
        }}

        window.removeEventListener('mousemove', move);
        window.removeEventListener('mouseup', up);
        resolve({{ start, end, shape,
                   naturalW: vid.videoWidth || {FW},
                   naturalH: vid.videoHeight || {FH} }});
      }};

      can.onmousedown = (e) => {{
        drawing = true;
        [sx,sy] = rel(e);
        start = vid.currentTime;    // mouse-down = clip start
        window.addEventListener('mousemove', move);
        window.addEventListener('mouseup', up);
      }};
    }});

    // Return to Streamlit
    res
    """

    result = st_javascript(js)

    # 4) When a draw completes, auto-save the cropped MP4
    if result and isinstance(result, dict):
        try:
            t_start = float(result.get("start", 0.0))
            t_end   = float(result.get("end", 0.0))
            shape   = result.get("shape", {}) or {}

            # Validate shape size
            if shape.get("type") == "rect":
                if shape.get("width", 0) < 2 or shape.get("height", 0) < 2:
                    st.warning("Rectangle too small; skipped.")
                    return
            elif shape.get("type") == "circle":
                if shape.get("radius", 0) < 2:
                    st.warning("Circle too small; skipped.")
                    return
            else:
                st.warning("Unsupported shape; use rect or circle.")
                return

            # Time → frames / duration
            start_f = max(0, int(round(t_start * fps)))
            end_f   = max(0, int(round(t_end   * fps)))
            if end_f <= start_f:
                end_f = start_f + 1
            dur_s = (end_f - start_f) / max(fps, 1e-6)

            outdir = _ensure_clip_dir(video_path)
            clip_path = _write_region_clip(
                video_path=video_path,
                obj=shape,
                frame_idx=end_f,             # anchor at end
                fps=fps,
                total_frames=total_frames,
                pre_sec=0.0,                 # start at mouse-down
                dur_sec=dur_s,               # end at mouse-up
                downscale=float(downscale),
                mask_outside=bool(mask_outside),
                outdir=outdir,
                frame_size_wh=(FW, FH),
            )

            st.success(f"Saved clip → {clip_path.name}  "
                       f"(frames {start_f}–{end_f}, ~{dur_s:.2f}s)")
            st.write(f"• {clip_path}")
            # (Optional) preview the saved clip:
            # st.video(str(clip_path))

            # (Optional) log to annotations in memory
            if "annotations" not in st.session_state:
                st.session_state.annotations = []
            st.session_state.annotations.append({
                "video": os.path.basename(video_path),
                "start_time_s": t_start,
                "end_time_s": t_end,
                "type": shape.get("type"),
                "left": shape.get("left"),
                "top": shape.get("top"),
                "width": shape.get("width"),
                "height": shape.get("height"),
                "radius": shape.get("radius"),
                "downscale": float(downscale),
                "mask_outside": bool(mask_outside),
            })

        except Exception as e:
            st.error(f"Clip save failed: {e}")

    # 5) Export annotations (optional)
    st.markdown("**3) Export log (optional)**")
    if "annotations" in st.session_state and st.session_state.annotations:
        fmt = st.radio("Format", ["csv", "json"], horizontal=True, index=0)
        if st.button("Export annotations"):
            out = _save_annotation(os.path.basename(video_path), st.session_state.annotations, fmt)
            st.success(f"Annotations saved: {out.name}")
            with open(out, "rb") as f:
                st.download_button("Download file", f, file_name=out.name)

    with st.expander("Current annotations in memory"):
        st.json(st.session_state.get("annotations", []))
