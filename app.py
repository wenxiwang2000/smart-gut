import os
import sys
import glob
import time
import shutil
import pathlib
import tempfile
import subprocess
import cv2
import numpy as np
import streamlit as st

# --- Annotation loader (forces loading the local annotation.py) ---
import importlib.util

def _load_annotation_module():
    here = pathlib.Path(__file__).parent
    ann_py = here / "annotation.py"
    if ann_py.exists():
        spec = importlib.util.spec_from_file_location("annotation_local", str(ann_py))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["annotation_local"] = mod
        spec.loader.exec_module(mod)
        return mod
    # Fallback: normal import if the file isn't next to app.py
    import importlib
    return importlib.import_module("annotation")

# ... (other imports and optional modules omitted for brevity) ...

# Set up Streamlit page configuration
st.set_page_config(page_title="Smart Gut — panels", layout="wide")

# ... (navigation bar code unchanged) ...

# --------------------------- Detect Panel Helpers ---------------------------

def _save_uploaded_video(uploaded_file) -> str:
    """Save uploaded file to a temporary path and return the path."""
    suffix = pathlib.Path(uploaded_file.name).suffix
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="smartgut_")
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tmp_path

def _ensure_dir(path: str) -> str:
    p = pathlib.Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return str(p.resolve())

def _try_python_api(video_path: str, output_dir: str):
    """
    If detection.py provides process_video(video_path, output_dir), use it.
    Otherwise return (False, None) to indicate fallback to CLI.
    """
    if detection is None:
        return False, None
    try:
        fn = getattr(detection, "process_video", None)
        if callable(fn):
            with st.spinner("Running detection (Python API)..."):
                csv_path = fn(video_path, output_dir)
            return True, csv_path
    except Exception as e:
        st.warning(f"Python API path failed, will try CLI next.\n\n{e}")
    return False, None

def _run_detection_cli(video_path: str, output_dir: str):
    """
    Fallback: call detection.py as a script if present.
    Adjust args to match detection.py CLI interface (video [overlay] [model] [thr]).
    """
    script_dir = pathlib.Path(__file__).parent
    script_path = script_dir / "detection.py"
    if not script_path.exists():
        st.error("detection.py not found. Please add it or provide a Python API.")
        return None
    # Snapshot existing CSV files to detect newly created ones
    before = set(glob.glob(str(script_dir / "*.csv"))) | set(
        glob.glob(str(script_dir / "**/*.csv"), recursive=True)
    )
    cmd = [sys.executable, str(script_path), video_path, "", "", "0.60"]
    with st.spinner("Running detection (CLI)..."):
        proc = subprocess.run(cmd, cwd=str(script_dir), capture_output=True, text=True)
    if proc.returncode != 0:
        st.error("Detection script failed.")
        if proc.stderr:
            st.code(proc.stderr)
        return None
    # Find new CSV file created by detection.py
    after = set(glob.glob(str(script_dir / "*.csv"))) | set(
        glob.glob(str(script_dir / "**/*.csv"), recursive=True)
    )
    new_files = sorted(list(after - before))
    if not new_files:
        return None
    return new_files[0]  # return the first new CSV as the main output

# --------------------------- Detect Panel (Streamlit UI) ---------------------------

def render_detect_panel():
    st.subheader("Detect — Video ➜ CSV")

    # (A) Video input
    uploaded = st.file_uploader("Drag & drop a video file", type=["mp4", "mov", "m4v", "avi"], key="det_file")
    if uploaded:
        # Save to temporary path
        tmp_path = _save_uploaded_video(uploaded)
        st.session_state.video_tmp = tmp_path
        st.session_state.video_name = uploaded.name

    # Display loaded video and option to clear
    if "video_tmp" in st.session_state:
        st.video(st.session_state.video_tmp)
        st.caption(f"Loaded: **{st.session_state.get('video_name', 'video')}**")
        if st.button("Clear video", type="secondary"):
            try:
                os.remove(st.session_state.video_tmp)
            except Exception:
                pass
            for k in ("video_tmp", "video_name", "last_csv"):
                st.session_state.pop(k, None)
            st.experimental_rerun()

    # (B) Output settings and detection mode
    col_out1, col_out2 = st.columns([1, 1.2], gap="large")
    with col_out1:
        st.markdown("#### 2) Choose detection mode and output folder")
        # Output directory selection
        default_out = str(pathlib.Path.cwd() / "outputs")
        outdir = st.text_input("Output folder", value=st.session_state.get("output_dir", default_out))
        use_cwd = st.checkbox("Use current working folder", value=False)
        if use_cwd:
            outdir = str(pathlib.Path.cwd())
        st.session_state.output_dir = outdir
        # Dropdown for detection mode
        mode = st.selectbox("Detection mode", ["Original (Motility events)", "HSV (blue contour tracking)"])
    with col_out2:
        # (You could include additional parameters or profiles if needed; for now this column can remain for future use)
        pass

    # (C) Run detection
    run_disabled = "video_tmp" not in st.session_state or not outdir.strip()
    run_btn = st.button("▶️ Run detection", type="primary", disabled=run_disabled)
    if run_btn:
        if "video_tmp" not in st.session_state:
            st.warning("Please upload a video first.")
        elif not outdir.strip():
            st.warning("Please specify an output folder.")
        else:
            outdir_path = _ensure_dir(outdir)
            video_path = st.session_state.video_tmp

            if mode.startswith("Original"):
                # Run the original detection pipeline (use Python API if available, else CLI)
                handled, csv_path = _try_python_api(video_path, outdir_path)
                if not handled:
                    csv_path = _run_detection_cli(video_path, outdir_path)
                if csv_path:
                    st.session_state.last_csv = csv_path
                    st.success(f"CSV saved to: `{csv_path}`")
                    st.markdown(f"[Open folder]({pathlib.Path(outdir_path).as_uri()})")
                    # Offer CSV download button
                    try:
                        with open(csv_path, "rb") as f:
                            st.download_button("Download CSV", f.read(), file_name=pathlib.Path(csv_path).name, mime="text/csv")
                    except Exception:
                        pass
            else:
                # Run HSV-based tracking mode
                try:
                    boxes_csv, contr_csv, traced_video = detection.run_hsv_mode(video_path, outdir_path)
                except Exception as e:
                    st.error(f"HSV tracking failed: {e}")
                else:
                    st.session_state.last_csv = contr_csv  # track contractions CSV as primary result
                    st.success(f"HSV tracking completed. Outputs saved to `{outdir_path}/hsv_mode/`")
                    st.markdown(f"[Open folder]({pathlib.Path(outdir_path, 'hsv_mode').as_uri()})")
                    # Display video with tracked gut
                    st.video(str(traced_video))
                    # Offer download buttons for all outputs
                    try:
                        with open(boxes_csv, "rb") as fb:
                            st.download_button("Download bounding boxes CSV", fb.read(), file_name=pathlib.Path(boxes_csv).name, mime="text/csv")
                        with open(contr_csv, "rb") as fc:
                            st.download_button("Download contractions CSV", fc.read(), file_name=pathlib.Path(contr_csv).name, mime="text/csv")
                        with open(traced_video, "rb") as fv:
                            st.download_button("Download traced video", fv.read(), file_name=pathlib.Path(traced_video).name, mime="video/mp4")
                    except Exception as ex:
                        st.warning(f"Outputs saved, but failed to load download buttons: {ex}")

    # (D) Optional preview (for original mode only)
    if "video_tmp" in st.session_state:
        if mode.startswith("Original"):
            # Show motion preview controls for the original detection (uses ACTIVE profile settings)
            active_params = _load_active()  # load profile (if using profile JSON system)
            with st.expander("Preview with active profile", expanded=True):
                # Preview window slider
                try:
                    _, _, _, _, dur = _probe_video_meta(st.session_state.video_tmp)
                except Exception:
                    dur = 0.0
                ws = float(active_params.get("window_start_s", 0.0))
                we = float(active_params.get("window_end_s", 8.0 if dur <= 0 else min(8.0, dur)))
                if dur > 0:
                    ws, we = st.slider("Preview window (seconds)", 0.0, float(max(0.5, dur)), (ws, we), 0.5)
                    active_params["window_start_s"] = ws
                    active_params["window_end_s"] = we
                build = st.button("Build previews", use_container_width=True)
                if build:
                    out_dir = pathlib.Path.cwd()
                    color_out = out_dir / "_preview_motion_color"
                    gray_out  = out_dir / "_preview_motion_gray"
                    try:
                        with st.spinner("Rendering previews with active profile…"):
                            color_path, gray_path = _build_motion_previews(
                                st.session_state.video_tmp,
                                str(color_out), str(gray_out), active_params
                            )
                        st.success("Previews ready")
                        st.markdown("**Movement overlay (full color)**")
                        st.video(str(color_path))
                        st.download_button(
                            "Download color preview",
                            data=open(color_path, "rb").read(),
                            file_name=os.path.basename(color_path),
                            mime="video/mp4" if str(color_path).endswith(".mp4") else "video/avi",
                        )
                        st.markdown("**Grayscale + tracing**")
                        st.video(str(gray_path))
                        st.download_button(
                            "Download grayscale preview",
                            data=open(gray_path, "rb").read(),
                            file_name=os.path.basename(gray_path),
                            mime="video/mp4" if str(gray_path).endswith(".mp4") else "video/avi",
                        )
                    except Exception as e:
                        st.error(f"Preview failed: {e}")
        else:
            # HSV mode: preview not applicable (the tracking uses color segmentation rather than threshold profile)
            st.info("Preview is not available for HSV tracking mode.")

    # (E) Show info about last result
    if "last_csv" in st.session_state:
        st.info(f"Last result: `{st.session_state.last_csv}`")

# ... (router logic remains unchanged) ...

if st.session_state["panel"] == "detect":
    render_detect_panel()
elif st.session_state["panel"] == "training":
    # ... unchanged ...
    pass

elif st.session_state["panel"] == "annotation":
    st.subheader("Annotation")
    try:
        mod = _load_annotation_module()
        if hasattr(mod, "render") and callable(mod.render):
            mod.render()
        else:
            st.warning("annotation.render() not found. Make sure annotation.py defines a top-level def render().")
    except Exception as e:
        st.error("Could not load the annotation module.")
        st.exception(e)

# (Other panels unchanged)
