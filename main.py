# main.py — Single-window pywebview mode (recommended) or classic Tk launcher with pop-outs.

import os, sys, socket, shutil, subprocess, pathlib, webbrowser

HERE = pathlib.Path(__file__).parent.resolve()
APP_SCRIPT = "app.py"

# ---- Toggle here ----
SINGLE_WINDOW = True   # True => one Smart Gut window via pywebview; False => Tk launcher + pop-outs
# ---------------------

# ----------------------- helpers -----------------------
def free_port(start=8501, end=8599):
    for p in range(start, end + 1):
        try:
            with socket.socket() as s:
                s.bind(("127.0.0.1", p))
                return p
        except OSError:
            continue
    return None

def find_streamlit_cli():
    exe = shutil.which("streamlit")
    if exe:
        return exe
    py = pathlib.Path(sys.executable)
    candidate = py.parent / "Scripts" / ("streamlit.exe" if os.name == "nt" else "streamlit")
    if candidate.exists():
        return str(candidate)
    candidates = [
        pathlib.Path(os.environ.get("CONDA_PREFIX", "")) / "Scripts" / "streamlit.exe",
        pathlib.Path.home() / "anaconda3" / "Scripts" / "streamlit.exe",
        pathlib.Path("C:/ProgramData/anaconda3/Scripts/streamlit.exe"),
    ]
    for c in candidates:
        if c and c.exists():
            return str(c)
    return None

def can_run_python_m_streamlit():
    try:
        subprocess.check_call(
            [sys.executable, "-m", "streamlit", "--version"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(HERE)
        )
        return True
    except Exception:
        return False

def launch_streamlit_panel(panel: str):
    """Start Streamlit via CLI or 'python -m streamlit' and return the http://localhost:PORT/?panel=... URL."""
    port = free_port()
    if port is None:
        messagebox.showerror("Port error", "No free port between 8501–8599.")
        return None

    # Add your desired max upload size here (MB)
    extra = ["--server.maxUploadSize", "4096"]

    cli = find_streamlit_cli()
    if cli:
        cmd = [cli, "run", APP_SCRIPT, "--server.port", str(port), "--server.headless", "true", *extra]
    elif can_run_python_m_streamlit():
        cmd = [sys.executable, "-m", "streamlit", "run", APP_SCRIPT,
               "--server.port", str(port), "--server.headless", "true", *extra]
    else:
        messagebox.showerror(
            "Streamlit not found",
            "Streamlit is not available in this Python environment.\n\n"
            "Run once:\n  python -m pip install -r requirements.txt"
        )
        return None

    try:
        subprocess.Popen(cmd, cwd=str(HERE))
    except Exception as e:
        messagebox.showerror("Launch failed", f"Couldn't start Streamlit:\n{e}")
        return None

    return f"http://localhost:{port}/?panel={panel}"


# ----------------------- single-window mode -----------------------
def run_single_window(default_panel="detect"):
    try:
        import webview
    except ModuleNotFoundError:
        # Last-resort fallback: open in browser
        url = launch_streamlit_panel(default_panel)
        webbrowser.open(url)
        print("pywebview not installed. Opened in your browser instead. Install with: pip install pywebview")
        return

    url = launch_streamlit_panel(default_panel)
    # Create one Smart Gut window that hosts the full UI
    webview.create_window("Smart Gut", url=url, width=1200, height=800, resizable=True)
    webview.start(debug=False)

# ----------------------- classic Tk launcher -----------------------
def run_classic_launcher():
    import tkinter as tk
    from tkinter import messagebox, filedialog

    def open_panel_in_embedded_window(panel: str, title: str):
        try:
            import importlib
            importlib.import_module("webview")
        except Exception:
            url = launch_streamlit_panel(panel)
            messagebox.showwarning(
                "Webview Not Installed",
                "pywebview is not installed.\n\nOpening in your default browser instead.\n"
                "To enable embedded windows, run:\n    pip install pywebview"
            )
            webbrowser.open(url)
            return
        # Spawn a child process that owns the webview GUI loop
        url = launch_streamlit_panel(panel)
        subprocess.Popen([sys.executable, str(HERE / "main.py"), "--popout", title, url], cwd=str(HERE))

    def run_native_detection():
        video = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video", "*.mp4 *.mov *.m4v *.avi"), ("All files", "*.*")]
        )
        if not video:
            return
        from tkinter import simpledialog
        save_overlay = messagebox.askyesno("Overlay video", "Save overlay MP4 (drawn boxes/paths)?")
        overlay = filedialog.asksaveasfilename(title="Overlay output", defaultextension=".mp4",
                                               filetypes=[("MP4", "*.mp4")]) if save_overlay else ""
        use_model = messagebox.askyesno("Supervised model", "Load a trained model (.joblib)?")
        model = filedialog.askopenfilename(title="Select model bundle (.joblib)",
                                           filetypes=[("Joblib / Pickle", "*.joblib *.pkl"), ("All files", "*.*")]) if use_model else ""
        prob_thr = simpledialog.askstring("Probability threshold", "Event probability threshold (0.10 – 0.95):", initialvalue="0.60") or "0.60"
        try:
            subprocess.Popen([sys.executable, "detection.py", video, overlay or "", model or "", prob_thr], cwd=str(HERE))
            messagebox.showinfo("Running", "Detection started.\nClose the OpenCV window (or press 'q') to finish.")
        except Exception as e:
            messagebox.showerror("Detection failed", str(e))

    def run_native_training():
        video = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video", "*.mp4 *.mov *.m4v *.avi"), ("All files", "*.*")]
        )
        if not video:
            return
        save_overlay = messagebox.askyesno("Overlay video", "Save overlay MP4 while training?")
        overlay = filedialog.asksaveasfilename(title="Overlay output", defaultextension=".mp4",
                                               filetypes=[("MP4", "*.mp4")]) if save_overlay else ""
        try:
            subprocess.Popen([sys.executable, "training.py", video] + ([overlay] if overlay else []), cwd=str(HERE))
            messagebox.showinfo("Running", "Training started.\nClose the OpenCV window (or press 'q') to finish.")
        except Exception as e:
            messagebox.showerror("Training failed", str(e))

    class Launcher(tk.Tk):
        def __init__(self):
            super().__init__()
            self.geometry("980x190")
            self.minsize(760, 160)
            self.title("Smart Gut")
            self.configure(bg="#f6f8fb")

            top = tk.Frame(self, bg="#eef3ff")
            top.pack(fill=tk.X, side=tk.TOP)
            tk.Label(top, text="  Smart Gut — Modules", font=("Segoe UI", 12, "bold"), bg="#eef3ff").pack(side=tk.LEFT, padx=6, pady=6)

            right = tk.Frame(top, bg="#eef3ff")
            right.pack(side=tk.RIGHT, padx=6, pady=6)

            mb_detect = tk.Menubutton(right, text="🔍 Detect ▾", relief=tk.RAISED)
            m1 = tk.Menu(mb_detect, tearoff=0)
            m1.add_command(label="Open Panel (Embedded Window)", command=lambda: open_panel_in_embedded_window("detect", "Smart Gut — Detect"))
            m1.add_command(label="Run Native (cv2 window)", command=run_native_detection)
            mb_detect.config(menu=m1); mb_detect.pack(side=tk.RIGHT, padx=4)

            mb_train = tk.Menubutton(right, text="🧪 Training ▾", relief=tk.RAISED)
            m2 = tk.Menu(mb_train, tearoff=0)
            m2.add_command(label="Open Panel (Embedded Window)", command=lambda: open_panel_in_embedded_window("training", "Smart Gut — Training"))
            m2.add_command(label="Run Native (cv2 window)", command=run_native_training)
            mb_train.config(menu=m2); mb_train.pack(side=tk.RIGHT, padx=4)

            mb_adj = tk.Menubutton(right, text="🎛️ Adjusting ▾", relief=tk.RAISED)
            m3 = tk.Menu(mb_adj, tearoff=0)
            m3.add_command(label="Open Panel (Embedded Window)", command=lambda: open_panel_in_embedded_window("adjusting", "Smart Gut — Adjusting"))
            mb_adj.config(menu=m3); mb_adj.pack(side=tk.RIGHT, padx=4)

            mb_annot = tk.Menubutton(right, text="✏️ Annotation ▾", relief=tk.RAISED)
            m4 = tk.Menu(mb_annot, tearoff=0)
            m4.add_command(label="Open Panel (Embedded Window)",
                           command=lambda: open_panel_in_embedded_window("annotation", "Smart Gut — Annotation"))
            mb_annot.config(menu=m4);
            mb_annot.pack(side=tk.RIGHT, padx=4)

            body = tk.Frame(self, bg="#f6f8fb", padx=12, pady=12)
            body.pack(fill=tk.BOTH, expand=True)
            tk.Label(
                body,
                text="Use the menus to open each module as an embedded window (pywebview) or run native (OpenCV) for Detect/Training.",
                bg="#f6f8fb", wraplength=860, justify="left"
            ).pack(anchor="w")

    # pop-out child process (webview owner)
    if len(sys.argv) >= 4 and sys.argv[1] == "--popout":
        title = sys.argv[2]
        url = sys.argv[3]
        try:
            import webview
            webview.create_window(title, url=url, width=1100, height=720, resizable=True)
            webview.start(debug=False)
        except ModuleNotFoundError:
            print("pywebview is not installed; opening panel in browser instead.")
            webbrowser.open(url)
        sys.exit(0)

    # normal launcher
    app = Launcher()
    app.mainloop()

# ----------------------- entrypoint -----------------------
if __name__ == "__main__":
    os.chdir(str(HERE))
    if SINGLE_WINDOW:
        run_single_window(default_panel="detect")
    else:
        run_classic_launcher()
