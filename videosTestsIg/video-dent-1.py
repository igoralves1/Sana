"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SANA VIDEO — FUTURISTIC DENTAL OFFICE ADVERTISEMENT GENERATOR              ║
║  Version  : 1.0                                                              ║
║  Date     : 2026-05-23                                                       ║
║  Author   : Dr. Igor Lemos Alves                                             ║
║  Target   : 1-minute cinematic | 704×1280 | 16fps | SANA-Video 2B 720p      ║
║  Hardware : RTX 5080 16GB | Ryzen 9 9900X | 32GB DDR5 | Windows             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CHANGELOG                                                                   ║
║  v1.0  2026-05-23  Initial versioned release.                                ║
║        FIX: StatisticsTracker.hms() declared as @staticmethod but called     ║
║             as StatisticsTracker.hms(None, est_time_s) in preflight(),       ║
║             passing two arguments to a one-argument method → TypeError.      ║
║        FIX: Added None-guard inside hms() so it returns "N/A" instead of    ║
║             crashing on int(None).                                           ║
║        FIX: Corrected preflight() call to StatisticsTracker.hms(est_time_s).║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ARCHITECTURE OVERVIEW                                                       ║
║  • 12 SCENES × 81 frames each = 972 frames = ~60.75 seconds at 16 fps       ║
║  • CONTEXT BRIDGE: last frame of scene N → I2V conditioning for scene N+1   ║
║  • MOTION SCORE TOKEN appended per-scene (10–25 range, scene-calibrated)    ║
║  • SCENE STATE JSON: crash-resume — re-run skips completed scenes            ║
║  • SEED CONTINUITY: seed = 2025 + scene_id (monotonic drift)                ║
║  • VRAM: bfloat16 transformer | float32 VAE | cpu_offload | attn_slicing    ║
║  • ENCODING: H.264 CRF-18, slow preset, +faststart (web-ready)              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SANA PROMPT ANATOMY (NVIDIA official technique)                             ║
║  [LIGHTING] → [SHOT TYPE] → [COMPOSITION] → [SUBJECT+ACTION]                ║
║  → [ENVIRONMENT] → [CAMERA] → [MOOD+COLOR] + " motion score: N."            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  INSTALL DEPENDENCIES                                                        ║
║  pip install git+https://github.com/huggingface/diffusers                   ║
║  pip install torch imageio[ffmpeg] psutil GPUtil Pillow numpy                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

__version__ = "1.0"
__author__  = "Dr. Igor Lemos Alves"
__date__    = "2026-05-23"

import torch
import gc
import os
import json
import time
import shutil
import threading
import numpy as np
import imageio
import psutil
from PIL import Image
from datetime import datetime
from collections import deque
from pathlib import Path

# ── optional GPU monitoring ──────────────────────────────────────────────────
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

# ── diffusers imports (lazy-checked at runtime) ───────────────────────────────
try:
    from diffusers import SanaVideoPipeline, SanaImageToVideoPipeline
    from diffusers.utils import export_to_video, load_image
    DIFFUSERS_OK = True
except ImportError:
    DIFFUSERS_OK = False


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ANSI COLORS                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class C:
    CYAN   = '\033[96m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    BLUE   = '\033[94m'
    BOLD   = '\033[1m'
    DIM    = '\033[2m'
    END    = '\033[0m'

def c(color, text):       return f"{color}{text}{C.END}"
def header(text):         print(f"\n{C.BOLD}{C.CYAN}{'═'*72}\n  {text}\n{'═'*72}{C.END}")
def ok(text):             print(f"  {C.GREEN}✔{C.END}  {text}")
def warn(text):           print(f"  {C.YELLOW}⚠{C.END}  {text}")
def err(text):            print(f"  {C.RED}✘{C.END}  {text}")
def step(n, t, text):     print(f"\n{C.BOLD}{C.BLUE}[{n}/{t}]{C.END} {text}")
def bar(frac, w=44):
    f = int(w * max(0.0, min(1.0, frac)))
    return f"{C.GREEN}{'█'*f}{C.END}{'░'*(w-f)}"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SCENE SCRIPT — 12 × 5-SECOND NARRATIVE BEATS                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# SANA official negative prompt (verbatim from NVIDIA documentation)
NEGATIVE_PROMPT = (
    "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, "
    "sudden disappearance, jump cuts, jerky movements, rapid shot changes, "
    "frames out of sync, inconsistent character shapes, temporal artifacts, "
    "jitter, and ghosting effects, creating a disorienting visual experience. "
    "overexposed, underexposed, noise, grain, watermark, text overlay, "
    "cartoon, anime, low resolution, blurry, anatomically incorrect, "
    "extra limbs, duplicate faces, ugly, deformed."
)

# Prompt anatomy per scene:
#   LIGHTING → SHOT TYPE → COMPOSITION → SUBJECT+ACTION
#   → ENVIRONMENT → CAMERA → MOOD+COLOR  +  " motion score: N."
SCENES = [
    {
        "id":    0,
        "name":  "Opening — City of the Future",
        "beats": "Aerial establishing shot of the futuristic city",
        "motion_score": 20,
        "prompt": (
            "Soft golden sunrise light, sweeping aerial wide shot, centered horizon composition. "
            "A luminous megacity of 2250 stretches below, crystal towers and bio-domes glowing amber. "
            "Sleek autonomous vehicles trace light trails through elevated glass highways. "
            "Camera drifts slowly forward over the skyline, slight downward tilt. "
            "Warm utopian color grading, cinematic anamorphic lens flare. "
            "motion score: 20."
        ),
    },
    {
        "id":    1,
        "name":  "Clinic Exterior — SmartSmile 2250",
        "beats": "Reveal the dental clinic facade",
        "motion_score": 15,
        "prompt": (
            "Soft diffused daylight, medium wide shot, symmetrical architectural composition. "
            "A breathtaking dental clinic facade of pure white biopolymer and blue-tinted smart glass, "
            "hovering holographic logo reading 'SmartSmile 2250' rotates gently above the entrance. "
            "Patients in elegant attire approach through a garden of bioluminescent trees. "
            "Camera slowly pushes forward toward the entrance. "
            "Cool aquamarine and white color palette, pristine and welcoming. "
            "motion score: 15."
        ),
    },
    {
        "id":    2,
        "name":  "Reception — AI Concierge",
        "beats": "Patient checks in with holographic AI receptionist",
        "motion_score": 18,
        "prompt": (
            "Cool blue ambient light with warm accent panels, medium close-up, rule-of-thirds. "
            "An elegant patient stands at a floating glass reception desk, interacting with a translucent "
            "holographic AI concierge avatar that smiles and gestures warmly. "
            "Soft neon medical data floats around the room. Staff in crisp white uniforms move in background. "
            "Camera holds steady with a slow gentle push-in. "
            "Clinical yet warm color grading, ultra-clean whites and soft blues. "
            "motion score: 18."
        ),
    },
    {
        "id":    3,
        "name":  "Diagnostics — AI Oral Scan",
        "beats": "Real-time AI full-mouth diagnostic scanning",
        "motion_score": 22,
        "prompt": (
            "Soft cool clinical lighting, close-up macro shot, centered composition. "
            "A cutting-edge AI oral scanner glides around a patient's open mouth, projecting "
            "real-time holographic 3D dental models in the air beside the chair. "
            "Colorful diagnostic data pulses over each tooth. The dentist reviews the hologram with a stylus. "
            "Camera slowly orbits the scanning device. "
            "High-tech blue and white color tones, precise medical aesthetic. "
            "motion score: 22."
        ),
    },
    {
        "id":    4,
        "name":  "Treatment Room — Nano-Robot Procedure",
        "beats": "Nano-robot swarm performs painless precision dental work",
        "motion_score": 25,
        "prompt": (
            "Diffused surgical lighting, medium shot with macro insert, clean single-subject framing. "
            "A patient reclines in a floating ergonomic chair, completely relaxed, eyes closed peacefully. "
            "A cloud of luminous nano-robots the size of dust motes swarms gracefully into the mouth, "
            "each one emitting a tiny blue glow as it works with nanometer precision. "
            "The attending dentist observes calmly on a transparent AR screen. "
            "Camera gently drifts from wide to close. "
            "Serene clinical blue-white palette, wonder and trust. "
            "motion score: 25."
        ),
    },
    {
        "id":    5,
        "name":  "Bioprinting — Tooth Regeneration",
        "beats": "Living tooth bioprinted in real time",
        "motion_score": 20,
        "prompt": (
            "Warm amber lab lighting, extreme close-up, centered macro composition. "
            "A state-of-the-art dental bioprinter deposits living tissue layer by layer, "
            "growing a perfect translucent tooth crown from base to tip in real time. "
            "The tooth glows faintly as stem cells activate, surrounded by a delicate scaffold of light. "
            "Camera holds static in macro focus with shallow depth of field. "
            "Rich amber and ivory tones, scientific wonder aesthetic. "
            "motion score: 20."
        ),
    },
    {
        "id":    6,
        "name":  "AI Dentist — Human + AI Collaboration",
        "beats": "Human dentist and AI working side by side",
        "motion_score": 18,
        "prompt": (
            "Soft split warm-cool lighting, medium two-shot, balanced composition. "
            "A skilled human dentist and a sleek humanoid AI dental assistant stand side by side, "
            "reviewing a rotating 3D holographic jaw model together, pointing and discussing with calm confidence. "
            "The AI's translucent face glows softly with data readouts. "
            "Camera slowly arcs from profile to frontal. "
            "Warm wood and cool steel clinic aesthetic, partnership and expertise. "
            "motion score: 18."
        ),
    },
    {
        "id":    7,
        "name":  "Patient Experience — Zero Pain",
        "beats": "Patient in comfortable pain-free treatment",
        "motion_score": 12,
        "prompt": (
            "Golden soft ambient light, medium close-up, slightly low angle, warm framing. "
            "A patient in the reclined chair smiles with eyes closed, completely at ease. "
            "A calming neural interface headband projects soft geometric light patterns above their forehead. "
            "The room's walls display a gentle animated underwater scene. "
            "Camera slowly pushes into the patient's serene face. "
            "Warm golden hour tones, complete comfort and peace. "
            "motion score: 12."
        ),
    },
    {
        "id":    8,
        "name":  "Results — The Perfect Smile",
        "beats": "Patient sees their transformed smile for the first time",
        "motion_score": 16,
        "prompt": (
            "Warm flattering beauty lighting, medium close-up, centered portrait composition. "
            "A patient holds up a sleek smart mirror and sees their radiant new smile for the first time, "
            "eyes widening with joy, breaking into a wide genuine laugh. "
            "The mirror overlays a health-score readout in soft green digits. "
            "Camera slowly pushes into the smile. "
            "Warm bright tones, pure joy and confidence. "
            "motion score: 16."
        ),
    },
    {
        "id":    9,
        "name":  "Community — Dental Health for All",
        "beats": "Diverse humanity receiving universal dental care",
        "motion_score": 22,
        "prompt": (
            "Bright even daylight, wide ensemble shot, dynamic group composition. "
            "A joyful multicultural crowd of patients of all ages walks through the SmartSmile 2250 park, "
            "children, elders, and adults all smiling brilliantly, passing holographic dental-health kiosks. "
            "Bioluminescent trees line the path. A SmartSmile drone distributes dental care packages above. "
            "Camera tracks alongside the crowd in a smooth lateral dolly. "
            "Vibrant warm-daylight palette, optimism and inclusivity. "
            "motion score: 22."
        ),
    },
    {
        "id":    10,
        "name":  "Data — Global Oral Health Dashboard",
        "beats": "Macro-scale planetary dental health data visualization",
        "motion_score": 18,
        "prompt": (
            "Deep blue ambient light, wide shot, global holographic map composition. "
            "A vast spherical holographic Earth rotates slowly in the center of a control room, "
            "with thousands of glowing points marking SmartSmile clinics worldwide. "
            "Curved data panels show real-time global oral health scores climbing toward 100 percent. "
            "Scientists in white move across the floor below. "
            "Camera slowly tracks backward to reveal the full globe. "
            "Deep navy, emerald, and white data-visualization tones, epic scale. "
            "motion score: 18."
        ),
    },
    {
        "id":    11,
        "name":  "Closing — Logo and Tagline",
        "beats": "Brand reveal with emotional closing message",
        "motion_score": 10,
        "prompt": (
            "Soft warm white studio light, ultra-wide clean shot, perfectly centered. "
            "A single tooth carved from pure crystal floats and rotates slowly against a pure white background, "
            "transforming into the glowing SmartSmile 2250 logo, which pulses gently. "
            "Clean sans-serif text materializes beneath: 'Every smile. Every human. Every future.' "
            "Camera holds completely static, logo gently breathes. "
            "Minimalist pure white and gold palette, timeless and iconic. "
            "motion score: 10."
        ),
    },
]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_config():
    """
    Production configuration for 1-minute 704×1280 dental advertisement.

    SANA-Video 2B 720p native resolution: 704×1280.
    81 frames at 16 fps = 5.0625 seconds per chunk — exactly one scene beat.
    12 scenes × 81 frames = 972 total frames = 60.75 seconds.
    """
    cfg = {
        # ── Model ────────────────────────────────────────────────────────
        "model_t2v": "Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
        "model_i2v": "Efficient-Large-Model/SANA-Video_2B_720p_diffusers",

        # ── Video spec ───────────────────────────────────────────────────
        "height":           704,
        "width":            1280,
        "fps":              16,          # SANA-Video native fps
        "frames_per_chunk": 81,          # SANA-Video native window (~5.06s)
        "num_scenes":       len(SCENES), # 12

        # ── Quality parameters (RTX 5080 optimized) ──────────────────────
        "guidance_scale":      6.0,      # NVIDIA default; raise to 7 for stricter adherence
        "num_inference_steps": 50,       # Quality sweet-spot (30=fast, 50=standard, 75=max)
        "flow_shift":          8,        # SANA default sampling schedule curvature
        "torch_dtype":         torch.bfloat16,  # RTX 5080 native precision
        "seed":                2025,     # Thematic seed (year of dental future)

        # ── Context bridge strategy ──────────────────────────────────────
        # Last frame of scene N → I2V conditioning image for scene N+1.
        # Maintains lighting, composition, and color continuity across chunks.
        "use_i2v_bridge":        True,
        "bridge_overlap_frames": 3,      # Set 0 to disable overlap blending

        # ── VRAM management ──────────────────────────────────────────────
        "cpu_offload":      True,        # Offload unused layers to RAM between steps
        "attention_slicing": True,       # Slice attention to reduce peak VRAM
        "vae_dtype":        torch.float32,  # VAE always float32 — never change

        # ── Storage ──────────────────────────────────────────────────────
        "output_dir":      "dental_ad_output",
        "frames_dir":      "dental_ad_output/frames",
        "bridges_dir":     "dental_ad_output/bridges",
        "checkpoints_dir": "dental_ad_output/checkpoints",
        "final_video":     "SmartSmile2250_Ad_1min_720p.mp4",
        "frame_format":    "PNG",        # Lossless intermediate storage
        "frame_quality":   95,           # JPEG fallback (unused when PNG)

        # ── Encoding ─────────────────────────────────────────────────────
        "crf":    18,       # H.264 quality (18=near-lossless, 23=default, 15=broadcast)
        "preset": "slow",   # x264 preset (slower = better compression ratio)
    }

    # Derived values
    cfg["total_frames"]           = cfg["num_scenes"] * cfg["frames_per_chunk"]
    cfg["total_duration_seconds"] = cfg["total_frames"] / cfg["fps"]
    cfg["num_chunks"]             = cfg["num_scenes"]

    return cfg


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STATISTICS TRACKER                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class StatisticsTracker:
    def __init__(self, cfg):
        self.cfg  = cfg
        self.t0   = None
        self.chunk_times     = []
        self.chunk_gpu_peaks = []
        self.chunk_ram_peaks = []
        self.gpu_hist  = deque(maxlen=200)
        self.ram_hist  = deque(maxlen=200)
        self.vram_hist = deque(maxlen=200)

    def start(self):
        self.t0 = time.time()

    def tick(self, gpu, ram, vram):
        self.gpu_hist.append(gpu)
        self.ram_hist.append(ram)
        self.vram_hist.append(vram)

    def end_chunk(self, elapsed):
        g = max(self.gpu_hist,  default=0)
        r = max(self.ram_hist,  default=0)
        v = max(self.vram_hist, default=0)
        self.chunk_times.append(elapsed)
        self.chunk_gpu_peaks.append(g)
        self.chunk_ram_peaks.append(r)
        self.vram_hist.append(v)

    def elapsed(self):
        return time.time() - self.t0 if self.t0 else 0

    def eta(self):
        done = len(self.chunk_times)
        if done < 1:
            return 0
        avg = sum(self.chunk_times[-5:]) / min(5, done)
        return avg * (self.cfg["num_chunks"] - done)

    @staticmethod
    def hms(s):
        """
        Format seconds as a human-readable string.

        Declared as @staticmethod so it can be called either as:
            StatisticsTracker.hms(seconds)   ← class-level call (no instance)
            self.hms(seconds)                ← instance call

        v1.0 FIX: Added None guard (returns "N/A") to prevent int(None)
                  TypeError when called before tracking has started.
        v1.0 FIX: Removed erroneous second positional argument from the
                  preflight() call site (StatisticsTracker.hms(None, x) → hms(x)).
        """
        if s is None:
            return "N/A"
        s = int(s)
        h, m, sec = s // 3600, (s % 3600) // 60, s % 60
        if h:  return f"{h}h {m:02d}m"
        if m:  return f"{m}m {sec:02d}s"
        return f"{sec}s"

    def summary(self):
        el      = self.elapsed()
        done    = len(self.chunk_times)
        avg     = sum(self.chunk_times) / done if done else 0
        fps_gen = (done * self.cfg["frames_per_chunk"]) / el if el > 0 else 0

        print(f"\n{C.BOLD}{C.CYAN}{'═'*72}{C.END}")
        print(f"{C.BOLD}{C.CYAN}  📊  GENERATION STATISTICS  —  v{__version__}{C.END}")
        print(f"{C.BOLD}{C.CYAN}{'═'*72}{C.END}")
        print(f"  {'Total time':<30} {self.hms(el)}")
        print(f"  {'Scenes completed':<30} {done}/{self.cfg['num_chunks']}")
        print(f"  {'Avg time / scene':<30} {avg:.1f}s")
        print(f"  {'Fastest scene':<30} {min(self.chunk_times, default=0):.1f}s")
        print(f"  {'Slowest scene':<30} {max(self.chunk_times, default=0):.1f}s")
        print(f"  {'Generation speed':<30} {fps_gen:.2f} frames/s")

        g_avg = sum(self.gpu_hist)  / len(self.gpu_hist)  if self.gpu_hist  else 0
        r_avg = sum(self.ram_hist)  / len(self.ram_hist)  if self.ram_hist  else 0
        v_avg = sum(self.vram_hist) / len(self.vram_hist) if self.vram_hist else 0

        print(f"  {'Avg GPU load':<30} {g_avg:.1f}%")
        print(f"  {'Peak GPU load':<30} {max(self.chunk_gpu_peaks, default=0):.1f}%")
        print(f"  {'Avg VRAM used':<30} {v_avg / 100 * 16:.1f} GB / 16 GB")
        print(f"  {'Avg RAM used':<30} {r_avg:.1f}%  "
              f"({r_avg / 100 * psutil.virtual_memory().total / 1e9:.1f} GB)")
        print(f"{C.BOLD}{C.CYAN}{'═'*72}{C.END}\n")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MONITORING THREAD                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_metrics     = {"gpu": deque(maxlen=300), "vram": deque(maxlen=300),
                "cpu": deque(maxlen=300), "ram":  deque(maxlen=300)}
_tracker_ref = None


def _monitor_loop(stop_evt):
    global _metrics, _tracker_ref
    while not stop_evt.is_set():
        try:
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            gpu = vram = 0
            if HAS_GPUTIL:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu  = gpus[0].load * 100
                    vram = (gpus[0].memoryUsed / gpus[0].memoryTotal) * 100
            _metrics["gpu"].append(gpu)
            _metrics["vram"].append(vram)
            _metrics["cpu"].append(cpu)
            _metrics["ram"].append(ram)
            if _tracker_ref:
                _tracker_ref.tick(gpu, ram, vram)
        except Exception:
            pass
        time.sleep(0.5)


def _print_live(chunk_idx, num_chunks, scene_name, stat):
    """Live dashboard — refreshes terminal on every scene transition."""
    os.system("cls" if os.name == "nt" else "clear")
    pct = chunk_idx / num_chunks if num_chunks else 0

    print(f"{C.BOLD}{C.CYAN}{'═'*72}{C.END}")
    print(f"{C.BOLD}{C.CYAN}  🦷  SmartSmile 2250 — Futuristic Dental Ad  v{__version__}{C.END}")
    print(f"{C.BOLD}{C.CYAN}{'═'*72}{C.END}")
    print(f"\n  {'Scene':<20} {chunk_idx}/{num_chunks}  —  {scene_name}")
    print(f"  {'Overall':<20} [{bar(pct)}]  {pct*100:.1f}%")

    eta = stat.eta()
    if eta > 0:
        print(f"  {'ETA':<20} {stat.hms(eta)}")
    print(f"  {'Elapsed':<20} {stat.hms(stat.elapsed())}")

    gpu  = list(_metrics["gpu"])[-1]  if _metrics["gpu"]  else 0
    vram = list(_metrics["vram"])[-1] if _metrics["vram"] else 0
    ram  = list(_metrics["ram"])[-1]  if _metrics["ram"]  else 0
    cpu  = list(_metrics["cpu"])[-1]  if _metrics["cpu"]  else 0

    print(f"\n  {C.BOLD}GPU  RTX 5080{C.END}")
    print(f"  {'  Load':<20} [{bar(gpu  / 100, 30)}]  {gpu:.0f}%")
    print(f"  {'  VRAM':<20} [{bar(vram / 100, 30)}]  {vram:.0f}%  ({vram/100*16:.1f}/16 GB)")
    print(f"\n  {C.BOLD}System{C.END}")
    print(f"  {'  CPU Ryzen 9900X':<20} [{bar(cpu / 100, 30)}]  {cpu:.0f}%")
    vm = psutil.virtual_memory()
    print(f"  {'  RAM DDR5':<20} [{bar(ram / 100, 30)}]  {ram:.0f}%  "
          f"({vm.used/1e9:.1f}/{vm.total/1e9:.0f} GB)")
    print(f"\n{C.BOLD}{C.CYAN}{'─'*72}{C.END}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CHECKPOINT HELPERS                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_state(cfg):
    """Load generation state — enables crash-resume across sessions."""
    state_path = os.path.join(cfg["checkpoints_dir"], "state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)
    return {"completed_scenes": [], "bridge_frames": {}}


def save_state(cfg, state):
    os.makedirs(cfg["checkpoints_dir"], exist_ok=True)
    with open(os.path.join(cfg["checkpoints_dir"], "state.json"), "w") as f:
        json.dump(state, f, indent=2)


def save_bridge_frame(frames, scene_id, cfg):
    """
    Save the last frame of chunk N as the I2V conditioning image for chunk N+1.
    This PNG is the primary mechanism for visual context continuity between scenes.
    """
    os.makedirs(cfg["bridges_dir"], exist_ok=True)
    last_frame = frames[-1]
    if not isinstance(last_frame, Image.Image):
        last_frame = Image.fromarray(np.array(last_frame, dtype=np.uint8))
    if last_frame.size != (cfg["width"], cfg["height"]):
        last_frame = last_frame.resize((cfg["width"], cfg["height"]), Image.LANCZOS)
    path = os.path.join(cfg["bridges_dir"], f"bridge_after_scene_{scene_id:02d}.png")
    last_frame.save(path, "PNG")
    return path


def save_scene_frames(frames, scene_id, cfg):
    """Save all frames from a generated scene as lossless PNG files."""
    scene_dir = os.path.join(cfg["frames_dir"], f"scene_{scene_id:02d}")
    os.makedirs(scene_dir, exist_ok=True)
    paths = []
    base_frame = scene_id * cfg["frames_per_chunk"]
    for i, frame in enumerate(frames):
        img  = frame if isinstance(frame, Image.Image) \
               else Image.fromarray(np.array(frame, dtype=np.uint8))
        path = os.path.join(scene_dir, f"frame_{base_frame + i:06d}.png")
        img.save(path, "PNG")
        paths.append(path)
    return paths


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PIPELINE LOADERS                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_t2v_pipeline(cfg):
    """Text-to-video pipeline with RTX 5080 optimizations."""
    ok(f"Loading T2V pipeline: {cfg['model_t2v'].split('/')[-1]}")
    pipe = SanaVideoPipeline.from_pretrained(
        cfg["model_t2v"],
        torch_dtype=cfg["torch_dtype"],
    )
    pipe.vae.to(cfg["vae_dtype"])            # VAE always float32
    pipe.text_encoder.to(cfg["torch_dtype"])
    pipe.to("cuda")
    if cfg["cpu_offload"]:
        pipe.enable_model_cpu_offload()
    if cfg["attention_slicing"]:
        pipe.enable_attention_slicing()
    torch.backends.cuda.matmul.allow_tf32 = True  # Blackwell TF32 speedup
    ok("T2V pipeline ready")
    return pipe


def load_i2v_pipeline(cfg):
    """
    Image-to-video pipeline for bridge-frame conditioning.
    Falls back gracefully to None if SanaImageToVideoPipeline is unavailable.
    """
    ok("Loading I2V pipeline for context bridging…")
    try:
        from diffusers import SanaImageToVideoPipeline
        pipe = SanaImageToVideoPipeline.from_pretrained(
            cfg["model_i2v"],
            torch_dtype=cfg["torch_dtype"],
        )
        pipe.transformer.to(cfg["torch_dtype"])
        pipe.text_encoder.to(cfg["torch_dtype"])
        pipe.vae.to(cfg["vae_dtype"])
        pipe.to("cuda")
        if cfg["cpu_offload"]:
            pipe.enable_model_cpu_offload()
        if cfg["attention_slicing"]:
            pipe.enable_attention_slicing()
        ok("I2V bridge pipeline ready")
        return pipe
    except Exception as e:
        warn(f"I2V pipeline unavailable ({e}). All scenes will use T2V.")
        return None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SCENE GENERATION                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def generate_scene_t2v(pipe, scene, cfg, seed):
    """Generate a scene from text — used for scene 0 and I2V fallback."""
    prompt    = scene["prompt"]
    generator = torch.Generator(device="cuda").manual_seed(seed)
    print(f"    Prompt excerpt : …{prompt[60:120]}…")
    print(f"    Motion score   : {scene['motion_score']}  |  seed: {seed}  "
          f"|  steps: {cfg['num_inference_steps']}")
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        height=cfg["height"],
        width=cfg["width"],
        frames=cfg["frames_per_chunk"],
        guidance_scale=cfg["guidance_scale"],
        num_inference_steps=cfg["num_inference_steps"],
        generator=generator,
    )
    return result.frames[0]


def generate_scene_i2v(pipe_i2v, pipe_t2v, bridge_image_path, scene, cfg, seed):
    """
    Generate a scene conditioned on the last frame of the previous scene.

    Context bridge mechanism:
        1. Load the bridge PNG (last frame of scene N-1).
        2. Resize to match the generation resolution exactly.
        3. Pass as conditioning image to SanaImageToVideoPipeline.
        4. The model continues the visual state: lighting, color, composition
           are inherited; the prompt navigates the narrative forward.
        5. Falls back to T2V if I2V pipeline is unavailable.
    """
    if pipe_i2v is None:
        warn("I2V unavailable — falling back to T2V for this scene")
        return generate_scene_t2v(pipe_t2v, scene, cfg, seed)

    prompt    = scene["prompt"]
    generator = torch.Generator(device="cuda").manual_seed(seed)

    try:
        from diffusers.utils import load_image
        bridge_img = load_image(bridge_image_path).resize(
            (cfg["width"], cfg["height"]), Image.LANCZOS
        )
    except Exception as e:
        warn(f"Could not load bridge image ({e}) — falling back to T2V")
        return generate_scene_t2v(pipe_t2v, scene, cfg, seed)

    print(f"    [I2V BRIDGE]   Conditioning on last frame of previous scene")
    print(f"    Prompt excerpt : …{prompt[60:120]}…")
    print(f"    Motion score   : {scene['motion_score']}  |  seed: {seed}  "
          f"|  steps: {cfg['num_inference_steps']}")

    result = pipe_i2v(
        image=bridge_img,
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        height=cfg["height"],
        width=cfg["width"],
        frames=cfg["frames_per_chunk"],
        guidance_scale=cfg["guidance_scale"],
        num_inference_steps=cfg["num_inference_steps"],
        generator=generator,
    )
    return result.frames[0]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VIDEO ASSEMBLY                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def assemble_video(cfg):
    """
    Stitch all saved PNG frames into the final MP4.
    H.264 CRF-18, slow preset, +faststart for web-ready delivery.
    """
    header("🎬  Assembling final video")
    final_path = os.path.join(cfg["output_dir"], cfg["final_video"])

    all_frames = []
    for s in range(cfg["num_scenes"]):
        scene_dir = os.path.join(cfg["frames_dir"], f"scene_{s:02d}")
        if not os.path.exists(scene_dir):
            warn(f"Scene {s} frames missing — skipping in assembly")
            continue
        files = sorted(
            [os.path.join(scene_dir, f) for f in os.listdir(scene_dir)
             if f.endswith((".png", ".jpg"))]
        )
        all_frames.extend(files)

    if not all_frames:
        err("No frames found. Cannot assemble video.")
        return None

    ok(f"Found {len(all_frames)} frames — encoding CRF {cfg['crf']}, preset {cfg['preset']}")

    writer = imageio.get_writer(
        final_path,
        fps=cfg["fps"],
        codec="libx264",
        quality=None,
        output_params=[
            "-crf",       str(cfg["crf"]),
            "-preset",    cfg["preset"],
            "-pix_fmt",   "yuv420p",
            "-profile:v", "high",
            "-level",     "4.2",          # 720p compatible
            "-movflags",  "+faststart",   # web-optimized
        ],
    )

    last_update = time.time()
    for i, fp in enumerate(all_frames):
        try:
            writer.append_data(imageio.imread(fp))
        except Exception as e:
            warn(f"Skipping corrupt frame {os.path.basename(fp)}: {e}")
        if time.time() - last_update > 3:
            pct = (i + 1) / len(all_frames)
            print(f"    [{bar(pct, 30)}]  {pct*100:.0f}%  frame {i+1}/{len(all_frames)}")
            last_update = time.time()

    writer.close()

    if os.path.exists(final_path):
        size_mb      = os.path.getsize(final_path) / 1e6
        duration_s   = len(all_frames) / cfg["fps"]
        bitrate_mbps = (os.path.getsize(final_path) * 8) / duration_s / 1e6
        ok(f"Video saved  : {final_path}")
        ok(f"Size         : {size_mb:.1f} MB  |  Duration: {duration_s:.1f}s  "
           f"|  Bitrate: {bitrate_mbps:.1f} Mbps")
    else:
        err("Video assembly failed — check ffmpeg installation.")
        return None

    return final_path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PREFLIGHT CHECKS                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def preflight(cfg):
    header("🔍  Preflight checks")

    if not DIFFUSERS_OK:
        err("diffusers not installed.")
        err("Run: pip install git+https://github.com/huggingface/diffusers")
        return False

    if not torch.cuda.is_available():
        err("CUDA not available. Check NVIDIA drivers and PyTorch CUDA build.")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    ok(f"GPU  : {gpu_name}  ({vram_gb:.1f} GB VRAM detected)")

    ram  = psutil.virtual_memory()
    disk = shutil.disk_usage(os.path.abspath("."))
    ok(f"RAM  : {ram.total/1e9:.1f} GB total  |  {ram.available/1e9:.1f} GB available")
    ok(f"Disk : {disk.free/1e9:.1f} GB free")
    ok(f"PyTorch {torch.__version__}  |  bfloat16 ready")

    # Estimate storage: PNG at 704×1280 ≈ 2.5 MB per frame
    est_frames_gb = cfg["total_frames"] * 2.5 / 1024

    print(f"\n  {C.BOLD}Target video:{C.END}")
    print(f"    Resolution    : {cfg['width']}×{cfg['height']} (720p portrait)")
    print(f"    Duration      : {cfg['total_duration_seconds']:.1f}s (~1 minute)")
    print(f"    FPS           : {cfg['fps']}")
    print(f"    Total frames  : {cfg['total_frames']}")
    print(f"    Scenes        : {cfg['num_scenes']} × {cfg['frames_per_chunk']} frames "
          f"({cfg['frames_per_chunk']/cfg['fps']:.1f}s each)")
    print(f"    Model         : SANA-Video 2B 720p")
    print(f"    Est. storage  : ~{est_frames_gb:.1f} GB (PNG frames)")
    print(f"    Context bridge: "
          f"{'I2V last-frame conditioning' if cfg['use_i2v_bridge'] else 'Disabled (T2V only)'}")

    if disk.free / 1e9 < est_frames_gb * 1.5:
        warn(f"Low disk space! Need ~{est_frames_gb*1.5:.0f} GB, "
             f"have {disk.free/1e9:.1f} GB")

    # v1.0 FIX: was StatisticsTracker.hms(None, est_time_s) — TypeError
    #           corrected to StatisticsTracker.hms(est_time_s)
    est_time_s = cfg["num_chunks"] * 36
    print(f"\n    Est. generation time : {StatisticsTracker.hms(est_time_s)} "
          f"(~36s/scene on RTX 5080)")
    return True


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN ENTRY POINT                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    global _tracker_ref

    print(f"{C.BOLD}{C.CYAN}  sana-dental-1.py  v{__version__}  —  {__date__}{C.END}")

    cfg          = get_config()
    stat         = StatisticsTracker(cfg)
    _tracker_ref = stat

    # ── Pre-flight ────────────────────────────────────────────────────────
    if not preflight(cfg):
        return

    # ── Create output directories ─────────────────────────────────────────
    for d in [cfg["output_dir"], cfg["frames_dir"],
              cfg["bridges_dir"], cfg["checkpoints_dir"]]:
        os.makedirs(d, exist_ok=True)

    # ── Resume check ──────────────────────────────────────────────────────
    state      = load_state(cfg)
    completed  = set(state.get("completed_scenes", []))
    bridge_map = state.get("bridge_frames", {})  # {str(scene_id) → path}

    if completed:
        ok(f"Resuming — {len(completed)}/{cfg['num_scenes']} scenes already done: "
           f"{sorted(completed)}")
    else:
        ok("Starting fresh generation")

    print(f"\n{C.YELLOW}  Press Enter to start…  (Ctrl+C to abort at any time){C.END}")
    try:
        input()
    except KeyboardInterrupt:
        print("\nAborted.")
        return

    # ── Start monitoring thread ───────────────────────────────────────────
    stop_evt   = threading.Event()
    mon_thread = threading.Thread(target=_monitor_loop, args=(stop_evt,), daemon=True)
    mon_thread.start()

    # ── Load pipelines ────────────────────────────────────────────────────
    header("📦  Loading pipelines")
    pipe_t2v = load_t2v_pipeline(cfg)
    pipe_i2v = load_i2v_pipeline(cfg) if cfg["use_i2v_bridge"] else None

    stat.start()

    # ── Scene generation loop ─────────────────────────────────────────────
    header("🎬  Generating 12 scenes — SmartSmile 2250")
    sid = 0  # declare outside loop so except block can reference it safely

    try:
        for scene in SCENES:
            sid    = scene["id"]
            s_name = scene["name"]

            if sid in completed:
                ok(f"Scene {sid:02d} '{s_name}' already done — skipping")
                continue

            _print_live(sid, cfg["num_scenes"], s_name, stat)
            step(sid + 1, cfg["num_scenes"], f"Scene {sid:02d}: {s_name}")
            print(f"    Beat: {scene['beats']}")

            chunk_seed  = cfg["seed"] + sid   # monotonic seed drift
            t_start     = time.time()

            # Context bridge decision:
            #   Scene 0 → always T2V (no prior frame exists)
            #   Scene N → I2V conditioned on bridge_after_scene_{N-1}.png
            bridge_path = bridge_map.get(str(sid - 1))
            use_bridge  = (
                sid > 0
                and cfg["use_i2v_bridge"]
                and bridge_path is not None
                and os.path.exists(bridge_path)
            )

            if use_bridge:
                frames = generate_scene_i2v(
                    pipe_i2v, pipe_t2v, bridge_path, scene, cfg, chunk_seed
                )
            else:
                if sid > 0 and cfg["use_i2v_bridge"]:
                    warn(f"No bridge frame for scene {sid} — using T2V fallback")
                frames = generate_scene_t2v(pipe_t2v, scene, cfg, chunk_seed)

            elapsed = time.time() - t_start
            ok(f"Scene {sid:02d} generated in {elapsed:.1f}s  ({len(frames)} frames)")

            # Save frames (lossless PNG)
            save_scene_frames(frames, sid, cfg)
            ok(f"Frames saved → scene_{sid:02d}/")

            # Save bridge frame for next scene
            bp = save_bridge_frame(frames, sid, cfg)
            bridge_map[str(sid)] = bp
            ok(f"Bridge saved → {os.path.basename(bp)}")

            # Persist state for crash-resume
            stat.end_chunk(elapsed)
            completed.add(sid)
            state["completed_scenes"] = sorted(completed)
            state["bridge_frames"]    = bridge_map
            save_state(cfg, state)

            # VRAM cleanup between scenes
            del frames
            gc.collect()
            torch.cuda.empty_cache()

            _print_live(sid + 1, cfg["num_scenes"], "Cleaning VRAM…", stat)
            time.sleep(1.5)  # Allow VRAM to fully drain before next scene

    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}  Interrupted — progress saved. Re-run to resume.{C.END}")
        stop_evt.set()
        stat.summary()
        return

    except Exception as e:
        err(f"Generation error at scene {sid}: {e}")
        import traceback; traceback.print_exc()
        stop_evt.set()
        warn("Progress saved — re-run to resume from the last completed scene.")
        stat.summary()
        return

    finally:
        stop_evt.set()

    # ── Free VRAM before assembly ─────────────────────────────────────────
    header("🧹  Freeing VRAM before video assembly")
    del pipe_t2v
    if pipe_i2v is not None:
        del pipe_i2v
    gc.collect()
    torch.cuda.empty_cache()
    ok("VRAM cleared")

    # ── Assemble final video ──────────────────────────────────────────────
    final_video = assemble_video(cfg)

    # ── Save metrics JSON report ──────────────────────────────────────────
    report_path = os.path.join(
        cfg["output_dir"],
        f"report_v{__version__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    report = {
        "version":          __version__,
        "timestamp":        datetime.now().isoformat(),
        "config":           {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict))
                             else v for k, v in cfg.items()},
        "scenes":           [s["name"] for s in SCENES],
        "completed_scenes": sorted(completed),
        "chunk_times_s":    stat.chunk_times,
        "total_elapsed_s":  stat.elapsed(),
        "final_video":      final_video,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    ok(f"Report saved: {report_path}")

    # ── Final statistics ──────────────────────────────────────────────────
    stat.summary()

    print(f"\n{C.BOLD}{C.GREEN}{'═'*72}{C.END}")
    print(f"{C.BOLD}{C.GREEN}  ✨  SmartSmile 2250 — 1-Minute Dental Ad — COMPLETE{C.END}")
    print(f"{C.BOLD}{C.GREEN}{'═'*72}{C.END}")
    print(f"\n  {C.BOLD}📹  Output  :{C.END}  {final_video}")
    print(f"  {C.BOLD}📐  Spec    :{C.END}  {cfg['width']}×{cfg['height']}  |  {cfg['fps']} fps  "
          f"|  ~{cfg['total_duration_seconds']:.0f}s")
    print(f"  {C.BOLD}🧬  Model   :{C.END}  SANA-Video 2B 720p  (bfloat16 + float32 VAE)")
    print(f"  {C.BOLD}🌉  Bridge  :{C.END}  I2V last-frame conditioning "
          f"(11 bridges across 12 scenes)")
    print(f"  {C.BOLD}🔖  Version :{C.END}  v{__version__}\n")


if __name__ == "__main__":
    main()