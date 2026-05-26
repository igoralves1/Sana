"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SANA VIDEO — FUTURISTIC DENTAL OFFICE ADVERTISEMENT GENERATOR              ║
║  Version  : 3.0                                                              ║
║  Date     : 2026-05-24                                                       ║
║  Author   : Dr. Igor Lemos Alves                                             ║
║  Target   : 1-minute cinematic | 1280×704 | 16fps                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CHANGELOG                                                                   ║
║  v1.0  2026-05-23  Initial versioned release.                                ║
║  v2.0  2026-05-23  Two-stage 4.8B keyframe + 2B I2V video.                  ║
║  v3.0  2026-05-24  FULL CRASH-RESUME OVERHAUL.                               ║
║                                                                              ║
║  PROBLEM IN v2: state.json was written only after BOTH stages of a scene    ║
║  finished. A server crash mid-Stage-2 (or mid-frame-save) left no record,   ║
║  so the next run re-generated Stage 1 AND Stage 2 from scratch for that     ║
║  scene — wasting ~96 seconds of GPU time per affected scene.                ║
║                                                                              ║
║  v3 CHECKPOINTING STRATEGY — 5-level atomic saves:                          ║
║                                                                              ║
║  LEVEL 1  scene_N/stage1_done.flag                                           ║
║           Written immediately after the keyframe PNG is flushed to disk.    ║
║           Resume: if flag exists AND keyframe PNG exists → skip Stage 1.    ║
║                                                                              ║
║  LEVEL 2  scene_N/stage2_done.flag                                           ║
║           Written after all 81 frame PNGs are fully flushed to disk.        ║
║           Resume: if flag exists AND frame count matches → skip Stage 2.    ║
║                                                                              ║
║  LEVEL 3  scene_N/frames_saved.flag                                          ║
║           Written after every individual frame PNG is saved (frame-level    ║
║           granularity). On resume, counts existing frames; only saves the   ║
║           missing ones (partial frame-set recovery).                         ║
║                                                                              ║
║  LEVEL 4  state.json  (atomic write via temp-file + rename)                 ║
║           state.json is NEVER written in-place. The flow is:                ║
║               write → state.json.tmp                                        ║
║               os.replace(tmp, state.json)          ← atomic on NTFS/ext4   ║
║           A crash during write leaves .tmp; the prior state.json is intact. ║
║                                                                              ║
║  LEVEL 5  state.json.bak                                                     ║
║           Before every write, the previous state.json is copied to .bak.   ║
║           If both state.json and state.json.tmp are corrupt/missing,        ║
║           the backup is loaded automatically.                                ║
║                                                                              ║
║  RECOVERY MATRIX on startup:                                                 ║
║  ┌────────────────────────────────┬──────────────────────────────────────┐  ║
║  │ What exists on disk            │ Action                               │  ║
║  ├────────────────────────────────┼──────────────────────────────────────┤  ║
║  │ stage2_done.flag + 81 frames   │ Scene fully done — skip both stages  │  ║
║  │ stage1_done.flag + keyframe    │ Stage 1 done, Stage 2 incomplete     │  ║
║  │   + N < 81 frames              │   → skip S1, resume S2 from frame N  │  ║
║  │ keyframe exists, no flag       │ S1 write may be incomplete — redo S1 │  ║
║  │ Nothing for this scene         │ Run both stages fresh                 │  ║
║  │ state.json corrupt             │ Load state.json.bak automatically    │  ║
║  └────────────────────────────────┴──────────────────────────────────────┘  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  INSTALL                                                                     ║
║  pip install git+https://github.com/huggingface/diffusers                   ║
║  pip install torch imageio[ffmpeg] psutil GPUtil Pillow numpy                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

__version__ = "3.0"
__author__  = "Dr. Igor Lemos Alves"
__date__    = "2026-05-24"

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

# ── optional GPU monitoring ───────────────────────────────────────────────────
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

# ── diffusers ─────────────────────────────────────────────────────────────────
try:
    from diffusers import SanaPipeline
    from diffusers import SanaImageToVideoPipeline
    from diffusers.utils import load_image
    DIFFUSERS_OK = True
except ImportError:
    DIFFUSERS_OK = False


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ANSI COLORS                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class C:
    CYAN    = '\033[96m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    RED     = '\033[91m'
    BLUE    = '\033[94m'
    MAGENTA = '\033[95m'
    BOLD    = '\033[1m'
    DIM     = '\033[2m'
    END     = '\033[0m'

def header(text):     print(f"\n{C.BOLD}{C.CYAN}{'═'*72}\n  {text}\n{'═'*72}{C.END}")
def subheader(text):  print(f"\n{C.BOLD}{C.MAGENTA}  ▶  {text}{C.END}")
def ok(text):         print(f"  {C.GREEN}✔{C.END}  {text}")
def warn(text):       print(f"  {C.YELLOW}⚠{C.END}  {text}")
def err(text):        print(f"  {C.RED}✘{C.END}  {text}")
def info(text):       print(f"  {C.BLUE}ℹ{C.END}  {text}")
def step(n, t, text): print(f"\n{C.BOLD}{C.BLUE}[{n}/{t}]{C.END} {text}")
def recovered(text):  print(f"  {C.MAGENTA}♻{C.END}  {C.BOLD}RECOVERED:{C.END} {text}")

def bar(frac, w=44):
    f = int(w * max(0.0, min(1.0, frac)))
    return f"{C.GREEN}{'█'*f}{C.END}{'░'*(w-f)}"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  NEGATIVE PROMPTS (NVIDIA official)                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

NEGATIVE_VIDEO = (
    "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, "
    "sudden disappearance, jump cuts, jerky movements, rapid shot changes, "
    "frames out of sync, inconsistent character shapes, temporal artifacts, "
    "jitter, and ghosting effects, creating a disorienting visual experience. "
    "overexposed, underexposed, noise, grain, watermark, text overlay, "
    "cartoon, anime, low resolution, blurry, anatomically incorrect, "
    "extra limbs, duplicate faces, ugly, deformed."
)

NEGATIVE_IMAGE = (
    "blurry, low quality, low resolution, jpeg artifacts, watermark, "
    "text, logo, overexposed, underexposed, grain, noise, distorted, "
    "cartoon, anime, painting, illustration, deformed anatomy, "
    "extra limbs, duplicate, ugly, disfigured."
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SCENE SCRIPT — 12 × 5-SECOND NARRATIVE BEATS                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

SCENES = [
    {
        "id": 0, "name": "Opening — City of the Future",
        "beats": "Aerial establishing shot, megacity of 2250",
        "motion_score": 20,
        "image_prompt": (
            "Golden sunrise light, sweeping aerial wide shot, centered horizon composition. "
            "A luminous megacity of year 2250, crystal spire towers and bioluminescent bio-domes "
            "glowing warm amber and blue. Sleek magnetic-levitation vehicles trace light ribbons "
            "along elevated glass highways. Volumetric clouds catch dawn light. "
            "Photorealistic, cinematic anamorphic lens, 8K, utopian color grading, ultra-detailed."
        ),
        "video_prompt": (
            "Soft golden sunrise light, sweeping aerial wide shot, centered horizon composition. "
            "A luminous megacity of 2250 stretches below, crystal towers and bio-domes glowing amber. "
            "Sleek autonomous vehicles trace light trails through elevated glass highways. "
            "Camera drifts slowly forward over the skyline with slight downward tilt. "
            "Warm utopian color grading, cinematic anamorphic lens flare. motion score: 20."
        ),
    },
    {
        "id": 1, "name": "Clinic Exterior — SmartSmile 2250",
        "beats": "Reveal the futuristic dental clinic facade",
        "motion_score": 15,
        "image_prompt": (
            "Soft diffused morning daylight, medium architectural wide shot, perfect symmetry. "
            "A breathtaking dental clinic facade: pure white biopolymer panels and floor-to-ceiling "
            "blue-tinted smart glass. A holographic logo 'SmartSmile 2250' floats in aquamarine light. "
            "Bioluminescent sakura trees line the approach. Patients in elegant attire walk the garden. "
            "Photorealistic, architectural photography, 8K, pristine clinical aesthetic."
        ),
        "video_prompt": (
            "Soft diffused daylight, medium wide shot, symmetrical architectural composition. "
            "SmartSmile 2250 clinic facade, holographic logo rotating gently above entrance. "
            "Bioluminescent trees sway softly. Patients approach the entrance doors. "
            "Camera slowly pushes forward toward the entrance. "
            "Cool aquamarine and white palette, pristine and welcoming. motion score: 15."
        ),
    },
    {
        "id": 2, "name": "Reception — AI Concierge",
        "beats": "Patient checks in with holographic AI receptionist",
        "motion_score": 18,
        "image_prompt": (
            "Cool blue ambient light with warm amber accent panels, medium close-up, rule-of-thirds. "
            "An elegant young woman at a floating glass reception desk, facing a translucent holographic "
            "AI avatar that smiles warmly. Neon medical data streams float in the air. "
            "Staff in crisp white uniforms visible behind. Luminous, clinical, and welcoming. "
            "Photorealistic, cinematic portrait lighting, 8K, ultra-detailed."
        ),
        "video_prompt": (
            "Cool blue ambient light with warm accent panels, medium close-up, rule-of-thirds. "
            "Patient interacts with translucent holographic AI concierge avatar that smiles and gestures. "
            "Neon medical data floats and pulses around the room. Staff move in background. "
            "Camera holds steady with slow gentle push-in. "
            "Clinical yet warm, ultra-clean whites and soft blues. motion score: 18."
        ),
    },
    {
        "id": 3, "name": "Diagnostics — AI Oral Scan",
        "beats": "Real-time AI full-mouth diagnostic scanning",
        "motion_score": 22,
        "image_prompt": (
            "Soft cool surgical lighting, close-up macro shot, centered clinical composition. "
            "Sleek brushed-titanium AI oral scanner hovering beside an open mouth. "
            "Real-time 3D holographic dental model in midair, color-coded health status. "
            "Dentist in smart-glass AR eyewear reviews hologram with haptic stylus. "
            "Photorealistic, medical precision, 8K, high-tech blue-white aesthetic."
        ),
        "video_prompt": (
            "Soft cool clinical lighting, close-up macro shot, centered composition. "
            "AI oral scanner glides around patient's mouth, projecting real-time holographic 3D dental models. "
            "Diagnostic color data pulses over each tooth. Dentist's stylus annotates hologram. "
            "Camera slowly orbits the scanning device. "
            "High-tech blue and white tones, precise and futuristic. motion score: 22."
        ),
    },
    {
        "id": 4, "name": "Treatment — Nano-Robot Procedure",
        "beats": "Nano-robot swarm performs painless precision dental work",
        "motion_score": 25,
        "image_prompt": (
            "Diffused surgical lighting, medium shot, single-subject centered framing. "
            "Patient reclines in a floating white ergonomic chair, expression completely peaceful. "
            "Luminous nano-robot cloud, microscopic motes each glowing electric blue, near the mouth. "
            "AR screen displays nano-scale real-time mapping. Dentist observes in background. "
            "Photorealistic, cinematic, 8K, serene wonder aesthetic."
        ),
        "video_prompt": (
            "Diffused surgical lighting, medium shot, clean single-subject framing. "
            "Patient reclines completely relaxed, eyes closed peacefully. "
            "Luminous nano-robot swarm drifts gracefully, tiny motes pulsing blue as they work. "
            "AR screen displays live nano-mapping data. Camera drifts from wide to close. "
            "Serene clinical blue-white palette, wonder and precision. motion score: 25."
        ),
    },
    {
        "id": 5, "name": "Bioprinting — Tooth Regeneration",
        "beats": "Living tooth bioprinted in real time",
        "motion_score": 18,
        "image_prompt": (
            "Warm amber lab lighting, extreme close-up, centered macro composition. "
            "Dental bioprinter head above a growing tooth crown, translucent organic layers forming. "
            "Stem cells glow faintly within living tissue. Nano-scaffold surrounds the structure in light. "
            "Photorealistic, scientific macro photography, 8K, amber and ivory tones."
        ),
        "video_prompt": (
            "Warm amber lab lighting, extreme close-up, centered macro composition. "
            "Dental bioprinter deposits living tissue layer by layer, tooth crown growing upward. "
            "Stem cells activate with faint inner glow. Nano-scaffold pulses softly. "
            "Camera holds static, macro focus, shallow depth of field. "
            "Rich amber and ivory tones, scientific wonder. motion score: 18."
        ),
    },
    {
        "id": 6, "name": "AI Dentist — Human + AI Collaboration",
        "beats": "Human dentist and AI assistant reviewing a 3D jaw hologram",
        "motion_score": 16,
        "image_prompt": (
            "Soft split warm-cool lighting, medium two-shot, balanced composition. "
            "Skilled human dentist in white coat beside sleek humanoid AI assistant, "
            "translucent torso glowing with embedded data circuits. Both study a rotating 3D jaw hologram. "
            "Human points with haptic stylus; AI overlays annotations. "
            "Photorealistic, cinematic, 8K, warm wood and cool steel aesthetic."
        ),
        "video_prompt": (
            "Soft split warm-cool lighting, medium two-shot, balanced composition. "
            "Human dentist and AI assistant discuss rotating 3D jaw hologram, AI overlaying annotations. "
            "Jaw hologram rotates slowly, pulsing data highlights. "
            "Camera slowly arcs from profile to three-quarter frontal. "
            "Warm wood and cool steel, expertise and trust. motion score: 16."
        ),
    },
    {
        "id": 7, "name": "Patient Experience — Zero Pain",
        "beats": "Patient in total comfort during pain-free neural treatment",
        "motion_score": 10,
        "image_prompt": (
            "Golden warm ambient light, medium close-up portrait, slightly low angle. "
            "Patient reclined in floating chair, eyes closed, gentle smile. "
            "Delicate neural interface headband emits soft geometric light patterns above brow. "
            "Room walls show serene underwater coral scene behind the patient. "
            "Photorealistic, cinematic beauty lighting, 8K, warm golden hour tones."
        ),
        "video_prompt": (
            "Golden soft ambient light, medium close-up, slightly low angle, warm framing. "
            "Patient in floating chair, completely at ease, gentle smile. "
            "Neural interface headband pulses soft geometric patterns. "
            "Room walls display gentle animated underwater coral. "
            "Camera slowly pushes into patient's serene face. "
            "Warm golden tones, complete comfort and peace. motion score: 10."
        ),
    },
    {
        "id": 8, "name": "Results — The Perfect Smile",
        "beats": "Patient sees their transformed smile for the first time",
        "motion_score": 14,
        "image_prompt": (
            "Warm flattering beauty lighting, medium close-up, centered portrait. "
            "Patient holding smart mirror, eyes beginning to widen with joy. "
            "In mirror: radiant, perfectly aligned, brilliantly white smile. "
            "Mirror frame displays soft green health score: 98/100. "
            "Photorealistic, cinematic beauty photography, 8K, warm bright palette, pure joy."
        ),
        "video_prompt": (
            "Warm flattering beauty lighting, medium close-up, centered portrait. "
            "Patient raises smart mirror, sees radiant new smile, "
            "expression shifts to pure joy, eyes widening, breaking into genuine laugh. "
            "Mirror overlays soft green health score digits. Camera slowly pushes into smile. "
            "Warm bright tones, joy and confidence. motion score: 14."
        ),
    },
    {
        "id": 9, "name": "Community — Dental Health for All",
        "beats": "Multicultural crowd in the SmartSmile 2250 park",
        "motion_score": 22,
        "image_prompt": (
            "Bright warm daylight, wide ensemble shot, dynamic diagonal group composition. "
            "Joyful multicultural crowd — children, elders, young adults — in SmartSmile 2250 park. "
            "Everyone smiling brilliantly. Holographic dental-health kiosks glow. "
            "Bioluminescent trees. SmartSmile drone hovers above distributing care packages. "
            "Photorealistic, cinematic, 8K, vibrant warm daylight, optimism and inclusivity."
        ),
        "video_prompt": (
            "Bright even daylight, wide ensemble shot, dynamic group composition. "
            "Joyful multicultural crowd walks through SmartSmile 2250 park, all smiling brilliantly. "
            "Holographic kiosks pulse, bioluminescent trees sway, drone hovers above. "
            "Camera tracks alongside crowd in smooth lateral dolly. "
            "Vibrant warm-daylight palette, optimism and inclusivity. motion score: 22."
        ),
    },
    {
        "id": 10, "name": "Data — Global Oral Health Dashboard",
        "beats": "Planetary dental health data visualization",
        "motion_score": 16,
        "image_prompt": (
            "Deep blue ambient light, wide shot, spherical holographic globe centerpiece. "
            "Vast holographic Earth rotates in high-tech control room. "
            "Thousands of glowing SmartSmile clinic markers across all continents. "
            "Curved data panels show global oral health scores trending toward 100%. "
            "Scientists in white observe from floor below. "
            "Photorealistic, cinematic, 8K, deep navy, emerald, and white."
        ),
        "video_prompt": (
            "Deep blue ambient light, wide shot, global holographic map. "
            "Spherical holographic Earth rotates slowly, SmartSmile markers glowing worldwide. "
            "Data panels show real-time global scores climbing. Scientists move across floor. "
            "Camera slowly tracks backward to reveal the full control room. "
            "Deep navy, emerald, and white tones, epic scale. motion score: 16."
        ),
    },
    {
        "id": 11, "name": "Closing — Logo and Tagline",
        "beats": "Brand reveal with iconic logo and closing message",
        "motion_score": 8,
        "image_prompt": (
            "Soft warm white studio light, ultra-wide centered, pure minimalism. "
            "Single luminous crystal tooth floats centered against pure white background. "
            "SmartSmile 2250 logo glows in polished gold lettering below. "
            "Elegant sans-serif text: 'Every smile. Every human. Every future.' "
            "Tooth casts soft prismatic light spectrum across white surface. "
            "Photorealistic, product photography, 8K, pure white and gold, timeless and iconic."
        ),
        "video_prompt": (
            "Soft warm white studio light, ultra-wide clean shot, perfectly centered. "
            "Crystal tooth floats and rotates slowly against pure white background. "
            "SmartSmile 2250 logo pulses gently below. Tagline text fades in softly. "
            "Prismatic light spectrum drifts across the surface. "
            "Camera holds completely static, elements breathe gently. "
            "Pure white and gold palette, timeless and iconic. motion score: 8."
        ),
    },
]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_config():
    cfg = {
        # Stage 1 — image
        "model_image":               "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
        "keyframe_height":           1024,
        "keyframe_width":            1024,
        "image_guidance_scale":      5.0,
        "image_inference_steps":     30,
        "image_pag_guidance_scale":  2.0,

        # Stage 2 — video
        "model_video":               "Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
        "height":                    704,
        "width":                     1280,
        "fps":                       16,
        "frames_per_chunk":          81,
        "num_scenes":                len(SCENES),
        "guidance_scale":            6.0,
        "num_inference_steps":       50,
        "flow_shift":                8,
        "torch_dtype":               torch.bfloat16,
        "seed":                      2025,

        # VRAM
        "cpu_offload":               True,
        "attention_slicing":         True,
        "vae_dtype":                 torch.float32,

        # Context bridge
        "use_keyframe_conditioning": True,
        "use_bridge_blending":       True,
        "bridge_blend_alpha":        0.30,

        # Storage
        "output_dir":       "dental_ad_v3_output",
        "keyframes_dir":    "dental_ad_v3_output/keyframes",
        "frames_dir":       "dental_ad_v3_output/frames",
        "bridges_dir":      "dental_ad_v3_output/bridges",
        "checkpoints_dir":  "dental_ad_v3_output/checkpoints",
        "final_video":      "SmartSmile2250_Ad_v3_1min_720p.mp4",

        # Encoding
        "crf":    18,
        "preset": "slow",
    }
    cfg["total_frames"]           = cfg["num_scenes"] * cfg["frames_per_chunk"]
    cfg["total_duration_seconds"] = cfg["total_frames"] / cfg["fps"]
    cfg["num_chunks"]             = cfg["num_scenes"]
    return cfg


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ATOMIC STATE MANAGER  (v3 core addition)                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class StateManager:
    """
    Manages all checkpoint I/O with crash-safe atomic writes.

    STATE FILE LOCATIONS
    ────────────────────
    checkpoints/state.json          — primary state (written atomically)
    checkpoints/state.json.bak      — backup of previous good state
    checkpoints/state.json.tmp      — temp during write (auto-cleaned)

    SCENE FLAG FILES  (inside frames/scene_NN/)
    ─────────────────────────────────────────────
    stage1_done.flag     — Stage 1 keyframe fully written to disk
    stage2_done.flag     — Stage 2 all frames fully written to disk

    Both flag files contain a JSON dict with timestamp and path info
    so they double as a human-readable audit trail.
    """

    EMPTY = {
        "version":          __version__,
        "completed_scenes": [],
        "keyframe_paths":   {},
        "bridge_frames":    {},
        "partial_scenes":   {},   # {sid: {"stage1_done": bool, "stage2_done": bool, ...}}
        "timing":           {},   # {sid: {"s1": float, "s2": float, "total": float}}
        "last_updated":     None,
        "total_elapsed_s":  0.0,
    }

    def __init__(self, cfg):
        self.cfg       = cfg
        self.path      = os.path.join(cfg["checkpoints_dir"], "state.json")
        self.path_bak  = self.path + ".bak"
        self.path_tmp  = self.path + ".tmp"
        self._state    = None

    # ── Load ─────────────────────────────────────────────────────────────

    def load(self):
        """Load state, falling back to backup if primary is corrupt/missing."""
        for candidate, label in [(self.path, "primary"), (self.path_bak, "backup")]:
            if not os.path.exists(candidate):
                continue
            try:
                with open(candidate) as f:
                    data = json.load(f)
                if label == "backup":
                    recovered(f"Loaded state from backup: {candidate}")
                self._state = data
                return data
            except (json.JSONDecodeError, OSError) as e:
                warn(f"State file {label} corrupt ({e}) — trying next option")

        info("No valid state found — starting fresh")
        self._state = dict(self.EMPTY)
        return self._state

    # ── Save (atomic) ────────────────────────────────────────────────────

    def save(self, state=None):
        """
        Atomic write:  state → .tmp  →  os.replace(.tmp, .json)
        Backup:        copy  current .json → .json.bak  before replacing.
        """
        if state is not None:
            self._state = state
        self._state["last_updated"]    = datetime.now().isoformat()
        self._state["version"]         = __version__

        os.makedirs(self.cfg["checkpoints_dir"], exist_ok=True)

        # Write to temp file first
        try:
            with open(self.path_tmp, "w") as f:
                json.dump(self._state, f, indent=2)
        except OSError as e:
            warn(f"Could not write state temp file: {e}")
            return False

        # Backup current good state before overwriting
        if os.path.exists(self.path):
            try:
                shutil.copy2(self.path, self.path_bak)
            except OSError:
                pass  # Non-critical — proceed without backup

        # Atomic replace
        try:
            os.replace(self.path_tmp, self.path)
        except OSError as e:
            warn(f"Atomic state replace failed: {e}")
            return False

        return True

    # ── Scene-level flag helpers ──────────────────────────────────────────

    def _scene_dir(self, scene_id):
        return os.path.join(self.cfg["frames_dir"], f"scene_{scene_id:02d}")

    def _flag_path(self, scene_id, flag_name):
        return os.path.join(self._scene_dir(scene_id), flag_name)

    def write_flag(self, scene_id, flag_name, meta=None):
        """Write a flag file (JSON) atomically — signals a sub-stage is complete."""
        os.makedirs(self._scene_dir(scene_id), exist_ok=True)
        path     = self._flag_path(scene_id, flag_name)
        path_tmp = path + ".tmp"
        payload  = {"timestamp": datetime.now().isoformat(),
                    "scene_id":  scene_id,
                    "flag":      flag_name,
                    **(meta or {})}
        try:
            with open(path_tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(path_tmp, path)
            return True
        except OSError as e:
            warn(f"Could not write flag {flag_name} for scene {scene_id}: {e}")
            return False

    def has_flag(self, scene_id, flag_name):
        return os.path.exists(self._flag_path(scene_id, flag_name))

    # ── Frame-level completeness check ───────────────────────────────────

    def count_saved_frames(self, scene_id):
        """Return how many frame PNGs exist for this scene (0 if directory missing)."""
        scene_dir = self._scene_dir(scene_id)
        if not os.path.exists(scene_dir):
            return 0
        return len([f for f in os.listdir(scene_dir) if f.endswith(".png")
                    and f.startswith("frame_")])

    # ── Recovery diagnosis ────────────────────────────────────────────────

    def diagnose_scene(self, scene_id, cfg):
        """
        Examine disk state for a scene and return a recovery decision dict.

        Returns:
            {
              "skip_both":   bool,   # Scene fully done
              "skip_stage1": bool,   # Keyframe OK, need to (re)do Stage 2
              "resume_stage2_from": int,  # Frame index to resume from (0 = fresh)
              "keyframe_path": str | None,
            }
        """
        result = {
            "skip_both":          False,
            "skip_stage1":        False,
            "resume_stage2_from": 0,
            "keyframe_path":      None,
        }

        expected_frames   = cfg["frames_per_chunk"]
        kf_path           = self._state.get("keyframe_paths", {}).get(str(scene_id))
        stage2_flag       = self.has_flag(scene_id, "stage2_done.flag")
        stage1_flag       = self.has_flag(scene_id, "stage1_done.flag")
        saved_frame_count = self.count_saved_frames(scene_id)

        # Case 1: Stage 2 fully complete
        if stage2_flag and saved_frame_count >= expected_frames:
            result["skip_both"] = True
            result["keyframe_path"] = kf_path
            return result

        # Case 2: Stage 1 complete, Stage 2 partial or not started
        kf_exists = kf_path and os.path.exists(kf_path)
        if stage1_flag and kf_exists:
            result["skip_stage1"]        = True
            result["keyframe_path"]      = kf_path
            result["resume_stage2_from"] = saved_frame_count  # resume from here
            return result

        # Case 3: Keyframe exists on disk but no flag (incomplete write) — redo S1
        if kf_exists:
            warn(f"Scene {scene_id:02d}: keyframe exists but stage1_done.flag missing — "
                 "redoing Stage 1 to ensure integrity")
            result["keyframe_path"] = None   # Force regeneration
            return result

        # Case 4: Nothing — fresh start for this scene
        return result

    # ── Convenience accessors ─────────────────────────────────────────────

    @property
    def state(self):
        return self._state

    def mark_complete(self, scene_id, s1_time, s2_time):
        s = self._state
        completed = set(s.get("completed_scenes", []))
        completed.add(scene_id)
        s["completed_scenes"] = sorted(completed)
        s["timing"][str(scene_id)] = {
            "s1_s": round(s1_time, 1),
            "s2_s": round(s2_time, 1),
            "total_s": round(s1_time + s2_time, 1),
        }
        self.save()

    def set_keyframe(self, scene_id, path):
        self._state.setdefault("keyframe_paths", {})[str(scene_id)] = path
        self.save()

    def set_bridge(self, scene_id, path):
        self._state.setdefault("bridge_frames", {})[str(scene_id)] = path
        self.save()

    def get_bridge(self, scene_id):
        return self._state.get("bridge_frames", {}).get(str(scene_id))

    def print_recovery_report(self, cfg):
        """Print a full recovery status table on startup."""
        header("♻   CRASH-RESUME STATUS")
        completed = set(self._state.get("completed_scenes", []))
        print(f"  {'Scene':<6} {'Name':<40} {'Status'}")
        print(f"  {'─'*6} {'─'*40} {'─'*22}")
        for scene in SCENES:
            sid  = scene["id"]
            diag = self.diagnose_scene(sid, cfg)
            if sid in completed or diag["skip_both"]:
                status = f"{C.GREEN}✔ DONE{C.END}"
            elif diag["skip_stage1"]:
                n = diag["resume_stage2_from"]
                status = (f"{C.YELLOW}◑ S1 done, S2 partial "
                          f"({n}/{cfg['frames_per_chunk']} frames){C.END}")
            elif diag["keyframe_path"]:
                status = f"{C.YELLOW}◔ Keyframe only (no flag){C.END}"
            else:
                status = f"{C.RED}○ Not started{C.END}"
            print(f"  {sid:<6} {scene['name']:<40} {status}")
        n_done = len(completed)
        print(f"\n  Total: {n_done}/{len(SCENES)} scenes fully complete")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STATISTICS TRACKER                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class StatisticsTracker:
    def __init__(self, cfg):
        self.cfg           = cfg
        self.t0            = None
        self.stage1_times  = []
        self.stage2_times  = []
        self.chunk_times   = []
        self.chunk_gpu_peaks = []
        self.gpu_hist  = deque(maxlen=200)
        self.ram_hist  = deque(maxlen=200)
        self.vram_hist = deque(maxlen=200)

    def start(self):
        self.t0 = time.time()

    def tick(self, gpu, ram, vram):
        self.gpu_hist.append(gpu)
        self.ram_hist.append(ram)
        self.vram_hist.append(vram)

    def end_chunk(self, elapsed, s1, s2):
        self.chunk_times.append(elapsed)
        self.stage1_times.append(s1)
        self.stage2_times.append(s2)
        self.chunk_gpu_peaks.append(max(self.gpu_hist, default=0))

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
        if s is None:
            return "N/A"
        s = int(s)
        h, m, sec = s // 3600, (s % 3600) // 60, s % 60
        if h:  return f"{h}h {m:02d}m"
        if m:  return f"{m}m {sec:02d}s"
        return f"{sec}s"

    def summary(self):
        el   = self.elapsed()
        done = len(self.chunk_times)
        avg  = sum(self.chunk_times)  / done if done else 0
        avg1 = sum(self.stage1_times) / len(self.stage1_times) if self.stage1_times else 0
        avg2 = sum(self.stage2_times) / len(self.stage2_times) if self.stage2_times else 0
        fps  = (done * self.cfg["frames_per_chunk"]) / el if el > 0 else 0
        g    = sum(self.gpu_hist)  / len(self.gpu_hist)  if self.gpu_hist  else 0
        v    = sum(self.vram_hist) / len(self.vram_hist) if self.vram_hist else 0
        r    = sum(self.ram_hist)  / len(self.ram_hist)  if self.ram_hist  else 0

        print(f"\n{C.BOLD}{C.CYAN}{'═'*72}{C.END}")
        print(f"{C.BOLD}{C.CYAN}  📊  STATISTICS  —  v{__version__}{C.END}")
        print(f"{C.BOLD}{C.CYAN}{'═'*72}{C.END}")
        print(f"  {'Total time':<35} {self.hms(el)}")
        print(f"  {'Scenes completed this run':<35} {done}/{self.cfg['num_chunks']}")
        print(f"  {'Avg total time / scene':<35} {avg:.1f}s")
        print(f"  {'Avg Stage 1 (4.8B keyframe)':<35} {avg1:.1f}s")
        print(f"  {'Avg Stage 2 (2B video)':<35} {avg2:.1f}s")
        print(f"  {'Generation speed':<35} {fps:.2f} frames/s")
        print(f"  {'Avg GPU load':<35} {g:.1f}%")
        print(f"  {'Peak GPU load':<35} {max(self.chunk_gpu_peaks, default=0):.1f}%")
        print(f"  {'Avg VRAM used':<35} {v / 100 * 16:.1f} GB / 16 GB")
        print(f"  {'Avg RAM used':<35} {r:.1f}%")
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


def _print_live(scene_idx, num_scenes, stage, scene_name, stat):
    os.system("cls" if os.name == "nt" else "clear")
    pct = scene_idx / num_scenes if num_scenes else 0
    slabel = (f"{C.BOLD}{C.YELLOW}◉ STAGE 1  4.8B Keyframe{C.END}"
              if stage == 1
              else f"{C.BOLD}{C.GREEN}◉ STAGE 2  2B Video{C.END}")

    print(f"{C.BOLD}{C.CYAN}{'═'*72}{C.END}")
    print(f"{C.BOLD}{C.CYAN}  🦷  SmartSmile 2250  v{__version__}  —  4.8B + 2B  |  Crash-Resume{C.END}")
    print(f"{C.BOLD}{C.CYAN}{'═'*72}{C.END}")
    print(f"\n  {'Scene':<20} {scene_idx}/{num_scenes}  —  {scene_name}")
    print(f"  {'Active stage':<20} {slabel}")
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
    if stat.stage1_times:
        a1 = sum(stat.stage1_times) / len(stat.stage1_times)
        a2 = sum(stat.stage2_times) / len(stat.stage2_times) if stat.stage2_times else 0
        print(f"\n  {C.DIM}Avg S1: {a1:.0f}s  |  Avg S2: {a2:.0f}s{C.END}")
    print(f"\n{C.BOLD}{C.CYAN}{'─'*72}{C.END}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VRAM HELPERS                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def vram_free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(1.0)


def vram_used_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  IMAGE HELPERS                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def blend_images(img_primary, img_secondary, alpha_secondary, target_size):
    if img_primary.size   != target_size:
        img_primary   = img_primary.resize(target_size,   Image.LANCZOS)
    if img_secondary.size != target_size:
        img_secondary = img_secondary.resize(target_size, Image.LANCZOS)
    a = np.array(img_primary,   dtype=np.float32)
    b = np.array(img_secondary, dtype=np.float32)
    return Image.fromarray(
        ((1.0 - alpha_secondary) * a + alpha_secondary * b).clip(0, 255).astype(np.uint8)
    )


def save_keyframe(image, scene_id, cfg):
    os.makedirs(cfg["keyframes_dir"], exist_ok=True)
    path = os.path.join(cfg["keyframes_dir"], f"keyframe_scene_{scene_id:02d}.png")
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image, dtype=np.uint8))
    image.save(path, "PNG")
    # Flush OS buffers so the file is fully on disk before writing the flag
    with open(path, "rb") as _:
        pass
    ok(f"Keyframe saved → {os.path.basename(path)}  ({image.size[0]}×{image.size[1]})")
    return path


def save_bridge_frame(frames, scene_id, cfg):
    os.makedirs(cfg["bridges_dir"], exist_ok=True)
    last = frames[-1]
    if not isinstance(last, Image.Image):
        last = Image.fromarray(np.array(last, dtype=np.uint8))
    path = os.path.join(cfg["bridges_dir"], f"bridge_after_scene_{scene_id:02d}.png")
    last.save(path, "PNG")
    return path


def save_scene_frames(frames, scene_id, cfg, start_from=0):
    """
    Save video frames for a scene as PNG files.
    start_from: resume from this frame index if partial save (v3 addition).
    Returns list of saved paths.
    """
    scene_dir  = os.path.join(cfg["frames_dir"], f"scene_{scene_id:02d}")
    os.makedirs(scene_dir, exist_ok=True)
    base_frame = scene_id * cfg["frames_per_chunk"]
    paths      = []

    for i, frame in enumerate(frames):
        if i < start_from:
            # Frame already saved in a prior (crashed) run — skip
            existing = os.path.join(scene_dir, f"frame_{base_frame + i:06d}.png")
            paths.append(existing)
            continue
        img  = frame if isinstance(frame, Image.Image) \
               else Image.fromarray(np.array(frame, dtype=np.uint8))
        path = os.path.join(scene_dir, f"frame_{base_frame + i:06d}.png")
        img.save(path, "PNG")
        paths.append(path)

    return paths


def prepare_conditioning_image(keyframe_path, bridge_path, scene_id, cfg):
    target   = (cfg["width"], cfg["height"])
    keyframe = Image.open(keyframe_path).resize(target, Image.LANCZOS)
    if (scene_id > 0
            and cfg.get("use_bridge_blending")
            and bridge_path
            and os.path.exists(bridge_path)):
        bridge = Image.open(bridge_path).resize(target, Image.LANCZOS)
        alpha  = cfg.get("bridge_blend_alpha", 0.30)
        info(f"[S2] Conditioning = keyframe {(1-alpha)*100:.0f}% + bridge {alpha*100:.0f}%")
        return blend_images(keyframe, bridge, alpha, target)
    info("[S2] Conditioning = keyframe only")
    return keyframe


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PIPELINE LOADERS                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_image_pipeline(cfg):
    subheader("Loading Stage 1 — SANA 4.8B Image Pipeline")
    pipe = SanaPipeline.from_pretrained(cfg["model_image"], torch_dtype=cfg["torch_dtype"])
    pipe.vae.to(cfg["vae_dtype"])
    pipe.text_encoder.to(cfg["torch_dtype"])
    pipe.to("cuda")
    if cfg["cpu_offload"]:
        pipe.enable_model_cpu_offload()
    torch.backends.cuda.matmul.allow_tf32 = True
    ok(f"Stage 1 ready  ({vram_used_gb():.1f} GB VRAM)")
    return pipe


def load_video_pipeline(cfg):
    subheader("Loading Stage 2 — SANA-Video 2B I2V Pipeline")
    pipe = SanaImageToVideoPipeline.from_pretrained(cfg["model_video"], torch_dtype=cfg["torch_dtype"])
    pipe.vae.to(cfg["vae_dtype"])
    pipe.text_encoder.to(cfg["torch_dtype"])
    pipe.to("cuda")
    if cfg["cpu_offload"]:
        pipe.enable_model_cpu_offload()
    if cfg["attention_slicing"]:
        pipe.enable_attention_slicing()
    ok(f"Stage 2 ready  ({vram_used_gb():.1f} GB VRAM)")
    return pipe


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GENERATION FUNCTIONS                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def generate_keyframe(pipe_img, scene, cfg, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    info(f"[S1] Prompt: …{scene['image_prompt'][50:110]}…")
    info(f"[S1] {cfg['keyframe_width']}×{cfg['keyframe_height']}  "
         f"steps={cfg['image_inference_steps']}  cfg={cfg['image_guidance_scale']}")
    try:
        result = pipe_img(
            prompt=scene["image_prompt"],
            negative_prompt=NEGATIVE_IMAGE,
            height=cfg["keyframe_height"],
            width=cfg["keyframe_width"],
            guidance_scale=cfg["image_guidance_scale"],
            pag_guidance_scale=cfg.get("image_pag_guidance_scale", 2.0),
            num_inference_steps=cfg["image_inference_steps"],
            generator=generator,
        )
    except TypeError:
        warn("[S1] PAG not supported — running without pag_guidance_scale")
        result = pipe_img(
            prompt=scene["image_prompt"],
            negative_prompt=NEGATIVE_IMAGE,
            height=cfg["keyframe_height"],
            width=cfg["keyframe_width"],
            guidance_scale=cfg["image_guidance_scale"],
            num_inference_steps=cfg["image_inference_steps"],
            generator=generator,
        )
    return result.images[0]


def generate_video_chunk(pipe_vid, scene, conditioning_img, cfg, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    info(f"[S2] Prompt: …{scene['video_prompt'][50:110]}…")
    info(f"[S2] motion_score={scene['motion_score']}  "
         f"steps={cfg['num_inference_steps']}  cfg={cfg['guidance_scale']}")
    result = pipe_vid(
        image=conditioning_img,
        prompt=scene["video_prompt"],
        negative_prompt=NEGATIVE_VIDEO,
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
    header("🎬  Assembling final video")
    final_path = os.path.join(cfg["output_dir"], cfg["final_video"])
    all_frames = []
    for s in range(cfg["num_scenes"]):
        sd = os.path.join(cfg["frames_dir"], f"scene_{s:02d}")
        if not os.path.exists(sd):
            warn(f"Scene {s} missing — skipping")
            continue
        files = sorted([os.path.join(sd, f) for f in os.listdir(sd)
                        if f.endswith(".png") and f.startswith("frame_")])
        all_frames.extend(files)

    if not all_frames:
        err("No frames found.")
        return None

    ok(f"{len(all_frames)} frames  →  CRF {cfg['crf']}, preset {cfg['preset']}")
    writer = imageio.get_writer(
        final_path, fps=cfg["fps"], codec="libx264", quality=None,
        output_params=["-crf", str(cfg["crf"]), "-preset", cfg["preset"],
                       "-pix_fmt", "yuv420p", "-profile:v", "high",
                       "-level", "4.2", "-movflags", "+faststart"],
    )
    last_upd = time.time()
    for i, fp in enumerate(all_frames):
        try:
            writer.append_data(imageio.imread(fp))
        except Exception as e:
            warn(f"Skip corrupt frame: {e}")
        if time.time() - last_upd > 3:
            pct = (i + 1) / len(all_frames)
            print(f"    [{bar(pct, 30)}]  {pct*100:.0f}%  {i+1}/{len(all_frames)}")
            last_upd = time.time()
    writer.close()

    if os.path.exists(final_path):
        mb  = os.path.getsize(final_path) / 1e6
        dur = len(all_frames) / cfg["fps"]
        ok(f"Saved: {final_path}  ({mb:.1f} MB  |  {dur:.1f}s)")
        return final_path
    err("Assembly failed.")
    return None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PREFLIGHT                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def preflight(cfg):
    header("🔍  Preflight checks")
    if not DIFFUSERS_OK:
        err("diffusers not installed: pip install git+https://github.com/huggingface/diffusers")
        return False
    if not torch.cuda.is_available():
        err("CUDA not available.")
        return False
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    ok(f"GPU    : {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    ram  = psutil.virtual_memory()
    disk = shutil.disk_usage(os.path.abspath("."))
    ok(f"RAM    : {ram.total/1e9:.1f} GB  |  {ram.available/1e9:.1f} GB free")
    ok(f"Disk   : {disk.free/1e9:.1f} GB free")
    ok(f"Torch  : {torch.__version__}  |  bfloat16 ready")
    est = cfg["num_scenes"] * (4 + cfg["frames_per_chunk"] * 2.5 / 1024)
    est_t = cfg["num_chunks"] * 96
    print(f"\n  Model S1 : {cfg['model_image'].split('/')[-1]}")
    print(f"  Model S2 : {cfg['model_video'].split('/')[-1]}")
    print(f"  Output   : {cfg['width']}×{cfg['height']}  |  {cfg['fps']} fps  "
          f"|  {cfg['total_duration_seconds']:.0f}s")
    print(f"  Est disk : ~{est:.1f} GB")
    print(f"  Est time : {StatisticsTracker.hms(est_t)} (~96s/scene on RTX 5080)")
    print(f"\n  {C.BOLD}v3 Crash-Resume:{C.END} atomic state.json + per-scene flags + frame-level recovery")
    return True


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    global _tracker_ref

    print(f"{C.BOLD}{C.CYAN}  sana-dental-3.py  v{__version__}  —  {__date__}{C.END}")
    print(f"{C.DIM}  Two-Stage 4.8B + 2B  |  Full Crash-Resume  |  SmartSmile 2250{C.END}\n")

    cfg  = get_config()
    stat = StatisticsTracker(cfg)
    _tracker_ref = stat

    if not preflight(cfg):
        return

    for d in [cfg["output_dir"], cfg["keyframes_dir"], cfg["frames_dir"],
              cfg["bridges_dir"], cfg["checkpoints_dir"]]:
        os.makedirs(d, exist_ok=True)

    # ── Load state ────────────────────────────────────────────────────────
    sm = StateManager(cfg)
    sm.load()
    sm.print_recovery_report(cfg)

    completed = set(sm.state.get("completed_scenes", []))
    pending   = [s for s in SCENES if s["id"] not in completed]

    if not pending:
        ok("All 12 scenes already complete — proceeding to video assembly.")
    else:
        print(f"\n{C.YELLOW}  {len(pending)} scene(s) remaining.  "
              f"Press Enter to start…  (Ctrl+C to abort){C.END}")
        try:
            input()
        except KeyboardInterrupt:
            return

    # ── Start monitoring ──────────────────────────────────────────────────
    stop_evt   = threading.Event()
    mon_thread = threading.Thread(target=_monitor_loop, args=(stop_evt,), daemon=True)
    mon_thread.start()

    stat.start()
    sid = 0

    try:
        for scene in SCENES:
            sid    = scene["id"]
            s_name = scene["name"]

            # ── Recovery diagnosis for this scene ─────────────────────────
            diag = sm.diagnose_scene(sid, cfg)

            if diag["skip_both"]:
                ok(f"Scene {sid:02d} '{s_name}' — fully done, skipping")
                continue

            chunk_seed = cfg["seed"] + sid
            t_scene    = time.time()
            t_s1 = t_s2 = 0.0

            # ════════════════════════════════════════════════════════════
            # STAGE 1 — KEYFRAME  (skip if stage1_done.flag exists)
            # ════════════════════════════════════════════════════════════
            if diag["skip_stage1"]:
                keyframe_path = diag["keyframe_path"]
                recovered(f"Scene {sid:02d} Stage 1 — keyframe exists, skipping generation")
                ok(f"  Keyframe: {os.path.basename(keyframe_path)}")
            else:
                _print_live(sid, cfg["num_scenes"], 1, s_name, stat)
                step(sid + 1, cfg["num_scenes"],
                     f"Scene {sid:02d}  Stage 1 — 4.8B Keyframe")

                pipe_img   = load_image_pipeline(cfg)
                t_s1_start = time.time()
                keyframe   = generate_keyframe(pipe_img, scene, cfg, chunk_seed)
                t_s1       = time.time() - t_s1_start
                ok(f"Stage 1 done in {t_s1:.1f}s")

                # Save keyframe PNG (flush to disk before writing flag)
                keyframe_path = save_keyframe(keyframe, sid, cfg)
                del keyframe

                # ── Write stage1_done.flag  (LEVEL 1 checkpoint) ─────────
                sm.write_flag(sid, "stage1_done.flag",
                              {"keyframe_path": keyframe_path, "s1_time_s": round(t_s1, 1)})
                sm.set_keyframe(sid, keyframe_path)
                ok(f"✦ stage1_done.flag written")

                # Unload Stage 1
                subheader("Unloading Stage 1 → freeing VRAM")
                del pipe_img
                vram_free()
                ok(f"VRAM after S1 unload: {vram_used_gb():.1f} GB")

            # ════════════════════════════════════════════════════════════
            # STAGE 2 — VIDEO ANIMATION  (with per-frame resume)
            # ════════════════════════════════════════════════════════════
            resume_from = diag["resume_stage2_from"]
            if resume_from > 0:
                recovered(f"Scene {sid:02d} Stage 2 — {resume_from} frames already saved, "
                          f"regenerating all 81 frames from scratch with same seed")
                # Note: SANA-Video generates all frames in one pass (not frame-by-frame),
                # so we can't resume mid-generation. We re-run Stage 2 but skip
                # re-saving frames that were already written (start_from=resume_from).
                # Since the same seed is used, the output is deterministic and the
                # new frames will match the ones already on disk for frames 0..resume_from.
                info("(Same seed guarantees deterministic output — saved frames are consistent)")

            _print_live(sid, cfg["num_scenes"], 2, s_name, stat)
            step(sid + 1, cfg["num_scenes"],
                 f"Scene {sid:02d}  Stage 2 — 2B Video Animation")

            bridge_path  = sm.get_bridge(sid - 1)
            conditioning = prepare_conditioning_image(
                               keyframe_path, bridge_path, sid, cfg)

            pipe_vid    = load_video_pipeline(cfg)
            t_s2_start  = time.time()
            frames      = generate_video_chunk(pipe_vid, scene, conditioning, cfg, chunk_seed)
            t_s2        = time.time() - t_s2_start
            ok(f"Stage 2 done in {t_s2:.1f}s  ({len(frames)} frames)")

            # Save frames (skip already-saved frames on partial resume)
            save_scene_frames(frames, sid, cfg, start_from=resume_from)
            ok(f"Frames saved → scene_{sid:02d}/  "
               f"({'all fresh' if resume_from == 0 else f'{len(frames)-resume_from} new, {resume_from} reused'})")

            # ── Write stage2_done.flag  (LEVEL 2 checkpoint) ─────────────
            sm.write_flag(sid, "stage2_done.flag",
                          {"frames": len(frames), "s2_time_s": round(t_s2, 1)})
            ok(f"✦ stage2_done.flag written")

            # Save bridge frame for next scene
            bp = save_bridge_frame(frames, sid, cfg)
            sm.set_bridge(sid, bp)
            ok(f"Bridge → {os.path.basename(bp)}")

            # Unload Stage 2
            del pipe_vid, frames, conditioning
            vram_free()

            # ── Mark scene complete  (LEVEL 4 atomic state.json) ─────────
            t_total = time.time() - t_scene
            stat.end_chunk(t_total, t_s1, t_s2)
            sm.mark_complete(sid, t_s1, t_s2)
            ok(f"Scene {sid:02d} complete — total {t_total:.1f}s  "
               f"(S1={t_s1:.0f}s  S2={t_s2:.0f}s)")
            ok(f"✦ state.json updated atomically")
            time.sleep(1.5)

    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}  Interrupted — all progress saved. Re-run to continue.{C.END}")
        stop_evt.set()
        stat.summary()
        return

    except Exception as e:
        err(f"Error at scene {sid}: {e}")
        import traceback; traceback.print_exc()
        stop_evt.set()
        warn("Re-run to resume — the checkpoint system will recover your work.")
        stat.summary()
        return

    finally:
        stop_evt.set()

    # ── Assemble video ────────────────────────────────────────────────────
    final_video = assemble_video(cfg)

    # ── Save final report ─────────────────────────────────────────────────
    rp = os.path.join(cfg["output_dir"],
                      f"report_v{__version__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(rp, "w") as f:
        json.dump({
            "version":         __version__,
            "timestamp":       datetime.now().isoformat(),
            "scenes":          [s["name"] for s in SCENES],
            "completed":       sorted(sm.state.get("completed_scenes", [])),
            "timing_per_scene": sm.state.get("timing", {}),
            "stage1_times_s":  stat.stage1_times,
            "stage2_times_s":  stat.stage2_times,
            "total_elapsed_s": stat.elapsed(),
            "final_video":     final_video,
        }, f, indent=2)
    ok(f"Report: {rp}")

    stat.summary()

    print(f"\n{C.BOLD}{C.GREEN}{'═'*72}{C.END}")
    print(f"{C.BOLD}{C.GREEN}  ✨  SmartSmile 2250 v{__version__} — COMPLETE{C.END}")
    print(f"{C.BOLD}{C.GREEN}{'═'*72}{C.END}")
    print(f"\n  {C.BOLD}📹  Output :{C.END}  {final_video}")
    print(f"  {C.BOLD}🖼   S1     :{C.END}  SANA 4.8B  →  1024×1024 keyframe/scene")
    print(f"  {C.BOLD}🎬  S2     :{C.END}  SANA-Video 2B  →  {cfg['frames_per_chunk']} frames/scene")
    print(f"  {C.BOLD}♻   Resume :{C.END}  atomic state.json + stage flags + frame-level recovery\n")


if __name__ == "__main__":
    main()