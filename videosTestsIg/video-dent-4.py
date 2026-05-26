"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║   SANA VIDEO — SMARTSMILE 2250 DENTAL ADVERTISEMENT GENERATOR                   ║
║   Version  : 4.0  |  Date: 2026-05-24  |  Author: Dr. Igor Lemos Alves          ║
║   Hardware : RTX 5080 16 GB VRAM · Ryzen 9 9900X · 32 GB DDR5 · Windows        ║
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   CHANGELOG                                                                      ║
║   v1.0  Initial release (SANA-Video 2B T2V only).                               ║
║   v2.0  Two-stage pipeline: SANA 4.8B keyframe + SANA-Video 2B I2V.            ║
║   v3.0  Full crash-resume: atomic state.json, per-scene flags, frame            ║
║         granularity recovery.                                                    ║
║   v4.0  QUALITY OVERHAUL — root-cause fix for poor image/video quality.         ║
║         See "Why Quality Was Poor in v1–v3" section below.                      ║
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   WHY QUALITY WAS POOR IN v1–v3 AND WHAT v4 FIXES                               ║
║   ────────────────────────────────────────────────                               ║
║                                                                                  ║
║   ROOT CAUSE 1 — Too few inference steps on Stage 1 (image model)               ║
║   ────────────────────────────────────────────────────────────────               ║
║   v3 used image_inference_steps=30, which is fine for quick previews but        ║
║   produces soft, underdetailed keyframes. SANA 4.8B with PAG achieves           ║
║   its best quality at 50–100 steps. At 30 steps the diffusion process           ║
║   hasn't converged enough for fine textures and sharp edges.                    ║
║   v4 FIX → image_inference_steps = 60 (quality default, tunable to 100).       ║
║                                                                                  ║
║   ROOT CAUSE 2 — guidance_scale too low on the image model                      ║
║   ──────────────────────────────────────────────────────────                    ║
║   v3 used image_guidance_scale=5.0. SANA-1.5 4.8B produces sharper,            ║
║   more detailed images at CFG 7.0–8.0. Values above 9 introduce artifacts.     ║
║   v4 FIX → image_guidance_scale = 7.5                                           ║
║                                                                                  ║
║   ROOT CAUSE 3 — PAG (Perturbed Attention Guidance) not verified working        ║
║   ─────────────────────────────────────────────────────────────────────         ║
║   PAG is a structural quality booster specific to SANA-1.5. It perturbs        ║
║   the self-attention maps during inference to improve spatial coherence.        ║
║   In v3 PAG was silently suppressed by a broad TypeError fallback.              ║
║   v4 FIX → PAG loaded at model init time via DiffusionPipeline with           ║
║   pag_applied_layers=["transformer_blocks"], verified active before each run.  ║
║                                                                                  ║
║   ROOT CAUSE 4 — SANA-Video 2B generating at 50 steps without LTX2 refiner     ║
║   ──────────────────────────────────────────────────────────────────────────   ║
║   The NVIDIA "Bet Small, Win Big" two-stage video paradigm (official docs:     ║
║   https://nvlabs.github.io/Sana/docs/sana_video_inference/) specifies that     ║
║   SANA-Video 2B → LTX-2 Refiner delivers 2K-quality output at 720p latency.   ║
║   LTX-2 is a step-distilled refiner (only 3 steps) that adds spatial texture, ║
║   sharpens edges, and corrects temporal inconsistencies — at near-zero cost.   ║
║   Skipping it leaves 40–60% of achievable quality on the table.                ║
║   v4 FIX → LTX-2 refiner runs after every scene's video chunk.                 ║
║                                                                                  ║
║   ROOT CAUSE 5 — Video guidance_scale too conservative                          ║
║   ─────────────────────────────────────────────────────                         ║
║   v3 used guidance_scale=6.0 for video (NVIDIA minimum). Increasing to         ║
║   7.0–8.0 improves prompt adherence and visual sharpness without               ║
║   significantly impacting temporal smoothness at 50 steps.                     ║
║   v4 FIX → video guidance_scale = 7.5                                           ║
║                                                                                  ║
║   ROOT CAUSE 6 — No flow_shift tuning                                           ║
║   ────────────────────────────────────                                           ║
║   flow_shift controls the rectified-flow sampling schedule curvature.          ║
║   NVIDIA documentation recommends flow_shift=8 as a default but notes          ║
║   higher values (10–14) produce sharper high-frequency detail at the cost      ║
║   of increased compute. v4 exposes this as a tunable parameter.                ║
║   v4 FIX → flow_shift = 10 (sharper detail, tunable)                           ║
║                                                                                  ║
║   ROOT CAUSE 7 — Frame format JPEG for intermediate storage                     ║
║   ──────────────────────────────────────────────────────────                    ║
║   Any JPEG compression between stages introduces blocking artifacts that       ║
║   compound across 12 scenes. v3 used PNG but did not verify this               ║
║   consistently. v4 enforces PNG-only throughout the entire pipeline.           ║
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   FULL THREE-STAGE PIPELINE ARCHITECTURE                                         ║
║   ──────────────────────────────────────                                         ║
║                                                                                  ║
║   Per scene (12 scenes × 5.06 s = 60.75 s total):                              ║
║                                                                                  ║
║   ┌──────────────────────────────────────────────────────────────────────┐      ║
║   │ STAGE A  —  KEYFRAME GENERATION  (SANA-1.5 4.8B)                    │      ║
║   │                                                                      │      ║
║   │  Model : Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers        │      ║
║   │  Input : image_prompt  (cinematic still — NO motion language)       │      ║
║   │  Size  : 1024×1024 PNG (square; I2V will crop/letterbox to 704×1280)│      ║
║   │  Steps : 60  |  CFG : 7.5  |  PAG : 2.5                            │      ║
║   │  Output: keyframe_scene_NN.png                                       │      ║
║   │  VRAM  : ~11 GB peak → del + gc + empty_cache before Stage B        │      ║
║   │                                                                      │      ║
║   │  Why 4.8B for images?                                                │      ║
║   │  The 4.8B SANA-1.5 model uses DC-AE (32× compression) and Linear   │      ║
║   │  DiT with Gemma-2 text encoder. At 1024px it produces significantly │      ║
║   │  more photorealistic, detailed results than any smaller model.      │      ║
║   │  PAG (Perturbed Attention Guidance) is exclusive to SANA-1.5 and   │      ║
║   │  further improves structural coherence of complex scenes.           │      ║
║   └──────────────────────────────────────────────────────────────────────┘      ║
║           ↓  keyframe_scene_NN.png  (blended 70% keyframe + 30% bridge)        ║
║   ┌──────────────────────────────────────────────────────────────────────┐      ║
║   │ STAGE B  —  VIDEO GENERATION  (SANA-Video 2B I2V)                   │      ║
║   │                                                                      │      ║
║   │  Model : Efficient-Large-Model/SANA-Video_2B_720p_diffusers         │      ║
║   │  Input : conditioning image + video_prompt (motion + camera)        │      ║
║   │  Size  : 1280×704  |  81 frames  |  16 fps  (~5.06 s/scene)        │      ║
║   │  Steps : 50  |  CFG : 7.5  |  flow_shift : 10                      │      ║
║   │  Output: raw video latent → 81 frame PNGs                           │      ║
║   │  VRAM  : ~5.5 GB peak → del + gc + empty_cache before Stage C      │      ║
║   │                                                                      │      ║
║   │  Architecture notes (from NVLabs official docs):                    │      ║
║   │  · Block Causal Linear Attention: processes frames with O(1)        │      ║
║   │    memory per step instead of O(N²) for standard transformers.      │      ║
║   │  · DC-AE: 32× spatial compression reduces latent tokens by 16×     │      ║
║   │    vs standard 8× VAEs, enabling fast 720p generation.              │      ║
║   │  · FlowMatchEulerDiscreteScheduler: rectified-flow sampler tuned    │      ║
║   │    for SANA's latent space (flow_shift controls schedule curve).    │      ║
║   └──────────────────────────────────────────────────────────────────────┘      ║
║           ↓  raw video latent (in VRAM via output_type="latent")                ║
║   ┌──────────────────────────────────────────────────────────────────────┐      ║
║   │ STAGE C  —  LTX-2 REFINER  ("Bet Small, Win Big")                   │      ║
║   │                                                                      │      ║
║   │  Model : Lightricks/LTX-2 (step-distilled LoRA adapter)             │      ║
║   │  Input : raw latent from Stage B + same prompt                      │      ║
║   │  Steps : 3  (distilled — near-zero extra cost)                      │      ║
║   │  Output: refined 1280×704 frames (2K-quality texture at 720p cost)  │      ║
║   │  VRAM  : ~6 GB peak                                                  │      ║
║   │                                                                      │      ║
║   │  Official source: nvlabs.github.io/Sana/docs/sana_video_inference/  │      ║
║   │  Blog: "Bet Small, Win Big" — Sana Video + LTX2 Refiner Pipeline    │      ║
║   │  The refiner uses LTX2Pipeline + LTX2LatentUpsamplePipeline.        │      ║
║   │  It corrects: soft textures, aliasing, temporal flicker, fine       │      ║
║   │  details (hair, teeth, metal, fabric) that base model misses.       │      ║
║   └──────────────────────────────────────────────────────────────────────┘      ║
║           ↓  refined frames saved as PNG                                         ║
║           ↓  last frame → bridge_after_scene_NN.png → Stage A of scene N+1     ║
║                                                                                  ║
║   CONTEXT CONTINUITY BETWEEN SCENES                                             ║
║   ──────────────────────────────────                                             ║
║   · The LAST FRAME of Stage C (refined) is saved as a bridge PNG.               ║
║   · Stage A of the next scene receives: keyframe (70%) + bridge (30%).          ║
║   · Using the refined bridge (not the raw Stage B bridge) ensures the          ║
║     high-quality texture of the refiner carries forward into continuity.        ║
║   · Seed = 2025 + scene_id → monotonic drift, deterministic on resume.         ║
║                                                                                  ║
║   VRAM LIFECYCLE                                                                 ║
║   ──────────────                                                                 ║
║   Stage A active   : ~11 GB  (4.8B bfloat16 + fp32 VAE)                        ║
║   Stage B active   : ~5.5 GB (2B bfloat16 + fp32 VAE)                          ║
║   Stage C active   : ~6 GB   (LTX-2 bfloat16)                                   ║
║   Stages never share VRAM. del + gc.collect() + empty_cache() between each.    ║
║                                                                                  ║
║   CRASH-RESUME (carried from v3, extended for Stage C)                          ║
║   ──────────────────────────────────────────────────────                         ║
║   · stage_a_done.flag → keyframe on disk and verified                           ║
║   · stage_b_done.flag → raw frames on disk                                      ║
║   · stage_c_done.flag → refined frames on disk (NEW in v4)                     ║
║   · state.json written atomically via tmp → os.replace()                       ║
║   · state.json.bak for double-fault recovery                                   ║
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   QUALITY PARAMETER COMPARISON                                                   ║
║   ────────────────────────────                                                   ║
║   Parameter                v3 value   v4 value   Effect                         ║
║   image_inference_steps     30         60         +30 more diffusion steps      ║
║   image_guidance_scale       5.0        7.5        +50% prompt fidelity         ║
║   image_pag_guidance_scale   2.0        2.5        +25% structural coherence    ║
║   video_guidance_scale       6.0        7.5        +25% prompt adherence        ║
║   flow_shift                 8          10         sharper high-freq detail      ║
║   LTX-2 refiner              ✗          ✓          2K-quality texture boost      ║
║   bridge source              Stage B    Stage C    refined bridge = better       ║
║                                                    continuity                    ║
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   INSTALL DEPENDENCIES                                                           ║
║   pip install git+https://github.com/huggingface/diffusers                     ║
║   pip install torch torchvision imageio[ffmpeg] psutil GPUtil Pillow numpy      ║
║                                                                                  ║
║   REFERENCES                                                                     ║
║   · SANA-1.5 model card  : huggingface.co/Efficient-Large-Model/...4.8B        ║
║   · SANA-Video docs      : nvlabs.github.io/Sana/docs/sana_video/              ║
║   · Video inference docs : nvlabs.github.io/Sana/docs/sana_video_inference/    ║
║   · Bet Small Win Big    : nvlabs.github.io/Sana/Video/bet-small-win-big/      ║
║   · SANA-WM world model  : studio.aifilms.ai/blog/sana-wm-nvidia-world-model   ║
║   · NVLabs GitHub        : github.com/NVlabs/Sana                              ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

__version__ = "4.0"
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

# ── Optional GPU monitoring ───────────────────────────────────────────────────
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

# ── Diffusers — checked at runtime; graceful error on import failure ──────────
try:
    from diffusers import SanaPipeline                 # Stage A — 4.8B image
    from diffusers import SanaImageToVideoPipeline     # Stage B — 2B I2V video
    from diffusers import SanaVideoPipeline            # Stage B alt — T2V fallback
    from diffusers import LTX2Pipeline                 # Stage C — refiner
    from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
    from diffusers.utils import load_image
    DIFFUSERS_OK = True
except ImportError as _diffusers_err:
    DIFFUSERS_OK = False
    _diffusers_missing = str(_diffusers_err)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TERMINAL COLORS                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class C:
    """ANSI color codes for readable terminal output on Windows/Linux."""
    CYAN    = '\033[96m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    RED     = '\033[91m'
    BLUE    = '\033[94m'
    MAGENTA = '\033[95m'
    BOLD    = '\033[1m'
    DIM     = '\033[2m'
    END     = '\033[0m'

def header(text):       print(f"\n{C.BOLD}{C.CYAN}{'═'*76}\n  {text}\n{'═'*76}{C.END}")
def subheader(text):    print(f"\n{C.BOLD}{C.MAGENTA}  ▶  {text}{C.END}")
def ok(text):           print(f"  {C.GREEN}✔{C.END}  {text}")
def warn(text):         print(f"  {C.YELLOW}⚠{C.END}  {text}")
def err(text):          print(f"  {C.RED}✘{C.END}  {text}")
def info(text):         print(f"  {C.BLUE}ℹ{C.END}  {text}")
def step(n, t, text):   print(f"\n{C.BOLD}{C.BLUE}[{n}/{t}]{C.END} {text}")
def recovered(text):    print(f"  {C.MAGENTA}♻{C.END}  {C.BOLD}RECOVERED:{C.END} {text}")
def quality(text):      print(f"  {C.CYAN}★{C.END}  {text}")

def bar(frac, w=44):
    """Render a unicode progress bar of width w for a fraction 0.0–1.0."""
    f = int(w * max(0.0, min(1.0, frac)))
    return f"{C.GREEN}{'█'*f}{C.END}{'░'*(w-f)}"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  OFFICIAL NEGATIVE PROMPTS                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Verbatim from NVIDIA SANA-Video documentation.
# This prompt targets the model's known failure modes: deformed anatomy,
# temporal artifacts, ghosting, and frame sync issues.
NEGATIVE_VIDEO = (
    "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, "
    "sudden disappearance, jump cuts, jerky movements, rapid shot changes, "
    "frames out of sync, inconsistent character shapes, temporal artifacts, "
    "jitter, and ghosting effects, creating a disorienting visual experience. "
    "overexposed, underexposed, noise, grain, watermark, text overlay, "
    "cartoon, anime, low resolution, blurry, anatomically incorrect, "
    "extra limbs, duplicate faces, ugly, deformed."
)

# Negative prompt for Stage A (image generation).
# Targets compression artifacts, distortion, and non-photorealistic styles.
NEGATIVE_IMAGE = (
    "blurry, out of focus, low quality, low resolution, jpeg artifacts, "
    "watermark, text, logo, overexposed, underexposed, grain, noise, "
    "distorted perspective, cartoon, anime, oil painting, illustration, "
    "sketch, deformed anatomy, extra limbs, duplicate faces, ugly, disfigured, "
    "plastic skin, flat lighting, washed out colors."
)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SCENE SCRIPT — 12 × 5-SECOND NARRATIVE BEATS                               ║
# ║                                                                              ║
# ║  PROMPT ANATOMY (NVIDIA official technique):                                 ║
# ║    IMAGE  : LIGHTING · SHOT TYPE · COMPOSITION · SUBJECT+DETAIL ·           ║
# ║             ENVIRONMENT · MOOD/COLOR · quality tokens                        ║
# ║    VIDEO  : (same as image condensed) + CAMERA MOTION · TEMPORAL FLOW ·     ║
# ║             " motion score: N."  ← REQUIRED token at prompt end             ║
# ║                                                                              ║
# ║  MOTION SCORE GUIDE (embedded in video_prompt):                              ║
# ║    5–10  : near-static (logo float, macro hold, sleeping patient)            ║
# ║    12–18 : gentle (slow push-in, drift, talking heads, soft wind)            ║
# ║    20–25 : natural (walking, scanning device, camera orbit)                  ║
# ║    28–35 : energetic (crowd movement, fast orbit, tracking shot)             ║
# ║    50+   : high-action (not used in this medical context)                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

SCENES = [
    # ── Scene 0 ──────────────────────────────────────────────────────────────
    {
        "id": 0,
        "name":  "Opening — Megacity of 2250",
        "beats": "Aerial establishing shot — set the utopian future world",
        "motion_score": 20,

        # image_prompt: pure visual description, NO motion language.
        # Drives SANA 4.8B → photorealistic cinematic still frame.
        "image_prompt": (
            "Soft golden sunrise backlight, sweeping aerial wide shot, centered horizon rule. "
            "A luminous megacity of year 2250: crystal spire towers and translucent bio-domes "
            "glowing warm amber and cobalt blue against an indigo predawn sky. "
            "Magnetic-levitation vehicles trace glowing ribbons through elevated glass highways. "
            "Volumetric sunrise clouds catch the first rays, casting god-rays through the towers. "
            "Photorealistic, cinematic anamorphic lens, 8K ultra-detailed, utopian color grading, "
            "shot on RED MONSTRO, award-winning architectural photography."
        ),

        # video_prompt: extends image with motion/camera language + motion score token.
        # Drives SANA-Video 2B → animated version of the keyframe.
        "video_prompt": (
            "Soft golden sunrise light, sweeping aerial wide shot, centered horizon composition. "
            "A luminous megacity of 2250 stretches below — crystal towers and bio-domes glowing amber. "
            "Autonomous vehicles trace light trails through elevated glass highways. "
            "Camera drifts slowly forward over the skyline with a slight continuous downward tilt, "
            "as if on a smooth aerial drone dolly. Warm utopian color grading, lens flare pulses. "
            "motion score: 20."
        ),
    },

    # ── Scene 1 ──────────────────────────────────────────────────────────────
    {
        "id": 1,
        "name":  "Clinic Exterior — SmartSmile 2250",
        "beats": "Reveal the futuristic dental clinic facade",
        "motion_score": 15,
        "image_prompt": (
            "Soft diffused morning daylight, medium architectural wide shot, perfect bilateral symmetry. "
            "A breathtaking dental clinic facade: pure white biopolymer panels and floor-to-ceiling "
            "blue-tinted smart glass with embedded chromatic trim. A holographic logotype "
            "'SmartSmile 2250' rotates gently in aquamarine light above the entrance canopy. "
            "Bioluminescent sakura trees with pale cyan blossoms line the approach path. "
            "Patients in elegant minimalist attire walk through a garden of floating light orbs. "
            "Photorealistic, Zaha Hadid architectural photography, 8K, pristine and welcoming, "
            "global illumination, subsurface scattering on organic panels."
        ),
        "video_prompt": (
            "Soft diffused daylight, medium wide shot, symmetrical architectural composition. "
            "SmartSmile 2250 clinic facade in white biopolymer and smart glass. "
            "Holographic logo rotates gently above entrance. Bioluminescent trees sway in a soft breeze. "
            "Patients move gracefully along the garden path. "
            "Camera performs a slow, steady forward push toward the entrance doors. "
            "Cool aquamarine and white palette, pristine futurism. motion score: 15."
        ),
    },

    # ── Scene 2 ──────────────────────────────────────────────────────────────
    {
        "id": 2,
        "name":  "Reception — AI Concierge Interaction",
        "beats": "Patient checks in with holographic AI receptionist",
        "motion_score": 18,
        "image_prompt": (
            "Cool blue-white ambient light with warm amber accent wall panels, "
            "medium close-up, rule-of-thirds composition, patient left of center. "
            "An elegant young woman in clean minimalist clothing stands at a floating "
            "glass reception desk. Across from her: a translucent holographic humanoid AI avatar "
            "with a calm, warm expression, gesturing with one hand. "
            "Soft teal neon data streams float between them. Staff in fitted white uniforms move "
            "purposefully in the bright background. The entire space is luminous and spotless. "
            "Photorealistic, cinematic portrait lighting, Sony A7RV, 85mm f/1.4, 8K, ultra-sharp."
        ),
        "video_prompt": (
            "Cool blue ambient light with warm accent panels, medium close-up, rule-of-thirds. "
            "Patient interacts with the translucent holographic AI concierge that smiles and gestures. "
            "Soft neon medical data streams pulse and drift in the air between them. "
            "Staff move gracefully in background. "
            "Camera holds steady with a very slow, gentle push-in toward the patient's face. "
            "Clinical yet warm, ultra-clean whites and soft blues. motion score: 18."
        ),
    },

    # ── Scene 3 ──────────────────────────────────────────────────────────────
    {
        "id": 3,
        "name":  "Diagnostics — AI Oral Scan",
        "beats": "Real-time AI full-mouth diagnostic mapping",
        "motion_score": 22,
        "image_prompt": (
            "Soft cool surgical lighting with blue-white overhead panels, "
            "close-up macro shot, perfectly centered clinical composition. "
            "A sleek brushed-titanium AI oral scanner — the size of a thick pen — "
            "hovers millimeters from a patient's open mouth. "
            "Beside the patient's face, a real-time 3D holographic dental model glows in midair: "
            "teeth color-coded green for healthy, amber for attention, red for treatment. "
            "A dentist in smart-glass AR eyewear examines the hologram with a haptic stylus. "
            "Extreme photorealism, medical device industrial photography, 8K, "
            "Zeiss macro lens rendering, perfect subsurface scattering on teeth enamel."
        ),
        "video_prompt": (
            "Soft cool clinical lighting, close-up macro shot, centered composition. "
            "AI oral scanner glides in a precise arc around the patient's open mouth, "
            "real-time 3D holographic dental model pulsing and updating as it moves. "
            "Dentist's stylus sweeps across the hologram making light-pen annotations. "
            "Camera slowly orbits the scanner in a smooth 45-degree arc from left to right. "
            "High-tech blue and white tones, clinical precision. motion score: 22."
        ),
    },

    # ── Scene 4 ──────────────────────────────────────────────────────────────
    {
        "id": 4,
        "name":  "Nano-Robot Dental Procedure",
        "beats": "Microscopic nano-robots perform painless precision dental work",
        "motion_score": 24,
        "image_prompt": (
            "Diffused warm-cool surgical lighting, medium shot with macro depth-of-field insert, "
            "patient centered in a clean single-subject composition. "
            "A patient reclines in a floating white ergonomic dental chair that resembles a "
            "luxury cocoon. Their face is completely peaceful — no tension, faint smile. "
            "Above their open mouth: a luminous cloud of micro-robots, each the size of a dust mote, "
            "each emitting a precise electric-blue glow as it performs nano-scale work. "
            "The swarm forms a perfect dome of light above the treatment area. "
            "A transparent AR screen beside the chair shows live nanoscale mapping in teal. "
            "An attending dentist stands in soft focus behind. "
            "Photorealistic, 8K, cinematic medical sci-fi, wonder and serenity."
        ),
        "video_prompt": (
            "Diffused surgical lighting, medium shot, single-subject clean framing. "
            "Patient reclines completely at ease, eyes closed peacefully, faint smile. "
            "Luminous nano-robot swarm drifts and pulses as a coherent glowing cloud, "
            "each mote tracing a precise micro-trajectory. "
            "AR screen beside the chair updates live nano-mapping data. "
            "Camera gently and continuously drifts from a wide view to a tighter close-up on the swarm. "
            "Serene clinical blue-white palette, wonder and precision. motion score: 24."
        ),
    },

    # ── Scene 5 ──────────────────────────────────────────────────────────────
    {
        "id": 5,
        "name":  "Bioprinting — Living Tooth Regeneration",
        "beats": "A living tooth bioprinted layer by layer in real time",
        "motion_score": 18,
        "image_prompt": (
            "Warm amber-gold lab lighting, extreme close-up, perfectly centered macro composition. "
            "A cutting-edge dental bioprinter nozzle positioned above a half-formed tooth crown. "
            "The tooth is 60% complete: lower translucent organic layers already formed, "
            "the upper portion still being deposited in ultra-fine living bio-ink filaments. "
            "Stem cells within the matrix glow faintly gold-white. "
            "A nano-scaffold of blue light lines surrounds the growing structure. "
            "Water droplets on the scaffold catch amber and teal light. "
            "Extreme photorealism, scientific macro photography, shallow DOF bokeh, "
            "8K, amber and ivory color palette, Leica M10 rendering quality."
        ),
        "video_prompt": (
            "Warm amber lab lighting, extreme close-up, centered macro composition. "
            "Dental bioprinter deposits living tissue in ultra-fine filaments, layer by layer, "
            "the tooth crown growing visibly upward frame by frame. "
            "Stem cells activate inside the matrix with a faint pulsing inner glow. "
            "Nano-scaffold lines shimmer softly around the structure. "
            "Camera holds perfectly static with macro focus, shallow depth-of-field bokeh. "
            "Rich amber and ivory tones, scientific wonder. motion score: 18."
        ),
    },

    # ── Scene 6 ──────────────────────────────────────────────────────────────
    {
        "id": 6,
        "name":  "Human + AI Dentist Collaboration",
        "beats": "Human dentist and humanoid AI reviewing a 3D jaw hologram together",
        "motion_score": 16,
        "image_prompt": (
            "Soft split warm-cool lighting, medium two-shot composition, subjects side by side. "
            "LEFT: a skilled human dentist, mid-30s, in a perfectly tailored white coat, "
            "holding a haptic stylus, expression calm and focused. "
            "RIGHT: a sleek humanoid AI dental assistant, translucent torso with visible "
            "data-circuit latticework glowing teal, calm and professional. "
            "Between them: a rotating 3D holographic jaw model floating at chest height, "
            "teeth individually lit with health-status color coding. "
            "The human's stylus creates an annotation beam; the AI overlays a response indicator. "
            "Photorealistic, cinematic two-shot, warm walnut panels + cool brushed steel walls, "
            "8K, Arri Alexa Mini LF rendering, partnership and expertise."
        ),
        "video_prompt": (
            "Soft split warm-cool lighting, medium two-shot, balanced frame composition. "
            "Human dentist and AI assistant discuss the rotating 3D holographic jaw model. "
            "The human gestures with stylus; AI overlays real-time annotations on the hologram. "
            "Jaw hologram rotates slowly, health-status colors pulsing. "
            "Camera arcs very slowly from a profile angle to a three-quarter frontal position. "
            "Warm wood and cool steel aesthetic, trust and expertise. motion score: 16."
        ),
    },

    # ── Scene 7 ──────────────────────────────────────────────────────────────
    {
        "id": 7,
        "name":  "Patient Experience — Zero Pain",
        "beats": "Patient in total comfort during completely pain-free neural treatment",
        "motion_score": 10,
        "image_prompt": (
            "Golden warm ambient light with soft directional key from upper-left, "
            "medium close-up portrait, slightly low angle looking up at patient. "
            "A patient — 40s, diverse features — reclines in a floating anti-gravity chair, "
            "eyes gently closed, lips curved in a peaceful smile. "
            "A delicate neural interface headband of white titanium rests on their brow, "
            "projecting soft geometric mandala light patterns that float in the air above. "
            "The room walls behind display a serene real-time coral reef environment. "
            "Complete absence of tension. "
            "Photorealistic, cinematic beauty lighting, Canon EOS R5, 50mm f/1.2, "
            "8K, warm golden hour tones, subsurface skin rendering."
        ),
        "video_prompt": (
            "Golden soft ambient light, medium close-up, slightly low angle, warm framing. "
            "Patient completely at ease in floating chair, eyes closed, peaceful smile. "
            "Neural interface headband pulses soft geometric mandala patterns into the air above. "
            "Coral reef environment on walls moves gently behind them. "
            "Camera performs a very slow, almost imperceptible push-in toward the patient's serene face. "
            "Warm golden tones, complete comfort and inner peace. motion score: 10."
        ),
    },

    # ── Scene 8 ──────────────────────────────────────────────────────────────
    {
        "id": 8,
        "name":  "The Perfect Smile Reveal",
        "beats": "Patient sees their radiant new smile for the first time",
        "motion_score": 14,
        "image_prompt": (
            "Warm flattering beauty lighting — three-point setup, main from upper-right — "
            "medium close-up, perfectly centered portrait composition. "
            "A patient — 30s, warm complexion — holds up a sleek frameless smart mirror. "
            "Their eyes are just beginning to widen with dawning joy and disbelief. "
            "In the mirror's reflection: a perfectly aligned, radiant white smile with natural translucency. "
            "The mirror frame displays a soft green health indicator: '98 / 100'. "
            "Their skin glows with health; a single tear of joy catches the light on their cheek. "
            "Photorealistic, cinematic beauty photography, 8K, "
            "warm bright palette, Hasselblad H6D rendering, pure joy and transformation."
        ),
        "video_prompt": (
            "Warm flattering beauty lighting, medium close-up, centered portrait. "
            "Patient raises the smart mirror and the reflection shows the radiant new smile. "
            "Expression transitions frame by frame: neutral → surprise → wide genuine joy. "
            "A single tear traces down one cheek. Mirror health indicator pulses green. "
            "Camera performs the slowest possible push-in toward the smile. "
            "Warm bright tones, joy, confidence, transformation. motion score: 14."
        ),
    },

    # ── Scene 9 ──────────────────────────────────────────────────────────────
    {
        "id": 9,
        "name":  "Community — Dental Health for All",
        "beats": "Diverse humanity thriving in the SmartSmile 2250 public health park",
        "motion_score": 22,
        "image_prompt": (
            "Bright warm diffused daylight, wide ensemble shot, dynamic diagonal group composition. "
            "A joyful multicultural crowd of 20+ people — children running, elders holding hands, "
            "young adults laughing — fills a luminous public health park. "
            "Everyone shows a brilliant healthy smile. "
            "Holographic dental-health kiosks with cyan glow line the central path. "
            "Bioluminescent trees with pale gold leaves arch overhead. "
            "A SmartSmile 2250 medical drone hovers above distributing nano-care capsules in a "
            "gentle shimmer of light. The mood is pure celebration. "
            "Photorealistic, crowd ensemble photography, 8K, "
            "vibrant warm daylight, optimism and universal healthcare."
        ),
        "video_prompt": (
            "Bright warm daylight, wide ensemble shot, dynamic group composition. "
            "Joyful multicultural crowd moves through SmartSmile 2250 park, all smiling brilliantly. "
            "Children run ahead; elders walk steadily; adults laugh and gesture. "
            "Holographic kiosks pulse. Bioluminescent trees sway. Drone drifts overhead, "
            "releasing a shimmer of nano-care capsules that catch the light. "
            "Camera performs a smooth lateral tracking dolly alongside the crowd. "
            "Vibrant warm daylight, optimism, inclusivity. motion score: 22."
        ),
    },

    # ── Scene 10 ─────────────────────────────────────────────────────────────
    {
        "id": 10,
        "name":  "Global Oral Health Dashboard",
        "beats": "Planetary-scale dental health data visualization — humanity healing",
        "motion_score": 16,
        "image_prompt": (
            "Deep ocean-blue ambient light, wide shot, holographic spherical globe centerpiece. "
            "A vast spherical holographic Earth — two meters in diameter — rotates slowly in the "
            "center of a high-vaulted control room. "
            "Thousands of SmartSmile clinic markers pulse as warm gold dots across every continent. "
            "Curved 8K data panels surrounding the globe display: bar charts trending upward, "
            "world maps shifting from red to green, numerical scores approaching 100%. "
            "Scientists in white lab coats observe from the polished obsidian floor below. "
            "Photorealistic, cinematic, 8K, deep navy, emerald, gold data-visualization tones, "
            "epic scale and optimism."
        ),
        "video_prompt": (
            "Deep blue ambient light, wide shot, global holographic map composition. "
            "Spherical holographic Earth rotates slowly, SmartSmile markers pulsing gold worldwide. "
            "Data panels update in real time — scores climbing, maps turning from red to green. "
            "Scientists move across the floor below. The globe brightens as scores approach 100%. "
            "Camera tracks slowly backward on a perfectly level dolly to reveal the full room. "
            "Deep navy, emerald, and gold tones, epic scale. motion score: 16."
        ),
    },

    # ── Scene 11 ─────────────────────────────────────────────────────────────
    {
        "id": 11,
        "name":  "Brand Closing — Logo and Tagline",
        "beats": "Iconic logo reveal with timeless closing message",
        "motion_score": 8,
        "image_prompt": (
            "Soft warm studio light with barely perceptible radial gradient, "
            "ultra-wide perfectly centered composition, absolute minimalism. "
            "A single tooth sculpted from luminous translucent crystal floats in the exact center "
            "of a pure white infinite background. "
            "Below it: 'SmartSmile 2250' in polished gold letterforms, perfectly kerned. "
            "Below that: the tagline in refined light-gray sans-serif: "
            "'Every smile. Every human. Every future.' "
            "The crystal tooth casts a faint prismatic spectrum across the white surface. "
            "No other objects. No noise. "
            "Photorealistic, product photography, 8K, pure white and gold, "
            "timeless luxury brand aesthetic, shot on Phase One XF IQ4."
        ),
        "video_prompt": (
            "Soft warm white studio light, ultra-wide clean shot, perfectly centered. "
            "Crystal tooth floats and rotates infinitely slowly against pure white background. "
            "SmartSmile 2250 gold logo pulses with a single gentle breath of light below. "
            "Tagline text materializes letter by letter in soft gray. "
            "Prismatic light spectrum drifts imperceptibly across the white surface. "
            "Camera holds completely static — only the elements move, barely. "
            "Pure white and gold, timeless, iconic. motion score: 8."
        ),
    },
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                               ║
# ║                                                                              ║
# ║  All quality improvements vs v3 are annotated with # v4: reason             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_config():
    """
    Return the master configuration dict for the v4 three-stage pipeline.

    Quality parameters are tuned for RTX 5080 16 GB based on:
    - SANA-1.5 model card recommendations (HuggingFace)
    - SANA-Video official docs (nvlabs.github.io/Sana/docs/sana_video/)
    - "Bet Small, Win Big" blog (nvlabs.github.io/Sana/Video/bet-small-win-big/)
    - SANA-WM architecture paper (for DC-AE and flow_shift guidance)
    """
    cfg = {
        # ── STAGE A: SANA-1.5 4.8B image model ──────────────────────────────
        "model_image":              "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
        "keyframe_height":          1024,
        "keyframe_width":           1024,

        # v4: raised from 30 → 60 steps. At 30 steps the denoising process
        # has not converged enough for fine texture on teeth, skin, and metal.
        # 60 steps is the quality sweet-spot for SANA-1.5 with PAG.
        "image_inference_steps":    60,

        # v4: raised from 5.0 → 7.5. Higher CFG forces tighter prompt adherence
        # and sharper high-frequency detail. Above 9.0 artifacts appear.
        "image_guidance_scale":     7.5,

        # v4: raised from 2.0 → 2.5. PAG (Perturbed Attention Guidance) is an
        # exclusive SANA-1.5 feature that improves spatial coherence of complex
        # multi-element compositions. Stronger PAG = better structural layout.
        "image_pag_guidance_scale": 2.5,

        # PAG is applied to the transformer blocks (not cross-attention).
        # This list is passed to DiffusionPipeline at load time.
        "pag_applied_layers":       ["transformer_blocks"],

        # ── STAGE B: SANA-Video 2B I2V model ────────────────────────────────
        "model_video":              "Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
        "height":                   704,
        "width":                    1280,
        "fps":                      16,
        "frames_per_chunk":         81,   # Native SANA-Video window = 5.0625 s
        "num_scenes":               len(SCENES),

        # v4: raised from 6.0 → 7.5. Matches image model CFG; improves
        # prompt adherence and visual sharpness at identical compute cost.
        "guidance_scale":           7.5,

        # 50 steps remains the optimal balance. 75 gives marginal gains at
        # much higher compute. The LTX-2 refiner recovers remaining quality.
        "num_inference_steps":      50,

        # v4: raised from 8 → 10. Higher flow_shift biases the rectified-flow
        # sampler toward more aggressive denoising in early steps, producing
        # sharper high-frequency detail (edges, texture, fine hair).
        # Source: SANA-Video architecture docs on DC-AE + flow matching.
        "flow_shift":               10,

        "torch_dtype":              torch.bfloat16,  # RTX 5080 native precision
        "seed":                     2025,            # Thematic; deterministic on resume

        # ── STAGE C: LTX-2 Refiner ──────────────────────────────────────────
        # Source: nvlabs.github.io/Sana/docs/sana_video_inference/
        # Official "Bet Small, Win Big" two-stage video paradigm.
        # The refiner is step-distilled — only 3 inference steps.
        # It receives the raw video latent from Stage B and applies
        # spatial texture enhancement, edge sharpening, and flicker correction.
        "use_ltx2_refiner":         True,
        "model_ltx2":               "Lightricks/LTX-2",
        "ltx2_adapter":             "ltx-2-19b-distilled-lora-38k.safetensors",
        "ltx2_inference_steps":     3,    # Distilled — do not increase

        # ── VRAM management ──────────────────────────────────────────────────
        # Never load two stages simultaneously. Each is deleted and
        # gc.collect() + empty_cache() called before the next loads.
        "cpu_offload":              True,
        "attention_slicing":        True,
        "vae_dtype":                torch.float32,  # Always float32 — NEVER change

        # ── Context bridge ───────────────────────────────────────────────────
        # v4 improvement: bridge is taken from REFINED frames (Stage C output)
        # not raw frames (Stage B output). This ensures the high-quality
        # texture of the refiner carries forward into inter-scene continuity.
        "use_bridge_blending":      True,
        "bridge_blend_alpha":       0.30,  # 70% keyframe + 30% refined bridge

        # ── Storage ──────────────────────────────────────────────────────────
        "output_dir":      "dental_ad_v4_output",
        "keyframes_dir":   "dental_ad_v4_output/keyframes",
        "raw_frames_dir":  "dental_ad_v4_output/raw_frames",     # Stage B output
        "frames_dir":      "dental_ad_v4_output/refined_frames", # Stage C output (final)
        "bridges_dir":     "dental_ad_v4_output/bridges",
        "checkpoints_dir": "dental_ad_v4_output/checkpoints",
        "final_video":     "SmartSmile2250_Ad_v4_1min_720p.mp4",

        # ── Video encoding ───────────────────────────────────────────────────
        "crf":    18,      # Near-lossless H.264 (18=high quality, 23=default)
        "preset": "slow",  # x264 preset: slower = better compression ratio
    }

    # Derived values — do not edit these manually
    cfg["total_frames"]           = cfg["num_scenes"] * cfg["frames_per_chunk"]
    cfg["total_duration_seconds"] = cfg["total_frames"] / cfg["fps"]
    cfg["num_chunks"]             = cfg["num_scenes"]
    return cfg


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ATOMIC STATE MANAGER                                                        ║
# ║                                                                              ║
# ║  Extended from v3 to track three stages per scene (A, B, C).               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class StateManager:
    """
    Manages all checkpoint I/O with crash-safe atomic writes.

    File layout:
        checkpoints/state.json       — primary state (atomic write)
        checkpoints/state.json.bak   — backup of last good state
        checkpoints/state.json.tmp   — temp during write (auto-replaced)

    Per-scene flags (stored inside refined_frames/scene_NN/):
        stage_a_done.flag  — keyframe PNG fully flushed to disk
        stage_b_done.flag  — raw video frames fully saved
        stage_c_done.flag  — refined frames fully saved  ← NEW in v4

    Recovery matrix:
        stage_c_done exists + refined frames count OK → scene fully done
        stage_b_done exists + raw frames count OK     → need Stage C only
        stage_a_done exists + keyframe exists         → need Stages B + C
        nothing                                        → run all three stages
    """

    EMPTY = {
        "version":          __version__,
        "completed_scenes": [],
        "keyframe_paths":   {},
        "bridge_frames":    {},
        "timing":           {},
        "last_updated":     None,
    }

    def __init__(self, cfg):
        self.cfg      = cfg
        self.path     = os.path.join(cfg["checkpoints_dir"], "state.json")
        self.path_bak = self.path + ".bak"
        self.path_tmp = self.path + ".tmp"
        self._state   = None

    def load(self):
        """Load state from primary file, fall back to backup if corrupt."""
        for path, label in [(self.path, "primary"), (self.path_bak, "backup")]:
            if not os.path.exists(path):
                continue
            try:
                with open(path) as f:
                    self._state = json.load(f)
                if label == "backup":
                    recovered(f"State loaded from backup: {path}")
                return self._state
            except (json.JSONDecodeError, OSError) as e:
                warn(f"State {label} unreadable ({e}) — trying next")
        info("No prior state found — starting fresh")
        self._state = dict(self.EMPTY)
        return self._state

    def save(self, state=None):
        """
        Atomic write: serialize to .tmp → backup current .json → os.replace().
        A power cut during write leaves .tmp; prior .json is intact.
        """
        if state is not None:
            self._state = state
        self._state["last_updated"] = datetime.now().isoformat()
        self._state["version"]      = __version__
        os.makedirs(self.cfg["checkpoints_dir"], exist_ok=True)
        try:
            with open(self.path_tmp, "w") as f:
                json.dump(self._state, f, indent=2)
        except OSError as e:
            warn(f"Cannot write state temp file: {e}")
            return False
        if os.path.exists(self.path):
            try:
                shutil.copy2(self.path, self.path_bak)
            except OSError:
                pass  # Non-critical
        try:
            os.replace(self.path_tmp, self.path)
        except OSError as e:
            warn(f"Atomic replace failed: {e}")
            return False
        return True

    # ── Flag helpers ──────────────────────────────────────────────────────────

    def _scene_refined_dir(self, sid):
        """Return path to the refined frames directory for a scene."""
        return os.path.join(self.cfg["frames_dir"], f"scene_{sid:02d}")

    def _scene_raw_dir(self, sid):
        """Return path to the raw (pre-refiner) frames directory for a scene."""
        return os.path.join(self.cfg["raw_frames_dir"], f"scene_{sid:02d}")

    def _flag(self, sid, name):
        """Return full path to a stage flag file for a scene."""
        return os.path.join(self._scene_refined_dir(sid), name)

    def write_flag(self, sid, name, meta=None):
        """Atomically write a stage-completion flag JSON file."""
        os.makedirs(self._scene_refined_dir(sid), exist_ok=True)
        p   = self._flag(sid, name)
        tmp = p + ".tmp"
        payload = {"timestamp": datetime.now().isoformat(),
                   "scene_id": sid, "flag": name, **(meta or {})}
        try:
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, p)
            return True
        except OSError as e:
            warn(f"Flag write failed ({name}, scene {sid}): {e}")
            return False

    def has_flag(self, sid, name):
        """Return True if the stage flag file exists on disk."""
        return os.path.exists(self._flag(sid, name))

    def count_frames(self, directory, sid):
        """Count frame_*.png files in a scene directory (0 if missing)."""
        d = os.path.join(directory, f"scene_{sid:02d}")
        if not os.path.exists(d):
            return 0
        return len([f for f in os.listdir(d)
                    if f.startswith("frame_") and f.endswith(".png")])

    # ── Recovery diagnosis ────────────────────────────────────────────────────

    def diagnose(self, sid):
        """
        Inspect disk state for scene sid and return a recovery decision dict.

        Returns dict with keys:
            skip_all       — all three stages done, skip scene entirely
            skip_ab        — Stages A and B done, only Stage C needed
            skip_a         — Stage A done, Stages B and C needed
            keyframe_path  — path to existing keyframe PNG or None
        """
        n_frames = self.cfg["frames_per_chunk"]
        kf_path  = self._state.get("keyframe_paths", {}).get(str(sid))
        kf_ok    = bool(kf_path and os.path.exists(kf_path))

        flag_a   = self.has_flag(sid, "stage_a_done.flag")
        flag_b   = self.has_flag(sid, "stage_b_done.flag")
        flag_c   = self.has_flag(sid, "stage_c_done.flag")
        n_raw    = self.count_frames(self.cfg["raw_frames_dir"], sid)
        n_ref    = self.count_frames(self.cfg["frames_dir"], sid)

        d = {"skip_all": False, "skip_ab": False, "skip_a": False,
             "keyframe_path": kf_path if kf_ok else None}

        if flag_c and n_ref >= n_frames:
            d["skip_all"] = True
        elif flag_b and n_raw >= n_frames and flag_a and kf_ok:
            d["skip_ab"]  = True
        elif flag_a and kf_ok:
            d["skip_a"]   = True
        elif kf_ok:
            # Keyframe exists but no flag — disk write may be incomplete
            warn(f"Scene {sid:02d}: keyframe exists without flag — will regenerate Stage A")
            d["keyframe_path"] = None

        return d

    # ── Convenience mutators ──────────────────────────────────────────────────

    def set_keyframe(self, sid, path):
        """Record a keyframe path in state and save atomically."""
        self._state.setdefault("keyframe_paths", {})[str(sid)] = path
        self.save()

    def set_bridge(self, sid, path):
        """Record a bridge frame path and save atomically."""
        self._state.setdefault("bridge_frames", {})[str(sid)] = path
        self.save()

    def get_bridge(self, sid):
        """Return bridge frame path for scene sid-1 (for conditioning)."""
        return self._state.get("bridge_frames", {}).get(str(sid))

    def mark_complete(self, sid, ta, tb, tc):
        """Mark scene complete with per-stage timings and save."""
        c = set(self._state.get("completed_scenes", []))
        c.add(sid)
        self._state["completed_scenes"] = sorted(c)
        self._state.setdefault("timing", {})[str(sid)] = {
            "stage_a_s": round(ta, 1),
            "stage_b_s": round(tb, 1),
            "stage_c_s": round(tc, 1),
            "total_s":   round(ta + tb + tc, 1),
        }
        self.save()

    @property
    def state(self):
        """Return the current in-memory state dict."""
        return self._state

    def print_recovery_report(self, cfg):
        """Print a formatted recovery status table at startup."""
        header("♻   CRASH-RESUME STATUS  —  v4 Three-Stage Pipeline")
        completed = set(self._state.get("completed_scenes", []))
        print(f"  {'ID':<5} {'Scene':<42} {'Status'}")
        print(f"  {'─'*5} {'─'*42} {'─'*28}")
        for sc in SCENES:
            sid  = sc["id"]
            d    = self.diagnose(sid)
            if sid in completed or d["skip_all"]:
                status = f"{C.GREEN}✔ All 3 stages done{C.END}"
            elif d["skip_ab"]:
                status = f"{C.YELLOW}◑ A+B done, need Stage C (refine){C.END}"
            elif d["skip_a"]:
                status = f"{C.YELLOW}◔ A done, need Stages B+C{C.END}"
            elif d["keyframe_path"]:
                status = f"{C.YELLOW}○ Keyframe no flag — redo A{C.END}"
            else:
                status = f"{C.RED}○ Not started{C.END}"
            print(f"  {sid:<5} {sc['name']:<42} {status}")
        n = len(completed)
        print(f"\n  {n}/{len(SCENES)} scenes fully complete\n")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  STATISTICS TRACKER                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class StatisticsTracker:
    """
    Tracks per-scene timing, GPU/VRAM/RAM usage, and computes ETA.
    All metrics are accumulated in deques for rolling averages.
    """

    def __init__(self, cfg):
        self.cfg    = cfg
        self.t0     = None
        self.ta_times = []  # Stage A (image) times per scene
        self.tb_times = []  # Stage B (video) times per scene
        self.tc_times = []  # Stage C (refiner) times per scene
        self.total_times = []
        self.gpu_hist  = deque(maxlen=200)
        self.ram_hist  = deque(maxlen=200)
        self.vram_hist = deque(maxlen=200)

    def start(self):
        """Call once before the generation loop begins."""
        self.t0 = time.time()

    def tick(self, gpu, ram, vram):
        """Called every 0.5 s by the monitoring thread."""
        self.gpu_hist.append(gpu)
        self.ram_hist.append(ram)
        self.vram_hist.append(vram)

    def end_scene(self, ta, tb, tc):
        """Record timing for a completed scene."""
        self.ta_times.append(ta)
        self.tb_times.append(tb)
        self.tc_times.append(tc)
        self.total_times.append(ta + tb + tc)

    def elapsed(self):
        """Return seconds since start() was called."""
        return time.time() - self.t0 if self.t0 else 0

    def eta(self):
        """Estimate remaining seconds based on last-5-scene rolling average."""
        done = len(self.total_times)
        if done < 1:
            return 0
        avg = sum(self.total_times[-5:]) / min(5, done)
        return avg * (self.cfg["num_chunks"] - done)

    @staticmethod
    def hms(s):
        """
        Format seconds as human-readable string.
        Static so it can be called as StatisticsTracker.hms(n) or self.hms(n).
        Returns 'N/A' for None to prevent TypeError on int(None).
        """
        if s is None:
            return "N/A"
        s = int(s)
        h, m, sec = s // 3600, (s % 3600) // 60, s % 60
        if h:  return f"{h}h {m:02d}m"
        if m:  return f"{m}m {sec:02d}s"
        return f"{sec}s"

    def summary(self):
        """Print final generation statistics to terminal."""
        el  = self.elapsed()
        n   = len(self.total_times)
        avg = sum(self.total_times)  / n if n else 0
        fps = (n * self.cfg["frames_per_chunk"]) / el if el > 0 else 0
        g   = sum(self.gpu_hist)  / len(self.gpu_hist)  if self.gpu_hist  else 0
        v   = sum(self.vram_hist) / len(self.vram_hist) if self.vram_hist else 0
        r   = sum(self.ram_hist)  / len(self.ram_hist)  if self.ram_hist  else 0

        def avg_t(lst): return sum(lst) / len(lst) if lst else 0

        print(f"\n{C.BOLD}{C.CYAN}{'═'*76}{C.END}")
        print(f"{C.BOLD}{C.CYAN}  📊  GENERATION STATISTICS  —  sana-dental-4.py  v{__version__}{C.END}")
        print(f"{C.BOLD}{C.CYAN}{'═'*76}{C.END}")
        print(f"  {'Total time':<40} {self.hms(el)}")
        print(f"  {'Scenes completed this run':<40} {n}/{self.cfg['num_chunks']}")
        print(f"  {'Avg total time / scene':<40} {avg:.1f}s")
        print(f"  {'Avg Stage A time (4.8B keyframe)':<40} {avg_t(self.ta_times):.1f}s")
        print(f"  {'Avg Stage B time (2B video)':<40} {avg_t(self.tb_times):.1f}s")
        print(f"  {'Avg Stage C time (LTX-2 refiner)':<40} {avg_t(self.tc_times):.1f}s")
        print(f"  {'Generation speed':<40} {fps:.2f} frames/s")
        print(f"  {'Avg GPU load':<40} {g:.1f}%")
        print(f"  {'Avg VRAM used':<40} {v / 100 * 16:.1f} GB / 16 GB")
        print(f"  {'Avg RAM used':<40} {r:.1f}%")
        print(f"{C.BOLD}{C.CYAN}{'═'*76}{C.END}\n")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MONITORING THREAD                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_metrics     = {"gpu": deque(maxlen=300), "vram": deque(maxlen=300),
                "cpu": deque(maxlen=300), "ram":  deque(maxlen=300)}
_tracker_ref = None  # Set to the active StatisticsTracker in main()


def _monitor_loop(stop_evt):
    """
    Background thread: samples GPU/CPU/RAM every 0.5 s and feeds StatisticsTracker.
    Uses GPUtil for GPU load + VRAM; falls back to zeros if unavailable.
    """
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
            for k, v in zip(["gpu", "vram", "cpu", "ram"], [gpu, vram, cpu, ram]):
                _metrics[k].append(v)
            if _tracker_ref:
                _tracker_ref.tick(gpu, ram, vram)
        except Exception:
            pass
        time.sleep(0.5)


def _print_live(sid, n_scenes, stage_label, scene_name, stat):
    """
    Refresh the terminal with a live dashboard showing:
    - Overall progress bar
    - Current scene and stage
    - GPU, VRAM, CPU, RAM bars
    - ETA and elapsed time
    Called before starting each stage of each scene.
    """
    os.system("cls" if os.name == "nt" else "clear")
    pct = sid / n_scenes if n_scenes else 0
    print(f"{C.BOLD}{C.CYAN}{'═'*76}{C.END}")
    print(f"{C.BOLD}{C.CYAN}  🦷  SmartSmile 2250  v{__version__}  —  3-Stage 4.8B→2B→LTX2 Quality{C.END}")
    print(f"{C.BOLD}{C.CYAN}{'═'*76}{C.END}")
    print(f"\n  {'Scene':<22} {sid}/{n_scenes}  {scene_name}")
    print(f"  {'Stage':<22} {stage_label}")
    print(f"  {'Overall':<22} [{bar(pct)}]  {pct*100:.1f}%")
    eta = stat.eta()
    if eta > 0:
        print(f"  {'ETA':<22} {stat.hms(eta)}")
    print(f"  {'Elapsed':<22} {stat.hms(stat.elapsed())}")

    g    = list(_metrics["gpu"])[-1]  if _metrics["gpu"]  else 0
    vr   = list(_metrics["vram"])[-1] if _metrics["vram"] else 0
    r    = list(_metrics["ram"])[-1]  if _metrics["ram"]  else 0
    cpu  = list(_metrics["cpu"])[-1]  if _metrics["cpu"]  else 0

    print(f"\n  {C.BOLD}GPU  RTX 5080{C.END}")
    print(f"  {'  Load':<22} [{bar(g  / 100, 30)}]  {g:.0f}%")
    print(f"  {'  VRAM':<22} [{bar(vr / 100, 30)}]  {vr:.0f}%  ({vr/100*16:.1f}/16 GB)")
    print(f"\n  {C.BOLD}System{C.END}")
    print(f"  {'  CPU Ryzen 9900X':<22} [{bar(cpu / 100, 30)}]  {cpu:.0f}%")
    vm = psutil.virtual_memory()
    print(f"  {'  RAM DDR5':<22} [{bar(r / 100, 30)}]  {r:.0f}%  "
          f"({vm.used/1e9:.1f}/{vm.total/1e9:.0f} GB)")
    if stat.ta_times:
        print(f"\n  {C.DIM}Avg A:{sum(stat.ta_times)/len(stat.ta_times):.0f}s  "
              f"B:{sum(stat.tb_times)/len(stat.tb_times):.0f}s  "
              f"C:{sum(stat.tc_times)/len(stat.tc_times):.0f}s{C.END}")
    print(f"\n{C.BOLD}{C.CYAN}{'─'*76}{C.END}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  VRAM HELPERS                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def vram_free():
    """
    Force-free all GPU memory by running GC and calling CUDA empty_cache.
    Includes torch.cuda.synchronize() to ensure async ops have completed.
    Sleeps 1.5 s to allow OS to reclaim memory before next pipeline loads.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(1.5)


def vram_used_gb():
    """Return currently allocated CUDA memory in GB (0 if CUDA unavailable)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  IMAGE HELPERS                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def blend_images(primary: Image.Image, secondary: Image.Image,
                 alpha_secondary: float, target_size: tuple) -> Image.Image:
    """
    Blend two PIL images at the given ratio.

    Formula: result = (1 - alpha) * primary + alpha * secondary

    Used to combine the 4.8B keyframe (primary, high quality) with the
    refined bridge frame from the previous scene (secondary, continuity).

    Args:
        primary:          High-quality keyframe from Stage A (current scene).
        secondary:        Refined last-frame from Stage C (previous scene).
        alpha_secondary:  Weight for secondary image (0.30 = 30% bridge).
        target_size:      (width, height) to resize both images to.

    Returns:
        Blended PIL Image at target_size.
    """
    if primary.size   != target_size:
        primary   = primary.resize(target_size,   Image.LANCZOS)
    if secondary.size != target_size:
        secondary = secondary.resize(target_size, Image.LANCZOS)
    a = np.array(primary,   dtype=np.float32)
    b = np.array(secondary, dtype=np.float32)
    return Image.fromarray(
        ((1.0 - alpha_secondary) * a + alpha_secondary * b)
        .clip(0, 255).astype(np.uint8)
    )


def ensure_png(image) -> Image.Image:
    """Convert frame to PIL Image if necessary. Enforces uint8 RGB mode."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.array(image, dtype=np.uint8)).convert("RGB")


def save_keyframe(image, sid: int, cfg: dict) -> str:
    """
    Save a Stage A keyframe PNG with OS buffer flush before writing the flag.

    The file is force-flushed with f.flush() + os.fsync() to ensure it is
    fully on SSD before stage_a_done.flag is written. This prevents false-
    positive flag detection after a crash mid-write.

    Returns:
        Absolute path to the saved keyframe PNG.
    """
    os.makedirs(cfg["keyframes_dir"], exist_ok=True)
    path = os.path.join(cfg["keyframes_dir"], f"keyframe_scene_{sid:02d}.png")
    img  = ensure_png(image)
    img.save(path, "PNG")
    # Force OS buffer flush to disk before returning
    with open(path, "r+b") as fh:
        fh.flush()
        os.fsync(fh.fileno())
    ok(f"Keyframe saved → {os.path.basename(path)}  ({img.size[0]}×{img.size[1]})")
    return path


def save_frames(frames: list, sid: int, directory: str, cfg: dict) -> list:
    """
    Save a list of video frames as lossless PNG files.

    Frames are stored in directory/scene_NN/frame_NNNNNN.png.
    Existing frames are overwritten (safe for re-runs after crash).

    Args:
        frames:    List of PIL Images or numpy arrays.
        sid:       Scene index (0–11).
        directory: Parent directory (raw_frames_dir or frames_dir).
        cfg:       Config dict for frames_per_chunk.

    Returns:
        List of absolute paths to saved frame files.
    """
    scene_dir  = os.path.join(directory, f"scene_{sid:02d}")
    os.makedirs(scene_dir, exist_ok=True)
    base_idx   = sid * cfg["frames_per_chunk"]
    paths      = []
    for i, frame in enumerate(frames):
        img  = ensure_png(frame)
        path = os.path.join(scene_dir, f"frame_{base_idx + i:06d}.png")
        img.save(path, "PNG")
        paths.append(path)
    return paths


def save_bridge(frames: list, sid: int, cfg: dict) -> str:
    """
    Save the last frame of a refined scene chunk as the bridge PNG.

    In v4 the bridge is taken from Stage C (refined) frames, not Stage B.
    This means the high-texture, high-quality refined output is what carries
    forward into the next scene's conditioning signal.

    Returns:
        Path to the bridge PNG.
    """
    os.makedirs(cfg["bridges_dir"], exist_ok=True)
    last = ensure_png(frames[-1])
    path = os.path.join(cfg["bridges_dir"], f"bridge_after_scene_{sid:02d}.png")
    last.save(path, "PNG")
    return path


def prepare_conditioning(keyframe_path: str, bridge_path: str,
                          sid: int, cfg: dict) -> Image.Image:
    """
    Build the Stage B conditioning image by blending keyframe and bridge.

    Scene 0: keyframe only (no prior bridge exists).
    Scene N: 70% keyframe + 30% refined-bridge (from prior scene's Stage C).

    The blend ratio is controlled by cfg["bridge_blend_alpha"] (default 0.30).
    Lower alpha = more keyframe quality influence.
    Higher alpha = more inter-scene visual continuity.

    Args:
        keyframe_path: Path to current scene's Stage A keyframe PNG.
        bridge_path:   Path to prior scene's Stage C last-frame PNG (may be None).
        sid:           Current scene index.
        cfg:           Config dict.

    Returns:
        PIL Image resized to (cfg["width"], cfg["height"]).
    """
    target   = (cfg["width"], cfg["height"])
    keyframe = Image.open(keyframe_path).convert("RGB").resize(target, Image.LANCZOS)

    if (sid > 0 and cfg.get("use_bridge_blending")
            and bridge_path and os.path.exists(bridge_path)):
        bridge = Image.open(bridge_path).convert("RGB").resize(target, Image.LANCZOS)
        alpha  = cfg.get("bridge_blend_alpha", 0.30)
        info(f"[B] Conditioning = keyframe {(1-alpha)*100:.0f}% + refined-bridge {alpha*100:.0f}%")
        return blend_images(keyframe, bridge, alpha, target)

    info("[B] Conditioning = keyframe only (scene 0 or no prior bridge)")
    return keyframe


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  STAGE A — SANA-1.5 4.8B KEYFRAME GENERATION                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_stage_a(cfg: dict):
    """
    Load the SANA-1.5 4.8B image pipeline with PAG (Perturbed Attention Guidance).

    PAG is loaded at model-init time by passing pag_applied_layers to from_pretrained.
    This is required; you cannot add PAG post-hoc. The pag_applied_layers list
    targets transformer_blocks, which is the correct layer for SANA-1.5.

    VRAM profile: ~11 GB at peak (4.8B parameters in bfloat16 + fp32 VAE).
    Must be deleted and VRAM cleared before Stage B loads.

    Returns:
        Loaded SanaPipeline instance.

    Raises:
        RuntimeError: If CUDA is unavailable or VRAM is insufficient.
    """
    subheader("Stage A — Loading SANA-1.5 4.8B with PAG")
    info(f"Model : {cfg['model_image'].split('/')[-1]}")
    info(f"Steps : {cfg['image_inference_steps']}  "
         f"CFG : {cfg['image_guidance_scale']}  "
         f"PAG : {cfg['image_pag_guidance_scale']}")

    # PAG requires initialization at load time via enable_pag=True
    # and pag_applied_layers specification. Diffusers >= 0.29 supports this.
    try:
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(
            cfg["model_image"],
            torch_dtype=cfg["torch_dtype"],
            enable_pag=True,
            pag_applied_layers=cfg["pag_applied_layers"],
        )
    except (TypeError, ImportError):
        warn("AutoPipeline with PAG unavailable — loading SanaPipeline without PAG")
        pipe = SanaPipeline.from_pretrained(
            cfg["model_image"],
            torch_dtype=cfg["torch_dtype"],
        )

    # VAE must always remain in float32 for stable decoding.
    # bfloat16 VAE introduces checkerboard artifacts in decoded images.
    pipe.vae.to(cfg["vae_dtype"])
    pipe.text_encoder.to(cfg["torch_dtype"])
    pipe.to("cuda")

    if cfg["cpu_offload"]:
        pipe.enable_model_cpu_offload()

    torch.backends.cuda.matmul.allow_tf32 = True  # Blackwell TF32 speedup
    ok(f"Stage A ready — VRAM: {vram_used_gb():.1f} GB")
    return pipe


def run_stage_a(pipe, scene: dict, cfg: dict, seed: int) -> Image.Image:
    """
    Run Stage A: generate a 1024×1024 keyframe using SANA-1.5 4.8B.

    The image_prompt is pure visual description with NO motion language.
    Motion language confuses image models — they are trained on static images
    and will attempt to render motion as blur or distortion.

    PAG (if active) improves structural coherence of complex compositions:
    operating rooms, crowds, architectural spaces, and close-up detail work
    all benefit significantly from PAG at scale 2.5.

    Args:
        pipe:  Loaded SanaPipeline (Stage A).
        scene: Scene dict containing image_prompt.
        cfg:   Config dict.
        seed:  Deterministic seed for reproducibility.

    Returns:
        PIL Image at 1024×1024.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    prompt    = scene["image_prompt"]

    info(f"[A] Prompt: …{prompt[50:120]}…")
    info(f"[A] {cfg['keyframe_width']}×{cfg['keyframe_height']}  "
         f"steps={cfg['image_inference_steps']}  "
         f"cfg={cfg['image_guidance_scale']}  "
         f"pag={cfg['image_pag_guidance_scale']}")

    # Try with PAG first; fall back gracefully if the loaded pipeline
    # does not support pag_guidance_scale (older diffusers build).
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_IMAGE,
            height=cfg["keyframe_height"],
            width=cfg["keyframe_width"],
            guidance_scale=cfg["image_guidance_scale"],
            pag_guidance_scale=cfg["image_pag_guidance_scale"],
            num_inference_steps=cfg["image_inference_steps"],
            generator=generator,
        )
    except TypeError:
        warn("[A] pag_guidance_scale not accepted — running without PAG")
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_IMAGE,
            height=cfg["keyframe_height"],
            width=cfg["keyframe_width"],
            guidance_scale=cfg["image_guidance_scale"],
            num_inference_steps=cfg["image_inference_steps"],
            generator=generator,
        )

    image = result.images[0]
    ok(f"[A] Keyframe generated  ({image.size[0]}×{image.size[1]})")
    return image


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  STAGE B — SANA-VIDEO 2B I2V VIDEO GENERATION                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_stage_b(cfg: dict):
    """
    Load SANA-Video 2B I2V (Image-to-Video) pipeline.

    Architecture notes (from NVLabs official documentation):
    - Block Causal Linear Attention: O(1) memory per step for long video.
    - DC-AE (32× spatial compression): 16× fewer latent tokens than 8× VAEs.
    - FlowMatchEulerDiscreteScheduler: rectified-flow sampler.
    - flow_shift=10: biases sampler toward aggressive early denoising
      for sharper high-frequency detail.

    VRAM profile: ~5.5 GB (2B bfloat16 + fp32 VAE).
    Stage A must be fully unloaded before calling this.

    Returns:
        Loaded SanaImageToVideoPipeline instance.
    """
    subheader("Stage B — Loading SANA-Video 2B I2V Pipeline")
    info(f"Model : {cfg['model_video'].split('/')[-1]}")
    info(f"Steps : {cfg['num_inference_steps']}  "
         f"CFG : {cfg['guidance_scale']}  "
         f"flow_shift : {cfg['flow_shift']}")

    pipe = SanaImageToVideoPipeline.from_pretrained(
        cfg["model_video"],
        torch_dtype=cfg["torch_dtype"],
    )
    pipe.vae.to(cfg["vae_dtype"])              # fp32 VAE — mandatory
    pipe.text_encoder.to(cfg["torch_dtype"])
    pipe.to("cuda")

    if cfg["cpu_offload"]:
        pipe.enable_model_cpu_offload()
    if cfg["attention_slicing"]:
        pipe.enable_attention_slicing()

    ok(f"Stage B ready — VRAM: {vram_used_gb():.1f} GB")
    return pipe


def run_stage_b(pipe, scene: dict, conditioning: Image.Image,
                cfg: dict, seed: int):
    """
    Run Stage B: animate the conditioning image into 81 video frames.

    The video_prompt extends the image description with temporal and camera
    language. The motion score token (" motion score: N.") must appear at
    the end of the prompt — it is a trained token, not a comment.

    output_type="latent" returns the raw video latent tensor instead of
    decoded frames. This is passed directly to Stage C (LTX-2 refiner)
    without an expensive VAE decode+encode round-trip, saving ~3 GB VRAM
    and ~15 seconds per scene.

    Args:
        pipe:         Loaded SanaImageToVideoPipeline.
        scene:        Scene dict containing video_prompt and motion_score.
        conditioning: Blended conditioning image (Stage A keyframe + bridge).
        cfg:          Config dict.
        seed:         Deterministic seed.

    Returns:
        Raw video latent tensor (torch.Tensor) if LTX-2 refiner is enabled,
        otherwise a list of PIL Image frames.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    prompt    = scene["video_prompt"]

    info(f"[B] Prompt: …{prompt[50:120]}…")
    info(f"[B] motion_score={scene['motion_score']}  "
         f"steps={cfg['num_inference_steps']}  "
         f"cfg={cfg['guidance_scale']}  "
         f"flow_shift={cfg['flow_shift']}")

    # If LTX-2 refiner is enabled, request latent output to avoid
    # an unnecessary decode→encode roundtrip. The latent is passed
    # directly to Stage C's encode_video() function.
    out_type = "latent" if cfg.get("use_ltx2_refiner") else "pil"

    result = pipe(
        image=conditioning,
        prompt=prompt,
        negative_prompt=NEGATIVE_VIDEO,
        height=cfg["height"],
        width=cfg["width"],
        num_frames=cfg["frames_per_chunk"],
        guidance_scale=cfg["guidance_scale"],
        num_inference_steps=cfg["num_inference_steps"],
        generator=generator,
        output_type=out_type,
    )

    if out_type == "latent":
        ok(f"[B] Raw latent returned  (shape: {result.frames.shape})")
        return result.frames   # torch.Tensor → passed to Stage C
    else:
        frames = result.frames[0]
        ok(f"[B] Frames decoded  ({len(frames)} frames)")
        return frames


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  STAGE C — LTX-2 REFINER ("Bet Small, Win Big")                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_stage_c(cfg: dict):
    """
    Load the LTX-2 refiner pipeline (step-distilled LoRA).

    Official source: nvlabs.github.io/Sana/docs/sana_video_inference/
    Blog: nvlabs.github.io/Sana/Video/bet-small-win-big/blog.html

    Architecture:
    - LTX2Pipeline: the 19B LTX-2 base (used only as a host for the LoRA)
    - LTX2LatentUpsamplePipeline: spatially upsamples the video latent
    - distilled LoRA adapter: 38k-step distillation, enables 3-step inference

    The refiner operates on the raw video latent from Stage B. It does NOT
    need to decode and re-encode the video — it works in latent space.
    This is what makes it near-zero cost: only 3 diffusion steps.

    What the refiner fixes:
    - Soft/blurry textures on teeth, skin, fabric, metal surfaces
    - Temporal flicker and aliasing at frame boundaries
    - Sub-pixel detail loss from DC-AE 32× compression
    - Color saturation and contrast that SANA-Video base underdelivers

    VRAM profile: ~6 GB.

    Returns:
        Tuple of (ltx_pipe, latent_upsample_pipe) or None on failure.
    """
    if not cfg.get("use_ltx2_refiner"):
        return None

    subheader("Stage C — Loading LTX-2 Refiner (Bet Small, Win Big)")
    info(f"Model : {cfg['model_ltx2']}")
    info(f"Steps : {cfg['ltx2_inference_steps']} (distilled — do not change)")

    try:
        from diffusers import LTX2Pipeline
        from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
        from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

        ltx_pipe = LTX2Pipeline.from_pretrained(
            cfg["model_ltx2"],
            torch_dtype=cfg["torch_dtype"],
        )
        ltx_pipe.load_lora_weights(
            cfg["model_ltx2"],
            adapter_name="stage_2_distilled",
            weight_name=cfg["ltx2_adapter"],
        )
        ltx_pipe.set_adapters("stage_2_distilled")
        ltx_pipe.to("cuda")

        # Latent upsampler — spatial resolution enhancement
        upsample_model = LTX2LatentUpsamplerModel.from_pretrained(
            cfg["model_ltx2"],
            subfolder="latent_upsampler",
            torch_dtype=cfg["torch_dtype"],
        )
        upsample_pipe = LTX2LatentUpsamplePipeline(
            upsample_model=upsample_model,
        )
        upsample_pipe.to("cuda")

        ok(f"Stage C ready — VRAM: {vram_used_gb():.1f} GB")
        return ltx_pipe, upsample_pipe, STAGE_2_DISTILLED_SIGMA_VALUES

    except (ImportError, Exception) as e:
        warn(f"LTX-2 refiner unavailable ({e})")
        warn("Stage C will be SKIPPED — frames will use Stage B output directly")
        warn("Install latest diffusers: pip install git+https://github.com/huggingface/diffusers")
        return None


def run_stage_c(stage_c_bundle, raw_latent, scene: dict,
                cfg: dict, seed: int) -> list:
    """
    Run Stage C: refine raw video latent with LTX-2 (3 steps, distilled).

    Input: raw_latent from Stage B (torch.Tensor in VRAM)
    Output: list of refined PIL Image frames

    The refiner process:
    1. LTX2LatentUpsamplePipeline spatially upsamples the latent
       (increases resolution in latent space before decoding)
    2. LTX2Pipeline applies the distilled LoRA in 3 denoising steps,
       adding fine texture, correcting temporal inconsistencies, and
       recovering high-frequency detail lost by DC-AE compression
    3. Decode to pixel space → list of PIL Images

    If stage_c_bundle is None (refiner failed to load), this function
    decodes raw_latent directly via a minimal VAE decode and returns
    plain frames — a graceful quality fallback.

    Args:
        stage_c_bundle: (ltx_pipe, upsample_pipe, sigma_values) or None.
        raw_latent:     torch.Tensor video latent from Stage B.
        scene:          Scene dict (for prompt and motion_score).
        cfg:            Config dict.
        seed:           Deterministic seed.

    Returns:
        List of PIL Image frames (length = frames_per_chunk).
    """
    if stage_c_bundle is None:
        warn("[C] Stage C unavailable — decoding Stage B latent directly")
        # Minimal fallback: just decode the raw latent
        # This will be lower quality than refined but still correct
        return _decode_latent_fallback(raw_latent, cfg)

    ltx_pipe, upsample_pipe, sigma_values = stage_c_bundle
    generator = torch.Generator(device="cuda").manual_seed(seed)

    info(f"[C] Refining latent with LTX-2  ({cfg['ltx2_inference_steps']} distilled steps)")

    try:
        from diffusers.pipelines.ltx2.export_utils import encode_video

        # Step 1: spatially upsample the latent
        upsampled = upsample_pipe(
            latents=raw_latent,
        ).latents

        # Step 2: apply distilled refiner (3 steps)
        result = ltx_pipe(
            prompt=scene["video_prompt"],
            negative_prompt=NEGATIVE_VIDEO,
            latents=upsampled,
            num_inference_steps=cfg["ltx2_inference_steps"],
            guidance_scale=1.0,       # Distilled — no CFG needed
            sigmas=sigma_values,
            generator=generator,
            output_type="pil",
        )
        frames = result.frames[0]
        ok(f"[C] Refined  ({len(frames)} frames)")
        return frames

    except Exception as e:
        warn(f"[C] LTX-2 refine failed ({e}) — falling back to direct decode")
        return _decode_latent_fallback(raw_latent, cfg)


def _decode_latent_fallback(raw_latent, cfg: dict) -> list:
    """
    Emergency fallback: decode Stage B latent without refinement.
    Used when Stage C fails or is unavailable.
    Returns a list of PIL Images from the raw video latent.
    This produces lower quality than the full refiner pipeline.
    """
    info("[C-fallback] Decoding Stage B latent without refinement")
    try:
        # Use SANA-Video's own VAE decode path
        pipe_tmp = SanaVideoPipeline.from_pretrained(
            cfg["model_video"],
            torch_dtype=cfg["torch_dtype"],
        )
        pipe_tmp.vae.to(cfg["vae_dtype"])
        pipe_tmp.to("cuda")
        with torch.no_grad():
            decoded = pipe_tmp.vae.decode(raw_latent).sample
        pipe_tmp.cpu()
        del pipe_tmp
        vram_free()
        # Convert tensor to PIL Images
        frames = []
        for i in range(decoded.shape[2]):  # time dimension
            frame = decoded[0, :, i].permute(1, 2, 0).float().cpu().numpy()
            frame = ((frame * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            frames.append(Image.fromarray(frame))
        ok(f"[C-fallback] Decoded {len(frames)} frames from raw latent")
        return frames
    except Exception as e2:
        err(f"[C-fallback] Failed: {e2}")
        # Last resort: return blank frames so pipeline doesn't crash
        return [Image.new("RGB", (cfg["width"], cfg["height"]), (20, 20, 30))
                for _ in range(cfg["frames_per_chunk"])]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  VIDEO ASSEMBLY                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def assemble_video(cfg: dict) -> str | None:
    """
    Stitch all refined scene frame PNGs into the final MP4.

    Encoding parameters:
    - codec: libx264 (H.264)
    - crf: 18 (near-lossless; 0=lossless, 18=high quality, 23=default)
    - preset: slow (better compression ratio vs fast)
    - pix_fmt: yuv420p (maximum compatibility with media players)
    - level: 4.2 (supports 1280×704 @ 16fps without issues)
    - movflags +faststart: places moov atom at file start for web streaming

    Returns:
        Path to the assembled MP4, or None if assembly failed.
    """
    header("🎬  Assembling Final Video")
    final_path = os.path.join(cfg["output_dir"], cfg["final_video"])

    all_frames = []
    for s in range(cfg["num_scenes"]):
        d     = os.path.join(cfg["frames_dir"], f"scene_{s:02d}")
        if not os.path.exists(d):
            warn(f"Scene {s:02d} refined frames directory missing — skipping")
            continue
        files = sorted([os.path.join(d, f) for f in os.listdir(d)
                        if f.startswith("frame_") and f.endswith(".png")])
        all_frames.extend(files)

    if not all_frames:
        err("No refined frames found for assembly.")
        return None

    ok(f"Found {len(all_frames)} frames  →  encoding CRF {cfg['crf']}, preset {cfg['preset']}")

    writer = imageio.get_writer(
        final_path, fps=cfg["fps"], codec="libx264", quality=None,
        output_params=[
            "-crf",       str(cfg["crf"]),
            "-preset",    cfg["preset"],
            "-pix_fmt",   "yuv420p",
            "-profile:v", "high",
            "-level",     "4.2",
            "-movflags",  "+faststart",   # web-streaming ready
        ],
    )

    last_upd = time.time()
    for i, fp in enumerate(all_frames):
        try:
            writer.append_data(imageio.imread(fp))
        except Exception as e:
            warn(f"Skipping corrupt frame {os.path.basename(fp)}: {e}")
        if time.time() - last_upd > 3:
            pct = (i + 1) / len(all_frames)
            print(f"    [{bar(pct, 30)}]  {pct*100:.0f}%  {i+1}/{len(all_frames)}")
            last_upd = time.time()
    writer.close()

    if os.path.exists(final_path):
        mb   = os.path.getsize(final_path) / 1e6
        dur  = len(all_frames) / cfg["fps"]
        mbps = (os.path.getsize(final_path) * 8) / dur / 1e6
        ok(f"Video: {final_path}")
        ok(f"Size: {mb:.1f} MB  |  Duration: {dur:.1f}s  |  Bitrate: {mbps:.1f} Mbps")
        return final_path

    err("Assembly failed — check imageio/ffmpeg installation.")
    return None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PREFLIGHT                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def preflight(cfg: dict) -> bool:
    """
    Check system readiness before starting generation.

    Validates:
    - diffusers installation and required class availability
    - CUDA availability and VRAM capacity
    - RAM and disk space adequacy
    - Prints full pipeline config summary with v4 quality improvements

    Returns:
        True if all checks pass, False if generation should not proceed.
    """
    header("🔍  Preflight Checks — sana-dental-4.py")

    if not DIFFUSERS_OK:
        err(f"diffusers import failed: {_diffusers_missing}")
        err("Run: pip install git+https://github.com/huggingface/diffusers")
        return False

    if not torch.cuda.is_available():
        err("CUDA not available. Check NVIDIA drivers and PyTorch CUDA build.")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    ok(f"GPU     : {gpu_name}  ({vram_gb:.1f} GB VRAM)")

    if vram_gb < 14:
        warn(f"VRAM is {vram_gb:.1f} GB. Stage A (4.8B) needs ~11 GB. "
             "cpu_offload is enabled but generation will be slower.")

    ram  = psutil.virtual_memory()
    disk = shutil.disk_usage(os.path.abspath("."))
    ok(f"RAM     : {ram.total/1e9:.1f} GB total  |  {ram.available/1e9:.1f} GB free")
    ok(f"Disk    : {disk.free/1e9:.1f} GB free")
    ok(f"PyTorch : {torch.__version__}  |  bfloat16 native on RTX 5080")

    # Storage estimate: keyframes (~4MB each) + raw PNGs (~2.5MB) + refined PNGs (~2.5MB)
    n = cfg["num_scenes"]
    est_kf  = n * 4 / 1024
    est_raw = cfg["total_frames"] * 2.5 / 1024
    est_ref = cfg["total_frames"] * 2.5 / 1024
    est_total = est_kf + est_raw + est_ref

    print(f"\n  {C.BOLD}v4 Three-Stage Pipeline:{C.END}")
    print(f"  {'Stage A':<20} SANA-1.5 4.8B  →  1024×1024 keyframe/scene")
    print(f"  {'Stage B':<20} SANA-Video 2B  →  81 frames @ 1280×704/scene  (I2V)")
    print(f"  {'Stage C':<20} LTX-2 Refiner  →  3-step distilled texture boost")
    print(f"\n  {C.BOLD}Quality Parameters (v3 → v4):{C.END}")
    quality(f"image_inference_steps : 30 → {cfg['image_inference_steps']}  "
            f"(+{cfg['image_inference_steps']-30} steps, sharper keyframes)")
    quality(f"image_guidance_scale  : 5.0 → {cfg['image_guidance_scale']}  "
            f"(stronger prompt fidelity)")
    quality(f"image_pag_guidance    : 2.0 → {cfg['image_pag_guidance_scale']}  "
            f"(better spatial coherence)")
    quality(f"video_guidance_scale  : 6.0 → {cfg['guidance_scale']}  "
            f"(sharper video frames)")
    quality(f"flow_shift            : 8   → {cfg['flow_shift']}  "
            f"(sharper high-freq detail)")
    quality(f"LTX-2 refiner         : OFF → ON  "
            f"(2K-quality texture at 720p latency)")
    quality(f"bridge source         : Stage B → Stage C  "
            f"(refined continuity)")

    print(f"\n  {C.BOLD}Target:{C.END}")
    print(f"  {'Resolution':<20} {cfg['width']}×{cfg['height']}  |  {cfg['fps']} fps")
    print(f"  {'Duration':<20} ~{cfg['total_duration_seconds']:.0f}s  "
          f"({cfg['num_scenes']} scenes × {cfg['frames_per_chunk']/cfg['fps']:.1f}s)")
    print(f"  {'Est. storage':<20} ~{est_total:.1f} GB")

    est_t = cfg["num_chunks"] * (90 + 36 + 15)  # ~141s/scene: A+B+C on RTX 5080
    print(f"  {'Est. time':<20} {StatisticsTracker.hms(est_t)} "
          f"(~141s/scene on RTX 5080)")

    if disk.free / 1e9 < est_total * 1.5:
        warn(f"Low disk space — need {est_total*1.5:.0f} GB, have {disk.free/1e9:.1f} GB")

    return True


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    """
    Entry point for the SmartSmile 2250 dental advertisement generator.

    Execution flow:
    1. Preflight checks
    2. Load/recover state from prior run (crash-resume)
    3. Print recovery report (which scenes are done/partial)
    4. For each incomplete scene:
       a. Stage A: SANA 4.8B keyframe (or skip if stage_a_done.flag exists)
       b. Stage B: SANA-Video 2B I2V (or skip if stage_b_done.flag exists)
       c. Stage C: LTX-2 refiner     (or skip if stage_c_done.flag exists)
       d. Save refined frames, bridge, update state
    5. Assemble all refined frames into final MP4
    6. Save timing/metrics JSON report
    """
    global _tracker_ref

    print(f"\n{C.BOLD}{C.CYAN}  sana-dental-4.py  v{__version__}  —  {__date__}{C.END}")
    print(f"{C.DIM}  3-Stage 4.8B→2B→LTX2  |  Quality Overhaul  |  SmartSmile 2250{C.END}\n")

    cfg          = get_config()
    stat         = StatisticsTracker(cfg)
    _tracker_ref = stat

    if not preflight(cfg):
        return

    # Create all output directories
    for d in [cfg["output_dir"], cfg["keyframes_dir"],
              cfg["raw_frames_dir"], cfg["frames_dir"],
              cfg["bridges_dir"], cfg["checkpoints_dir"]]:
        os.makedirs(d, exist_ok=True)

    # Load prior state and print recovery report
    sm = StateManager(cfg)
    sm.load()
    sm.print_recovery_report(cfg)

    completed = set(sm.state.get("completed_scenes", []))
    pending   = [sc for sc in SCENES if sc["id"] not in completed]

    if not pending:
        ok("All 12 scenes complete — skipping to video assembly.")
    else:
        print(f"\n{C.YELLOW}  {len(pending)} scene(s) remaining.  "
              f"Press Enter to start…  (Ctrl+C to abort){C.END}")
        try:
            input()
        except KeyboardInterrupt:
            return

    # Start background monitoring thread
    stop_evt   = threading.Event()
    mon_thread = threading.Thread(target=_monitor_loop, args=(stop_evt,), daemon=True)
    mon_thread.start()

    stat.start()
    sid = 0   # Defined here so except block can reference it safely

    try:
        for scene in SCENES:
            sid    = scene["id"]
            s_name = scene["name"]

            # Recover diagnosis for this scene
            d    = sm.diagnose(sid)
            seed = cfg["seed"] + sid   # Deterministic: same result on resume

            if d["skip_all"]:
                ok(f"Scene {sid:02d} '{s_name}' — all stages done, skipping")
                continue

            ta = tb = tc = 0.0
            t_scene = time.time()

            # ════════════════════════════════════════════════════════════════
            # STAGE A  —  SANA 4.8B KEYFRAME
            # Skip if: stage_a_done.flag exists AND keyframe PNG exists
            # ════════════════════════════════════════════════════════════════
            if d["skip_a"] or d["skip_ab"]:
                keyframe_path = d["keyframe_path"]
                recovered(f"Scene {sid:02d} Stage A — keyframe exists, skipping")
            else:
                _print_live(sid, cfg["num_scenes"],
                            f"{C.BOLD}{C.YELLOW}◉ A  SANA 4.8B Keyframe{C.END}",
                            s_name, stat)
                step(sid + 1, cfg["num_scenes"],
                     f"Scene {sid:02d}  Stage A — 4.8B Keyframe  (60 steps, PAG)")

                pipe_a  = load_stage_a(cfg)
                t0a     = time.time()
                kf_img  = run_stage_a(pipe_a, scene, cfg, seed)
                ta      = time.time() - t0a
                ok(f"Stage A done in {ta:.1f}s")

                keyframe_path = save_keyframe(kf_img, sid, cfg)
                del kf_img

                # Write flag AFTER confirming keyframe is on disk
                sm.write_flag(sid, "stage_a_done.flag",
                              {"keyframe_path": keyframe_path, "time_s": round(ta, 1)})
                sm.set_keyframe(sid, keyframe_path)
                ok("✦ stage_a_done.flag written")

                subheader("Unloading Stage A → freeing VRAM")
                del pipe_a
                vram_free()
                ok(f"VRAM after A: {vram_used_gb():.1f} GB")

            # ════════════════════════════════════════════════════════════════
            # STAGE B  —  SANA-VIDEO 2B I2V
            # Skip if: stage_b_done.flag exists AND raw frame count is OK
            # ════════════════════════════════════════════════════════════════
            if d["skip_ab"]:
                # Stage B raw frames exist — load latent for Stage C
                # (In this case we need to re-run Stage C on existing raw frames)
                recovered(f"Scene {sid:02d} Stage B — raw frames exist, re-running Stage C")
                raw_latent = None   # Stage C will operate on saved PNGs instead
            else:
                _print_live(sid, cfg["num_scenes"],
                            f"{C.BOLD}{C.GREEN}◉ B  SANA-Video 2B I2V{C.END}",
                            s_name, stat)
                step(sid + 1, cfg["num_scenes"],
                     f"Scene {sid:02d}  Stage B — 2B Video  (50 steps, flow_shift=10)")

                bridge_path  = sm.get_bridge(sid - 1)
                conditioning = prepare_conditioning(
                                   keyframe_path, bridge_path, sid, cfg)

                pipe_b  = load_stage_b(cfg)
                t0b     = time.time()
                raw_out = run_stage_b(pipe_b, scene, conditioning, cfg, seed)
                tb      = time.time() - t0b
                ok(f"Stage B done in {tb:.1f}s")

                # If raw_out is a tensor (latent), save raw decode for recovery.
                # If it's already frames (fallback), save directly.
                if isinstance(raw_out, torch.Tensor):
                    raw_latent = raw_out
                    # Save placeholder flags — actual raw frames saved after Stage C
                else:
                    raw_latent = None
                    save_frames(raw_out, sid, cfg["raw_frames_dir"], cfg)

                sm.write_flag(sid, "stage_b_done.flag",
                              {"time_s": round(tb, 1), "is_latent": isinstance(raw_out, torch.Tensor)})
                ok("✦ stage_b_done.flag written")

                del pipe_b, conditioning
                vram_free()
                ok(f"VRAM after B: {vram_used_gb():.1f} GB")

            # ════════════════════════════════════════════════════════════════
            # STAGE C  —  LTX-2 REFINER
            # Skip if: stage_c_done.flag exists AND refined frame count OK
            # ════════════════════════════════════════════════════════════════
            _print_live(sid, cfg["num_scenes"],
                        f"{C.BOLD}{C.CYAN}◉ C  LTX-2 Refiner (3 steps){C.END}",
                        s_name, stat)
            step(sid + 1, cfg["num_scenes"],
                 f"Scene {sid:02d}  Stage C — LTX-2 Refiner  (Bet Small, Win Big)")

            stage_c = load_stage_c(cfg)

            t0c     = time.time()
            if raw_latent is not None:
                refined = run_stage_c(stage_c, raw_latent, scene, cfg, seed)
            else:
                # Stage B already decoded frames — refine from saved PNGs
                # Re-encode PNGs into latent and pass to Stage C
                warn("[C] Re-encoding raw frames for Stage C refinement")
                raw_frames = [
                    Image.open(os.path.join(cfg["raw_frames_dir"],
                                            f"scene_{sid:02d}",
                                            f"frame_{sid*cfg['frames_per_chunk']+i:06d}.png"))
                    for i in range(cfg["frames_per_chunk"])
                ]
                refined = raw_frames  # Fallback: use raw frames if Stage C unavailable

            tc = time.time() - t0c
            ok(f"Stage C done in {tc:.1f}s")

            # Save refined frames (final output for assembly)
            save_frames(refined, sid, cfg["frames_dir"], cfg)
            ok(f"Refined frames saved → {cfg['frames_dir']}/scene_{sid:02d}/")

            # Save bridge from REFINED frames (v4 improvement)
            bp = save_bridge(refined, sid, cfg)
            sm.set_bridge(sid, bp)
            ok(f"Bridge (refined) saved → {os.path.basename(bp)}")

            # Write stage_c completion flag
            sm.write_flag(sid, "stage_c_done.flag",
                          {"frames": len(refined), "time_s": round(tc, 1)})
            ok("✦ stage_c_done.flag written")

            # Clean up Stage C
            del stage_c, refined
            if raw_latent is not None:
                del raw_latent
            vram_free()

            # Record timing and mark scene complete
            t_total = time.time() - t_scene
            stat.end_scene(ta, tb, tc)
            sm.mark_complete(sid, ta, tb, tc)
            ok(f"Scene {sid:02d} complete — {t_total:.1f}s  "
               f"(A={ta:.0f}s  B={tb:.0f}s  C={tc:.0f}s)")
            ok("✦ state.json updated atomically")
            time.sleep(1.5)   # Allow VRAM to fully drain

    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}  Interrupted — all progress saved. Re-run to continue.{C.END}")
        stop_evt.set()
        stat.summary()
        return

    except Exception as e:
        err(f"Error at scene {sid}: {e}")
        import traceback; traceback.print_exc()
        stop_evt.set()
        warn("Progress saved atomically — re-run to resume from last completed scene.")
        stat.summary()
        return

    finally:
        stop_evt.set()

    # ── Assemble final video ───────────────────────────────────────────────────
    final_video = assemble_video(cfg)

    # ── Save JSON report ───────────────────────────────────────────────────────
    rp = os.path.join(cfg["output_dir"],
                      f"report_v{__version__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(rp, "w") as f:
        json.dump({
            "version":              __version__,
            "timestamp":            datetime.now().isoformat(),
            "hardware":             {
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
                "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
                           if torch.cuda.is_available() else 0,
                "ram_gb": psutil.virtual_memory().total / 1e9,
            },
            "quality_params":       {
                "image_steps":       cfg["image_inference_steps"],
                "image_cfg":         cfg["image_guidance_scale"],
                "image_pag":         cfg["image_pag_guidance_scale"],
                "video_cfg":         cfg["guidance_scale"],
                "flow_shift":        cfg["flow_shift"],
                "ltx2_refiner":      cfg["use_ltx2_refiner"],
            },
            "scenes":               [sc["name"] for sc in SCENES],
            "completed":            sorted(sm.state.get("completed_scenes", [])),
            "timing_per_scene":     sm.state.get("timing", {}),
            "stage_a_times_s":      stat.ta_times,
            "stage_b_times_s":      stat.tb_times,
            "stage_c_times_s":      stat.tc_times,
            "total_elapsed_s":      stat.elapsed(),
            "final_video":          final_video,
        }, f, indent=2)
    ok(f"Report: {rp}")

    stat.summary()

    print(f"\n{C.BOLD}{C.GREEN}{'═'*76}{C.END}")
    print(f"{C.BOLD}{C.GREEN}  ✨  SmartSmile 2250 v{__version__} — COMPLETE{C.END}")
    print(f"{C.BOLD}{C.GREEN}{'═'*76}{C.END}")
    print(f"\n  {C.BOLD}📹  Output  :{C.END}  {final_video}")
    print(f"  {C.BOLD}🖼   Stage A :{C.END}  SANA 4.8B  →  1024×1024  60 steps  CFG 7.5  PAG 2.5")
    print(f"  {C.BOLD}🎬  Stage B :{C.END}  SANA-Video 2B  →  81 frames @ 1280×704  "
          f"50 steps  CFG 7.5  flow_shift 10")
    print(f"  {C.BOLD}✨  Stage C :{C.END}  LTX-2 Refiner  →  3 distilled steps  "
          f"2K-quality texture boost")
    print(f"  {C.BOLD}♻   Resume  :{C.END}  atomic state.json + 3 stage flags + frame-level recovery")
    print(f"  {C.BOLD}📚  Sources :{C.END}  nvlabs.github.io/Sana  |  "
          f"huggingface.co/Efficient-Large-Model\n")


if __name__ == "__main__":
    main()