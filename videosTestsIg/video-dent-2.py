"""
https://huggingface.co/Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers
https://studio.aifilms.ai/blog/sana-wm-nvidia-world-model
https://github.com/NVlabs
https://github.com/orgs/NVlabs/repositories
https://github.com/orgs/NVlabs/repositories?page=2
https://github.com/orgs/NVlabs/repositories?page=16
https://github.com/NVlabs/Sana

╔══════════════════════════════════════════════════════════════════════════════╗
║  SANA VIDEO — FUTURISTIC DENTAL OFFICE ADVERTISEMENT GENERATOR              ║
║  Version  : 2.0                                                              ║
║  Date     : 2026-05-23                                                       ║
║  Author   : Dr. Igor Lemos Alves                                             ║
║  Target   : 1-minute cinematic | 1280×704 | 16fps                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CHANGELOG                                                                   ║
║  v1.0  2026-05-23  Initial versioned release (SANA-Video 2B only).          ║
║  v2.0  2026-05-23  Two-stage 4.8B → 2B architecture.                        ║
║        NEW: Stage 1 — SANA 4.8B (SanaPipeline) generates a 1024px          ║
║             keyframe image for every scene with maximum quality.             ║
║        NEW: Stage 2 — SANA-Video 2B (SanaImageToVideoPipeline) animates     ║
║             each keyframe into 81 frames using I2V conditioning.             ║
║        NEW: Both pipelines swap in/out of VRAM sequentially so the          ║
║             16 GB budget is never exceeded (one model active at a time).    ║
║        NEW: Keyframe images saved to disk and reused on resume — Stage 1    ║
║             is skipped for scenes where the keyframe PNG already exists.    ║
║        NEW: Image-prompt tuned separately from video-prompt for each        ║
║             scene: image prompt maximises static visual quality; video      ║
║             prompt adds motion, camera, and temporal language.              ║
║        FIX: Carried forward all v1.0 fixes (hms staticmethod, etc.)        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  TWO-STAGE ARCHITECTURE                                                      ║
║                                                                              ║
║  STAGE 1  —  KEYFRAME GENERATION (per scene)                                ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  SanaPipeline (4.8B, bfloat16)                                      │    ║
║  │  Input : image_prompt (lighting, composition, subject, mood)        │    ║
║  │  Output: 1024×1024 PNG keyframe  (max SANA quality)                 │    ║
║  │  VRAM  : ~12 GB peak → offloaded to RAM after stage                 │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║           ↓  keyframe PNG                                                    ║
║  STAGE 2  —  VIDEO ANIMATION (per scene)                                    ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  SanaImageToVideoPipeline (2B, bfloat16)                            │    ║
║  │  Input : keyframe PNG + video_prompt (motion, camera, duration)     │    ║
║  │  Output: 81 frames @ 1280×704  (~5 seconds at 16 fps)               │    ║
║  │  VRAM  : ~10 GB peak → cleared after each scene                     │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║           ↓  bridge PNG (last frame)                                         ║
║  CONTEXT BRIDGE (scene N+1)                                                  ║
║    The last frame of scene N is fed back into Stage 2 of scene N+1          ║
║    as an additional I2V conditioning signal alongside the Stage 1            ║
║    keyframe, weighted 70%/30% (keyframe/bridge blend).                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  VRAM BUDGET  (RTX 5080 16 GB)                                               ║
║  Stage 1 active  : 4.8B (bfloat16) ≈ 9.6 GB + VAE fp32 ≈ 1.5 GB = ~11 GB  ║
║  Stage 2 active  : 2B  (bfloat16) ≈ 4.0 GB + VAE fp32 ≈ 1.5 GB = ~5.5 GB  ║
║  Never both active simultaneously — del + gc.collect() between stages.      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SANA PROMPT ANATOMY (NVIDIA official)                                       ║
║  IMAGE: [LIGHTING] [SHOT TYPE] [COMPOSITION] [SUBJECT] [ENVIRONMENT] [MOOD] ║
║  VIDEO: image_prompt + [CAMERA MOTION] [TEMPORAL FLOW] + " motion score: N" ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  INSTALL                                                                     ║
║  pip install git+https://github.com/huggingface/diffusers                   ║
║  pip install torch imageio[ffmpeg] psutil GPUtil Pillow numpy                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

__version__ = "2.0"
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

# ── optional GPU monitoring ──────────────────────────────────────────────────
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

# ── diffusers imports ─────────────────────────────────────────────────────────
try:
    from diffusers import SanaPipeline                  # Stage 1 — 4.8B image
    from diffusers import SanaImageToVideoPipeline      # Stage 2 — 2B I2V video
    from diffusers.utils import load_image
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
    MAGENTA= '\033[95m'
    BOLD   = '\033[1m'
    DIM    = '\033[2m'
    END    = '\033[0m'

def header(text):     print(f"\n{C.BOLD}{C.CYAN}{'═'*72}\n  {text}\n{'═'*72}{C.END}")
def subheader(text):  print(f"\n{C.BOLD}{C.MAGENTA}  ▶  {text}{C.END}")
def ok(text):         print(f"  {C.GREEN}✔{C.END}  {text}")
def warn(text):       print(f"  {C.YELLOW}⚠{C.END}  {text}")
def err(text):        print(f"  {C.RED}✘{C.END}  {text}")
def info(text):       print(f"  {C.BLUE}ℹ{C.END}  {text}")
def step(n, t, text): print(f"\n{C.BOLD}{C.BLUE}[{n}/{t}]{C.END} {text}")

def bar(frac, w=44):
    f = int(w * max(0.0, min(1.0, frac)))
    return f"{C.GREEN}{'█'*f}{C.END}{'░'*(w-f)}"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  OFFICIAL NEGATIVE PROMPT (NVIDIA verbatim)                             ║
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
#
#  Each scene has TWO prompts:
#
#  image_prompt  → drives Stage 1 (4.8B SanaPipeline)
#                  Anatomy: LIGHTING · SHOT TYPE · COMPOSITION ·
#                           SUBJECT+DETAIL · ENVIRONMENT · MOOD
#                  Goal: maximum photorealism, cinematic still frame
#
#  video_prompt  → drives Stage 2 (2B SanaImageToVideoPipeline)
#                  Anatomy: image_prompt condensed + CAMERA MOTION ·
#                           TEMPORAL FLOW + " motion score: N."
#                  Goal: describe how the scene MOVES from the keyframe
#
# ─────────────────────────────────────────────────────────────────────────

SCENES = [
    {
        "id":    0,
        "name":  "Opening — City of the Future",
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
            "Warm utopian color grading, cinematic anamorphic lens flare. "
            "motion score: 20."
        ),
    },
    {
        "id":    1,
        "name":  "Clinic Exterior — SmartSmile 2250",
        "beats": "Reveal the futuristic dental clinic facade",
        "motion_score": 15,

        "image_prompt": (
            "Soft diffused morning daylight, medium architectural wide shot, perfect symmetry. "
            "A breathtaking dental clinic facade: pure white biopolymer panels and floor-to-ceiling "
            "blue-tinted smart glass. A holographic logo 'SmartSmile 2250' floats above the entrance "
            "in aquamarine light. Bioluminescent sakura trees line the approach path. "
            "Patients in elegant minimalist attire walk through the garden. "
            "Photorealistic, architectural photography, 8K, pristine clinical aesthetic."
        ),
        "video_prompt": (
            "Soft diffused daylight, medium wide shot, symmetrical architectural composition. "
            "The SmartSmile 2250 clinic facade of white biopolymer and smart glass, "
            "holographic logo rotating gently above the entrance. "
            "Bioluminescent trees sway softly in a light breeze. Patients approach the entrance. "
            "Camera slowly pushes forward toward the entrance doors. "
            "Cool aquamarine and white color palette, pristine and welcoming. "
            "motion score: 15."
        ),
    },
    {
        "id":    2,
        "name":  "Reception — AI Concierge",
        "beats": "Patient checks in with holographic AI receptionist",
        "motion_score": 18,

        "image_prompt": (
            "Cool blue ambient light with warm amber accent panels, medium close-up, rule-of-thirds. "
            "An elegant young woman stands at a floating glass reception desk. "
            "Across from her, a translucent holographic AI avatar in humanoid form smiles with warmth. "
            "Soft neon medical data streams float in the air. Staff in crisp white uniforms visible behind. "
            "The space is luminous, clinical, and welcoming. "
            "Photorealistic, cinematic portrait lighting, 8K, ultra-detailed."
        ),
        "video_prompt": (
            "Cool blue ambient light with warm accent panels, medium close-up, rule-of-thirds. "
            "Patient interacts with the translucent holographic AI concierge avatar that smiles and gestures. "
            "Soft neon medical data floats and pulses around the room. Staff move gracefully in background. "
            "Camera holds steady with a slow gentle push-in toward the patient. "
            "Clinical yet warm color grading, ultra-clean whites and soft blues. "
            "motion score: 18."
        ),
    },
    {
        "id":    3,
        "name":  "Diagnostics — AI Oral Scan",
        "beats": "Real-time AI full-mouth diagnostic scanning",
        "motion_score": 22,

        "image_prompt": (
            "Soft cool surgical lighting, close-up macro shot, centered clinical composition. "
            "A state-of-the-art AI oral scanner: a sleek brushed-titanium device hovering beside "
            "an open mouth. A real-time 3D holographic dental model hovers in midair, color-coded "
            "by health status — green for healthy, amber for attention, blue for scan in progress. "
            "A dentist in smart-glass AR eyewear reviews the hologram with a haptic stylus. "
            "Photorealistic, medical precision, 8K, high-tech blue-white aesthetic."
        ),
        "video_prompt": (
            "Soft cool clinical lighting, close-up macro shot, centered composition. "
            "AI oral scanner glides smoothly around the patient's mouth, projecting real-time "
            "holographic 3D dental models that pulse with diagnostic color data. "
            "Dentist's stylus moves across the hologram making annotations. "
            "Camera slowly orbits the scanning device from one side to the other. "
            "High-tech blue and white tones, precise and futuristic. "
            "motion score: 22."
        ),
    },
    {
        "id":    4,
        "name":  "Treatment — Nano-Robot Procedure",
        "beats": "Nano-robot swarm performs painless precision dental work",
        "motion_score": 25,

        "image_prompt": (
            "Diffused surgical lighting, medium shot, single-subject centered framing. "
            "A patient reclines in a floating white ergonomic chair, expression completely peaceful. "
            "A luminous cloud of nano-robots — microscopic, each glowing a soft electric blue — "
            "forms a precise swarm near the patient's mouth. An AR screen beside the chair displays "
            "nano-scale real-time mapping. Attending dentist observes calmly in background. "
            "Photorealistic, cinematic, 8K, serene wonder aesthetic."
        ),
        "video_prompt": (
            "Diffused surgical lighting, medium shot, clean single-subject framing. "
            "Patient reclines completely relaxed, eyes closed peacefully. "
            "Luminous nano-robot swarm drifts gracefully, each tiny mote pulsing blue as it works. "
            "AR screen beside the chair displays live nano-mapping data. "
            "Camera gently drifts from wide to close on the swarm. "
            "Serene clinical blue-white palette, wonder and precision. "
            "motion score: 25."
        ),
    },
    {
        "id":    5,
        "name":  "Bioprinting — Tooth Regeneration",
        "beats": "Living tooth bioprinted in real time",
        "motion_score": 18,

        "image_prompt": (
            "Warm amber lab lighting, extreme close-up, centered macro composition. "
            "A cutting-edge dental bioprinter head positioned above a growing tooth crown. "
            "The tooth is mid-print: translucent organic layers already formed, top still "
            "being deposited in fine bio-ink filaments. Stem cells glow faintly within the "
            "living tissue. A luminous nano-scaffold surrounds the structure like a cage of light. "
            "Photorealistic, scientific macro photography, 8K, amber and ivory tones."
        ),
        "video_prompt": (
            "Warm amber lab lighting, extreme close-up, centered macro composition. "
            "Dental bioprinter deposits living tissue layer by layer, tooth crown growing upward. "
            "Stem cells activate with a faint inner glow. Nano-scaffold pulses softly around the structure. "
            "Camera holds static in macro focus, shallow depth of field. "
            "Rich amber and ivory tones, scientific wonder. "
            "motion score: 18."
        ),
    },
    {
        "id":    6,
        "name":  "AI Dentist — Human + AI Collaboration",
        "beats": "Human dentist and AI assistant reviewing a 3D jaw hologram",
        "motion_score": 16,

        "image_prompt": (
            "Soft split warm-cool lighting, medium two-shot, balanced frame composition. "
            "A skilled human dentist in a crisp white coat stands beside a sleek humanoid "
            "AI dental assistant — translucent torso glowing with embedded data circuits, "
            "calm and professional. Both look toward a rotating 3D holographic jaw model "
            "floating between them. The human points with a haptic stylus; the AI overlays "
            "annotations in real time. "
            "Photorealistic, cinematic, 8K, warm wood and cool steel aesthetic."
        ),
        "video_prompt": (
            "Soft split warm-cool lighting, medium two-shot, balanced composition. "
            "Human dentist and AI dental assistant discuss the rotating 3D holographic jaw model, "
            "the AI overlaying annotations as the dentist gestures. "
            "The jaw hologram rotates slowly, pulsing data highlights. "
            "Camera slowly arcs from profile to three-quarter frontal. "
            "Warm wood and cool steel clinic aesthetic, expertise and trust. "
            "motion score: 16."
        ),
    },
    {
        "id":    7,
        "name":  "Patient Experience — Zero Pain",
        "beats": "Patient in total comfort during pain-free neural treatment",
        "motion_score": 10,

        "image_prompt": (
            "Golden warm ambient light, medium close-up portrait, slightly low angle. "
            "A patient reclined in a floating chair, eyes closed, gentle smile on their face. "
            "A delicate neural interface headband emits soft geometric light patterns above their brow. "
            "The room walls behind display a serene underwater coral scene. "
            "Complete peace, zero tension. "
            "Photorealistic, cinematic beauty lighting, 8K, warm golden hour tones."
        ),
        "video_prompt": (
            "Golden soft ambient light, medium close-up, slightly low angle, warm framing. "
            "Patient in floating chair, completely at ease, gentle smile. "
            "Neural interface headband pulses soft geometric patterns above their brow. "
            "Room walls display gentle animated underwater coral scene. "
            "Camera slowly pushes into the patient's serene face. "
            "Warm golden tones, complete comfort and peace. "
            "motion score: 10."
        ),
    },
    {
        "id":    8,
        "name":  "Results — The Perfect Smile",
        "beats": "Patient sees their transformed smile for the first time",
        "motion_score": 14,

        "image_prompt": (
            "Warm flattering beauty lighting, medium close-up, centered portrait composition. "
            "A patient holding up a sleek smart mirror, eyes just beginning to widen with joy. "
            "In the mirror: a radiant, perfectly aligned, brilliantly white smile. "
            "The mirror frame displays a soft green health score: 98/100. "
            "The patient's face glows with happiness. "
            "Photorealistic, cinematic beauty photography, 8K, warm bright palette, pure joy."
        ),
        "video_prompt": (
            "Warm flattering beauty lighting, medium close-up, centered portrait. "
            "Patient raises the smart mirror and sees their radiant new smile. "
            "Expression shifts from neutral to pure joy, eyes widening, breaking into a genuine laugh. "
            "Mirror overlays soft green health score digits. "
            "Camera slowly pushes into the smile. "
            "Warm bright tones, joy and confidence. "
            "motion score: 14."
        ),
    },
    {
        "id":    9,
        "name":  "Community — Dental Health for All",
        "beats": "Multicultural crowd in the SmartSmile 2250 park",
        "motion_score": 22,

        "image_prompt": (
            "Bright warm daylight, wide ensemble shot, dynamic diagonal group composition. "
            "A joyful multicultural crowd — children, elders, young adults, diverse backgrounds — "
            "walking through the SmartSmile 2250 public health park. Everyone smiling brilliantly. "
            "Holographic dental-health kiosks glow along the path. Bioluminescent trees line the way. "
            "A SmartSmile drone hovers above distributing care packages. "
            "Photorealistic, cinematic, 8K, vibrant warm daylight, optimism and inclusivity."
        ),
        "video_prompt": (
            "Bright even daylight, wide ensemble shot, dynamic group composition. "
            "Joyful multicultural crowd walks through the SmartSmile 2250 park, all smiling brilliantly. "
            "Holographic kiosks pulse with health data. Bioluminescent trees sway. Drone hovers above. "
            "Camera tracks alongside the crowd in a smooth lateral dolly. "
            "Vibrant warm-daylight palette, optimism and inclusivity. "
            "motion score: 22."
        ),
    },
    {
        "id":    10,
        "name":  "Data — Global Oral Health Dashboard",
        "beats": "Planetary dental health data visualization",
        "motion_score": 16,

        "image_prompt": (
            "Deep blue ambient light, wide shot, spherical holographic globe centerpiece. "
            "A vast spherical holographic Earth rotates in the center of a high-tech control room. "
            "Thousands of glowing SmartSmile clinic markers across every continent. "
            "Curved data panels surrounding the globe show global oral health scores: "
            "graphs, charts, numbers all trending upward toward 100%. "
            "Scientists in white observe from the floor below. "
            "Photorealistic, cinematic, 8K, deep navy, emerald, and white data-visualization tones."
        ),
        "video_prompt": (
            "Deep blue ambient light, wide shot, global holographic map composition. "
            "Spherical holographic Earth rotates slowly, SmartSmile clinic markers glowing worldwide. "
            "Data panels show real-time global oral health scores climbing toward 100 percent. "
            "Scientists move across the floor. Globe slowly brightens as scores rise. "
            "Camera slowly tracks backward to reveal the full control room. "
            "Deep navy, emerald, and white tones, epic scale. "
            "motion score: 16."
        ),
    },
    {
        "id":    11,
        "name":  "Closing — Logo and Tagline",
        "beats": "Brand reveal with iconic logo and closing message",
        "motion_score": 8,

        "image_prompt": (
            "Soft warm white studio light, ultra-wide centered shot, pure minimalism. "
            "A single tooth carved from luminous crystal floats centered against a pure white background. "
            "The SmartSmile 2250 logo glows below it in polished gold lettering. "
            "Beneath the logo, elegant sans-serif text: 'Every smile. Every human. Every future.' "
            "The tooth casts a soft prismatic light spectrum across the white surface. "
            "Photorealistic, product photography, 8K, pure white and gold, timeless and iconic."
        ),
        "video_prompt": (
            "Soft warm white studio light, ultra-wide clean shot, perfectly centered. "
            "Crystal tooth floats and rotates slowly against pure white background. "
            "SmartSmile 2250 logo pulses gently below. Tagline text fades in softly. "
            "Prismatic light spectrum drifts across the surface. "
            "Camera holds completely static, elements breathe gently. "
            "Pure white and gold palette, timeless and iconic. "
            "motion score: 8."
        ),
    },
]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_config():
    """
    Two-stage pipeline configuration.

    Stage 1  SANA 4.8B → 1024×1024 keyframe PNG (maximum image quality)
    Stage 2  SANA-Video 2B → 1280×704 animated 81-frame chunk (I2V from keyframe)

    VRAM budget strategy (RTX 5080 16 GB):
      Stage 1 loaded  →  Stage 1 del + gc.collect() + empty_cache()
      Stage 2 loaded  →  Stage 2 del + gc.collect() + empty_cache()
      Never both pipelines in VRAM simultaneously.
    """
    cfg = {
        # ── Stage 1: Image model ─────────────────────────────────────────
        "model_image": "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
        "keyframe_height": 1024,
        "keyframe_width":  1024,
        "image_guidance_scale":      5.0,   # SANA-1.5 recommended (4.5–6.0)
        "image_inference_steps":     30,    # 18 = fast, 30 = quality, 50 = max
        "image_pag_guidance_scale":  2.0,   # Perturbed Attention Guidance (PAG)

        # ── Stage 2: Video model ─────────────────────────────────────────
        "model_video": "Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
        "height":           704,
        "width":            1280,
        "fps":              16,
        "frames_per_chunk": 81,             # SANA-Video native window
        "num_scenes":       len(SCENES),

        # ── Video quality params (RTX 5080 optimized) ────────────────────
        "guidance_scale":      6.0,
        "num_inference_steps": 50,
        "flow_shift":          8,
        "torch_dtype":         torch.bfloat16,
        "seed":                2025,

        # ── VRAM management ──────────────────────────────────────────────
        "cpu_offload":       True,
        "attention_slicing": True,
        "vae_dtype":         torch.float32,  # Always keep VAE in fp32

        # ── Context bridge ───────────────────────────────────────────────
        # The keyframe (Stage 1) is the PRIMARY conditioning image for each scene.
        # The bridge (last frame of prior scene) is blended in as SECONDARY context.
        # This maintains both maximum quality AND inter-scene continuity.
        "use_keyframe_conditioning": True,
        "use_bridge_blending":       True,
        "bridge_blend_alpha":        0.30,  # 30% bridge / 70% keyframe

        # ── Storage ──────────────────────────────────────────────────────
        "output_dir":       "dental_ad_v2_output",
        "keyframes_dir":    "dental_ad_v2_output/keyframes",
        "frames_dir":       "dental_ad_v2_output/frames",
        "bridges_dir":      "dental_ad_v2_output/bridges",
        "checkpoints_dir":  "dental_ad_v2_output/checkpoints",
        "final_video":      "SmartSmile2250_Ad_v2_1min_720p.mp4",

        # ── Encoding ─────────────────────────────────────────────────────
        "crf":    18,
        "preset": "slow",
    }

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
        self.stage1_times = []
        self.stage2_times = []
        self.chunk_times  = []
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

    def end_chunk(self, elapsed, s1_time, s2_time):
        self.chunk_times.append(elapsed)
        self.stage1_times.append(s1_time)
        self.stage2_times.append(s2_time)
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
        """Format seconds → human string. @staticmethod: call as hms(s) or self.hms(s)."""
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
        avg  = sum(self.chunk_times) / done if done else 0
        avg1 = sum(self.stage1_times) / len(self.stage1_times) if self.stage1_times else 0
        avg2 = sum(self.stage2_times) / len(self.stage2_times) if self.stage2_times else 0
        fps  = (done * self.cfg["frames_per_chunk"]) / el if el > 0 else 0

        print(f"\n{C.BOLD}{C.CYAN}{'═'*72}{C.END}")
        print(f"{C.BOLD}{C.CYAN}  📊  GENERATION STATISTICS  —  v{__version__}{C.END}")
        print(f"{C.BOLD}{C.CYAN}{'═'*72}{C.END}")
        print(f"  {'Total time':<35} {self.hms(el)}")
        print(f"  {'Scenes completed':<35} {done}/{self.cfg['num_chunks']}")
        print(f"  {'Avg total time / scene':<35} {avg:.1f}s")
        print(f"  {'Avg Stage 1 (4.8B keyframe)':<35} {avg1:.1f}s")
        print(f"  {'Avg Stage 2 (2B video)':<35} {avg2:.1f}s")
        print(f"  {'Generation speed':<35} {fps:.2f} frames/s")
        g_avg = sum(self.gpu_hist)  / len(self.gpu_hist)  if self.gpu_hist  else 0
        v_avg = sum(self.vram_hist) / len(self.vram_hist) if self.vram_hist else 0
        r_avg = sum(self.ram_hist)  / len(self.ram_hist)  if self.ram_hist  else 0
        print(f"  {'Avg GPU load':<35} {g_avg:.1f}%")
        print(f"  {'Peak GPU load':<35} {max(self.chunk_gpu_peaks, default=0):.1f}%")
        print(f"  {'Avg VRAM used':<35} {v_avg / 100 * 16:.1f} GB / 16 GB")
        print(f"  {'Avg RAM used':<35} {r_avg:.1f}%")
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

    stage_color = C.YELLOW if stage == 1 else C.GREEN
    stage_label = (f"{C.BOLD}{stage_color}◉ STAGE 1  4.8B Keyframe{C.END}"
                   if stage == 1
                   else f"{C.BOLD}{stage_color}◉ STAGE 2  2B Video{C.END}")

    print(f"{C.BOLD}{C.CYAN}{'═'*72}{C.END}")
    print(f"{C.BOLD}{C.CYAN}  🦷  SmartSmile 2250  v{__version__}  —  Two-Stage 4.8B + 2B{C.END}")
    print(f"{C.BOLD}{C.CYAN}{'═'*72}{C.END}")
    print(f"\n  {'Scene':<20} {scene_idx}/{num_scenes}  —  {scene_name}")
    print(f"  {'Active stage':<20} {stage_label}")
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

    # Per-scene timing table if data available
    if stat.stage1_times:
        avg1 = sum(stat.stage1_times) / len(stat.stage1_times)
        avg2 = sum(stat.stage2_times) / len(stat.stage2_times) if stat.stage2_times else 0
        print(f"\n  {C.DIM}Avg Stage 1 (keyframe): {avg1:.0f}s  "
              f"|  Avg Stage 2 (video): {avg2:.0f}s{C.END}")
    print(f"\n{C.BOLD}{C.CYAN}{'─'*72}{C.END}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CHECKPOINT HELPERS                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_state(cfg):
    path = os.path.join(cfg["checkpoints_dir"], "state.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed_scenes": [], "keyframe_paths": {}, "bridge_frames": {}}


def save_state(cfg, state):
    os.makedirs(cfg["checkpoints_dir"], exist_ok=True)
    with open(os.path.join(cfg["checkpoints_dir"], "state.json"), "w") as f:
        json.dump(state, f, indent=2)


def save_keyframe(image, scene_id, cfg):
    """Save 4.8B-generated keyframe image to disk."""
    os.makedirs(cfg["keyframes_dir"], exist_ok=True)
    path = os.path.join(cfg["keyframes_dir"], f"keyframe_scene_{scene_id:02d}.png")
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image, dtype=np.uint8))
    image.save(path, "PNG")
    ok(f"Keyframe saved → {os.path.basename(path)}  ({image.size[0]}×{image.size[1]})")
    return path


def blend_images(img_primary, img_secondary, alpha_secondary, target_size):
    """
    Blend two PIL images: result = (1-alpha)*primary + alpha*secondary.
    Used to mix the 4.8B keyframe (primary) with the bridge frame (secondary).
    """
    if img_primary.size != target_size:
        img_primary   = img_primary.resize(target_size, Image.LANCZOS)
    if img_secondary.size != target_size:
        img_secondary = img_secondary.resize(target_size, Image.LANCZOS)
    primary_arr   = np.array(img_primary,   dtype=np.float32)
    secondary_arr = np.array(img_secondary, dtype=np.float32)
    blended = ((1.0 - alpha_secondary) * primary_arr
               + alpha_secondary * secondary_arr).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def save_bridge_frame(frames, scene_id, cfg):
    """Save the last frame of a scene as the bridge image for the next scene."""
    os.makedirs(cfg["bridges_dir"], exist_ok=True)
    last = frames[-1]
    if not isinstance(last, Image.Image):
        last = Image.fromarray(np.array(last, dtype=np.uint8))
    path = os.path.join(cfg["bridges_dir"], f"bridge_after_scene_{scene_id:02d}.png")
    last.save(path, "PNG")
    return path


def save_scene_frames(frames, scene_id, cfg):
    """Save all video frames for a scene as lossless PNGs."""
    scene_dir  = os.path.join(cfg["frames_dir"], f"scene_{scene_id:02d}")
    os.makedirs(scene_dir, exist_ok=True)
    base_frame = scene_id * cfg["frames_per_chunk"]
    paths = []
    for i, frame in enumerate(frames):
        img  = frame if isinstance(frame, Image.Image) \
               else Image.fromarray(np.array(frame, dtype=np.uint8))
        path = os.path.join(scene_dir, f"frame_{base_frame + i:06d}.png")
        img.save(path, "PNG")
        paths.append(path)
    return paths


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
# ║  STAGE 1 — KEYFRAME GENERATION (SANA 4.8B)                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_image_pipeline(cfg):
    """
    Load SANA 4.8B image pipeline.
    Must be fully deleted and VRAM cleared before loading Stage 2.
    """
    subheader("Loading Stage 1 — SANA 4.8B Image Pipeline")
    info(f"Model: {cfg['model_image']}")
    info(f"Expected VRAM: ~11 GB peak (bfloat16 + fp32 VAE)")

    pipe = SanaPipeline.from_pretrained(
        cfg["model_image"],
        torch_dtype=cfg["torch_dtype"],
    )
    # VAE stays float32 for stability — critical
    pipe.vae.to(cfg["vae_dtype"])
    pipe.text_encoder.to(cfg["torch_dtype"])
    pipe.to("cuda")
    if cfg["cpu_offload"]:
        pipe.enable_model_cpu_offload()
    torch.backends.cuda.matmul.allow_tf32 = True
    ok(f"Stage 1 pipeline ready  (VRAM used: {vram_used_gb():.1f} GB)")
    return pipe


def generate_keyframe(pipe_img, scene, cfg, seed):
    """
    Stage 1: generate a high-quality 1024×1024 keyframe with SANA 4.8B.

    The image_prompt is crafted for maximum photorealism:
    LIGHTING · SHOT · COMPOSITION · SUBJECT DETAIL · ENVIRONMENT · MOOD
    No motion language — this is a cinematic still frame.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    prompt    = scene["image_prompt"]

    info(f"[S1] Prompt excerpt: …{prompt[50:110]}…")
    info(f"[S1] Size: {cfg['keyframe_width']}×{cfg['keyframe_height']}  "
         f"|  Steps: {cfg['image_inference_steps']}  "
         f"|  CFG: {cfg['image_guidance_scale']}  "
         f"|  PAG: {cfg.get('image_pag_guidance_scale', 2.0)}")

    # SanaPipeline supports PAG (Perturbed Attention Guidance) for quality boost
    try:
        result = pipe_img(
            prompt=prompt,
            negative_prompt=NEGATIVE_IMAGE,
            height=cfg["keyframe_height"],
            width=cfg["keyframe_width"],
            guidance_scale=cfg["image_guidance_scale"],
            pag_guidance_scale=cfg.get("image_pag_guidance_scale", 2.0),
            num_inference_steps=cfg["image_inference_steps"],
            generator=generator,
        )
    except TypeError:
        # Older diffusers build may not support pag_guidance_scale
        warn("[S1] PAG not supported in this diffusers build — running without PAG")
        result = pipe_img(
            prompt=prompt,
            negative_prompt=NEGATIVE_IMAGE,
            height=cfg["keyframe_height"],
            width=cfg["keyframe_width"],
            guidance_scale=cfg["image_guidance_scale"],
            num_inference_steps=cfg["image_inference_steps"],
            generator=generator,
        )

    image = result.images[0]
    ok(f"[S1] Keyframe generated  ({image.size[0]}×{image.size[1]})")
    return image


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 2 — VIDEO ANIMATION (SANA-Video 2B)                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_video_pipeline(cfg):
    """
    Load SANA-Video 2B I2V pipeline.
    Stage 1 must be deleted and VRAM cleared before calling this.
    """
    subheader("Loading Stage 2 — SANA-Video 2B I2V Pipeline")
    info(f"Model: {cfg['model_video']}")
    info(f"Expected VRAM: ~5.5 GB peak (bfloat16 + fp32 VAE)")

    pipe = SanaImageToVideoPipeline.from_pretrained(
        cfg["model_video"],
        torch_dtype=cfg["torch_dtype"],
    )
    pipe.vae.to(cfg["vae_dtype"])
    pipe.text_encoder.to(cfg["torch_dtype"])
    pipe.to("cuda")
    if cfg["cpu_offload"]:
        pipe.enable_model_cpu_offload()
    if cfg["attention_slicing"]:
        pipe.enable_attention_slicing()
    ok(f"Stage 2 pipeline ready  (VRAM used: {vram_used_gb():.1f} GB)")
    return pipe


def prepare_conditioning_image(keyframe_path, bridge_path, scene_id, cfg):
    """
    Build the conditioning image for Stage 2.

    Strategy:
      Scene 0          → keyframe only (no bridge exists yet)
      Scene N (N > 0)  → blend(keyframe 70%, bridge 30%)
                         The keyframe drives visual quality and scene identity.
                         The bridge drives temporal continuity from the prior scene.

    Both images are resized to match the video generation resolution (1280×704)
    before blending, because the I2V pipeline expects the conditioning image
    at the same resolution as the output frames.
    """
    target = (cfg["width"], cfg["height"])
    keyframe = Image.open(keyframe_path).resize(target, Image.LANCZOS)

    if (scene_id > 0
            and cfg.get("use_bridge_blending")
            and bridge_path
            and os.path.exists(bridge_path)):
        bridge = Image.open(bridge_path).resize(target, Image.LANCZOS)
        alpha  = cfg.get("bridge_blend_alpha", 0.30)
        conditioning = blend_images(keyframe, bridge, alpha, target)
        info(f"[S2] Conditioning = keyframe {(1-alpha)*100:.0f}% + bridge {alpha*100:.0f}%")
    else:
        conditioning = keyframe
        info(f"[S2] Conditioning = keyframe only (scene 0 or no bridge)")

    return conditioning


def generate_video_chunk(pipe_vid, scene, conditioning_img, cfg, seed):
    """
    Stage 2: animate the conditioning image into 81 frames using SANA-Video 2B.

    The video_prompt adds temporal language (camera motion, movement description)
    on top of the scene content described in the image_prompt.
    The motion score token is embedded in every video_prompt.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    prompt    = scene["video_prompt"]

    info(f"[S2] Prompt excerpt: …{prompt[50:110]}…")
    info(f"[S2] motion score: {scene['motion_score']}  "
         f"|  Steps: {cfg['num_inference_steps']}  "
         f"|  CFG: {cfg['guidance_scale']}  "
         f"|  Seed: {seed}")

    result = pipe_vid(
        image=conditioning_img,
        prompt=prompt,
        negative_prompt=NEGATIVE_VIDEO,
        height=cfg["height"],
        width=cfg["width"],
        frames=cfg["frames_per_chunk"],
        guidance_scale=cfg["guidance_scale"],
        num_inference_steps=cfg["num_inference_steps"],
        generator=generator,
    )
    frames = result.frames[0]
    ok(f"[S2] Video chunk generated  ({len(frames)} frames)")
    return frames


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VIDEO ASSEMBLY                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def assemble_video(cfg):
    header("🎬  Assembling final video")
    final_path = os.path.join(cfg["output_dir"], cfg["final_video"])

    all_frames = []
    for s in range(cfg["num_scenes"]):
        scene_dir = os.path.join(cfg["frames_dir"], f"scene_{s:02d}")
        if not os.path.exists(scene_dir):
            warn(f"Scene {s} frames missing — skipping")
            continue
        files = sorted([os.path.join(scene_dir, f) for f in os.listdir(scene_dir)
                        if f.endswith((".png", ".jpg"))])
        all_frames.extend(files)

    if not all_frames:
        err("No frames found. Cannot assemble.")
        return None

    ok(f"Found {len(all_frames)} frames — encoding CRF {cfg['crf']}, preset {cfg['preset']}")

    writer = imageio.get_writer(
        final_path, fps=cfg["fps"], codec="libx264", quality=None,
        output_params=[
            "-crf",       str(cfg["crf"]),
            "-preset",    cfg["preset"],
            "-pix_fmt",   "yuv420p",
            "-profile:v", "high",
            "-level",     "4.2",
            "-movflags",  "+faststart",
        ],
    )
    last_update = time.time()
    for i, fp in enumerate(all_frames):
        try:
            writer.append_data(imageio.imread(fp))
        except Exception as e:
            warn(f"Skipping corrupt frame: {e}")
        if time.time() - last_update > 3:
            pct = (i + 1) / len(all_frames)
            print(f"    [{bar(pct, 30)}]  {pct*100:.0f}%  {i+1}/{len(all_frames)}")
            last_update = time.time()
    writer.close()

    if os.path.exists(final_path):
        size_mb = os.path.getsize(final_path) / 1e6
        dur     = len(all_frames) / cfg["fps"]
        mbps    = (os.path.getsize(final_path) * 8) / dur / 1e6
        ok(f"Video: {final_path}")
        ok(f"Size: {size_mb:.1f} MB  |  Duration: {dur:.1f}s  |  Bitrate: {mbps:.1f} Mbps")
        return final_path
    else:
        err("Assembly failed.")
        return None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PREFLIGHT                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def preflight(cfg):
    header("🔍  Preflight checks")

    if not DIFFUSERS_OK:
        err("diffusers not installed.")
        err("Run: pip install git+https://github.com/huggingface/diffusers")
        return False

    if not torch.cuda.is_available():
        err("CUDA not available.")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    ok(f"GPU    : {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    ram  = psutil.virtual_memory()
    disk = shutil.disk_usage(os.path.abspath("."))
    ok(f"RAM    : {ram.total/1e9:.1f} GB total  |  {ram.available/1e9:.1f} GB free")
    ok(f"Disk   : {disk.free/1e9:.1f} GB free")
    ok(f"Torch  : {torch.__version__}  |  bfloat16 ready")

    if vram_gb < 14:
        warn(f"VRAM is {vram_gb:.1f} GB — Stage 1 (4.8B) needs ~11 GB. "
             "Consider enabling cpu_offload.")

    est_kf_gb  = cfg["num_scenes"] * 4 / 1024         # ~4 MB per 1024px PNG
    est_vid_gb = cfg["total_frames"] * 2.5 / 1024     # ~2.5 MB per 720p PNG
    est_total  = est_kf_gb + est_vid_gb

    print(f"\n  {C.BOLD}Two-Stage Pipeline:{C.END}")
    print(f"    Stage 1   : SANA 4.8B → {cfg['keyframe_width']}×{cfg['keyframe_height']} "
          f"keyframe per scene  ({cfg['image_inference_steps']} steps)")
    print(f"    Stage 2   : SANA-Video 2B → {cfg['width']}×{cfg['height']} "
          f"{cfg['frames_per_chunk']} frames per scene  ({cfg['num_inference_steps']} steps)")
    print(f"    Scenes    : {cfg['num_scenes']}  ×  5.1s  =  "
          f"~{cfg['total_duration_seconds']:.0f}s total")
    print(f"    Est. disk : ~{est_total:.1f} GB  "
          f"(keyframes {est_kf_gb:.1f} GB + frames {est_vid_gb:.1f} GB)")
    print(f"    Bridge    : keyframe 70% + prior-scene-last-frame 30%  (blended)")

    # Stage 1: ~60s per scene on RTX 5080 @ 30 steps
    # Stage 2: ~36s per scene on RTX 5080 @ 50 steps
    est_s = cfg["num_chunks"] * (60 + 36)
    print(f"\n    Est. generation time : {StatisticsTracker.hms(est_s)} "
          f"(~96s/scene on RTX 5080)")

    if disk.free / 1e9 < est_total * 1.5:
        warn(f"Low disk space — need ~{est_total*1.5:.0f} GB, have {disk.free/1e9:.1f} GB")

    return True


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    global _tracker_ref

    print(f"{C.BOLD}{C.CYAN}  sana-dental-2.py  v{__version__}  —  {__date__}{C.END}")
    print(f"{C.DIM}  Two-Stage 4.8B Keyframe + 2B Video  |  SmartSmile 2250{C.END}\n")

    cfg          = get_config()
    stat         = StatisticsTracker(cfg)
    _tracker_ref = stat

    if not preflight(cfg):
        return

    for d in [cfg["output_dir"], cfg["keyframes_dir"], cfg["frames_dir"],
              cfg["bridges_dir"], cfg["checkpoints_dir"]]:
        os.makedirs(d, exist_ok=True)

    state        = load_state(cfg)
    completed    = set(state.get("completed_scenes", []))
    keyframe_map = state.get("keyframe_paths", {})
    bridge_map   = state.get("bridge_frames",  {})

    if completed:
        ok(f"Resuming — {len(completed)}/{cfg['num_scenes']} scenes done: {sorted(completed)}")
    else:
        ok("Starting fresh generation")

    print(f"\n{C.YELLOW}  Press Enter to start…  (Ctrl+C to abort){C.END}")
    try:
        input()
    except KeyboardInterrupt:
        return

    stop_evt   = threading.Event()
    mon_thread = threading.Thread(target=_monitor_loop, args=(stop_evt,), daemon=True)
    mon_thread.start()

    stat.start()
    sid = 0  # keep in outer scope for except block

    try:
        for scene in SCENES:
            sid    = scene["id"]
            s_name = scene["name"]

            if sid in completed:
                ok(f"Scene {sid:02d} '{s_name}' already done — skipping")
                continue

            chunk_seed = cfg["seed"] + sid
            t_scene    = time.time()

            # ── STAGE 1: Generate keyframe with 4.8B ─────────────────────
            keyframe_path = keyframe_map.get(str(sid))
            if keyframe_path and os.path.exists(keyframe_path):
                ok(f"Scene {sid:02d}  Stage 1 keyframe already exists — reusing")
                t_s1 = 0.0
            else:
                _print_live(sid, cfg["num_scenes"], 1, s_name, stat)
                step(sid + 1, cfg["num_scenes"], f"Scene {sid:02d}  Stage 1 — 4.8B Keyframe")

                pipe_img = load_image_pipeline(cfg)
                t_s1_start = time.time()

                keyframe   = generate_keyframe(pipe_img, scene, cfg, chunk_seed)
                t_s1       = time.time() - t_s1_start
                ok(f"Stage 1 complete in {t_s1:.1f}s")

                keyframe_path = save_keyframe(keyframe, sid, cfg)
                keyframe_map[str(sid)] = keyframe_path

                # ── Unload Stage 1 before loading Stage 2 ────────────────
                subheader("Unloading Stage 1 → freeing VRAM for Stage 2")
                del pipe_img, keyframe
                vram_free()
                ok(f"VRAM after Stage 1 unload: {vram_used_gb():.1f} GB")

                # Save keyframe path immediately for crash-resume
                state["keyframe_paths"] = keyframe_map
                save_state(cfg, state)

            # ── STAGE 2: Animate keyframe with 2B I2V ────────────────────
            _print_live(sid, cfg["num_scenes"], 2, s_name, stat)
            step(sid + 1, cfg["num_scenes"], f"Scene {sid:02d}  Stage 2 — 2B Video Animation")

            bridge_path    = bridge_map.get(str(sid - 1))
            conditioning   = prepare_conditioning_image(
                                 keyframe_path, bridge_path, sid, cfg)

            pipe_vid    = load_video_pipeline(cfg)
            t_s2_start  = time.time()

            frames      = generate_video_chunk(pipe_vid, scene, conditioning, cfg, chunk_seed)
            t_s2        = time.time() - t_s2_start
            ok(f"Stage 2 complete in {t_s2:.1f}s")

            # ── Save frames and bridge ────────────────────────────────────
            save_scene_frames(frames, sid, cfg)
            ok(f"Frames saved → scene_{sid:02d}/")

            bp = save_bridge_frame(frames, sid, cfg)
            bridge_map[str(sid)] = bp
            ok(f"Bridge saved → {os.path.basename(bp)}")

            # ── Unload Stage 2 ────────────────────────────────────────────
            del pipe_vid, frames, conditioning
            vram_free()

            # ── Update stats and checkpoint ───────────────────────────────
            t_total = time.time() - t_scene
            stat.end_chunk(t_total, t_s1, t_s2)
            completed.add(sid)
            state["completed_scenes"] = sorted(completed)
            state["bridge_frames"]    = bridge_map
            save_state(cfg, state)

            ok(f"Scene {sid:02d} done in {t_total:.1f}s  "
               f"(S1={t_s1:.0f}s + S2={t_s2:.0f}s)")
            time.sleep(1.5)

    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}  Interrupted — progress saved. Re-run to resume.{C.END}")
        stop_evt.set()
        stat.summary()
        return

    except Exception as e:
        err(f"Error at scene {sid}: {e}")
        import traceback; traceback.print_exc()
        stop_evt.set()
        warn("Re-run to resume from last completed scene.")
        stat.summary()
        return

    finally:
        stop_evt.set()

    # ── Assemble ──────────────────────────────────────────────────────────
    final_video = assemble_video(cfg)

    # ── Report ────────────────────────────────────────────────────────────
    rp = os.path.join(cfg["output_dir"],
                      f"report_v{__version__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(rp, "w") as f:
        json.dump({
            "version":        __version__,
            "timestamp":      datetime.now().isoformat(),
            "scenes":         [s["name"] for s in SCENES],
            "completed":      sorted(completed),
            "stage1_times_s": stat.stage1_times,
            "stage2_times_s": stat.stage2_times,
            "total_elapsed_s": stat.elapsed(),
            "final_video":    final_video,
        }, f, indent=2)
    ok(f"Report: {rp}")

    stat.summary()

    print(f"\n{C.BOLD}{C.GREEN}{'═'*72}{C.END}")
    print(f"{C.BOLD}{C.GREEN}  ✨  SmartSmile 2250 — 1-Minute Dental Ad v{__version__} — COMPLETE{C.END}")
    print(f"{C.BOLD}{C.GREEN}{'═'*72}{C.END}")
    print(f"\n  {C.BOLD}📹  Output     :{C.END}  {final_video}")
    print(f"  {C.BOLD}📐  Spec       :{C.END}  {cfg['width']}×{cfg['height']}  "
          f"|  {cfg['fps']} fps  |  ~{cfg['total_duration_seconds']:.0f}s")
    print(f"  {C.BOLD}🖼   Stage 1    :{C.END}  SANA 4.8B  →  {cfg['keyframe_width']}×"
          f"{cfg['keyframe_height']} keyframe/scene")
    print(f"  {C.BOLD}🎬  Stage 2    :{C.END}  SANA-Video 2B  →  "
          f"{cfg['frames_per_chunk']} frames/scene  (I2V)")
    print(f"  {C.BOLD}🌉  Bridge     :{C.END}  Keyframe 70% + prior-scene bridge 30%\n")


if __name__ == "__main__":
    main()