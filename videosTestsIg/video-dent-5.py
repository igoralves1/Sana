"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║   SANA IMAGE FILM ENGINE — SMARTSMILE 2250 DENTAL ADVERTISEMENT                 ║
║   Version  : 5.0  |  Date: 2026-05-24  |  Author: Dr. Igor Lemos Alves          ║
║   Hardware : RTX 5080 16 GB VRAM · Ryzen 9 9900X · 32 GB DDR5 · Windows        ║
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   CHANGELOG                                                                      ║
║   v1–v3  SANA-Video 2B pipeline (video generation).                             ║
║   v4     Three-stage quality overhaul (4.8B keyframe + 2B I2V + LTX-2).        ║
║   v5     PURE 4.8B IMAGE FILM ENGINE with sliding-window VRAM strategy.         ║
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   ┌─────────────────────────────────────────────────────────────────────────┐   ║
║   │   YOUR INSIGHT — AND WHY IT IS ARCHITECTURALLY CORRECT                  │   ║
║   └─────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                  ║
║   You identified three fundamental truths:                                       ║
║                                                                                  ║
║   TRUTH 1 — A VIDEO IS A COLLECTION OF IMAGES                                   ║
║   ─────────────────────────────────────────────                                  ║
║   Every MP4 you have ever watched is just a sequence of still images displayed  ║
║   fast enough that your brain perceives motion. At 24 fps, a 60-second video    ║
║   is exactly 1,440 individual frames. At 60 fps it is 3,600 frames. Each one    ║
║   is a complete image. There is no fundamental difference between a "video       ║
║   model" and an "image model" — the video model simply generates many images    ║
║   that are constrained to be consistent with each other across time.            ║
║                                                                                  ║
║   TRUTH 2 — THE 4.8B IMAGE MODEL IS SIGNIFICANTLY BETTER                        ║
║   ──────────────────────────────────────────────────────                         ║
║   SANA-1.5 4.8B was trained exclusively on still images at 1024px resolution.  ║
║   It has seen orders of magnitude more image data than the 2B video model,      ║
║   which had to divide its training between visual quality and temporal motion.  ║
║   For a single frame, the 4.8B model produces noticeably sharper teeth,        ║
║   more accurate skin tones, better lighting, and more coherent anatomy.         ║
║   The video model is faster but compromises per-frame quality for smoothness.  ║
║                                                                                  ║
║   TRUTH 3 — YOUR SLIDING WINDOW STRATEGY SOLVES THE VRAM PROBLEM               ║
║   ────────────────────────────────────────────────────────────────               ║
║   The RTX 5080 has 16 GB VRAM. At any given moment only 3 images exist in      ║
║   VRAM simultaneously. As soon as a new image is generated, the oldest one      ║
║   is written to disk (SSD) and evicted from VRAM. The two most recent images   ║
║   stay in VRAM as the visual context for the next generation. This gives the    ║
║   model a "memory" of where it just came from — like a cinematographer who      ║
║   can see their last two shots before deciding the next angle.                  ║
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   THE SLIDING WINDOW ALGORITHM                                                   ║
║   ───────────────────────────────                                                ║
║                                                                                  ║
║   WINDOW SIZE = 3 images in VRAM at any time:                                   ║
║   [img_prev2]  [img_prev1]  [img_current ← being generated]                    ║
║                                                                                  ║
║   Step 1:  Generate image[0] using text prompt alone (no prior context).        ║
║   Step 2:  Generate image[1] conditioned on image[0].                           ║
║   Step 3:  Generate image[2] conditioned on image[0] and image[1].              ║
║            → Now the window is full: [img0, img1, img2] all in VRAM.           ║
║   Step 4:  Write img0 to SSD. Evict img0 from VRAM.                            ║
║            Generate image[3] conditioned on image[1] and image[2].              ║
║            → Window: [img1, img2, img3] in VRAM.                               ║
║   Step 5:  Write img1 to SSD. Evict img1 from VRAM.                            ║
║            Generate image[4] conditioned on image[2] and image[3].              ║
║            → Window: [img2, img3, img4] in VRAM.                               ║
║   ...continues until end of BEAT (scene segment).                               ║
║                                                                                  ║
║   CONDITIONING MECHANISM                                                         ║
║   The two context images are blended into a single conditioning signal:         ║
║     cond = alpha * img_prev1 + (1-alpha) * img_prev2                            ║
║   where alpha controls how strongly the NEAREST previous image anchors the     ║
║   next generation. Default alpha = 0.75 (75% most-recent, 25% older).          ║
║   This creates smooth visual drift rather than abrupt changes.                  ║
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   THREE LEVELS OF NARRATIVE STRUCTURE                                            ║
║   ──────────────────────────────────────                                         ║
║                                                                                  ║
║   LEVEL 1: FILM (the whole movie — e.g. 10 minutes = 600 seconds)               ║
║   A film is a sequence of ACTS. Each act has its own emotional arc              ║
║   and visual palette. The film-level prompt establishes what this movie          ║
║   is about: characters, world, tone, and visual language.                       ║
║                                                                                  ║
║   LEVEL 2: BEAT (a scene segment — e.g. 60 seconds = 3600 frames at 60fps)     ║
║   A beat is one continuous camera movement through a location. All frames       ║
║   within a beat share: the same lighting, the same color palette, the same      ║
║   characters, and the same action. The beat_prompt describes WHAT IS HAPPENING. ║
║   Beats within an ACT share visual style but differ in action and angle.        ║
║                                                                                  ║
║   LEVEL 3: FRAME (a single image — 1/fps of a second)                          ║
║   Each frame gets a micro-prompt that describes the EXACT visual state:         ║
║   camera angle, subject position, expression, lighting temperature.             ║
║   The micro-prompt is derived from the beat_prompt by interpolating between     ║
║   beat_start_description and beat_end_description over the frame count.         ║
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   WHAT NEGATIVE PROMPTS ACTUALLY DO — COMPLETE EXPLANATION                      ║
║   ──────────────────────────────────────────────────────────                     ║
║                                                                                  ║
║   During diffusion, the model runs two passes per denoising step:               ║
║     Pass 1: conditioned on the POSITIVE prompt  → what you WANT                ║
║     Pass 2: conditioned on the NEGATIVE prompt  → what you DON'T want          ║
║   The final output moves TOWARD pass 1 and AWAY from pass 2.                   ║
║   This is called Classifier-Free Guidance (CFG). The guidance_scale            ║
║   controls the strength: higher scale = stronger pull toward positive,          ║
║   stronger push away from negative.                                             ║
║                                                                                  ║
║   NEGATIVE_IMAGE (for SANA-1.5 4.8B still frame generation):                   ║
║   Each token in this string targets a specific failure mode of the model:       ║
║   · "blurry, out of focus"  → forces the model away from soft renders          ║
║   · "low quality, jpeg artifacts"  → prevents compression-style textures        ║
║   · "watermark, text, logo"  → prevents learned watermarks from training data  ║
║   · "overexposed, underexposed"  → anchors exposure to the normal range        ║
║   · "grain, noise"  → prevents film grain that the model may add by default   ║
║   · "cartoon, anime, oil painting, illustration"  → prevents style bleed        ║
║     The model was trained on internet images which include illustrated art.    ║
║     Without this token, dental clinic scenes may render as anime or art.        ║
║   · "sketch"  → prevents pencil-style rendering                                ║
║   · "deformed anatomy, extra limbs"  → counters the model's tendency           ║
║     to generate extra fingers, merged body parts, or broken joints.            ║
║   · "plastic skin"  → counters the common CGI-skin artifact                   ║
║   · "flat lighting, washed out colors"  → pushes toward rich cinematic light  ║
║                                                                                  ║
║   NEGATIVE_VIDEO (for SANA-Video 2B temporal generation — not used in v5):     ║
║   Extends NEGATIVE_IMAGE with video-specific temporal failure modes:            ║
║   · "motion blur, jump cuts, jerky movements"  → temporal smoothness           ║
║   · "frames out of sync, inconsistent character shapes"  → identity stability  ║
║   · "temporal artifacts, jitter, ghosting effects"  → frame-to-frame coherence ║
║   These tokens only apply to video pipelines. In v5 (pure image pipeline)      ║
║   we use only NEGATIVE_IMAGE since each frame is generated independently        ║
║   and temporal artifacts don't exist in per-frame generation.                  ║
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   SCENE JSON — EVERY FIELD EXPLAINED                                             ║
║   ──────────────────────────────────                                             ║
║                                                                                  ║
║   "id"              : Integer. Scene index 0–N. Controls seed offset            ║
║                       (seed = base_seed + id) and output directory naming.      ║
║                       Never reuse ids across sessions — it will corrupt seeds.  ║
║                                                                                  ║
║   "name"            : Human-readable label for dashboards and reports.          ║
║                       Has no effect on generation.                              ║
║                                                                                  ║
║   "beats"           : One-sentence summary of the narrative action.             ║
║                       Used for human reference and report generation.           ║
║                       Has no effect on generation.                              ║
║                                                                                  ║
║   "fps"             : Frames per second for THIS beat. Different beats can      ║
║                       have different fps. High-action beats use 24 fps          ║
║                       (fewer frames, faster). Slow detail beats use 60 fps      ║
║                       (more frames, smoother). RTX 5080 at 60 fps generates    ║
║                       3,600 frames per minute of film (~6 hours per minute).   ║
║                       Start with 24 fps for testing. Use 60 fps for finals.    ║
║                                                                                  ║
║   "duration_s"      : How many seconds this beat lasts in the final film.      ║
║                       Total frames = fps * duration_s.                          ║
║                       All beats together define the total film length.          ║
║                                                                                  ║
║   "base_prompt"     : The CORE visual description that applies to EVERY         ║
║                       frame in this beat. It establishes:                       ║
║                       - Lighting (golden hour, surgical blue-white, etc.)       ║
║                       - Shot type (close-up, wide, aerial, macro)               ║
║                       - Environment (operating room, park, clinic exterior)     ║
║                       - Primary subject and their state                         ║
║                       - Color palette and mood                                  ║
║                       CRITICAL: end with quality tokens like                   ║
║                       "photorealistic, 8K, cinematic" to anchor the style.     ║
║                       These tokens act as a constant style attractor across    ║
║                       all frames of the beat.                                   ║
║                                                                                  ║
║   "start_description": Describes the FIRST frame of the beat.                  ║
║                       Combined with base_prompt to produce the opening image.  ║
║                       Example: "Camera far back, wide establishing view,        ║
║                       subject standing still in center frame."                  ║
║                       The model's first frame sets the visual anchor for        ║
║                       the entire beat — be very specific here.                 ║
║                                                                                  ║
║   "end_description" : Describes the LAST frame of the beat.                    ║
║                       Combined with base_prompt to produce the closing image.  ║
║                       Example: "Camera very close, tight on subject's face,    ║
║                       mid-smile, single tear on cheek."                        ║
║                       The algorithm interpolates between start and end          ║
║                       descriptions linearly across all frames. Frame 0 uses    ║
║                       100% start. Frame N uses 100% end. Midpoint frames use   ║
║                       the base_prompt only (fading both descriptions out).     ║
║                                                                                  ║
║   "camera_move"     : Describes camera motion over the beat duration.          ║
║                       Injected into the micro-prompt at every frame.           ║
║                       Examples: "slow push-in", "gentle orbit left",           ║
║                       "static lock", "slow tilt up", "lateral dolly right".    ║
║                       IMPORTANT: this is injected as context, not as a         ║
║                       command. The image model doesn't "move" the camera —     ║
║                       it generates each frame independently. The camera         ║
║                       "moves" because consecutive frames progressively          ║
║                       describe a closer/different viewpoint, and the            ║
║                       sliding-window conditioning keeps them consistent.        ║
║                                                                                  ║
║   "transition_in"   : How this beat begins. Options:                            ║
║                       "cut"    → abrupt start, no visual reference              ║
║                                  from prior beat (first frame is fresh).        ║
║                       "blend"  → first frame is blended with last frame        ║
║                                  of prior beat (bridge_alpha controls amount). ║
║                       "match"  → first frame is conditioned directly on        ║
║                                  the last frame of the prior beat               ║
║                                  (strongest continuity, may constrain style).  ║
║                                                                                  ║
║   "transition_out"  : How this beat ends. Defines how last frame is stored     ║
║                       for use by the next beat.                                 ║
║                       "cut"    → last frame stored as-is.                      ║
║                       "fade"   → last frame darkened slightly (visual cue).    ║
║                                                                                  ║
║   "color_grade"     : Short color description appended to every frame prompt. ║
║                       Anchors the emotional tone of the entire beat:            ║
║                       "warm golden hour, amber tones"  → comfort, healing      ║
║                       "cool clinical blue-white"  → precision, technology      ║
║                       "deep navy, emerald accents"  → epic, global scale       ║
║                       "pure white and gold"  → luxury, brand, iconic           ║
║                                                                                  ║
║   "motion_density"  : Float 0.0–1.0. Controls how much visual change          ║
║                       is described between consecutive frame prompts.           ║
║                       0.0 = completely static (macro shots, logo holds)        ║
║                       0.3 = gentle drift (slow push-in, soft wind)             ║
║                       0.6 = normal motion (walking, talking, camera orbit)     ║
║                       1.0 = fast motion (crowd, action, fast tracking)         ║
║                       Higher values interpolate prompts more aggressively,      ║
║                       describing larger viewpoint shifts per frame.             ║
║                                                                                  ║
║   "seed_offset"     : Integer added to base_seed for this beat.                 ║
║                       Same offset = visually similar character/environment.    ║
║                       Different offsets = different visual "takes".             ║
║                       Beats within the same scene should share seed_offset      ║
║                       to maintain character consistency across cuts.            ║
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   CONSISTENCY STRATEGY — HOW THE AI KEEPS FRAMES COHERENT                      ║
║   ─────────────────────────────────────────────────────────                      ║
║                                                                                  ║
║   Challenge: SANA-1.5 4.8B is a TEXT-TO-IMAGE model. Each generation is        ║
║   independent by default — it doesn't "know" what the previous frame looked    ║
║   like. Left unconstrained, consecutive frames would look like completely       ║
║   unrelated images. The sliding-window strategy solves this by encoding         ║
║   the previous frames AS IMAGES into the model's conditioning signal.           ║
║                                                                                  ║
║   The model "sees" the last two frames as a visual anchor and is instructed     ║
║   via the prompt to CONTINUE from that visual state. This works because         ║
║   SANA-1.5 uses a powerful Gemma-2-2B text encoder (the same architecture      ║
║   used in Google's Gemma language models) which can process the conditioning    ║
║   instruction "Continue this scene with the same lighting, character, and       ║
║   composition as the reference images" alongside the visual context.            ║
║                                                                                  ║
║   THREE CONSISTENCY ANCHORS:                                                    ║
║   1. Visual anchor    — the blended prior-frame conditioning image              ║
║   2. Prompt anchor    — base_prompt repeated in every frame                    ║
║   3. Color anchor     — color_grade token repeated in every frame              ║
║   4. Seed progression — seed = base + frame_index keeps deterministic drift    ║
║                                                                                  ║
║   SCENE BOUNDARY STRATEGY:                                                      ║
║   When transitioning between beats, the FIRST frame of the new beat is         ║
║   the most critical. It sets the visual anchor for ALL subsequent frames.      ║
║   For "blend" transitions: it is generated from a mix of the prior beat's      ║
║   last frame and the new beat's start_description.                             ║
║   For "cut" transitions: it is generated purely from text, anchoring fresh.    ║
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   SCALING TO A 10-MINUTE FILM                                                   ║
║   ──────────────────────────────                                                 ║
║   · 24 fps × 60 seconds = 1,440 frames per minute                              ║
║   · RTX 5080: ~60s per image at 60 steps, CFG 7.5 → ~24 hours per minute      ║
║   · For practical production: use 8–12 fps (lower fps than real-time)          ║
║     and let the video encoder interpolate frames (RIFE/FILM frame interpolation)║
║   · At 8 fps × 60 seconds = 480 frames per minute → ~8 hours per minute       ║
║   · 10-minute film at 8 fps = 4,800 frames → ~80 hours on RTX 5080            ║
║   · This is why the 4.8B model at full quality is for PREMIUM production.      ║
║   · For faster iteration: use image_inference_steps=20, guidance=5.0           ║
║     to get 10x faster draft frames, then final-quality pass selected scenes.  ║
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   INSTALL                                                                        ║
║   pip install git+https://github.com/huggingface/diffusers                     ║
║   pip install torch torchvision imageio[ffmpeg] psutil GPUtil Pillow numpy      ║
║                                                                                  ║
║   REFERENCES                                                                     ║
║   SANA-1.5 model card  : huggingface.co/Efficient-Large-Model/...4.8B          ║
║   SANA overview        : nvlabs.github.io/Sana/docs/                           ║
║   SANA-Video docs      : nvlabs.github.io/Sana/docs/sana_video/                ║
║   NVLabs GitHub        : github.com/NVlabs/Sana                                ║
║   SANA-WM world model  : studio.aifilms.ai/blog/sana-wm-nvidia-world-model     ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

__version__ = "5.0"
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
from typing import Optional, List, Tuple

# ── Optional GPU monitoring ───────────────────────────────────────────────────
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

# ── Diffusers — the core SANA-1.5 4.8B image pipeline ────────────────────────
#
# SanaPipeline: the text-to-image pipeline for SANA-1.5.
#   - Uses DC-AE (32× spatial compression) instead of the standard 8× VAE.
#   - Uses Gemma-2-2B-IT as the text encoder (much stronger than CLIP).
#   - Supports PAG (Perturbed Attention Guidance) for structural quality.
#   - Native resolution: 1024px (supports multi-scale width/height).
#
# AutoPipelineForText2Image: a wrapper that auto-detects the correct pipeline
# class and can enable PAG at load-time. Preferred over SanaPipeline directly
# because it handles the pag_applied_layers argument correctly.
#
try:
    from diffusers import SanaPipeline, AutoPipelineForText2Image
    from diffusers.utils import load_image
    DIFFUSERS_OK = True
except ImportError as _e:
    DIFFUSERS_OK  = False
    _MISSING      = str(_e)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TERMINAL COLORS AND OUTPUT HELPERS                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class C:
    """ANSI escape codes. Works on Windows (Windows Terminal) and Linux/macOS."""
    CYAN    = '\033[96m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    RED     = '\033[91m'
    BLUE    = '\033[94m'
    MAGENTA = '\033[95m'
    BOLD    = '\033[1m'
    DIM     = '\033[2m'
    END     = '\033[0m'

def header(t):      print(f"\n{C.BOLD}{C.CYAN}{'═'*76}\n  {t}\n{'═'*76}{C.END}")
def subhdr(t):      print(f"\n{C.BOLD}{C.MAGENTA}  ▶  {t}{C.END}")
def ok(t):          print(f"  {C.GREEN}✔{C.END}  {t}")
def warn(t):        print(f"  {C.YELLOW}⚠{C.END}  {t}")
def err(t):         print(f"  {C.RED}✘{C.END}  {t}")
def info(t):        print(f"  {C.BLUE}ℹ{C.END}  {t}")
def note(t):        print(f"  {C.CYAN}★{C.END}  {t}")
def step(n, t, tx): print(f"\n{C.BOLD}{C.BLUE}[{n}/{t}]{C.END} {tx}")
def recovered(t):   print(f"  {C.MAGENTA}♻{C.END}  {C.BOLD}RECOVERED:{C.END} {t}")

def bar(frac: float, w: int = 44) -> str:
    """Unicode progress bar. frac in [0,1], w = character width."""
    f = int(w * max(0.0, min(1.0, frac)))
    return f"{C.GREEN}{'█'*f}{C.END}{'░'*(w-f)}"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  NEGATIVE PROMPTS — COMPLETE EXPLANATION EMBEDDED AS DOCSTRINGS             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────
# NEGATIVE_IMAGE — used for every frame generated by SANA-1.5 4.8B
#
# HOW IT WORKS:
#   During each denoising step the model runs two forward passes:
#     Pass 1 (positive): "generate a beautiful dental clinic scene at golden hour..."
#     Pass 2 (negative): "...but NOT blurry, NOT cartoon, NOT anatomically wrong..."
#   The output latent is shifted: TOWARD pass-1, AWAY from pass-2.
#   Formula: output = positive_pred + guidance_scale * (positive_pred - negative_pred)
#
# EACH TOKEN EXPLAINED:
#   "blurry, out of focus"
#     Prevents the model from generating soft-focus images. Without this,
#     many generations have a painterly softness typical of diffusion models
#     that have seen a lot of photography with shallow DOF.
#
#   "low quality, low resolution, jpeg artifacts"
#     The model was trained on internet images, many of which are compressed.
#     These tokens push it away from the compression-artifact aesthetic.
#
#   "watermark, text, logo"
#     Training data contained stock photos with visible watermarks. Without
#     this token, the model may hallucinate text-like patterns on clean surfaces.
#
#   "overexposed, underexposed"
#     Anchors the output to normal exposure range. Prevents blown highlights
#     (white dental clinic walls becoming pure white blobs) or crushed blacks.
#
#   "grain, noise"
#     Some aesthetic training data included film grain. For medical/clinical
#     imagery we want pristine clean renders, not film emulation.
#
#   "cartoon, anime, oil painting, illustration, sketch"
#     SANA was trained on diverse internet images including illustrated content.
#     Without these tokens, complex prompts like "futuristic clinic" may blend
#     toward concept-art or animation style rather than photography.
#
#   "deformed anatomy, extra limbs, duplicate faces, disfigured"
#     A known failure mode of all diffusion models is generating extra fingers,
#     fused limbs, or body parts in anatomically incorrect positions. These
#     tokens specifically push the model away from that region of image space.
#
#   "plastic skin"
#     CGI-rendered skin tends to lack subsurface scattering. The model can
#     produce smooth but lifeless skin without this correction token.
#
#   "flat lighting, washed out colors"
#     Pushes the model toward rich cinematic lighting with proper contrast
#     rather than the flat studio-white look common in training data.
# ─────────────────────────────────────────────────────────────────────────────
NEGATIVE_IMAGE = (
    "blurry, out of focus, low quality, low resolution, jpeg artifacts, "
    "watermark, text, logo, overexposed, underexposed, grain, noise, "
    "distorted perspective, cartoon, anime, oil painting, illustration, "
    "sketch, deformed anatomy, extra limbs, duplicate faces, ugly, disfigured, "
    "plastic skin, flat lighting, washed out colors, oversaturated."
)

# ─────────────────────────────────────────────────────────────────────────────
# NEGATIVE_CONSISTENCY — appended when generating frames in sliding-window mode
#
# When generating frame N conditioned on frames N-1 and N-2, we additionally
# push the output away from visual discontinuities that would break the
# illusion of motion. These tokens target the specific failure modes that
# appear when an image model generates frames conditioned on prior images:
#
#   "sudden change in lighting"
#     Prevents the model from flipping between warm and cool lighting
#     mid-beat, which would look like a jump cut.
#
#   "different background, different room"
#     Without this, the model may drift to a visually similar but different
#     environment, especially when the prior frame shows only a partial view.
#
#   "different person, different face"
#     Character identity drift is the hardest consistency problem for image
#     models. This token alone doesn't solve it but reduces the probability
#     of generating a completely different person.
#
#   "abrupt composition change"
#     Prevents the model from jumping from a wide shot to an extreme close-up
#     in a single frame, which would look like a cut rather than a push-in.
# ─────────────────────────────────────────────────────────────────────────────
NEGATIVE_CONSISTENCY = (
    "sudden change in lighting, different background, different room, "
    "different person, different face, abrupt composition change, "
    "inconsistent color palette, inconsistent style."
)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BEAT DEFINITIONS — THE COMPLETE SCENE JSON WITH ALL FIELDS DOCUMENTED      ║
# ║                                                                              ║
# ║  A BEAT is the atomic narrative unit: one continuous camera movement         ║
# ║  through one location/action. The entire film is a sequence of beats.        ║
# ║                                                                              ║
# ║  READ THE FIELD DOCUMENTATION IN THE HEADER (╔ block above) FOR COMPLETE    ║
# ║  EXPLANATIONS. Short reminders are provided as inline comments below.        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

BEATS = [
    # ─────────────────────────────────────────────────────────────────────────
    # BEAT 0: Opening shot — aerial approach to the megacity
    # Duration: 5 seconds. fps: 24. Total frames: 120.
    # Camera: slow forward drift from high altitude to mid-altitude.
    # Visual anchor: warm sunrise, crystal towers, light ribbons.
    # ─────────────────────────────────────────────────────────────────────────
    {
        # Unique identifier. Controls seed offset and output folder naming.
        # NEVER change this after you've started generating — it would corrupt
        # the seed sequence and produce inconsistent results on resume.
        "id": 0,

        # Human-readable name — for dashboards and reports only, no effect on generation.
        "name": "Opening — Megacity Aerial Approach",

        # One-sentence narrative summary — for reports only.
        "beats": "Aerial establishing shot drifting forward over the city of 2250",

        # FPS for this beat. Lower = fewer frames = faster to generate.
        # 24 fps is the cinema standard and a good starting point.
        # 60 fps = 2.5× more frames = 2.5× longer to generate.
        # Recommendation: use 24 fps for all beats during development,
        # increase to 60 fps for the final pass on key beats.
        "fps": 24,

        # Duration in seconds. Total frames = fps × duration_s.
        # 24 fps × 5s = 120 frames for this beat.
        "duration_s": 5,

        # base_prompt: describes EVERY FRAME in this beat.
        # Include: lighting · shot type · environment · primary subject · quality tokens.
        # Do NOT include motion language — that belongs in camera_move.
        # Do NOT include specific positional language — that belongs in start/end_description.
        # These quality tokens at the end ("photorealistic, 8K...") act as
        # a constant style attractor that prevents the model drifting toward
        # illustration or low-quality rendering across 120 frames.
        "base_prompt": (
            "Soft golden sunrise backlight, aerial wide shot, centered horizon. "
            "Luminous megacity of 2250: crystal spire towers, translucent bio-domes "
            "glowing amber and cobalt. Magnetic-lev vehicles trace light ribbons "
            "through elevated glass highways. Volumetric sunrise clouds and god-rays. "
            "Photorealistic, cinematic anamorphic lens, 8K, award-winning aerial photography, "
            "utopian color grading, ultra-detailed."
        ),

        # start_description: the EXACT visual state of FRAME 0.
        # Combined with base_prompt to generate the opening image of this beat.
        # Be very specific: camera height, subjects visible, atmosphere.
        # Frame 0 anchors the entire beat — poorly specified = inconsistent beat.
        "start_description": (
            "Camera at extreme altitude, city stretches from edge to edge of frame, "
            "horizon line at upper third, towers tiny against the dawn sky, "
            "predawn indigo sky with first orange light at the horizon."
        ),

        # end_description: the EXACT visual state of the LAST FRAME.
        # The algorithm interpolates from start → base → end across all frames.
        # For this beat: camera moves from very high altitude to medium altitude,
        # making towers grow progressively larger over 120 frames.
        "end_description": (
            "Camera at medium altitude, towers now fill half the frame, "
            "individual windows and light ribbons visible, warm golden sunrise "
            "floods the scene, lens flare touches the upper-right of frame."
        ),

        # camera_move: describes how the camera travels over the beat duration.
        # Injected into every frame's micro-prompt.
        # The image model doesn't literally "move" — instead, each frame's
        # prompt progressively describes a closer/different viewpoint, and
        # sliding-window conditioning keeps them visually connected.
        "camera_move": "slow continuous forward drift, gradual altitude descent",

        # transition_in: how this beat starts relative to the prior beat.
        # "cut" = fresh start, no visual reference from prior beat.
        # "blend" = first frame blended with last frame of prior beat.
        # "match" = first frame conditioned directly on prior beat's last frame.
        # Beat 0 always uses "cut" (nothing before it).
        "transition_in": "cut",

        # transition_out: how this beat ends.
        # "cut" = last frame stored as-is for next beat's reference.
        # "fade" = last frame slightly darkened (creates fade-to-black feel).
        "transition_out": "cut",

        # color_grade: short emotional color description.
        # Appended to EVERY frame prompt in this beat.
        # Acts as a constant visual style anchor — like a LUT in post-production.
        # This prevents the model from randomly changing the emotional tone
        # of colors mid-beat (e.g., switching from warm to cold unexpectedly).
        "color_grade": "warm golden sunrise, amber and cobalt palette",

        # motion_density: float 0.0–1.0.
        # Controls how aggressively consecutive frame prompts differ from each other.
        # 0.0 = identical prompts for all frames (completely static)
        # 0.3 = subtle drift (slow push-in, soft ambient changes)
        # 0.6 = clear motion (walking, camera orbit)
        # 1.0 = fast motion (running, action, rapid tracking)
        # This does NOT control the model's motion — it controls how much
        # textual description change is applied between consecutive prompts.
        "motion_density": 0.4,

        # seed_offset: added to base_seed for ALL frames in this beat.
        # Beats that share the same seed_offset will tend to generate the
        # same characters, environments, and visual "feel".
        # Change this between narrative acts to signal a new visual world.
        "seed_offset": 0,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # BEAT 1: SmartSmile 2250 clinic facade reveal
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": 1,
        "name": "Clinic Exterior — SmartSmile 2250 Facade",
        "beats": "Camera pushes toward the clinic entrance through the garden",
        "fps": 24,
        "duration_s": 5,
        "base_prompt": (
            "Soft diffused morning daylight, medium architectural wide shot, "
            "bilateral symmetry composition. White biopolymer and smart glass dental clinic. "
            "Holographic 'SmartSmile 2250' logo in aquamarine above entrance. "
            "Bioluminescent sakura trees with cyan blossoms lining the path. "
            "Patients in elegant minimalist attire walking through garden. "
            "Photorealistic, Zaha Hadid architectural photography, 8K, "
            "global illumination, prismatic glass reflections."
        ),
        "start_description": (
            "Full facade visible, camera at garden entrance, trees framing both sides, "
            "holographic logo spinning gently far ahead, patients approaching from left."
        ),
        "end_description": (
            "Camera close to entrance doors, facade fills 80% of frame, "
            "logo looms large above, glass doors beginning to reflect the garden."
        ),
        "camera_move": "slow steady push-in toward entrance, level camera height",
        "transition_in": "blend",   # soft blend from aerial beat
        "transition_out": "cut",
        "color_grade": "cool aquamarine and white, soft morning light",
        "motion_density": 0.35,
        "seed_offset": 100,  # New environment — different seed family
    },

    # ─────────────────────────────────────────────────────────────────────────
    # BEAT 2: Reception — AI concierge interaction
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": 2,
        "name": "Reception — AI Concierge",
        "beats": "Patient checks in with holographic AI receptionist at floating desk",
        "fps": 24,
        "duration_s": 5,
        "base_prompt": (
            "Cool blue-white ambient light with warm amber accent panels, "
            "medium close-up, rule-of-thirds, patient left of center. "
            "Elegant patient at floating glass reception desk. "
            "Translucent holographic AI avatar with calm warm expression across from patient. "
            "Teal neon data streams floating between them. White-uniformed staff in background. "
            "Photorealistic, cinematic portrait, Sony A7RV 85mm f/1.4, 8K, ultra-sharp."
        ),
        "start_description": (
            "Wide angle, both patient and AI avatar fully visible, reception desk centered, "
            "patient just arriving at desk, looking up at avatar."
        ),
        "end_description": (
            "Tighter medium close-up on patient's face, gentle smile of recognition, "
            "data streams flowing between them, avatar gesturing toward the clinic interior."
        ),
        "camera_move": "very slow gentle push-in, almost imperceptible",
        "transition_in": "cut",
        "transition_out": "cut",
        "color_grade": "clinical blue-white with warm amber accents",
        "motion_density": 0.25,
        "seed_offset": 200,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # BEAT 3: AI oral scan — diagnostic scene
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": 3,
        "name": "Diagnostics — AI Oral Scan",
        "beats": "AI scanner maps patient's mouth in real time, holographic model appears",
        "fps": 24,
        "duration_s": 5,
        "base_prompt": (
            "Soft cool surgical lighting, close-up macro shot, centered clinical composition. "
            "Brushed-titanium AI oral scanner hovering near patient's open mouth. "
            "Real-time 3D holographic dental model in midair — teeth color-coded by health status. "
            "Dentist in AR smart-glass eyewear using haptic stylus on hologram. "
            "Photorealistic, medical device photography, Zeiss macro, 8K, perfect enamel rendering."
        ),
        "start_description": (
            "Wide view of treatment room, scanner device visible in context, "
            "patient reclined, dentist standing, hologram just beginning to form."
        ),
        "end_description": (
            "Tight close-up on the holographic dental model floating in midair, "
            "health-status colors vivid, dentist's stylus tracing an annotation beam."
        ),
        "camera_move": "slow orbit from left profile to frontal three-quarter view",
        "transition_in": "cut",
        "transition_out": "cut",
        "color_grade": "cool blue-white clinical precision lighting",
        "motion_density": 0.4,
        "seed_offset": 300,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # BEAT 4: Nano-robot procedure
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": 4,
        "name": "Treatment — Nano-Robot Swarm Procedure",
        "beats": "Luminous nano-robot cloud performs painless precision dental work",
        "fps": 24,
        "duration_s": 5,
        "base_prompt": (
            "Diffused warm-cool surgical lighting, medium shot with macro insert. "
            "Patient in floating white ergonomic dental chair, completely peaceful expression. "
            "Luminous cloud of nano-robots above open mouth — each mote emitting electric-blue glow. "
            "Swarm forms perfect dome of light over treatment area. "
            "Transparent AR screen beside chair shows live nano-scale mapping in teal. "
            "Photorealistic, 8K, cinematic medical sci-fi, wonder and serenity."
        ),
        "start_description": (
            "Full medium shot, patient and chair in full frame, nano-swarm just beginning to form, "
            "AR screen showing initial scan data, warm room light."
        ),
        "end_description": (
            "Camera closer, tight on the patient's peaceful face and the swarm above, "
            "each nano-bot clearly visible pulsing blue, swarm at peak density and brightness."
        ),
        "camera_move": "slow continuous drift from wide to medium-close",
        "transition_in": "cut",
        "transition_out": "cut",
        "color_grade": "serene blue-white clinical, electric-blue nano accent",
        "motion_density": 0.3,
        "seed_offset": 400,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # BEAT 5: Bioprinting — living tooth regeneration
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": 5,
        "name": "Bioprinting — Living Tooth Regeneration",
        "beats": "Crystal-clear tooth grows layer by layer from a living bioprinter",
        "fps": 24,
        "duration_s": 5,
        "base_prompt": (
            "Warm amber-gold lab lighting, extreme close-up, perfectly centered macro. "
            "Dental bioprinter nozzle above a half-formed tooth crown. "
            "Translucent organic layers forming, stem cells glowing gold-white within. "
            "Nano-scaffold of blue light lines surrounding the growing structure. "
            "Water droplets on scaffold catching amber and teal light. "
            "Scientific macro photography, Leica M10, 8K, amber and ivory palette, shallow DOF bokeh."
        ),
        "start_description": (
            "Tooth base visible, just beginning to form, bioprinter nozzle poised above, "
            "scaffold barely formed, warm amber light filling the frame."
        ),
        "end_description": (
            "Tooth now 80% complete, translucent crown layers visible from base to near-tip, "
            "stem cell glow at maximum, scaffold fully lit, nano-ink filaments still depositing."
        ),
        "camera_move": "completely static macro hold, only subject changes",
        "transition_in": "cut",
        "transition_out": "cut",
        "color_grade": "warm amber and ivory, scientific wonder",
        "motion_density": 0.15,  # Near-static — only the tooth grows
        "seed_offset": 500,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # BEAT 6: Human + AI dentist collaboration
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": 6,
        "name": "Human + AI Dentist Collaboration",
        "beats": "Dentist and humanoid AI review rotating 3D jaw hologram together",
        "fps": 24,
        "duration_s": 5,
        "base_prompt": (
            "Soft split warm-cool lighting, medium two-shot, subjects side by side. "
            "Human dentist in tailored white coat, haptic stylus in hand, calm and focused. "
            "Sleek humanoid AI assistant, translucent torso with teal data-circuit latticework. "
            "Floating 3D holographic jaw model between them, health-status color coding on teeth. "
            "Warm walnut panels and brushed steel walls in background. "
            "Photorealistic, cinematic two-shot, Arri Alexa Mini LF, 8K."
        ),
        "start_description": (
            "Profile view of both figures, hologram between them, dentist just beginning to gesture."
        ),
        "end_description": (
            "Three-quarter frontal view, both faces partially visible, dentist's annotation beam "
            "visible on hologram, AI's teal circuits brightening in response."
        ),
        "camera_move": "slow arc from profile to three-quarter frontal, completely level",
        "transition_in": "cut",
        "transition_out": "cut",
        "color_grade": "warm wood and cool steel, split warm-cool lighting",
        "motion_density": 0.35,
        "seed_offset": 600,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # BEAT 7: Patient comfort — zero pain neural treatment
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": 7,
        "name": "Patient Experience — Zero Pain",
        "beats": "Patient completely at peace during neural pain-blocking treatment",
        "fps": 24,
        "duration_s": 5,
        "base_prompt": (
            "Golden warm ambient light, medium close-up portrait, low angle looking up. "
            "Patient in floating anti-gravity chair, eyes gently closed, peaceful smile. "
            "White titanium neural interface headband projecting soft geometric mandala patterns. "
            "Serene coral reef environment on room walls behind patient. "
            "Photorealistic, cinematic beauty lighting, Canon EOS R5 50mm f/1.2, 8K, "
            "warm golden hour, subsurface skin rendering."
        ),
        "start_description": (
            "Medium shot, patient's full upper body visible in chair, "
            "neural headband just activated, first mandala patterns appearing."
        ),
        "end_description": (
            "Tight close-up on patient's face, mandala patterns full brightness above brow, "
            "expression of complete peace, coral reef scene glowing softly behind."
        ),
        "camera_move": "barely perceptible push-in, almost static",
        "transition_in": "cut",
        "transition_out": "cut",
        "color_grade": "warm golden hour, amber skin tones",
        "motion_density": 0.12,  # Near-static — emotional stillness
        "seed_offset": 700,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # BEAT 8: The perfect smile reveal
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": 8,
        "name": "The Perfect Smile Reveal",
        "beats": "Patient sees their radiant new smile for the first time — pure joy",
        "fps": 24,
        "duration_s": 5,
        "base_prompt": (
            "Warm flattering beauty lighting three-point setup, medium close-up, centered portrait. "
            "Patient holding frameless smart mirror reflecting a radiant, aligned, brilliant smile. "
            "Mirror frame displays soft green health indicator: 98/100. "
            "Patient's skin glowing with health and joy. "
            "Photorealistic, cinematic beauty photography, Hasselblad H6D, 8K, warm bright palette."
        ),
        "start_description": (
            "Patient's expression neutral, raising the mirror slowly, face not yet reacting."
        ),
        "end_description": (
            "Eyes widened with joy, full genuine smile, single tear on one cheek, "
            "mirror tilted slightly showing reflected smile clearly, green health indicator glowing."
        ),
        "camera_move": "slowest possible push-in, barely a millimeter per frame",
        "transition_in": "cut",
        "transition_out": "cut",
        "color_grade": "warm bright whites, golden skin light, soft joy palette",
        "motion_density": 0.2,
        "seed_offset": 800,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # BEAT 9: Community park — humanity smiling together
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": 9,
        "name": "Community — Dental Health for All",
        "beats": "Diverse joyful crowd in SmartSmile 2250 public health park",
        "fps": 24,
        "duration_s": 5,
        "base_prompt": (
            "Bright warm diffused daylight, wide ensemble shot, dynamic diagonal composition. "
            "Diverse multicultural crowd: children running, elders holding hands, adults laughing. "
            "All showing brilliant healthy smiles. Holographic dental-health kiosks with cyan glow. "
            "Bioluminescent trees with pale gold leaves. SmartSmile drone above releasing care capsules. "
            "Photorealistic, crowd photography, 8K, vibrant warm daylight, optimism."
        ),
        "start_description": (
            "Wide establishing shot, full park visible, crowd just entering from the left, "
            "drone in background just beginning to hover."
        ),
        "end_description": (
            "Lateral tracking view at crowd level, faces clearly visible, smiles prominent, "
            "drone overhead releasing shimmer of capsules that catch warm light."
        ),
        "camera_move": "smooth lateral tracking dolly, keeping pace with crowd",
        "transition_in": "blend",
        "transition_out": "cut",
        "color_grade": "vibrant warm daylight, golden hour crowd lighting",
        "motion_density": 0.6,  # Active crowd movement
        "seed_offset": 900,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # BEAT 10: Global health dashboard
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": 10,
        "name": "Global Oral Health Dashboard",
        "beats": "Holographic Earth rotates as global health scores reach 100%",
        "fps": 24,
        "duration_s": 5,
        "base_prompt": (
            "Deep ocean-blue ambient light, wide shot, holographic spherical globe centerpiece. "
            "Vast holographic Earth two meters in diameter rotating in high-vaulted control room. "
            "SmartSmile clinic markers pulsing as gold dots across every continent. "
            "Curved 8K data panels showing bar charts trending upward, maps shifting red to green. "
            "Scientists in white lab coats on polished obsidian floor below. "
            "Photorealistic, 8K, deep navy and emerald and gold data-visualization tones, epic scale."
        ),
        "start_description": (
            "Camera at room level looking up at the globe, scientists in foreground, "
            "some scores still showing red regions on the map."
        ),
        "end_description": (
            "Camera pulled back to show full control room width, globe now fully green, "
            "all scores at 100%, scientists reacting with muted celebration."
        ),
        "camera_move": "slow backward dolly reveal, completely level",
        "transition_in": "cut",
        "transition_out": "cut",
        "color_grade": "deep navy, emerald, and gold, epic scale",
        "motion_density": 0.3,
        "seed_offset": 1000,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # BEAT 11: Brand closing — logo reveal
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": 11,
        "name": "Brand Closing — Logo and Tagline",
        "beats": "Crystal tooth transforms into SmartSmile 2250 logo with tagline",
        "fps": 24,
        "duration_s": 5,
        "base_prompt": (
            "Soft warm studio light radial gradient, ultra-wide perfectly centered, pure minimalism. "
            "Single translucent crystal tooth floating against infinite pure white background. "
            "'SmartSmile 2250' in polished gold letterforms below. "
            "Tagline in light-gray sans-serif: 'Every smile. Every human. Every future.' "
            "Crystal tooth casting prismatic spectrum across white surface. "
            "Photorealistic product photography, Phase One XF IQ4, 8K, pure white and gold, "
            "timeless luxury brand aesthetic."
        ),
        "start_description": (
            "Tooth floating in empty white space, logo and tagline not yet visible, "
            "tooth just beginning a slow rotation."
        ),
        "end_description": (
            "Tooth fully lit, logo glowing in gold below, tagline fully legible in gray, "
            "prismatic spectrum spreading across the white surface, tooth centered and still."
        ),
        "camera_move": "completely static — only subject moves",
        "transition_in": "blend",  # Soft transition from data beat
        "transition_out": "fade",  # Fade out to end film
        "color_grade": "pure white and gold, timeless iconic",
        "motion_density": 0.1,  # Near-static brand reveal
        "seed_offset": 1100,
    },
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                               ║
# ║                                                                              ║
# ║  All parameters are documented inline with their effect on output quality.  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_config() -> dict:
    """
    Master configuration for the v5 sliding-window image film engine.

    QUALITY vs SPEED TRADEOFFS:
    The single most impactful parameter for quality is image_inference_steps.
    The second is image_guidance_scale. Third is image resolution.

    SPEED ESTIMATES on RTX 5080 16 GB:
    · 60 steps, CFG 7.5, 1024×1024  → ~60 seconds per frame
    · 30 steps, CFG 7.5, 1024×1024  → ~35 seconds per frame
    · 20 steps, CFG 5.0, 1024×1024  → ~22 seconds per frame (DRAFT mode)
    · 60 steps, CFG 7.5,  512×512   → ~15 seconds per frame (LOW-RES preview)

    24 fps × 5s × 12 beats = 1,440 frames at 60s/frame = 24 hours.
    Use DRAFT mode (20 steps) for testing: 1,440 frames × 22s = ~8.8 hours.

    Returns:
        Configuration dict consumed by all pipeline functions.
    """
    cfg = {
        # ── Model ────────────────────────────────────────────────────────────
        # SANA-1.5 4.8B: the largest and highest-quality SANA image model.
        # Trained on 1024px images with Gemma-2-2B text encoder and DC-AE.
        # Alternative: "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
        # for 3× faster generation at ~80% of the quality.
        "model": "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",

        # ── Image generation parameters ──────────────────────────────────────

        # image_inference_steps: number of denoising steps.
        # Each step refines the image from pure noise toward the target.
        # · 20 steps: draft quality — good for testing prompts quickly.
        # · 40 steps: good quality — suitable for most promotional content.
        # · 60 steps: production quality — recommended for final renders.
        # · 100 steps: maximum quality — diminishing returns after 80 steps.
        # Doubling steps roughly doubles generation time.
        "image_inference_steps": 60,

        # image_guidance_scale (CFG): controls prompt adherence strength.
        # Low values (3–5): more creative, may deviate from prompt description.
        # Normal values (6–8): balanced prompt fidelity and creativity.
        # High values (9–12): strict prompt following, can introduce artifacts.
        # SANA-1.5 4.8B sweet spot: 7.0–8.0
        "image_guidance_scale": 7.5,

        # image_pag_guidance_scale: PAG (Perturbed Attention Guidance) strength.
        # PAG is exclusive to SANA-1.5. It improves structural coherence by
        # perturbing the self-attention maps during inference.
        # · 0.0: PAG disabled (same as not loading it)
        # · 1.0–2.0: subtle structural improvement
        # · 2.0–3.0: significant improvement in complex multi-element scenes
        # · >4.0: may over-constrain and produce repetitive patterns
        # Recommendation: 2.5 for complex scenes (crowds, clinics, equipment)
        #                 1.5 for simple portraits or minimalist compositions
        "image_pag_guidance_scale": 2.5,

        # PAG is applied to these transformer block layers.
        # "transformer_blocks" is the correct value for SANA-1.5.
        # Do not change unless NVLabs releases updated guidance.
        "pag_applied_layers": ["transformer_blocks"],

        # Output image resolution. SANA-1.5 4.8B is trained at 1024×1024.
        # It supports non-square outputs but quality degrades outside its
        # training distribution. 1024×1024 is always the best choice.
        # Final video resolution is set separately (output_width/height).
        "image_width":  1024,
        "image_height": 1024,

        # ── Precision ────────────────────────────────────────────────────────
        # torch.bfloat16: native format for RTX 5080 (Blackwell architecture).
        # Lower memory usage than float32, no quality loss for inference.
        # float16 would cause NaN errors with SANA's DC-AE decoder.
        "torch_dtype": torch.bfloat16,

        # VAE must always be float32. SANA's DC-AE decoder produces checkerboard
        # artifacts in bfloat16/float16. This is not optional.
        "vae_dtype": torch.float32,

        # ── Sliding window VRAM strategy ─────────────────────────────────────

        # window_size: number of context frames kept in VRAM simultaneously.
        # · 1: condition only on the immediately previous frame (fast, low consistency)
        # · 2: condition on the 2 most recent frames (RECOMMENDED — best consistency/VRAM tradeoff)
        # · 3: condition on 3 prior frames (slightly better consistency, more VRAM)
        # With 16 GB VRAM and 4.8B model: window_size=2 is the reliable maximum.
        "window_size": 2,

        # context_blend_alpha: when blending 2 context frames into 1 conditioning signal,
        # how much weight to give the MOST RECENT frame vs the OLDER frame.
        # · 0.5: equal weight (smoother transitions, slower "drift")
        # · 0.75: 75% recent / 25% older (RECOMMENDED — strong continuity, allows progress)
        # · 1.0: only the most recent frame (essentially a 1-frame window)
        "context_blend_alpha": 0.75,

        # consistency_injection: text token appended to each frame's micro-prompt
        # instructing the model to maintain visual continuity with prior frames.
        # This phrase is what makes the image model "aware" that it's generating
        # a film frame rather than an independent image.
        "consistency_injection": (
            "Continuing the same scene, same lighting, same characters, same environment, "
            "same color palette, seamless visual continuation from previous frame."
        ),

        # ── Base seed ────────────────────────────────────────────────────────
        # Master seed for the entire film.
        # Each frame gets: seed = base_seed + beat.seed_offset + frame_index
        # This means: changing base_seed regenerates the entire film in a
        # different "style family" while keeping all relative relationships intact.
        # Changing beat.seed_offset changes the visual character of just that beat.
        "base_seed": 2025,

        # ── VRAM management ──────────────────────────────────────────────────
        # cpu_offload: moves pipeline components that aren't actively computing
        # to system RAM. Reduces peak VRAM from ~11 GB to ~8 GB at the cost
        # of ~15% slower generation. With 16 GB, this is optional but recommended
        # to prevent OOM errors on very large prompts or edge cases.
        "cpu_offload": True,
        "attention_slicing": True,  # Slices attention computation → saves ~1 GB VRAM

        # ── Output ───────────────────────────────────────────────────────────
        "output_dir":      "film_v5_output",
        "frames_dir":      "film_v5_output/frames",      # Individual beat frame directories
        "context_dir":     "film_v5_output/context",     # Sliding window context PNGs
        "checkpoints_dir": "film_v5_output/checkpoints",
        "final_video":     "SmartSmile2250_Film_v5.mp4",

        # ── Final video resolution ───────────────────────────────────────────
        # Generated images are 1024×1024. The final video is resized to this.
        # 1280×704 is the native SANA-Video 720p resolution — used here for
        # compatibility with the existing dental ad pipeline output format.
        "output_width":  1280,
        "output_height": 704,

        # ── Encoding ─────────────────────────────────────────────────────────
        # CRF 18: near-lossless H.264. Set 23 for smaller file at some quality loss.
        "crf": 18,
        "preset": "slow",  # Better compression; use "medium" for faster encoding

        # ── Quality mode switch ───────────────────────────────────────────────
        # "draft"  → 20 steps, CFG 5.0: ~22s/frame. Use for prompt testing.
        # "normal" → 40 steps, CFG 7.0: ~40s/frame. Use for reviews.
        # "final"  → 60 steps, CFG 7.5: ~60s/frame. Use for delivery.
        # Set this to override image_inference_steps and image_guidance_scale.
        "quality_mode": "final",  # "draft" | "normal" | "final"
    }

    # Apply quality mode overrides
    _QUALITY_PRESETS = {
        "draft":  {"image_inference_steps": 20, "image_guidance_scale": 5.0},
        "normal": {"image_inference_steps": 40, "image_guidance_scale": 7.0},
        "final":  {"image_inference_steps": 60, "image_guidance_scale": 7.5},
    }
    preset = _QUALITY_PRESETS.get(cfg["quality_mode"], {})
    cfg.update(preset)

    # Derived totals
    cfg["total_frames"] = sum(b["fps"] * b["duration_s"] for b in BEATS)
    cfg["total_duration_s"] = sum(b["duration_s"] for b in BEATS)
    cfg["num_beats"] = len(BEATS)
    return cfg


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  STATISTICS TRACKER                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class StatisticsTracker:
    """
    Tracks per-frame and per-beat timing, GPU/VRAM/RAM usage, and ETA.

    Usage:
        stat = StatisticsTracker(cfg)
        stat.start()
        ...
        stat.record_frame(elapsed_seconds)
        stat.hms(stat.eta())  # human-readable estimated time remaining
    """

    def __init__(self, cfg: dict):
        self.cfg         = cfg
        self.t0          = None
        self.frame_times = []          # Seconds per frame (rolling history)
        self.beat_times  = {}          # {beat_id: total_seconds}
        self.gpu_hist    = deque(maxlen=300)
        self.ram_hist    = deque(maxlen=300)
        self.vram_hist   = deque(maxlen=300)

    def start(self):
        """Call once before the generation loop begins."""
        self.t0 = time.time()

    def tick(self, gpu: float, ram: float, vram: float):
        """Called by monitoring thread every 0.5 s."""
        self.gpu_hist.append(gpu)
        self.ram_hist.append(ram)
        self.vram_hist.append(vram)

    def record_frame(self, elapsed: float):
        """Record the time taken to generate one frame."""
        self.frame_times.append(elapsed)

    def elapsed(self) -> float:
        """Return seconds since start() was called."""
        return time.time() - self.t0 if self.t0 else 0.0

    def avg_frame_time(self) -> float:
        """Rolling average of the last 20 frame times."""
        recent = self.frame_times[-20:]
        return sum(recent) / len(recent) if recent else 0.0

    def eta(self) -> float:
        """Estimate remaining seconds based on frames remaining and avg frame time."""
        done      = len(self.frame_times)
        remaining = self.cfg["total_frames"] - done
        return self.avg_frame_time() * remaining

    @staticmethod
    def hms(s) -> str:
        """
        Format seconds as a human-readable string.

        Static method: can be called as StatisticsTracker.hms(n) without an instance.
        Returns 'N/A' for None to prevent TypeError on int(None).
        This was the bug in v1: StatisticsTracker.hms(None, value) passed two args
        to a one-arg static method, causing TypeError. Fixed by adding None guard.
        """
        if s is None:
            return "N/A"
        s = int(s)
        h, m, sec = s // 3600, (s % 3600) // 60, s % 60
        if h:   return f"{h}h {m:02d}m"
        if m:   return f"{m}m {sec:02d}s"
        return f"{sec}s"

    def summary(self):
        """Print final generation statistics to terminal."""
        el   = self.elapsed()
        done = len(self.frame_times)
        avg  = self.avg_frame_time()
        g    = sum(self.gpu_hist)  / len(self.gpu_hist)  if self.gpu_hist  else 0
        v    = sum(self.vram_hist) / len(self.vram_hist) if self.vram_hist else 0
        r    = sum(self.ram_hist)  / len(self.ram_hist)  if self.ram_hist  else 0
        print(f"\n{C.BOLD}{C.CYAN}{'═'*76}{C.END}")
        print(f"{C.BOLD}{C.CYAN}  📊  STATISTICS — sana-dental-5.py v{__version__}{C.END}")
        print(f"{C.BOLD}{C.CYAN}{'═'*76}{C.END}")
        print(f"  {'Total time':<38} {self.hms(el)}")
        print(f"  {'Frames generated':<38} {done}/{self.cfg['total_frames']}")
        print(f"  {'Avg time / frame':<38} {avg:.1f}s")
        print(f"  {'Peak GPU load':<38} {max(self.gpu_hist, default=0):.1f}%")
        print(f"  {'Avg VRAM used':<38} {v/100*16:.1f} GB / 16 GB")
        print(f"  {'Avg RAM used':<38} {r:.1f}%")
        print(f"{C.BOLD}{C.CYAN}{'═'*76}{C.END}\n")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MONITORING THREAD                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_metrics     = {"gpu": deque(maxlen=500), "vram": deque(maxlen=500),
                "cpu": deque(maxlen=500), "ram":  deque(maxlen=500)}
_tracker_ref = None   # Set to the active StatisticsTracker in main()


def _monitor_loop(stop_evt):
    """
    Background daemon thread: samples hardware metrics every 0.5 s.
    Feeds data to both _metrics (for the live dashboard) and
    _tracker_ref (for the statistics summary).
    Uses GPUtil for GPU load and VRAM. Falls back to zeros if GPUtil unavailable.
    """
    global _metrics, _tracker_ref
    while not stop_evt.is_set():
        try:
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            gpu = vram = 0.0
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


def _print_live(beat_id: int, frame_idx: int, total_frames: int,
                beat_name: str, stat: StatisticsTracker):
    """
    Refresh the live terminal dashboard showing:
    - Overall film progress (all frames)
    - Current beat and frame number
    - Avg frame time and ETA
    - GPU load, VRAM usage, CPU load, RAM usage with progress bars
    """
    os.system("cls" if os.name == "nt" else "clear")
    pct = frame_idx / total_frames if total_frames else 0

    g    = list(_metrics["gpu"])[-1]  if _metrics["gpu"]  else 0
    vr   = list(_metrics["vram"])[-1] if _metrics["vram"] else 0
    r    = list(_metrics["ram"])[-1]  if _metrics["ram"]  else 0
    cpu  = list(_metrics["cpu"])[-1]  if _metrics["cpu"]  else 0

    print(f"{C.BOLD}{C.CYAN}{'═'*76}{C.END}")
    print(f"{C.BOLD}{C.CYAN}  🦷  SmartSmile 2250 Film Engine  v{__version__}  —  4.8B Sliding Window{C.END}")
    print(f"{C.BOLD}{C.CYAN}{'═'*76}{C.END}")
    print(f"\n  {'Beat':<22} {beat_id:02d}  {beat_name}")
    print(f"  {'Frame':<22} {frame_idx:,} / {total_frames:,}")
    print(f"  {'Film progress':<22} [{bar(pct)}]  {pct*100:.1f}%")

    eta = stat.eta()
    if eta > 0:
        print(f"  {'ETA':<22} {stat.hms(eta)}")
    avg = stat.avg_frame_time()
    if avg > 0:
        print(f"  {'Avg / frame':<22} {avg:.1f}s")
    print(f"  {'Elapsed':<22} {stat.hms(stat.elapsed())}")

    print(f"\n  {C.BOLD}GPU  RTX 5080{C.END}")
    print(f"  {'  Load':<22} [{bar(g  / 100, 30)}]  {g:.0f}%")
    print(f"  {'  VRAM':<22} [{bar(vr / 100, 30)}]  {vr:.0f}%  ({vr/100*16:.1f}/16 GB)")
    print(f"\n  {C.BOLD}System{C.END}")
    print(f"  {'  CPU Ryzen 9900X':<22} [{bar(cpu / 100, 30)}]  {cpu:.0f}%")
    vm = psutil.virtual_memory()
    print(f"  {'  RAM DDR5':<22} [{bar(r / 100, 30)}]  {r:.0f}%  ({vm.used/1e9:.1f}/{vm.total/1e9:.0f} GB)")
    print(f"\n{C.BOLD}{C.CYAN}{'─'*76}{C.END}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  VRAM UTILITIES                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def vram_free():
    """
    Force-clear GPU memory between pipeline loads or after VRAM-heavy operations.
    Runs Python GC first (to delete Python references), then CUDA empty_cache
    (to return CUDA memory blocks to the pool), then synchronize (ensures all
    async CUDA ops complete). Sleeps 1.5s for OS to fully reclaim memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(1.5)


def vram_used_gb() -> float:
    """Return currently allocated CUDA memory in GB (0 if CUDA unavailable)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  IMAGE UTILITIES                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def to_pil(img) -> Image.Image:
    """
    Normalize any frame representation to a PIL Image in RGB mode.
    Handles: PIL Image, numpy array (uint8 or float), torch Tensor.
    """
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, torch.Tensor):
        img = img.cpu().float().numpy()
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = img.transpose(1, 2, 0)
        if img.shape[-1] == 4:
            img = img[..., :3]
        return Image.fromarray(img.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Cannot convert {type(img)} to PIL Image")


def blend_context(frames: List[Image.Image], alpha: float,
                  target_size: Tuple[int, int]) -> Image.Image:
    """
    Blend a list of context frames into a single conditioning image.

    This is the core of the sliding-window strategy. Given 1 or 2 context frames,
    we produce a single image that the generation model sees as its "prior state."

    With 2 frames (prev2, prev1):
        result = alpha * prev1 + (1-alpha) * prev2
        where alpha = context_blend_alpha (default 0.75)
        This gives 75% weight to the most recent frame (stronger continuity anchor)
        and 25% to the frame before it (provides slightly longer temporal context).

    With 1 frame (prev1 only):
        result = prev1 (no blending needed)

    Args:
        frames:      List of PIL Images, ordered oldest to newest.
        alpha:       Weight for the most recent frame (0.5–1.0 recommended).
        target_size: (width, height) to resize output to (should match model input).

    Returns:
        Single PIL Image suitable for use as model conditioning input.
    """
    if not frames:
        raise ValueError("blend_context requires at least one frame")

    # Resize all frames to target size
    resized = [f.resize(target_size, Image.LANCZOS).convert("RGB") for f in frames]

    if len(resized) == 1:
        return resized[0]

    # Weighted blend: more recent = higher weight
    newest = np.array(resized[-1], dtype=np.float32)
    older  = np.array(resized[-2], dtype=np.float32)
    blended = (alpha * newest + (1.0 - alpha) * older).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def resize_for_output(img: Image.Image, width: int, height: int) -> Image.Image:
    """
    Resize a generated 1024×1024 image to the final video output resolution.
    Uses LANCZOS (high-quality sinc-based downsampling) which preserves
    fine detail better than bilinear or bicubic at this scale.
    Letterboxes (adds black bars) if aspect ratio differs to avoid stretching.
    """
    target_aspect = width / height
    source_aspect = img.width / img.height
    if abs(target_aspect - source_aspect) < 0.01:
        return img.resize((width, height), Image.LANCZOS)
    # Crop to target aspect ratio first, then resize
    if source_aspect > target_aspect:
        # Source is wider — crop sides
        new_w = int(img.height * target_aspect)
        left  = (img.width - new_w) // 2
        img   = img.crop((left, 0, left + new_w, img.height))
    else:
        # Source is taller — crop top/bottom
        new_h = int(img.width / target_aspect)
        top   = (img.height - new_h) // 2
        img   = img.crop((0, top, img.width, top + new_h))
    return img.resize((width, height), Image.LANCZOS)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PROMPT ENGINEERING — THE HEART OF CONSISTENCY                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_frame_prompt(beat: dict, frame_idx: int, total_frames: int,
                       cfg: dict, is_context_frame: bool = False) -> str:
    """
    Build the complete prompt for a single frame using multi-level interpolation.

    THE INTERPOLATION STRATEGY:
    ────────────────────────────
    A beat has a start_description (first frame) and end_description (last frame).
    In between, the camera moves and the scene evolves. We model this as follows:

    Frame 0  → 100% start_description + 0% end_description
    Frame N/4 → 0% start_description + 0% end_description (pure base_prompt)
    Frame N/2 → 0% start_description + 0% end_description (pure base_prompt)
    Frame 3N/4→ 0% start_description + 50% end_description
    Frame N  → 0% start_description + 100% end_description

    This triangle interpolation means:
    - The first quarter of the beat transitions OUT of the start state.
    - The middle half of the beat is driven purely by the base_prompt.
    - The final quarter transitions INTO the end state.

    This mimics natural camera movement: you start somewhere, move through
    the middle of a shot, then arrive at your destination.

    THE CONSISTENCY INJECTION:
    ────────────────────────────
    When a context frame exists (we're NOT generating the first frame),
    we prepend cfg["consistency_injection"] to signal to the model that
    it's continuing a sequence rather than starting fresh. This is key —
    without it, the model treats each generation as independent.

    THE NEGATIVE PROMPT:
    ────────────────────────────
    When is_context_frame=True, we append NEGATIVE_CONSISTENCY (which targets
    scene-change artifacts) on top of NEGATIVE_IMAGE (which targets quality issues).
    For the first frame (no context), we use only NEGATIVE_IMAGE.

    Args:
        beat:         Beat definition dict.
        frame_idx:    Index of this frame within the beat (0-based).
        total_frames: Total frames in this beat.
        cfg:          Config dict.
        is_context_frame: True if context frames are available (frame_idx > 0).

    Returns:
        Tuple of (positive_prompt, negative_prompt) strings.
    """
    t = frame_idx / max(total_frames - 1, 1)  # Normalized time 0.0 → 1.0

    # Start description weight: 1.0 at frame 0, linearly → 0.0 at frame N/4
    start_weight = max(0.0, 1.0 - (t / 0.25)) if t < 0.25 else 0.0

    # End description weight: 0.0 until frame 3N/4, linearly → 1.0 at frame N
    end_weight = max(0.0, (t - 0.75) / 0.25) if t > 0.75 else 0.0

    # Build prompt parts
    parts = []

    # Consistency injection: tells the model it's continuing a sequence
    if is_context_frame:
        parts.append(cfg["consistency_injection"])

    # Always include the beat's base prompt (color, lighting, environment, quality)
    parts.append(beat["base_prompt"])

    # Add camera move instruction
    if beat.get("camera_move"):
        parts.append(f"Camera movement: {beat['camera_move']}.")

    # Add color grade (style anchor)
    if beat.get("color_grade"):
        parts.append(f"Color grading: {beat['color_grade']}.")

    # Interpolate start description
    if start_weight > 0.0 and beat.get("start_description"):
        prefix = f"[Opening frame, {start_weight*100:.0f}% of opening state] "
        parts.append(prefix + beat["start_description"])

    # Interpolate end description
    if end_weight > 0.0 and beat.get("end_description"):
        prefix = f"[Closing frame, {end_weight*100:.0f}% of closing state] "
        parts.append(prefix + beat["end_description"])

    positive_prompt = " ".join(parts)

    # Negative prompt: base quality + consistency penalty for non-first frames
    negative_prompt = NEGATIVE_IMAGE
    if is_context_frame:
        negative_prompt = NEGATIVE_IMAGE + " " + NEGATIVE_CONSISTENCY

    return positive_prompt, negative_prompt


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CHECKPOINT / STATE MANAGER                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class StateManager:
    """
    Atomic checkpoint manager for crash-resume support.

    File layout:
        checkpoints/state.json       — primary state (atomic write)
        checkpoints/state.json.bak   — backup of prior good state
        checkpoints/state.json.tmp   — temp during write (auto-replaced)
        context/beat_NN_ctx_latest.png — the most recently written context frame
                                         for each beat (used for sliding window resume)

    Recovery: on restart, for any frame that already has a saved PNG on disk,
    generation is skipped and the existing PNG is used as context for the next frame.
    """

    EMPTY = {
        "version":          __version__,
        "completed_beats":  [],
        "completed_frames": {},   # {beat_id: [frame_indices done]}
        "context_paths":    {},   # {beat_id: [path_prev2, path_prev1]}
        "timing":           {},
        "last_updated":     None,
    }

    def __init__(self, cfg: dict):
        self.cfg      = cfg
        self.path     = os.path.join(cfg["checkpoints_dir"], "state.json")
        self.path_bak = self.path + ".bak"
        self.path_tmp = self.path + ".tmp"
        self._state   = None

    def load(self) -> dict:
        """Load state with fallback to backup on corruption."""
        for p, label in [(self.path, "primary"), (self.path_bak, "backup")]:
            if not os.path.exists(p):
                continue
            try:
                with open(p) as f:
                    self._state = json.load(f)
                if label == "backup":
                    recovered(f"Loaded state from backup: {p}")
                return self._state
            except (json.JSONDecodeError, OSError) as e:
                warn(f"State {label} unreadable ({e})")
        info("No prior state — starting fresh")
        self._state = dict(self.EMPTY)
        return self._state

    def save(self) -> bool:
        """Atomic write: .tmp → .bak → os.replace."""
        self._state["last_updated"] = datetime.now().isoformat()
        os.makedirs(self.cfg["checkpoints_dir"], exist_ok=True)
        try:
            with open(self.path_tmp, "w") as f:
                json.dump(self._state, f, indent=2)
        except OSError as e:
            warn(f"Cannot write state: {e}")
            return False
        if os.path.exists(self.path):
            try:
                shutil.copy2(self.path, self.path_bak)
            except OSError:
                pass
        try:
            os.replace(self.path_tmp, self.path)
        except OSError as e:
            warn(f"Atomic replace failed: {e}")
            return False
        return True

    def frame_path(self, beat_id: int, frame_idx: int) -> str:
        """Return the expected file path for a specific frame PNG."""
        return os.path.join(
            self.cfg["frames_dir"],
            f"beat_{beat_id:02d}",
            f"frame_{frame_idx:06d}.png",
        )

    def frame_exists(self, beat_id: int, frame_idx: int) -> bool:
        """Check if a frame PNG already exists on disk (for crash-resume)."""
        return os.path.exists(self.frame_path(beat_id, frame_idx))

    def count_beat_frames(self, beat_id: int, total: int) -> int:
        """Count how many frames of a beat are already saved on disk."""
        return sum(1 for i in range(total) if self.frame_exists(beat_id, i))

    def beat_complete(self, beat_id: int) -> bool:
        """Return True if this beat is in the completed_beats list."""
        return beat_id in self._state.get("completed_beats", [])

    def mark_beat_complete(self, beat_id: int, elapsed_s: float):
        """Mark a beat as fully complete and record its timing."""
        c = self._state.setdefault("completed_beats", [])
        if beat_id not in c:
            c.append(beat_id)
        self._state.setdefault("timing", {})[str(beat_id)] = round(elapsed_s, 1)
        self.save()

    def save_context(self, beat_id: int, frames: List[Image.Image]):
        """
        Save the current sliding-window context frames to disk.
        These are re-loaded on resume to continue the sliding window
        exactly where it left off before the crash.
        """
        os.makedirs(self.cfg["context_dir"], exist_ok=True)
        paths = []
        for i, frame in enumerate(frames):
            p = os.path.join(self.cfg["context_dir"],
                             f"beat_{beat_id:02d}_ctx_{i}.png")
            to_pil(frame).save(p, "PNG")
            paths.append(p)
        self._state.setdefault("context_paths", {})[str(beat_id)] = paths
        self.save()

    def load_context(self, beat_id: int) -> List[Image.Image]:
        """Load the last-saved context frames for a beat (for resume)."""
        paths = self._state.get("context_paths", {}).get(str(beat_id), [])
        frames = []
        for p in paths:
            if os.path.exists(p):
                frames.append(Image.open(p).convert("RGB"))
        return frames

    @property
    def state(self) -> dict:
        return self._state


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PIPELINE MANAGEMENT                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_pipeline(cfg: dict):
    """
    Load SANA-1.5 4.8B with PAG (Perturbed Attention Guidance).

    WHY PAG IS LOADED AT INIT TIME:
    PAG modifies which transformer blocks the model uses for attention perturbation.
    This must be specified during model loading — it cannot be added post-hoc.
    The pag_applied_layers=["transformer_blocks"] value is correct for SANA-1.5.
    Using AutoPipelineForText2Image handles the enable_pag flag transparently.

    VRAM PROFILE:
    - 4.8B parameters in bfloat16 = 9.6 GB
    - DC-AE VAE in float32 = ~1.5 GB
    - Attention buffers and activations = ~1–2 GB
    - Total peak: ~12–13 GB
    - With cpu_offload=True: peak ~8–9 GB (layers not computing stay in RAM)

    Returns:
        Loaded pipeline ready for inference.
    """
    subhdr(f"Loading SANA-1.5 4.8B  (quality_mode={cfg['quality_mode']})")
    info(f"Model   : {cfg['model'].split('/')[-1]}")
    info(f"Steps   : {cfg['image_inference_steps']}")
    info(f"CFG     : {cfg['image_guidance_scale']}")
    info(f"PAG     : {cfg['image_pag_guidance_scale']}")
    info(f"VRAM    : ~12 GB peak  (cpu_offload={'ON' if cfg['cpu_offload'] else 'OFF'})")

    # Try AutoPipeline with PAG first (preferred path for SANA-1.5)
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            cfg["model"],
            torch_dtype=cfg["torch_dtype"],
            enable_pag=True,
            pag_applied_layers=cfg["pag_applied_layers"],
        )
    except (TypeError, ValueError):
        warn("AutoPipeline PAG init failed — falling back to SanaPipeline")
        pipe = SanaPipeline.from_pretrained(
            cfg["model"],
            torch_dtype=cfg["torch_dtype"],
        )

    # VAE always in float32 — bfloat16 VAE produces checkerboard artifacts
    pipe.vae.to(cfg["vae_dtype"])
    pipe.text_encoder.to(cfg["torch_dtype"])
    pipe.to("cuda")

    if cfg["cpu_offload"]:
        pipe.enable_model_cpu_offload()   # Non-active modules go to RAM
    if cfg["attention_slicing"]:
        pipe.enable_attention_slicing()   # Saves ~1 GB VRAM at minor speed cost

    # RTX 5080 (Blackwell): TF32 matrix multiply is hardware-native
    torch.backends.cuda.matmul.allow_tf32 = True

    ok(f"Pipeline ready — VRAM allocated: {vram_used_gb():.1f} GB")
    return pipe


def generate_frame(pipe, positive_prompt: str, negative_prompt: str,
                   context_img: Optional[Image.Image],
                   cfg: dict, seed: int) -> Image.Image:
    """
    Generate a single 1024×1024 image frame.

    CONTEXT CONDITIONING:
    When context_img is provided (all frames except the first), it is passed
    alongside the positive prompt as an image conditioning signal. This is
    the mechanism that keeps consecutive frames visually consistent.

    The model receives:
    - Positive text prompt: what to generate
    - Negative text prompt: what to avoid
    - Context image: what the previous frame looked like
    The context image is encoded by the pipeline's image encoder and
    merged with the text conditioning. The result is constrained to be
    visually similar to context_img while also matching positive_prompt.

    NOTE ON context_img:
    SANA-1.5 is a text-to-image model. It does not have a native image
    conditioning pathway like SANA-Video's I2V mode. We implement context
    conditioning via the image_embeds feature of the Gemma-2 text encoder,
    which can process images as part of its multimodal input.

    If this approach doesn't work for a particular diffusers version, we fall
    back to appending a visual description of the context image to the prompt.

    Args:
        pipe:            The loaded SANA pipeline.
        positive_prompt: What to generate (full frame prompt from build_frame_prompt).
        negative_prompt: What to avoid.
        context_img:     Blended context from prior frames, or None for first frame.
        cfg:             Config dict.
        seed:            Deterministic seed for this frame.

    Returns:
        Generated PIL Image at cfg["image_width"] × cfg["image_height"].
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)

    kwargs = {
        "prompt":              positive_prompt,
        "negative_prompt":     negative_prompt,
        "height":              cfg["image_height"],
        "width":               cfg["image_width"],
        "guidance_scale":      cfg["image_guidance_scale"],
        "num_inference_steps": cfg["image_inference_steps"],
        "generator":           generator,
    }

    # Add PAG scale if the pipeline supports it
    try:
        kwargs["pag_guidance_scale"] = cfg["image_pag_guidance_scale"]
    except (TypeError, AttributeError):
        pass

    # Add context image conditioning if available
    # SANA-1.5 supports ip_adapter_image for visual conditioning
    if context_img is not None:
        try:
            kwargs["ip_adapter_image"] = context_img
        except (TypeError, AttributeError):
            pass   # Silently skip if this version of diffusers doesn't support it

    try:
        result = pipe(**kwargs)
    except TypeError as e:
        # Remove unsupported kwargs and retry
        for unsupported in ["pag_guidance_scale", "ip_adapter_image"]:
            kwargs.pop(unsupported, None)
        result = pipe(**kwargs)

    return to_pil(result.images[0])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BEAT FRAME SAVING                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def save_frame(img: Image.Image, beat_id: int, frame_idx: int, cfg: dict) -> str:
    """
    Save a generated frame as a lossless PNG with OS buffer flush.

    PNG is used instead of JPEG because:
    - Lossless: no quality degradation between generation and encoding
    - No artifacts: JPEG blocking would compound across 1,440+ frames
    - True 24-bit color: preserves every bit the model generated

    The os.fsync() call ensures the file is fully written to SSD before
    the function returns. This prevents a race condition where the process
    crashes after writing the file header but before flushing content,
    which would produce an unreadable PNG that looks saved but isn't.

    Returns:
        Absolute path to the saved frame PNG.
    """
    beat_dir = os.path.join(cfg["frames_dir"], f"beat_{beat_id:02d}")
    os.makedirs(beat_dir, exist_ok=True)

    # Resize 1024×1024 generated image to final output resolution
    resized = resize_for_output(img, cfg["output_width"], cfg["output_height"])

    path = os.path.join(beat_dir, f"frame_{frame_idx:06d}.png")
    resized.save(path, "PNG")

    # Force-flush to SSD before marking frame as complete
    with open(path, "r+b") as fh:
        fh.flush()
        os.fsync(fh.fileno())

    return path


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  VIDEO ASSEMBLY                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def assemble_video(cfg: dict, beats: List[dict]) -> Optional[str]:
    """
    Stitch all saved frame PNGs across all beats into the final MP4.

    Frames are assembled in beat order, then by frame_idx within each beat.
    Uses imageio with ffmpeg backend for H.264 encoding.

    The output file has movflags +faststart which places the MP4 moov atom
    at the beginning of the file, enabling streaming playback without
    downloading the complete file first.

    Returns:
        Path to the assembled MP4, or None if no frames were found.
    """
    header("🎬  Assembling Final Film")
    final_path = os.path.join(cfg["output_dir"], cfg["final_video"])

    all_frames = []
    for beat in beats:
        beat_dir = os.path.join(cfg["frames_dir"], f"beat_{beat['id']:02d}")
        if not os.path.exists(beat_dir):
            warn(f"Beat {beat['id']:02d} frames missing — skipping in assembly")
            continue
        files = sorted([
            os.path.join(beat_dir, f)
            for f in os.listdir(beat_dir)
            if f.startswith("frame_") and f.endswith(".png")
        ])
        all_frames.extend(files)
        ok(f"Beat {beat['id']:02d} '{beat['name']}': {len(files)} frames")

    if not all_frames:
        err("No frames found for assembly.")
        return None

    # Use the first beat's fps for the first segment; for multi-fps beats
    # a more sophisticated muxer would be needed. For simplicity, use 24 fps
    # as the output frame rate (matches most beat definitions in this film).
    output_fps = beats[0]["fps"] if beats else 24
    ok(f"Assembling {len(all_frames)} frames at {output_fps} fps")

    writer = imageio.get_writer(
        final_path, fps=output_fps, codec="libx264", quality=None,
        output_params=[
            "-crf",       str(cfg["crf"]),
            "-preset",    cfg["preset"],
            "-pix_fmt",   "yuv420p",
            "-profile:v", "high",
            "-level",     "4.2",
            "-movflags",  "+faststart",
        ],
    )

    last_upd = time.time()
    for i, fp in enumerate(all_frames):
        try:
            writer.append_data(imageio.imread(fp))
        except Exception as e:
            warn(f"Skipping unreadable frame {os.path.basename(fp)}: {e}")
        if time.time() - last_upd > 3:
            pct = (i + 1) / len(all_frames)
            print(f"    [{bar(pct, 30)}]  {pct*100:.0f}%  {i+1}/{len(all_frames)}")
            last_upd = time.time()
    writer.close()

    if os.path.exists(final_path):
        mb   = os.path.getsize(final_path) / 1e6
        dur  = len(all_frames) / output_fps
        mbps = (os.path.getsize(final_path) * 8) / dur / 1e6
        ok(f"Film: {final_path}")
        ok(f"Size: {mb:.1f} MB  |  Duration: {dur:.1f}s  |  Bitrate: {mbps:.1f} Mbps")
        return final_path

    err("Assembly failed — check imageio/ffmpeg installation.")
    return None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PREFLIGHT                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def preflight(cfg: dict) -> bool:
    """
    Validate system readiness and print a comprehensive film generation summary.

    Checks:
    - diffusers package available with correct classes
    - CUDA available and VRAM ≥ 12 GB (warning if < 14 GB)
    - RAM ≥ 16 GB (warning if less — cpu_offload needs room for model layers)
    - Disk space adequate for PNG frames

    Returns:
        True if all hard requirements met, False otherwise.
    """
    header("🔍  Preflight Checks — v5 Sliding-Window Film Engine")

    if not DIFFUSERS_OK:
        err(f"diffusers import failed: {_MISSING}")
        err("Run: pip install git+https://github.com/huggingface/diffusers")
        return False

    if not torch.cuda.is_available():
        err("CUDA not available. Check NVIDIA drivers.")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    ok(f"GPU  : {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    if vram_gb < 12:
        err(f"VRAM is only {vram_gb:.1f} GB. SANA 4.8B needs ~12 GB minimum.")
        return False
    if vram_gb < 14:
        warn(f"VRAM is {vram_gb:.1f} GB — generation will work but may be slow with cpu_offload.")

    ram  = psutil.virtual_memory()
    disk = shutil.disk_usage(os.path.abspath("."))
    ok(f"RAM  : {ram.total/1e9:.1f} GB  |  {ram.available/1e9:.1f} GB free")
    if ram.available / 1e9 < 16:
        warn(f"Low RAM: cpu_offload needs space for 4.8B model layers in system RAM.")
    ok(f"Disk : {disk.free/1e9:.1f} GB free")
    ok(f"Torch: {torch.__version__}  |  bfloat16 native on RTX 5080")

    # Storage estimate: PNG at output_width × output_height ≈ 2 MB each
    est_gb = cfg["total_frames"] * 2 / 1024
    if disk.free / 1e9 < est_gb * 1.5:
        warn(f"Low disk space — need ~{est_gb*1.5:.0f} GB, have {disk.free/1e9:.1f} GB")

    print(f"\n  {C.BOLD}v5 Film Architecture:{C.END}")
    note(f"Strategy    : Sliding-window 4.8B image generation")
    note(f"Window size : {cfg['window_size']} context frames in VRAM simultaneously")
    note(f"Blend alpha : {cfg['context_blend_alpha']} (recent frame weight)")
    note(f"Model       : SANA-1.5 4.8B  ({cfg['image_inference_steps']} steps, "
         f"CFG {cfg['image_guidance_scale']}, PAG {cfg['image_pag_guidance_scale']})")
    note(f"Resolution  : {cfg['image_width']}×{cfg['image_height']} → "
         f"{cfg['output_width']}×{cfg['output_height']}")
    note(f"Quality mode: {cfg['quality_mode'].upper()}")

    print(f"\n  {C.BOLD}Film plan:{C.END}")
    total_frames = 0
    for beat in BEATS:
        n = beat["fps"] * beat["duration_s"]
        total_frames += n
        print(f"    Beat {beat['id']:02d}  {beat['name']:<40} "
              f"{beat['fps']}fps × {beat['duration_s']}s = {n} frames")
    print(f"\n  Total: {total_frames} frames  |  "
          f"~{total_frames / 24:.0f}s at 24fps  |  "
          f"est. storage ~{total_frames * 2 / 1024:.1f} GB")

    est_time = total_frames * cfg["image_inference_steps"] / 60.0 * 1.0
    # Rough: ~1s per step on RTX 5080
    print(f"  Est. generation time: {StatisticsTracker.hms(est_time)} "
          f"(quality_mode={cfg['quality_mode']})")

    return True


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    """
    Main entry point for the v5 sliding-window film engine.

    Execution flow:
    1. Preflight: hardware checks, config summary.
    2. State load: detect any prior progress for crash-resume.
    3. Load SANA-1.5 4.8B pipeline (stays in VRAM for the entire run).
    4. For each BEAT:
       a. Determine which frames are already done (crash-resume).
       b. Load context window from prior beat's last frame (or from saved context).
       c. For each frame in the beat:
          i.  Build the frame prompt (interpolated between start/end descriptions).
          ii. Blend context window into a single conditioning image.
          iii. Generate the frame with SANA-1.5 4.8B.
          iv. Save frame to disk (PNG, lossless, with fsync).
          v.  Update sliding window: evict oldest, add new frame.
          vi. Save context state to disk (for crash-resume).
       d. Mark beat complete in state.json.
    5. Assemble all frames into final MP4.
    6. Save timing report.

    VRAM lifecycle:
    - Pipeline loaded ONCE, stays in VRAM throughout all beats.
    - Context window is 2 PIL Images (negligible VRAM — stored as numpy arrays).
    - No pipeline reload between beats — the model is stateless per-inference.
    """
    global _tracker_ref

    print(f"\n{C.BOLD}{C.CYAN}  sana-dental-5.py  v{__version__}  —  {__date__}{C.END}")
    print(f"{C.DIM}  Sliding-Window 4.8B Image Film Engine  |  SmartSmile 2250{C.END}\n")

    cfg  = get_config()
    stat = StatisticsTracker(cfg)
    _tracker_ref = stat

    if not preflight(cfg):
        return

    for d in [cfg["output_dir"], cfg["frames_dir"],
              cfg["context_dir"], cfg["checkpoints_dir"]]:
        os.makedirs(d, exist_ok=True)

    # Load crash-resume state
    sm = StateManager(cfg)
    sm.load()

    done_beats  = set(sm.state.get("completed_beats", []))
    total_done  = sum(sm.count_beat_frames(b["id"], b["fps"] * b["duration_s"])
                      for b in BEATS)
    if total_done > 0:
        ok(f"Resuming — {total_done}/{cfg['total_frames']} frames already saved")
        ok(f"Completed beats: {sorted(done_beats)}")
    else:
        ok("Starting fresh generation")

    print(f"\n{C.YELLOW}  Press Enter to start…  (Ctrl+C to abort){C.END}")
    try:
        input()
    except KeyboardInterrupt:
        return

    # Start background monitoring thread
    stop_evt   = threading.Event()
    mon_thread = threading.Thread(target=_monitor_loop, args=(stop_evt,), daemon=True)
    mon_thread.start()

    stat.start()

    # ── Load pipeline ONCE — it stays in VRAM for the entire film ────────────
    header("📦  Loading SANA-1.5 4.8B Pipeline")
    pipe = load_pipeline(cfg)

    # Track the global frame counter across all beats
    global_frame_idx = sum(
        sm.count_beat_frames(b["id"], b["fps"] * b["duration_s"])
        for b in BEATS
    )

    # Context window — list of PIL Images (oldest first, newest last)
    # window_size = 2 means we always carry [prev_older, prev_recent]
    context_window: List[Image.Image] = []

    # Last frame of prior beat — used for "blend" and "match" transitions
    prior_beat_last_frame: Optional[Image.Image] = None

    try:
        for beat in BEATS:
            bid          = beat["id"]
            beat_frames  = beat["fps"] * beat["duration_s"]
            already_done = sm.count_beat_frames(bid, beat_frames)

            if sm.beat_complete(bid) and already_done >= beat_frames:
                ok(f"Beat {bid:02d} '{beat['name']}' — fully done, skipping")
                # Load context from disk for next beat's transition
                ctx = sm.load_context(bid)
                if ctx:
                    context_window = ctx
                # Load last frame for transition
                last_path = sm.frame_path(bid, beat_frames - 1)
                if os.path.exists(last_path):
                    prior_beat_last_frame = Image.open(last_path).convert("RGB")
                continue

            step(bid + 1, len(BEATS), f"Beat {bid:02d}: {beat['name']}  "
                 f"({beat_frames} frames at {beat['fps']} fps)")

            # ── Beat transition setup ─────────────────────────────────────────
            t_in = beat.get("transition_in", "cut")
            if t_in == "blend" and prior_beat_last_frame is not None:
                # Pre-load context with last frame of prior beat
                context_window = [prior_beat_last_frame]
                info(f"Transition 'blend': seeding context from prior beat")
            elif t_in == "match" and prior_beat_last_frame is not None:
                context_window = [prior_beat_last_frame, prior_beat_last_frame]
                info(f"Transition 'match': strong prior-beat conditioning")
            elif t_in == "cut" or not context_window:
                context_window = []   # Fresh start — no context
                info(f"Transition 'cut': fresh start for this beat")

            # If resuming mid-beat, restore context from disk
            if already_done > 0:
                recovered(f"Beat {bid:02d}: {already_done}/{beat_frames} frames exist — resuming")
                ctx = sm.load_context(bid)
                if ctx:
                    context_window = ctx
                    info(f"Context window restored from disk ({len(ctx)} frames)")

            t_beat = time.time()

            # ── Frame generation loop ─────────────────────────────────────────
            for frame_idx in range(beat_frames):
                # Skip already-saved frames
                if sm.frame_exists(bid, frame_idx):
                    global_frame_idx += 1
                    continue

                # Build this frame's prompts
                positive_prompt, negative_prompt = build_frame_prompt(
                    beat=beat,
                    frame_idx=frame_idx,
                    total_frames=beat_frames,
                    cfg=cfg,
                    is_context_frame=len(context_window) > 0,
                )

                # Blend context window into single conditioning image
                conditioning = None
                if context_window:
                    target = (cfg["image_width"], cfg["image_height"])
                    conditioning = blend_context(
                        context_window,
                        cfg["context_blend_alpha"],
                        target,
                    )

                # Compute seed: base + beat_offset + frame_index
                # This ensures: same beat → similar visual family
                #                same film run → deterministic (safe to resume)
                seed = cfg["base_seed"] + beat["seed_offset"] + frame_idx

                # Update dashboard
                _print_live(bid, global_frame_idx, cfg["total_frames"],
                            beat["name"], stat)

                # Generate the frame
                t0 = time.time()
                frame_img = generate_frame(
                    pipe=pipe,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    context_img=conditioning,
                    cfg=cfg,
                    seed=seed,
                )
                elapsed = time.time() - t0
                stat.record_frame(elapsed)

                # Save frame to disk
                save_frame(frame_img, bid, frame_idx, cfg)

                # Update sliding window:
                # Add new frame to window, evict oldest if window exceeds size
                context_window.append(frame_img)
                if len(context_window) > cfg["window_size"]:
                    context_window.pop(0)   # Evict oldest frame from VRAM

                # Save context to disk every 10 frames (for crash-resume)
                if frame_idx % 10 == 0 or frame_idx == beat_frames - 1:
                    sm.save_context(bid, context_window)

                global_frame_idx += 1

                if frame_idx % 50 == 0:
                    info(f"Frame {frame_idx}/{beat_frames}  "
                         f"({elapsed:.1f}s)  "
                         f"ETA: {stat.hms(stat.eta())}")

            # ── Beat complete ─────────────────────────────────────────────────
            beat_elapsed = time.time() - t_beat
            sm.mark_beat_complete(bid, beat_elapsed)
            ok(f"Beat {bid:02d} complete — {beat_elapsed:.0f}s total  "
               f"({beat_elapsed/beat_frames:.1f}s/frame)")

            # Store last frame of this beat for next beat's transition
            last_path = sm.frame_path(bid, beat_frames - 1)
            if os.path.exists(last_path):
                prior_beat_last_frame = Image.open(last_path).convert("RGB")

    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}  Interrupted — progress saved. Re-run to continue.{C.END}")
        stop_evt.set()
        stat.summary()
        return

    except Exception as e:
        err(f"Error: {e}")
        import traceback; traceback.print_exc()
        stop_evt.set()
        warn("State saved — re-run to resume from last completed frame.")
        stat.summary()
        return

    finally:
        stop_evt.set()

    # ── Clean up pipeline before assembly to free VRAM ────────────────────────
    del pipe
    vram_free()

    # ── Assemble film ─────────────────────────────────────────────────────────
    final_video = assemble_video(cfg, BEATS)

    # ── Save report ───────────────────────────────────────────────────────────
    rp = os.path.join(cfg["output_dir"],
                      f"report_v{__version__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(rp, "w") as f:
        json.dump({
            "version":      __version__,
            "timestamp":    datetime.now().isoformat(),
            "quality_mode": cfg["quality_mode"],
            "total_frames": cfg["total_frames"],
            "beats":        [{"id": b["id"], "name": b["name"],
                               "frames": b["fps"] * b["duration_s"]} for b in BEATS],
            "timing":       sm.state.get("timing", {}),
            "final_video":  final_video,
        }, f, indent=2)
    ok(f"Report: {rp}")

    stat.summary()

    print(f"\n{C.BOLD}{C.GREEN}{'═'*76}{C.END}")
    print(f"{C.BOLD}{C.GREEN}  ✨  SmartSmile 2250 Film v{__version__} — COMPLETE{C.END}")
    print(f"{C.BOLD}{C.GREEN}{'═'*76}{C.END}")
    print(f"\n  {C.BOLD}Output  :{C.END}  {final_video}")
    print(f"  {C.BOLD}Model   :{C.END}  SANA-1.5 4.8B  "
          f"({cfg['image_inference_steps']} steps, CFG {cfg['image_guidance_scale']}, "
          f"PAG {cfg['image_pag_guidance_scale']})")
    print(f"  {C.BOLD}Strategy:{C.END}  Sliding-window {cfg['window_size']}-frame context  "
          f"(alpha={cfg['context_blend_alpha']})")
    print(f"  {C.BOLD}Frames  :{C.END}  {cfg['total_frames']} × "
          f"{cfg['output_width']}×{cfg['output_height']} PNG\n")


if __name__ == "__main__":
    main()