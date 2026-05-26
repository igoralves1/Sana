import torch
from diffusers import SanaVideoPipeline, FlowMatchEulerDiscreteScheduler
import imageio
import numpy as np
from PIL import Image

print("=" * 60)
print("🎬 SANA-Video: War Thunder Cinematic from Text")
print("=" * 60)

prompt = """Cinematic War Thunder air battle between F-4 Phantom and MiG-21 jets over mountains at sunset. 
Camera tracks alongside the Phantom's wing as tracers fly past. 
Afterburners glow bright orange. Missile launches and streaks toward target. 
Epic dogfight, smooth camera motion, dramatic lighting, 8k quality."""

print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"📝 Prompt: {prompt[:100]}...")

print("\n📦 Loading SANA-Video model...")

pipe = SanaVideoPipeline.from_pretrained(
    "Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.scheduler = FlowMatchEulerDiscreteScheduler()
pipe.to("cuda")
pipe.enable_model_cpu_offload()

print("🎬 Generating 5-second War Thunder cinematic...")

# Generate video - the output is a list of PIL Images
output = pipe(
    prompt=prompt,
    height=704,
    width=1280,
    frames=81,
    guidance_scale=6.0,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(42),
)

# The frames are in output.frames[0] as a list of PIL Images
video_frames = output.frames[0]

print(f"✅ Generated {len(video_frames)} frames")

# Save as MP4 using imageio
print("💾 Saving video...")
writer = imageio.get_writer("warthunder_direct.mp4", fps=16, codec='libx264')

for frame in video_frames:
    # Convert PIL Image to numpy array
    frame_np = np.array(frame)
    writer.append_data(frame_np)

writer.close()

print("\n✅ Video saved as: warthunder_direct.mp4")
print(f"📐 Resolution: 1280x704 pixels")
print(f"⏱️ Duration: 5 seconds")
print(f"🎬 Frames: {len(video_frames)}")