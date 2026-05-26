import torch
from diffusers import SanaPipeline
import time

print("=" * 60)
print("🎬 SANA - War Thunder Cinematic Generator")
print("=" * 60)

# Your cinematic prompt - edit this to change the scene!
prompt = """Cinematic War Thunder air battle between F-4 Phantom and MiG-21 jets over mountains at sunset. 
Camera tracks alongside the Phantom's wing. Afterburners glow bright orange. 
Tracers fly past. Missile launches from Phantom, streaks across frame, strikes MiG in fireball. 
Epic dogfight, photorealistic graphics, dramatic lighting, cinematic, 8k quality, highly detailed."""

print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
print(f"✅ PyTorch Version: {torch.__version__}")
print(f"\n📝 Your Prompt:")
print(f"   {prompt[:150]}...")

print("\n📦 Loading SANA 1.5 1.6B model...")
print("   ⏱️  First download takes 2-3 minutes (model size: ~3GB)")
print("   🔥 Your RTX 5080 will crush this!\n")

start_time = time.time()

# Load the pipeline (this works on Windows!)
pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

# Optimize for memory (optional, but good practice)
pipe.enable_attention_slicing()

load_time = time.time() - start_time
print(f"   ✅ Model loaded in {load_time:.1f} seconds")

print("\n🎨 Generating your War Thunder cinematic image...")
print("   ⏱️  This takes 1-3 seconds on RTX 5080")
print("   🎬 Creating epic dogfight scene...\n")

gen_start = time.time()

# Generate the image
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=4.5,
    num_inference_steps=20,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

gen_time = time.time() - gen_start

# Save the image
image.save("warthunder_final.png")
total_time = time.time() - start_time

print("=" * 60)
print("✅ SUCCESS! Your War Thunder cinematic is ready!")
print("=" * 60)
print(f"\n📁 File: warthunder_final.png")
print(f"📐 Resolution: 1024x1024 pixels")
print(f"⏱️  Generation time: {gen_time:.1f} seconds")
print(f"⏱️  Total time: {total_time:.1f} seconds")
print(f"🎮 GPU: NVIDIA GeForce RTX 5080")
print(f"💾 VRAM used: ~8-10 GB")

print("\n🎬 NEXT STEPS TO MAKE A VIDEO:")
print("=" * 40)
print("1. Generate 5-10 images with different angles:")
print("   - Change the prompt to 'wide shot', 'close up', 'missile view'")
print("2. Download CapCut (free) or DaVinci Resolve")
print("3. Import all images into the editor")
print("4. Add zoom/pan effects to each image (Ken Burns effect)")
print("5. Add transitions between scenes")
print("6. Add jet engine sounds and epic music")
print("7. Export as MP4 - YOUR CINEMATIC IS DONE!")

print("\n🔥 QUICK TIPS FOR BETTER RESULTS:")
print("   • Change the random seed for different results")
print("   • Add 'cinematic lighting' to your prompt")
print("   • Try 'first-person cockpit view' for intense action")
print("   • Generate 20 images for a 30-second video")

print("\n🎉 Your RTX 5080 is a beast! Enjoy your cinematic!")