import torch
from diffusers import SanaPipeline

print("=" * 50)
print("SANA Hello World")
print("=" * 50)

print(f"\n✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

print("\n📦 Loading Sana model (1.6B)...")
print("   First download will take a few minutes...")

pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

# Optimize for memory
pipe.enable_attention_slicing()

print("\n🎨 Generating image...")
prompt = "a cute cat wearing a spacesuit, digital art, high quality, 4k"

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=4.5,
    num_inference_steps=20,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

image.save("sana_hello_world.png")
print(f"\n✅ Image saved as: sana_hello_world.png")
print("   Location: C:\\Users\\alexa\\Documents\\Sana\\sana_hello_world.png")
print("\n🎉 Success! Your first Sana image has been generated!")