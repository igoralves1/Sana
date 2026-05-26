import torch
from diffusers import SanaVideoPipeline, FlowMatchEulerDiscreteScheduler
import imageio
import numpy as np
from PIL import Image
import psutil
import time
import threading
import os
from datetime import datetime
import GPUtil

# Color codes for beautiful terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Metrics storage
metrics = {
    'gpu_usage': [],
    'gpu_memory': [],
    'cpu_usage': [],
    'ram_usage': [],
    'bandwidth': [],
    'timestamps': []
}

last_net_io = psutil.net_io_counters()
last_time = time.time()

def get_bandwidth():
    global last_net_io, last_time
    current_net_io = psutil.net_io_counters()
    current_time = time.time()
    
    download_speed = (current_net_io.bytes_recv - last_net_io.bytes_recv) / (current_time - last_time) / 1024 / 1024
    upload_speed = (current_net_io.bytes_sent - last_net_io.bytes_sent) / (current_time - last_time) / 1024 / 1024
    
    last_net_io = current_net_io
    last_time = current_time
    
    return download_speed, upload_speed

def monitor_resources(stop_event):
    """Live monitoring thread"""
    while not stop_event.is_set():
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_usage = gpu.load * 100
                gpu_memory = (gpu.memoryUsed / gpu.memoryTotal) * 100
            else:
                gpu_usage = 0
                gpu_memory = 0
            
            cpu_usage = psutil.cpu_percent()
            ram_usage = psutil.virtual_memory().percent
            
            dl_speed, ul_speed = get_bandwidth()
            
            metrics['gpu_usage'].append(gpu_usage)
            metrics['gpu_memory'].append(gpu_memory)
            metrics['cpu_usage'].append(cpu_usage)
            metrics['ram_usage'].append(ram_usage)
            metrics['bandwidth'].append(dl_speed)
            metrics['timestamps'].append(time.time())
            
            time.sleep(0.5)
            
        except:
            pass

def print_dashboard(step, total_steps, step_name):
    """Beautiful dashboard display"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}🎬 SANA VIDEO CINEMATIC GENERATOR - LIVE DASHBOARD{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")
    
    progress = (step / total_steps) * 100
    bar_length = 40
    filled = int(bar_length * step // total_steps)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"{Colors.BOLD}📊 Progress: {Colors.END}{bar} {progress:.1f}%")
    print(f"{Colors.BOLD}📍 Current Step: {Colors.GREEN}{step_name}{Colors.END}\n")
    
    if metrics['gpu_usage']:
        current_gpu = metrics['gpu_usage'][-1]
        current_gpu_mem = metrics['gpu_memory'][-1]
        avg_gpu = sum(metrics['gpu_usage']) / len(metrics['gpu_usage'])
        
        print(f"{Colors.BOLD}{Colors.YELLOW}🎮 NVIDIA RTX 5080{Colors.END}")
        print(f"   GPU Usage:    [{Colors.GREEN}{'█' * int(current_gpu/4)}{Colors.END}{'░' * (25 - int(current_gpu/4))}] {current_gpu:.1f}%")
        print(f"   GPU Memory:   [{Colors.CYAN}{'█' * int(current_gpu_mem/4)}{Colors.END}{'░' * (25 - int(current_gpu_mem/4))}] {current_gpu_mem:.1f}%")
        print(f"   Avg GPU Load: {avg_gpu:.1f}%\n")
    
    current_cpu = metrics['cpu_usage'][-1] if metrics['cpu_usage'] else 0
    current_ram = metrics['ram_usage'][-1] if metrics['ram_usage'] else 0
    
    print(f"{Colors.BOLD}{Colors.YELLOW}💻 SYSTEM RESOURCES{Colors.END}")
    print(f"   CPU Usage:    [{Colors.GREEN}{'█' * int(current_cpu/4)}{Colors.END}{'░' * (25 - int(current_cpu/4))}] {current_cpu:.1f}%")
    print(f"   RAM Usage:    [{Colors.CYAN}{'█' * int(current_ram/4)}{Colors.END}{'░' * (25 - int(current_ram/4))}] {current_ram:.1f}%")
    print(f"   RAM Total:    {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"   RAM Free:     {psutil.virtual_memory().available / (1024**3):.1f} GB\n")
    
    if metrics['bandwidth']:
        current_bw = metrics['bandwidth'][-1]
        print(f"{Colors.BOLD}{Colors.YELLOW}🌐 NETWORK BANDWIDTH{Colors.END}")
        print(f"   Download:     {Colors.GREEN}{current_bw:.2f}{Colors.END} MB/s\n")
    
    if len(metrics['gpu_usage']) > 1:
        elapsed = metrics['timestamps'][-1] - metrics['timestamps'][0]
        print(f"{Colors.BOLD}{Colors.YELLOW}⏱️  SESSION STATISTICS{Colors.END}")
        print(f"   Monitoring:   {elapsed:.1f} seconds")
        print(f"   Samples:      {len(metrics['gpu_usage'])}")
        print(f"   Peak GPU:     {max(metrics['gpu_usage']):.1f}%")
        print(f"   Peak RAM:     {max(metrics['ram_usage']):.1f}%\n")
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'─'*60}{Colors.END}")

def save_metrics_report():
    """Save metrics to a file"""
    report_name = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_name, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SANA VIDEO GENERATION - PERFORMANCE REPORT\n")
        f.write("="*60 + "\n\n")
        
        if metrics['gpu_usage']:
            f.write(f"Average GPU Usage:  {sum(metrics['gpu_usage'])/len(metrics['gpu_usage']):.1f}%\n")
            f.write(f"Peak GPU Usage:     {max(metrics['gpu_usage']):.1f}%\n")
            f.write(f"Average GPU Memory: {sum(metrics['gpu_memory'])/len(metrics['gpu_memory']):.1f}%\n")
            f.write(f"Peak GPU Memory:    {max(metrics['gpu_memory']):.1f}%\n")
            f.write(f"Average CPU Usage:  {sum(metrics['cpu_usage'])/len(metrics['cpu_usage']):.1f}%\n")
            f.write(f"Peak CPU Usage:     {max(metrics['cpu_usage']):.1f}%\n")
            f.write(f"Average RAM Usage:  {sum(metrics['ram_usage'])/len(metrics['ram_usage']):.1f}%\n")
            f.write(f"Peak RAM Usage:     {max(metrics['ram_usage']):.1f}%\n")
            
            if metrics['bandwidth']:
                valid_bw = [b for b in metrics['bandwidth'] if b > 0]
                if valid_bw:
                    f.write(f"Peak Bandwidth:     {max(valid_bw):.2f} MB/s\n")
        
        f.write(f"\nTotal Duration:     {metrics['timestamps'][-1] - metrics['timestamps'][0]:.1f} seconds\n")
    
    print(f"\n{Colors.GREEN}📊 Performance report saved to: {report_name}{Colors.END}")
    return report_name

# ============================================================
# 🎬 CONFIGURATION FUNCTION - EDIT ONLY THIS SECTION! 🎬
# ============================================================

def get_video_config():
    """
    Configure your video generation parameters here.
    Change these values to create different videos!
    """
    
    config = {
        # Model selection - USING WORKING SanaVideoPipeline
        "model_name": "Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
        
        # Video duration and quality
        "duration_seconds": 10,       # Length of video in seconds (start with 5 for testing)
        "fps": 60,                   # Frames per second (16, 24, 30, or 60)
        "height": 704,               # Video height in pixels
        "width": 1280,               # Video width in pixels
        
        # Generation quality (higher = better but slower)
        "guidance_scale": 6.0,       # How closely to follow prompt (5-10)
        "num_inference_steps": 50,   # Quality steps (30-100, higher = slower)
        
        # Random seed (change for different results, or use None for random)
        "seed": 42,                  # Use None for random results
        
        # ========== YOUR PROMPT - EDIT THIS! ==========
        "prompt": """Cinematic War Thunder air battle between F-4 Phantom and MiG-21 jets over mountains at sunset. 
Camera tracks alongside the Phantom's wing. Afterburners glow bright orange. Tracers fly past. 
Missile launches from Phantom, streaks across frame, strikes MiG in fireball. 
Epic dogfight, photorealistic graphics, dramatic lighting, smooth camera movement, 8k quality.""",
    }
    
    # Calculate number of frames based on duration and fps
    config["num_frames"] = config["duration_seconds"] * config["fps"]
    
    # Estimated generation time (rough estimate)
    config["estimated_minutes"] = (config["num_frames"] / 60) * 0.15
    
    return config

# ============================================================
# 🚀 MAIN EXECUTION
# ============================================================

def main():
    # Load configuration
    config = get_video_config()
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}🎬 SANA Video CINEMATIC GENERATOR{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    
    print(f"\n{Colors.BOLD}✅ System Check:{Colors.END}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    print(f"   CPU Cores: {psutil.cpu_count()}")
    print(f"   RAM Total: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    print(f"\n{Colors.BOLD}🎬 Video Configuration:{Colors.END}")
    print(f"   Model: SANA-Video 2B")
    print(f"   Duration: {config['duration_seconds']} seconds")
    print(f"   FPS: {config['fps']} fps")
    print(f"   Total Frames: {config['num_frames']}")
    print(f"   Resolution: {config['width']}x{config['height']}")
    print(f"   Estimated time: ~{config['estimated_minutes']:.1f} minutes")
    
    # Start monitoring thread
    stop_monitoring = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_monitoring,))
    monitor_thread.start()
    
    try:
        print(f"\n{Colors.BOLD}📦 Loading SANA-Video model...{Colors.END}")
        print(f"   First download takes ~10GB - monitoring active!\n")
        
        # Step 1: Load model
        print_dashboard(1, 5, "Loading Model (Downloading weights...)")
        
        pipe = SanaVideoPipeline.from_pretrained(
            config["model_name"],
            torch_dtype=torch.bfloat16,
        )
        pipe.scheduler = FlowMatchEulerDiscreteScheduler()
        
        # Step 2: Move to GPU
        print_dashboard(2, 5, "Moving model to GPU...")
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        
        # Step 3: Generate video
        print_dashboard(3, 5, "Generating video frames...")
        print(f"\n{Colors.BOLD}📝 Your prompt:{Colors.END}\n   {config['prompt']}\n")
        
        # Set up generator with seed if provided
        generator = None
        if config["seed"] is not None:
            generator = torch.Generator(device="cuda").manual_seed(config["seed"])
        
        output = pipe(
            prompt=config["prompt"],
            height=config["height"],
            width=config["width"],
            frames=config["num_frames"],
            guidance_scale=config["guidance_scale"],
            num_inference_steps=config["num_inference_steps"],
            generator=generator,
        )
        
        # The frames are in output.frames[0] as a list of PIL Images
        video_frames = output.frames[0]
        
        # Step 4: Save video
        print_dashboard(4, 5, "Saving video as MP4...")
        
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        video_filename = f"{script_name}_sana_{config['duration_seconds']}s_{config['fps']}fps.mp4"
        
        # Save using imageio (handles PIL Images correctly)
        writer = imageio.get_writer(video_filename, fps=config["fps"], codec='libx264')
        for frame in video_frames:
            frame_np = np.array(frame)
            writer.append_data(frame_np)
        writer.close()
        
        # Step 5: Finalize
        print_dashboard(5, 5, "Finalizing...")
        time.sleep(1)
        
        # Stop monitoring
        stop_monitoring.set()
        monitor_thread.join()
        
        # Save metrics report
        report_name = save_metrics_report()
        
        # Final success screen
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.GREEN}✨ SUCCESS! Your SANA cinematic is ready! ✨{Colors.END}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.END}\n")
        
        print(f"{Colors.BOLD}🎬 Video Details:{Colors.END}")
        print(f"   📹 File: {Colors.CYAN}{video_filename}{Colors.END}")
        size_mb = os.path.getsize(video_filename) / (1024 * 1024)
        print(f"   💾 Size: {size_mb:.1f} MB")
        print(f"   ⏱️  Duration: {config['duration_seconds']} seconds")
        print(f"   🎞️  FPS: {config['fps']}")
        print(f"   🎨 Resolution: {config['width']}x{config['height']}")
        print(f"   🖼️  Total Frames: {len(video_frames)}")
        
        print(f"\n{Colors.BOLD}📊 Performance Summary:{Colors.END}")
        
        if metrics['gpu_usage']:
            avg_gpu = sum(metrics['gpu_usage']) / len(metrics['gpu_usage'])
            peak_gpu = max(metrics['gpu_usage'])
            avg_ram = sum(metrics['ram_usage']) / len(metrics['ram_usage'])
            print(f"   🎮 Avg GPU Load: {avg_gpu:.1f}% (Peak: {peak_gpu:.1f}%)")
            print(f"   💾 Avg RAM Usage: {avg_ram:.1f}%")
            print(f"   📈 Performance report: {report_name}")
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}🎉 Your RTX 5080 crushed it! Watch your cinematic now!{Colors.END}")
        print(f"\n{Colors.BOLD}🔥 Quick Tips:{Colors.END}")
        print(f"   • Edit the 'prompt' in get_video_config() for different videos")
        print(f"   • Change duration_seconds to 20 for longer videos")
        print(f"   • Increase fps to 30 or 60 for smoother motion")
        
    except Exception as e:
        stop_monitoring.set()
        monitor_thread.join()
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.END}")

if __name__ == "__main__":
    main()