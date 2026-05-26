import torch
from diffusers import SanaPipeline, SanaVideoPipeline, SanaImageToVideoPipeline, FlowMatchEulerDiscreteScheduler
import imageio
import numpy as np
from PIL import Image
import psutil
import time
import threading
import os
from datetime import datetime
import GPUtil
import json
import gc
from collections import deque

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

# Statistics tracker
class StatisticsTracker:
    def __init__(self, config):
        self.config = config
        self.start_time = None
        self.stage_times = []
        self.gpu_usage_history = deque(maxlen=100)
        self.ram_usage_history = deque(maxlen=100)
        self.vram_usage_history = deque(maxlen=100)
        
    def start(self):
        self.start_time = time.time()
        
    def add_stage_time(self, stage_name, duration):
        self.stage_times.append({'stage': stage_name, 'duration': duration})
        
    def update_metrics(self, gpu_usage, ram_usage, vram_usage):
        self.gpu_usage_history.append(gpu_usage)
        self.ram_usage_history.append(ram_usage)
        self.vram_usage_history.append(vram_usage)
        
    def get_statistics(self):
        total_time = time.time() - self.start_time if self.start_time else 0
        return {
            'total_time': total_time,
            'total_time_formatted': self.format_time(total_time),
            'stage_times': self.stage_times,
            'avg_gpu': sum(self.gpu_usage_history) / len(self.gpu_usage_history) if self.gpu_usage_history else 0,
            'peak_gpu': max(self.gpu_usage_history) if self.gpu_usage_history else 0,
            'avg_ram': sum(self.ram_usage_history) / len(self.ram_usage_history) if self.ram_usage_history else 0,
            'peak_ram': max(self.ram_usage_history) if self.ram_usage_history else 0,
            'avg_vram': sum(self.vram_usage_history) / len(self.vram_usage_history) if self.vram_usage_history else 0,
            'peak_vram': max(self.vram_usage_history) if self.vram_usage_history else 0,
        }
        
    def format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.0f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
            
    def print_summary(self):
        stats = self.get_statistics()
        config = self.config
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}📊 TWO-STAGE GENERATION STATISTICS{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
        
        print(f"\n{Colors.BOLD}🎬 FINAL VIDEO SPECIFICATIONS{Colors.END}")
        print(f"   {'Duration:':<25} {config['total_duration_seconds']} seconds")
        print(f"   {'FPS:':<25} {config['fps']} fps")
        print(f"   {'Resolution:':<25} {config['width']}x{config['height']}")
        
        print(f"\n{Colors.BOLD}🎨 STAGE 1: 4.8B KEYFRAMES{Colors.END}")
        print(f"   {'Model:':<25} SANA1.5 4.8B (Image Generation)")
        print(f"   {'Keyframes Generated:':<25} {config['num_keyframes']}")
        print(f"   {'Guidance Scale:':<25} {config['image_guidance_scale']}")
        print(f"   {'Inference Steps:':<25} {config['image_inference_steps']}")
        
        print(f"\n{Colors.BOLD}🎥 STAGE 2: 2B VIDEO ANIMATION{Colors.END}")
        print(f"   {'Model:':<25} SANA-Video 2B")
        print(f"   {'Frames per keyframe:':<25} {config['frames_per_keyframe']}")
        print(f"   {'Guidance Scale:':<25} {config['video_guidance_scale']}")
        print(f"   {'Inference Steps:':<25} {config['video_inference_steps']}")
        
        print(f"\n{Colors.BOLD}⏱️ TIME STATISTICS{Colors.END}")
        print(f"   {'Total Time:':<25} {stats['total_time_formatted']}")
        for stage in stats['stage_times']:
            print(f"   {stage['stage']:<24} {stage['duration']:.1f}s")
        
        print(f"\n{Colors.BOLD}🎮 GPU PERFORMANCE{Colors.END}")
        print(f"   {'Average GPU Usage:':<25} {stats['avg_gpu']:.1f}%")
        print(f"   {'Peak GPU Usage:':<25} {stats['peak_gpu']:.1f}%")
        print(f"   {'Average VRAM Usage:':<25} {stats['avg_vram']:.1f}%")
        print(f"   {'Peak VRAM Usage:':<25} {stats['peak_vram']:.1f}%")
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")

# Global statistics tracker
stats_tracker = None

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
                if stats_tracker:
                    stats_tracker.update_metrics(gpu_usage, psutil.virtual_memory().percent, gpu_memory)
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

def print_dashboard(stage, stage_name, progress, details=None):
    """Beautiful dashboard display"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}🎬 TWO-STAGE CINEMATIC GENERATOR{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")
    
    print(f"{Colors.BOLD}{Colors.YELLOW}📊 CURRENT STAGE: {stage_name}{Colors.END}")
    
    # Progress bar
    bar_length = 40
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"   {'Progress:':<20} [{bar}] {progress*100:.1f}%")
    
    if details:
        for key, value in details.items():
            print(f"   {key:<20} {value}")
    
    # GPU Stats
    if metrics['gpu_usage']:
        current_gpu = metrics['gpu_usage'][-1]
        current_gpu_mem = metrics['gpu_memory'][-1]
        avg_gpu = sum(metrics['gpu_usage']) / len(metrics['gpu_usage']) if metrics['gpu_usage'] else 0
        
        print(f"\n{Colors.BOLD}{Colors.YELLOW}🎮 NVIDIA RTX 5080 (16GB){Colors.END}")
        print(f"   {'GPU Usage:':<20} [{Colors.GREEN}{'█' * int(current_gpu/4)}{Colors.END}{'░' * (25 - int(current_gpu/4))}] {current_gpu:.1f}%")
        print(f"   {'VRAM Usage:':<20} [{Colors.CYAN}{'█' * int(current_gpu_mem/4)}{Colors.END}{'░' * (25 - int(current_gpu_mem/4))}] {current_gpu_mem:.1f}%")
        print(f"   {'Avg GPU Load:':<20} {avg_gpu:.1f}%")
    
    # CPU & RAM
    current_cpu = metrics['cpu_usage'][-1] if metrics['cpu_usage'] else 0
    current_ram = metrics['ram_usage'][-1] if metrics['ram_usage'] else 0
    
    print(f"\n{Colors.BOLD}{Colors.YELLOW}💻 SYSTEM RESOURCES{Colors.END}")
    print(f"   {'CPU Usage:':<20} [{Colors.GREEN}{'█' * int(current_cpu/4)}{Colors.END}{'░' * (25 - int(current_cpu/4))}] {current_cpu:.1f}%")
    print(f"   {'RAM Usage:':<20} [{Colors.CYAN}{'█' * int(current_ram/4)}{Colors.END}{'░' * (25 - int(current_ram/4))}] {current_ram:.1f}%")
    print(f"   {'RAM Total:':<20} {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"   {'RAM Free:':<20} {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'─'*70}{Colors.END}")

def save_metrics_report(stats_tracker, config):
    """Save comprehensive metrics report to file"""
    stats = stats_tracker.get_statistics() if stats_tracker else {}
    report_name = f"report_two_stage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'statistics': stats,
        'metrics_summary': {
            'gpu_usage': {
                'avg': sum(metrics['gpu_usage']) / len(metrics['gpu_usage']) if metrics['gpu_usage'] else 0,
                'max': max(metrics['gpu_usage']) if metrics['gpu_usage'] else 0,
            },
            'gpu_memory': {
                'avg': sum(metrics['gpu_memory']) / len(metrics['gpu_memory']) if metrics['gpu_memory'] else 0,
                'max': max(metrics['gpu_memory']) if metrics['gpu_memory'] else 0,
            },
            'ram_usage': {
                'avg': sum(metrics['ram_usage']) / len(metrics['ram_usage']) if metrics['ram_usage'] else 0,
                'max': max(metrics['ram_usage']) if metrics['ram_usage'] else 0,
            },
        }
    }
    
    with open(report_name, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{Colors.GREEN}📊 Comprehensive report saved to: {report_name}{Colors.END}")
    return report_name

# ============================================================
# 🎬 TWO-STAGE CONFIGURATION - 4.8B KEYFRAMES + 2B VIDEO
# ============================================================

def get_video_config():
    """
    TWO-STAGE CONFIGURATION:
    Stage 1: Generate high-quality keyframes using 4.8B image model
    Stage 2: Animate keyframes using 2B video model
    """
    
    config = {
        # STAGE 1: 4.8B Image Model (for keyframes)
        "image_model_name": "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
        
        # STAGE 2: 2B Video Model (for animation)
        "video_model_name": "Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
        
        # FINAL VIDEO SETTINGS
        "total_duration_seconds": 10,     # Total video length
        "fps": 30,                        # Frames per second
        "height": 720,                    # Video height
        "width": 1280,                    # Video width
        
        # KEYFRAME SETTINGS
        "num_keyframes": 5,               # Number of keyframes to generate
        "frames_per_keyframe": 60,        # Frames per keyframe (2 seconds at 30fps)
        
        # IMAGE GENERATION (4.8B) SETTINGS
        "image_guidance_scale": 12.0,     # Higher = better prompt following
        "image_inference_steps": 100,     # More steps = better quality
        
        # VIDEO ANIMATION (2B) SETTINGS
        "video_guidance_scale": 8.0,
        "video_inference_steps": 50,
        
        # Random seed
        "seed": 42,
        
        # Prompts for each keyframe (different camera angles)
        "keyframe_prompts": [
            """Wide shot: F-4 Phantom and MiG-21 jets dogfighting over mountains at sunset. 
            Epic cinematic, 4K, photorealistic, dramatic orange sky, volumetric clouds.""",
            
            """Close up: F-4 Phantom cockpit view, pilot looking intensely, enemy MiG in crosshairs. 
            Cinematic lighting, detailed cockpit instruments, dramatic.""",
            
            """Action shot: Missile launching from F-4 Phantom wing, smoke trail, afterburners glowing bright orange. 
            High speed, dramatic motion blur, cinematic.""",
            
            """Explosion: MiG-21 being struck by missile, massive fireball, debris falling, smoke cloud. 
            Epic destruction, dramatic lighting, cinematic.""",
            
            """Victory flyby: F-4 Phantom flying away from explosion, sunset in background, contrails. 
            Heroic shot, cinematic, epic atmosphere."""
        ],
        
        # Motion prompt for animating each keyframe
        "motion_prompt": """Camera slowly pans and zooms in smoothly. 
        Clouds drift past in the background. 
        Atmospheric motion, cinematic camera movement, smooth and dramatic.""",
        
        # Output settings
        "output_dir": "two_stage_output",
    }
    
    # Calculate totals
    config["total_frames"] = config["total_duration_seconds"] * config["fps"]
    config["frames_per_keyframe"] = min(config["frames_per_keyframe"], config["total_frames"] // config["num_keyframes"])
    
    return config

def generate_keyframes(pipe_image, config, stop_event, monitor_thread):
    """Stage 1: Generate high-quality keyframes using 4.8B model"""
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}🎨 STAGE 1: Generating 4.8B Keyframes{Colors.END}")
    
    keyframes = []
    total_keyframes = config['num_keyframes']
    
    for i in range(total_keyframes):
        if stop_event.is_set():
            break
            
        progress = (i + 1) / total_keyframes
        print_dashboard(1, f"Generating Keyframe {i+1}/{total_keyframes}", progress, {
            'Keyframe:': f"{i+1}/{total_keyframes}",
            'Prompt:': config['keyframe_prompts'][i][:50] + "..."
        })
        
        generator = torch.Generator(device="cuda").manual_seed(config["seed"] + i)
        
        # Generate image using 4.8B model
        image = pipe_image(
            prompt=config['keyframe_prompts'][i],
            height=config['height'],
            width=config['width'],
            guidance_scale=config['image_guidance_scale'],
            num_inference_steps=config['image_inference_steps'],
            generator=generator,
        ).images[0]
        
        # Save keyframe
        keyframe_path = os.path.join(config['output_dir'], f"keyframe_{i:03d}.png")
        image.save(keyframe_path)
        keyframes.append(image)
        
        print(f"\n   ✅ Keyframe {i+1} saved to {keyframe_path}")
        
    return keyframes

def animate_keyframes(pipe_i2v, keyframes, config, stop_event, monitor_thread):
    """Stage 2: Animate keyframes using 2B video model"""
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}🎥 STAGE 2: Animating Keyframes with 2B Model{Colors.END}")
    
    all_frames = []
    total_keyframes = len(keyframes)
    
    for i, keyframe in enumerate(keyframes):
        if stop_event.is_set():
            break
            
        progress = (i + 1) / total_keyframes
        print_dashboard(2, f"Animating Keyframe {i+1}/{total_keyframes}", progress, {
            'Keyframe:': f"{i+1}/{total_keyframes}",
            'Frames:': f"{config['frames_per_keyframe']}",
            'Duration:': f"{config['frames_per_keyframe'] / config['fps']:.1f}s"
        })
        
        generator = torch.Generator(device="cuda").manual_seed(config["seed"] + i + 100)
        
        # Generate video from keyframe
        output = pipe_i2v(
            image=keyframe,
            prompt=config['motion_prompt'],
            height=config['height'],
            width=config['width'],
            frames=config['frames_per_keyframe'],
            guidance_scale=config['video_guidance_scale'],
            num_inference_steps=config['video_inference_steps'],
            generator=generator,
        )
        
        frames = output.frames[0]
        all_frames.extend(frames)
        
        print(f"\n   ✅ Keyframe {i+1} animated: {len(frames)} frames")
        
        # Clear memory between animations
        torch.cuda.empty_cache()
        gc.collect()
    
    return all_frames

def assemble_video(all_frames, config, output_path):
    """Assemble final video from all animated frames"""
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}🎬 Assembling final video...{Colors.END}")
    
    # Trim to exact desired length if needed
    if len(all_frames) > config['total_frames']:
        all_frames = all_frames[:config['total_frames']]
    
    writer = imageio.get_writer(
        output_path, 
        fps=config["fps"], 
        codec='libx264',
        quality=10,
        pixelformat='yuv420p',
        output_params=['-crf', '18', '-preset', 'medium']
    )
    
    for i, frame in enumerate(all_frames):
        if isinstance(frame, Image.Image):
            writer.append_data(np.array(frame))
        else:
            writer.append_data(frame)
        
        if (i + 1) % 100 == 0:
            print(f"   📹 Assembled {i+1}/{len(all_frames)} frames ({(i+1)/len(all_frames)*100:.1f}%)")
    
    writer.close()
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n{Colors.GREEN}✅ Final video assembled: {output_path} ({size_mb:.1f} MB){Colors.END}")
    return output_path

# ============================================================
# 🚀 MAIN EXECUTION - TWO-STAGE WORKFLOW
# ============================================================

def main():
    global stats_tracker
    
    # Load configuration
    config = get_video_config()
    
    # Initialize statistics tracker
    stats_tracker = StatisticsTracker(config)
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}🎬 TWO-STAGE CINEMATIC GENERATOR{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    
    print(f"\n{Colors.BOLD}✅ System Check:{Colors.END}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: 16GB")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    print(f"   CPU Cores: {psutil.cpu_count()}")
    print(f"   RAM Total: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    print(f"\n{Colors.BOLD}🎬 TWO-STAGE WORKFLOW:{Colors.END}")
    print(f"   {'Stage 1:':<20} 4.8B Image Model - Generate Keyframes")
    print(f"   {'Stage 2:':<20} 2B Video Model - Animate Keyframes")
    print(f"   {'Final:':<20} Assemble {config['total_duration_seconds']}s video at {config['fps']}fps")
    
    print(f"\n{Colors.BOLD}📊 Configuration:{Colors.END}")
    print(f"   {'Keyframes:':<20} {config['num_keyframes']}")
    print(f"   {'Frames per keyframe:':<20} {config['frames_per_keyframe']}")
    print(f"   {'Total frames:':<20} {config['total_frames']}")
    print(f"   {'Resolution:':<20} {config['width']}x{config['height']}")
    
    # Ask for confirmation
    print(f"\n{Colors.YELLOW}⚠️ This will generate {config['num_keyframes']} keyframes using the 4.8B model{Colors.END}")
    print(f"{Colors.YELLOW}   then animate them using the 2B video model{Colors.END}")
    response = input(f"\n{Colors.BOLD}Ready to start? (y/n): {Colors.END}")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Start monitoring thread
    stop_monitoring = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_monitoring,))
    monitor_thread.start()
    
    try:
        stats_tracker.start()
        
        # ========== STAGE 1: Load 4.8B Model and Generate Keyframes ==========
        stage_start = time.time()
        
        print(f"\n{Colors.BOLD}📦 Loading 4.8B Image Model...{Colors.END}")
        print_dashboard(0, "Loading 4.8B Model", 0)
        
        pipe_image = SanaPipeline.from_pretrained(
            config["image_model_name"],
            torch_dtype=torch.bfloat16,
        )
        pipe_image.to("cuda")
        
        print_dashboard(0.1, "4.8B Model Loaded", 0.1)
        
        keyframes = generate_keyframes(pipe_image, config, stop_monitoring, monitor_thread)
        
        stage_time = time.time() - stage_start
        stats_tracker.add_stage_time("Stage 1: 4.8B Keyframes", stage_time)
        
        # Clear memory
        del pipe_image
        torch.cuda.empty_cache()
        gc.collect()
        
        # ========== STAGE 2: Load 2B Model and Animate Keyframes ==========
        stage_start = time.time()
        
        print(f"\n{Colors.BOLD}📦 Loading 2B Video Model...{Colors.END}")
        print_dashboard(0, "Loading 2B Model", 0)
        
        # Load image-to-video pipeline
        pipe_i2v = SanaImageToVideoPipeline.from_pretrained(
            config["video_model_name"],
            torch_dtype=torch.bfloat16,
        )
        pipe_i2v.to("cuda")
        pipe_i2v.enable_model_cpu_offload()
        
        print_dashboard(0.1, "2B Model Loaded", 0.1)
        
        all_frames = animate_keyframes(pipe_i2v, keyframes, config, stop_monitoring, monitor_thread)
        
        stage_time = time.time() - stage_start
        stats_tracker.add_stage_time("Stage 2: 2B Animation", stage_time)
        
        # Clear memory
        del pipe_i2v
        torch.cuda.empty_cache()
        gc.collect()
        
        # ========== STAGE 3: Assemble Final Video ==========
        stage_start = time.time()
        
        output_path = os.path.join(config['output_dir'], f"final_{config['total_duration_seconds']}s_{config['fps']}fps_two_stage.mp4")
        final_video = assemble_video(all_frames, config, output_path)
        
        stage_time = time.time() - stage_start
        stats_tracker.add_stage_time("Stage 3: Assembly", stage_time)
        
        # Stop monitoring
        stop_monitoring.set()
        monitor_thread.join()
        
        # Save comprehensive report
        report_name = save_metrics_report(stats_tracker, config)
        
        # Print final statistics
        stats_tracker.print_summary()
        
        # Final success screen
        print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.GREEN}✨ SUCCESS! Your two-stage cinematic is ready! ✨{Colors.END}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.END}")
        print(f"\n{Colors.BOLD}🎬 FINAL VIDEO:{Colors.END}")
        print(f"   📹 File: {final_video}")
        print(f"   🎞️  Resolution: {config['width']}x{config['height']}")
        print(f"   ⏱️  Duration: {config['total_duration_seconds']} seconds")
        print(f"   🎯 FPS: {config['fps']}")
        print(f"   🎨 Keyframes: {config['num_keyframes']} (generated with 4.8B)")
        print(f"   🎥 Animation: 2B Video Model")
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}🎉 Your RTX 5080 delivered 4.8B quality with smooth 2B animation!{Colors.END}")
        
    except Exception as e:
        stop_monitoring.set()
        monitor_thread.join()
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.END}")
        print(f"\n{Colors.YELLOW}💡 Troubleshooting:{Colors.END}")
        print(f"   • The two-stage workflow is designed to handle any video length")
        print(f"   • Check available SSD space in {config['output_dir']}")
        print(f"   • Reduce num_keyframes or frames_per_keyframe if memory is low")

if __name__ == "__main__":
    main()