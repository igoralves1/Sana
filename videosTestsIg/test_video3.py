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
        self.chunk_times = []
        self.frame_times = []
        self.gpu_usage_history = deque(maxlen=100)
        self.ram_usage_history = deque(maxlen=100)
        self.vram_usage_history = deque(maxlen=100)
        self.chunk_gpu_peaks = []
        self.chunk_ram_peaks = []
        
    def start(self):
        self.start_time = time.time()
        
    def add_chunk_time(self, duration, gpu_peak, ram_peak, vram_peak):
        self.chunk_times.append(duration)
        self.chunk_gpu_peaks.append(gpu_peak)
        self.chunk_ram_peaks.append(ram_peak)
        self.vram_usage_history.append(vram_peak)
        
    def add_frame_time(self, duration):
        self.frame_times.append(duration)
        
    def update_metrics(self, gpu_usage, ram_usage, vram_usage):
        self.gpu_usage_history.append(gpu_usage)
        self.ram_usage_history.append(ram_usage)
        self.vram_usage_history.append(vram_usage)
        
    def get_statistics(self):
        total_time = time.time() - self.start_time if self.start_time else 0
        avg_chunk_time = sum(self.chunk_times) / len(self.chunk_times) if self.chunk_times else 0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
        
        return {
            'total_time': total_time,
            'total_time_formatted': self.format_time(total_time),
            'num_chunks': len(self.chunk_times),
            'avg_chunk_time': avg_chunk_time,
            'avg_chunk_time_formatted': f"{avg_chunk_time:.1f}s",
            'fastest_chunk': min(self.chunk_times) if self.chunk_times else 0,
            'slowest_chunk': max(self.chunk_times) if self.chunk_times else 0,
            'avg_frame_time': avg_frame_time,
            'avg_frame_time_formatted': f"{avg_frame_time*1000:.1f}ms",
            'avg_gpu': sum(self.gpu_usage_history) / len(self.gpu_usage_history) if self.gpu_usage_history else 0,
            'peak_gpu': max(self.gpu_usage_history) if self.gpu_usage_history else 0,
            'avg_ram': sum(self.ram_usage_history) / len(self.ram_usage_history) if self.ram_usage_history else 0,
            'peak_ram': max(self.ram_usage_history) if self.ram_usage_history else 0,
            'avg_vram': sum(self.vram_usage_history) / len(self.vram_usage_history) if self.vram_usage_history else 0,
            'peak_vram': max(self.vram_usage_history) if self.vram_usage_history else 0,
            'estimated_remaining': self.estimate_remaining(),
            'completion_percent': (len(self.chunk_times) / self.config['num_chunks']) * 100 if self.config['num_chunks'] > 0 else 0,
        }
        
    def estimate_remaining(self):
        if len(self.chunk_times) < 2:
            return 0
        avg_chunk = sum(self.chunk_times[-3:]) / min(3, len(self.chunk_times))
        remaining_chunks = self.config['num_chunks'] - len(self.chunk_times)
        return avg_chunk * remaining_chunks
        
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
        print(f"{Colors.BOLD}{Colors.CYAN}📊 COMPREHENSIVE GENERATION STATISTICS{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
        
        # Video Specifications
        print(f"\n{Colors.BOLD}🎬 VIDEO SPECIFICATIONS{Colors.END}")
        print(f"   {'Target Duration:':<25} {config['total_duration_seconds']} seconds")
        print(f"   {'Target FPS:':<25} {config['fps']} fps")
        print(f"   {'Total Frames:':<25} {config['total_frames']:,}")
        print(f"   {'Resolution:':<25} {config['width']}x{config['height']}")
        print(f"   {'Total Pixels:':<25} {config['width'] * config['height']:,} per frame")
        print(f"   {'Total Pixels (all frames):':<25} {config['width'] * config['height'] * config['total_frames']:,} ({((config['width'] * config['height'] * config['total_frames']) / 1e6):.1f} MP)")
        
        # Generation Configuration
        print(f"\n{Colors.BOLD}⚙️ GENERATION CONFIGURATION{Colors.END}")
        print(f"   {'Model:':<25} {config['model_name'].split('/')[-1]}")
        print(f"   {'Chunk Size:':<25} {config['frames_per_chunk']} frames ({config['chunk_duration_seconds']:.1f} seconds)")
        print(f"   {'Total Chunks:':<25} {config['num_chunks']}")
        print(f"   {'Guidance Scale:':<25} {config['guidance_scale']}")
        print(f"   {'Inference Steps:':<25} {config['num_inference_steps']}")
        print(f"   {'Seed:':<25} {config['seed']}")
        
        # Time Statistics
        print(f"\n{Colors.BOLD}⏱️ TIME STATISTICS{Colors.END}")
        print(f"   {'Total Time:':<25} {stats['total_time_formatted']}")
        print(f"   {'Average Chunk Time:':<25} {stats['avg_chunk_time_formatted']}")
        print(f"   {'Fastest Chunk:':<25} {stats['fastest_chunk']:.1f}s")
        print(f"   {'Slowest Chunk:':<25} {stats['slowest_chunk']:.1f}s")
        print(f"   {'Average Frame Time:':<25} {stats['avg_frame_time_formatted']}")
        print(f"   {'Estimated Remaining:':<25} {self.format_time(stats['estimated_remaining'])}")
        print(f"   {'Completion:':<25} {stats['completion_percent']:.1f}%")
        
        # GPU & Memory Statistics
        print(f"\n{Colors.BOLD}🎮 GPU PERFORMANCE{Colors.END}")
        print(f"   {'Average GPU Usage:':<25} {stats['avg_gpu']:.1f}%")
        print(f"   {'Peak GPU Usage:':<25} {stats['peak_gpu']:.1f}%")
        print(f"   {'Average VRAM Usage:':<25} {stats['avg_vram']:.1f}%")
        print(f"   {'Peak VRAM Usage:':<25} {stats['peak_vram']:.1f}%")
        print(f"   {'VRAM Used:':<25} {(stats['peak_vram'] / 100) * 16:.1f} GB / 16 GB")
        
        print(f"\n{Colors.BOLD}💻 SYSTEM MEMORY{Colors.END}")
        print(f"   {'Average RAM Usage:':<25} {stats['avg_ram']:.1f}%")
        print(f"   {'Peak RAM Usage:':<25} {stats['peak_ram']:.1f}%")
        
        # Chunk Performance Breakdown
        if len(self.chunk_times) > 1:
            print(f"\n{Colors.BOLD}📈 CHUNK PERFORMANCE BREAKDOWN{Colors.END}")
            print(f"   {'Chunk':<8} {'Time (s)':<12} {'GPU Peak':<12} {'RAM Peak':<12} {'FPS Speed':<12}")
            print(f"   {'─'*8} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
            for i, (chunk_time, gpu_peak, ram_peak) in enumerate(zip(self.chunk_times, self.chunk_gpu_peaks, self.chunk_ram_peaks)):
                fps_speed = config['frames_per_chunk'] / chunk_time if chunk_time > 0 else 0
                print(f"   {i+1:<8} {chunk_time:<12.1f} {gpu_peak:<12.1f}% {ram_peak:<12.1f}% {fps_speed:<12.1f} fps")
        
        # Performance Metrics
        frames_per_second = config['total_frames'] / stats['total_time'] if stats['total_time'] > 0 else 0
        seconds_per_frame = stats['total_time'] / config['total_frames'] if config['total_frames'] > 0 else 0
        
        print(f"\n{Colors.BOLD}🚀 PERFORMANCE METRICS{Colors.END}")
        print(f"   {'Overall Generation Speed:':<30} {frames_per_second:.2f} frames/second")
        print(f"   {'Time Per Frame:':<30} {seconds_per_frame*1000:.1f} ms/frame")
        print(f"   {'Real-time Factor:':<30} {config['total_duration_seconds'] / stats['total_time']:.2f}x")
        print(f"   {'Frames Generated Per Minute:':<30} {frames_per_second * 60:.0f} frames/min")
        
        # Storage Statistics
        print(f"\n{Colors.BOLD}💾 STORAGE STATISTICS{Colors.END}")
        if os.path.exists(config['frames_dir']):
            import shutil
            try:
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(config['frames_dir']):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total_size += os.path.getsize(fp)
                print(f"   {'Frames Directory Size:':<25} {total_size / (1024**3):.2f} GB")
                print(f"   {'Average Frame Size:':<25} {total_size / config['total_frames'] / 1024:.1f} KB")
            except:
                pass
                
        # Final Video Info
        final_video = f"final_{config['total_duration_seconds']}s_{config['fps']}fps.mp4"
        if os.path.exists(final_video):
            video_size = os.path.getsize(final_video)
            print(f"\n{Colors.BOLD}🎥 FINAL VIDEO{Colors.END}")
            print(f"   {'Filename:':<25} {final_video}")
            print(f"   {'Size:':<25} {video_size / (1024**3):.2f} GB")
            print(f"   {'Bitrate:':<25} {(video_size * 8) / (config['total_duration_seconds']) / 1e6:.1f} Mbps")
        
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

def print_dashboard(step, total_steps, step_name, chunk_info=None, frame_info=None, stats=None):
    """Beautiful dashboard display"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}🎬 SANA VIDEO CINEMATIC GENERATOR - LIVE DASHBOARD{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")
    
    # Progress Section
    if chunk_info:
        print(f"{Colors.BOLD}{Colors.YELLOW}📊 OVERALL PROGRESS{Colors.END}")
        print(f"   {'Total Duration:':<20} {chunk_info.get('total_duration', 0)} seconds")
        print(f"   {'Total Frames:':<20} {chunk_info.get('total_frames', 0):,}")
        print(f"   {'Chunks:':<20} {chunk_info['current']}/{chunk_info['total']}")
        print(f"   {'Progress:':<20} {chunk_info['percent']:.1f}%")
        
        # Progress bar
        bar_length = 40
        filled = int(bar_length * chunk_info['current'] / chunk_info['total'])
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"   {'':<20} [{bar}]")
        
        if stats:
            if stats.get('estimated_remaining', 0) > 0:
                remaining = stats_tracker.format_time(stats['estimated_remaining']) if stats_tracker else "Unknown"
                print(f"   {'Est. Remaining:':<20} {remaining}")
            if stats.get('completion_percent', 0) > 0:
                print(f"   {'Completion:':<20} {stats['completion_percent']:.1f}%")
        print()
    
    # Generation Progress
    progress = (step / total_steps) * 100
    bar_length = 40
    filled = int(bar_length * step // total_steps)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"{Colors.BOLD}{Colors.YELLOW}⚙️ CURRENT OPERATION{Colors.END}")
    print(f"   {'Step:':<20} {step_name}")
    print(f"   {'Progress:':<20} {bar} {progress:.1f}%")
    
    if frame_info:
        print(f"   {'Frame:':<20} {frame_info['current']}/{frame_info['total']} ({frame_info['percent']:.1f}%)")
    print()
    
    # GPU Stats
    if metrics['gpu_usage']:
        current_gpu = metrics['gpu_usage'][-1]
        current_gpu_mem = metrics['gpu_memory'][-1]
        avg_gpu = sum(metrics['gpu_usage']) / len(metrics['gpu_usage']) if metrics['gpu_usage'] else 0
        
        print(f"{Colors.BOLD}{Colors.YELLOW}🎮 NVIDIA RTX 5080{Colors.END}")
        print(f"   {'GPU Usage:':<20} [{Colors.GREEN}{'█' * int(current_gpu/4)}{Colors.END}{'░' * (25 - int(current_gpu/4))}] {current_gpu:.1f}%")
        print(f"   {'GPU Memory:':<20} [{Colors.CYAN}{'█' * int(current_gpu_mem/4)}{Colors.END}{'░' * (25 - int(current_gpu_mem/4))}] {current_gpu_mem:.1f}%")
        print(f"   {'Avg GPU Load:':<20} {avg_gpu:.1f}%")
        if stats:
            print(f"   {'Peak GPU:':<20} {stats.get('peak_gpu', 0):.1f}%")
        print()
    
    # CPU & RAM
    current_cpu = metrics['cpu_usage'][-1] if metrics['cpu_usage'] else 0
    current_ram = metrics['ram_usage'][-1] if metrics['ram_usage'] else 0
    
    print(f"{Colors.BOLD}{Colors.YELLOW}💻 SYSTEM RESOURCES{Colors.END}")
    print(f"   {'CPU Usage:':<20} [{Colors.GREEN}{'█' * int(current_cpu/4)}{Colors.END}{'░' * (25 - int(current_cpu/4))}] {current_cpu:.1f}%")
    print(f"   {'RAM Usage:':<20} [{Colors.CYAN}{'█' * int(current_ram/4)}{Colors.END}{'░' * (25 - int(current_ram/4))}] {current_ram:.1f}%")
    print(f"   {'RAM Total:':<20} {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"   {'RAM Free:':<20} {psutil.virtual_memory().available / (1024**3):.1f} GB")
    if stats:
        print(f"   {'Peak RAM:':<20} {stats.get('peak_ram', 0):.1f}%")
    print()
    
    # Network
    if metrics['bandwidth']:
        current_bw = metrics['bandwidth'][-1]
        print(f"{Colors.BOLD}{Colors.YELLOW}🌐 NETWORK{Colors.END}")
        print(f"   {'Download:':<20} {Colors.GREEN}{current_bw:.2f}{Colors.END} MB/s\n")
    
    # Session Statistics
    if len(metrics['gpu_usage']) > 1:
        elapsed = metrics['timestamps'][-1] - metrics['timestamps'][0]
        print(f"{Colors.BOLD}{Colors.YELLOW}⏱️ SESSION STATISTICS{Colors.END}")
        print(f"   {'Monitoring:':<20} {elapsed:.1f} seconds")
        print(f"   {'Samples:':<20} {len(metrics['gpu_usage'])}")
        
        if stats:
            print(f"   {'Total Time:':<20} {stats_tracker.format_time(stats['total_time']) if stats_tracker else 'N/A'}")
            print(f"   {'Est. Remaining:':<20} {stats_tracker.format_time(stats['estimated_remaining']) if stats_tracker and stats.get('estimated_remaining') else 'N/A'}")
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'─'*70}{Colors.END}")

def save_metrics_report(stats_tracker, config):
    """Save comprehensive metrics report to file"""
    stats = stats_tracker.get_statistics() if stats_tracker else {}
    report_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'statistics': stats,
        'detailed_chunks': {
            'chunk_times': stats_tracker.chunk_times if stats_tracker else [],
            'chunk_gpu_peaks': stats_tracker.chunk_gpu_peaks if stats_tracker else [],
            'chunk_ram_peaks': stats_tracker.chunk_ram_peaks if stats_tracker else [],
        },
        'metrics_summary': {
            'gpu_usage': {
                'avg': sum(metrics['gpu_usage']) / len(metrics['gpu_usage']) if metrics['gpu_usage'] else 0,
                'max': max(metrics['gpu_usage']) if metrics['gpu_usage'] else 0,
                'min': min(metrics['gpu_usage']) if metrics['gpu_usage'] else 0,
            },
            'gpu_memory': {
                'avg': sum(metrics['gpu_memory']) / len(metrics['gpu_memory']) if metrics['gpu_memory'] else 0,
                'max': max(metrics['gpu_memory']) if metrics['gpu_memory'] else 0,
            },
            'cpu_usage': {
                'avg': sum(metrics['cpu_usage']) / len(metrics['cpu_usage']) if metrics['cpu_usage'] else 0,
                'max': max(metrics['cpu_usage']) if metrics['cpu_usage'] else 0,
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
# 🎬 CONFIGURATION FUNCTION - EDIT ONLY THIS SECTION! 🎬
# ============================================================

def get_video_config():
    """
    Configure your video generation parameters here.
    Change these values to create different videos!
    """
    
    config = {
        # Model selection
        "model_name": "Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
        
        # HIGH QUALITY: Your desired output
        "total_duration_seconds": 30,    # Total video length in seconds
        "fps": 60,                       # 60fps for ultra-smooth gaming quality
        "height": 704,                   # Video height in pixels
        "width": 1280,                   # Video width in pixels
        
        # CHUNK SETTINGS - Each chunk generates this many frames at once
        "frames_per_chunk": 60,          # 1 second per chunk (adjust based on your VRAM)
        
        # Generation quality
        "guidance_scale": 6.0,
        "num_inference_steps": 50,
        
        # Random seed (change for different results)
        "seed": 42,
        
        # ========== YOUR PROMPT - EDIT THIS! ==========
        "prompt": """Cinematic War Thunder air battle between F-4 Phantom and MiG-21 jets over mountains at sunset. 
Camera tracks alongside the Phantom's wing. Afterburners glow bright orange. Tracers fly past. 
Missile launches from Phantom, streaks across frame, strikes MiG in fireball. 
Epic dogfight, photorealistic graphics, dramatic lighting, smooth camera movement, 8k quality.""",
        
        # Output settings
        "frames_dir": "video_frames",    # Directory to save individual frames
    }
    
    # Calculate totals
    config["total_frames"] = config["total_duration_seconds"] * config["fps"]
    config["frames_per_chunk"] = min(config["frames_per_chunk"], config["total_frames"])
    config["chunk_duration_seconds"] = config["frames_per_chunk"] / config["fps"]
    config["num_chunks"] = (config["total_frames"] + config["frames_per_chunk"] - 1) // config["frames_per_chunk"]
    
    return config

def save_frame_batch(frames, chunk_idx, frames_dir, config):
    """Save a batch of frames to SSD"""
    chunk_dir = os.path.join(frames_dir, f"chunk_{chunk_idx:04d}")
    os.makedirs(chunk_dir, exist_ok=True)
    
    saved_paths = []
    start_frame = chunk_idx * config["frames_per_chunk"]
    
    frame_start_time = time.time()
    
    for i, frame in enumerate(frames):
        global_frame_num = start_frame + i
        frame_path = os.path.join(chunk_dir, f"frame_{global_frame_num:06d}.png")
        
        if isinstance(frame, Image.Image):
            frame.save(frame_path, "PNG", compress_level=1)
        else:
            Image.fromarray(frame).save(frame_path, "PNG", compress_level=1)
        
        saved_paths.append(frame_path)
        
        if stats_tracker:
            stats_tracker.add_frame_time(time.time() - frame_start_time)
            frame_start_time = time.time()
    
    return saved_paths

def assemble_video_from_frames(frames_dir, config):
    """Assemble final video from all saved frames"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}🎬 Assembling final video from {config['total_frames']} frames...{Colors.END}")
    
    final_video_path = f"final_{config['total_duration_seconds']}s_{config['fps']}fps.mp4"
    
    writer = imageio.get_writer(
        final_video_path, 
        fps=config["fps"], 
        codec='libx264',
        quality=10,
        pixelformat='yuv420p',
        output_params=['-crf', '18', '-preset', 'medium']
    )
    
    frame_count = 0
    total_frames = config["total_frames"]
    last_update = time.time()
    
    for chunk_idx in range(config["num_chunks"]):
        chunk_dir = os.path.join(frames_dir, f"chunk_{chunk_idx:04d}")
        if not os.path.exists(chunk_dir):
            continue
            
        frame_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.png')])
        
        for frame_file in frame_files:
            frame_path = os.path.join(chunk_dir, frame_file)
            frame = imageio.imread(frame_path)
            writer.append_data(frame)
            frame_count += 1
            
            if time.time() - last_update > 2:  # Update every 2 seconds
                print(f"   📹 Assembled {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
                last_update = time.time()
    
    writer.close()
    
    print(f"\n{Colors.GREEN}✅ Final video assembled: {final_video_path}{Colors.END}")
    return final_video_path

# ============================================================
# 🚀 MAIN EXECUTION WITH COMPREHENSIVE STATISTICS
# ============================================================

def main():
    global stats_tracker
    
    # Load configuration
    config = get_video_config()
    
    # Initialize statistics tracker
    stats_tracker = StatisticsTracker(config)
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}🎬 SANA Video - HIGH QUALITY 60fps CINEMATIC GENERATOR{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    
    print(f"\n{Colors.BOLD}✅ System Check:{Colors.END}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    print(f"   CPU Cores: {psutil.cpu_count()}")
    print(f"   RAM Total: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # Check SSD space
    frames_dir = config["frames_dir"]
    print(f"\n{Colors.BOLD}💾 Storage Check:{Colors.END}")
    try:
        os.makedirs(frames_dir, exist_ok=True)
        import shutil
        disk_usage = shutil.disk_usage(frames_dir)
        free_gb = disk_usage.free / (1024**3)
        print(f"   Free space on SSD: {free_gb:.1f} GB")
        
        estimated_size_gb = (config["total_frames"] * config["width"] * config["height"] * 3) / (1024**3) * 0.3
        print(f"   Estimated storage for frames: ~{estimated_size_gb:.1f} GB")
        
        if free_gb < estimated_size_gb * 1.2:
            print(f"   {Colors.YELLOW}⚠️ Warning: Low disk space! Consider reducing duration or resolution{Colors.END}")
    except:
        pass
    
    print(f"\n{Colors.BOLD}🎬 Video Configuration:{Colors.END}")
    print(f"   Model: SANA-Video 2B")
    print(f"   Total Duration: {config['total_duration_seconds']} seconds")
    print(f"   FPS: {config['fps']} fps")
    print(f"   Total Frames: {config['total_frames']:,}")
    print(f"   Resolution: {config['width']}x{config['height']}")
    print(f"   Chunk Size: {config['frames_per_chunk']} frames ({config['chunk_duration_seconds']:.1f} seconds/chunk)")
    print(f"   Total Chunks: {config['num_chunks']}")
    print(f"   Frames Directory: {config['frames_dir']}/")
    
    # Ask for confirmation
    print(f"\n{Colors.YELLOW}⚠️ This will generate {config['total_frames']:,} frames ({config['total_duration_seconds']}s at {config['fps']}fps){Colors.END}")
    response = input(f"\n{Colors.BOLD}Ready to start? (y/n): {Colors.END}")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Start monitoring thread
    stop_monitoring = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_monitoring,))
    monitor_thread.start()
    
    try:
        print(f"\n{Colors.BOLD}📦 Loading SANA-Video model...{Colors.END}")
        print(f"   First download takes ~10GB - monitoring active!\n")
        
        # Load model
        print_dashboard(1, 4, "Loading Model (Downloading weights...)")
        
        pipe = SanaVideoPipeline.from_pretrained(
            config["model_name"],
            torch_dtype=torch.bfloat16,
        )
        pipe.scheduler = FlowMatchEulerDiscreteScheduler()
        
        print_dashboard(2, 4, "Moving model to GPU...")
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        
        # Start statistics tracking
        stats_tracker.start()
        
        # Generate and save frames chunk by chunk
        total_chunks = config['num_chunks']
        all_frame_paths = []
        
        for chunk_idx in range(total_chunks):
            chunk_start_time = time.time()
            
            chunk_info = {
                'current': chunk_idx + 1,
                'total': total_chunks,
                'percent': ((chunk_idx) / total_chunks) * 100,
                'total_frames': config['total_frames'],
                'total_duration': config['total_duration_seconds']
            }
            
            # Get current stats for display
            current_stats = stats_tracker.get_statistics()
            
            print_dashboard(3, 4, f"Generating Chunk {chunk_idx + 1}/{total_chunks}", chunk_info, None, current_stats)
            print(f"\n{Colors.BOLD}{Colors.CYAN}🎬 Generating chunk {chunk_idx + 1}/{total_chunks}{Colors.END}")
            print(f"   Frames in chunk: {min(config['frames_per_chunk'], config['total_frames'] - chunk_idx * config['frames_per_chunk'])}")
            print(f"   Remaining chunks: {total_chunks - chunk_idx - 1}")
            if current_stats.get('estimated_remaining', 0) > 0:
                print(f"   Estimated remaining time: {stats_tracker.format_time(current_stats['estimated_remaining'])}")
            
            # Set different seed for each chunk
            chunk_seed = config["seed"] + chunk_idx
            generator = torch.Generator(device="cuda").manual_seed(chunk_seed)
            
            # Calculate actual frames for last chunk
            frames_this_chunk = min(config["frames_per_chunk"], config['total_frames'] - chunk_idx * config["frames_per_chunk"])
            
            # Generate this chunk
            output = pipe(
                prompt=config["prompt"],
                height=config["height"],
                width=config["width"],
                frames=frames_this_chunk,
                guidance_scale=config["guidance_scale"],
                num_inference_steps=config["num_inference_steps"],
                generator=generator,
            )
            
            chunk_frames = output.frames[0]
            
            # Save frames to SSD
            print(f"   💾 Saving {len(chunk_frames)} frames to SSD...")
            saved_paths = save_frame_batch(chunk_frames, chunk_idx, config['frames_dir'], config)
            all_frame_paths.extend(saved_paths)
            
            chunk_time = time.time() - chunk_start_time
            
            # Get GPU and RAM peaks for this chunk
            recent_gpu = metrics['gpu_usage'][-len(chunk_frames):] if metrics['gpu_usage'] else [0]
            recent_ram = metrics['ram_usage'][-len(chunk_frames):] if metrics['ram_usage'] else [0]
            recent_vram = metrics['gpu_memory'][-len(chunk_frames):] if metrics['gpu_memory'] else [0]
            
            gpu_peak = max(recent_gpu) if recent_gpu else 0
            ram_peak = max(recent_ram) if recent_ram else 0
            vram_peak = max(recent_vram) if recent_vram else 0
            
            stats_tracker.add_chunk_time(chunk_time, gpu_peak, ram_peak, vram_peak)
            
            print(f"   ✅ Chunk complete: {chunk_time:.1f}s ({(frames_this_chunk / chunk_time):.1f} frames/sec)")
            
            # Clear GPU memory between chunks
            torch.cuda.empty_cache()
            gc.collect()
            
            # Save progress metadata
            progress_data = {
                'completed_chunks': chunk_idx + 1,
                'total_chunks': total_chunks,
                'total_frames_saved': len(all_frame_paths),
                'expected_total_frames': config['total_frames'],
                'chunk_size_frames': config['frames_per_chunk'],
                'fps': config['fps'],
                'duration_seconds': config['total_duration_seconds'],
                'last_update': datetime.now().isoformat(),
                'statistics': stats_tracker.get_statistics()
            }
            
            with open(os.path.join(config['frames_dir'], 'progress.json'), 'w') as f:
                json.dump(progress_data, f, indent=2)
        
        # Assemble final video
        print_dashboard(4, 4, "Assembling final video from frames...")
        final_video = assemble_video_from_frames(config['frames_dir'], config)
        
        # Stop monitoring
        stop_monitoring.set()
        monitor_thread.join()
        
        # Save comprehensive report
        report_name = save_metrics_report(stats_tracker, config)
        
        # Print final statistics
        stats_tracker.print_summary()
        
        # Final success screen
        print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.GREEN}✨ SUCCESS! Your HIGH QUALITY {config['total_duration_seconds']}s {config['fps']}fps cinematic is ready! ✨{Colors.END}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.END}")
        
    except Exception as e:
        stop_monitoring.set()
        monitor_thread.join()
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.END}")
        print(f"\n{Colors.YELLOW}💡 Troubleshooting:{Colors.END}")
        print(f"   • Try reducing 'frames_per_chunk' to 30 (0.5 seconds per chunk)")
        print(f"   • Check available SSD space")
        print(f"   • The frame-saving approach is designed to handle any video length!")

if __name__ == "__main__":
    main()