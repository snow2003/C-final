import os, time, psutil, torch, gc

def print_mem(tag: str):
    """打印当前进程的 CPU 内存和 GPU 显存，占位 tag 随意写。"""
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss / 1024**3            # GB
    gpu = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    print(f"[{time.strftime('%H:%M:%S')}] {tag:<15} | RAM={rss:5.2f} GB | GPU={gpu:5.2f} GB", flush=True)
    gc.collect()
    torch.cuda.empty_cache()
