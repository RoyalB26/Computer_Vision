import sys
import os
import gc
import json
import time
import torch
from pathlib import Path

# Thêm đường dẫn project
workspace_dir = Path(r"D:\Study\Projects\Computer_Vision")
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))

from visdrone_mobilenetv3.model import VisdroneDetector
from torch.utils.flop_counter import FlopCounterMode
from thop import profile

def benchmark_single(dim: int, num_points: int, img_size: int, device: torch.device):
    torch.cuda.empty_cache()
    gc.collect()
    
    # 1. Khởi tạo mô hình
    model = VisdroneDetector(
        num_classes=10,
        dim=dim,
        num_heads=4,
        num_points=num_points,
        head_type="DeformableHead"
    ).to(device)
    
    params_count = sum(p.numel() for p in model.parameters())
    params_m = params_count / 1e6
    
    # 2. FLOPs đo bằng thop và FlopCounterMode
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    
    try:
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        g_macs = macs / 1e9
        g_flops_thop = 2.0 * g_macs
    except Exception as e:
        g_macs = None
        g_flops_thop = None

    try:
        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                map_pred, feat_deform = model.extract_features(dummy_input)
                _ = model.head(map_pred, feat_deform)
        g_flops_torch = fc.get_total_flops() / 1e9
    except Exception as e:
        g_flops_torch = None
        
    # 3. Đo VRAM Inference
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            map_pred, feat_deform = model.extract_features(dummy_input)
            _ = model.head(map_pred, feat_deform)
    torch.cuda.synchronize()
    
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    iters = 10
    with torch.no_grad():
        for _ in range(iters):
            map_pred, feat_deform = model.extract_features(dummy_input)
            _ = model.head(map_pred, feat_deform)
    torch.cuda.synchronize()
    latency_ms = (time.time() - start_time) / iters * 1000
    
    inf_peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    inf_peak_res = torch.cuda.max_memory_reserved() / (1024 ** 2)

    # 4. Đo VRAM Training (Forward + Backward, Batch size 1)
    model.train()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    train_input = torch.randn(1, 3, img_size, img_size, device=device)
    dummy_target = [{
        'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1]], device=device),
        'labels': torch.tensor([1], device=device)
    }]
    
    try:
        loss_dict = model(train_input, targets=dummy_target)
        loss = loss_dict['loss']
        loss.backward()
        torch.cuda.synchronize()
        train_peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        train_peak_res = torch.cuda.max_memory_reserved() / (1024 ** 2)
    except torch.cuda.OutOfMemoryError:
        train_peak_alloc = "OOM"
        train_peak_res = "OOM"
    except Exception as e:
        train_peak_alloc = str(e)
        train_peak_res = str(e)

    # Cleanup
    del model, dummy_input, train_input
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        "img_size": img_size,
        "dim": dim,
        "num_points": num_points,
        "params_m": round(params_m, 2),
        "g_macs": round(g_macs, 3) if g_macs is not None else None,
        "g_flops_thop": round(g_flops_thop, 3) if g_flops_thop is not None else None,
        "g_flops_torch": round(g_flops_torch, 3) if g_flops_torch is not None else None,
        "inf_peak_alloc_mb": round(inf_peak_alloc, 1),
        "inf_peak_res_mb": round(inf_peak_res, 1),
        "train_peak_alloc_mb": round(train_peak_alloc, 1) if isinstance(train_peak_alloc, (int, float)) else train_peak_alloc,
        "train_peak_res_mb": round(train_peak_res, 1) if isinstance(train_peak_res, (int, float)) else train_peak_res,
        "latency_ms": round(latency_ms, 2)
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    img_sizes = [640, 960]
    dims = [32, 64, 96, 128]
    num_points_list = [4, 8, 12, 16]
    
    results = []
    
    for size in img_sizes:
        print(f"\n================ IMAGE SIZE {size}x{size} ================")
        for d in dims:
            for p in num_points_list:
                print(f"Running size={size}, dim={d}, num_points={p}...", end="", flush=True)
                res = benchmark_single(dim=d, num_points=p, img_size=size, device=device)
                results.append(res)
                print(f" Done! FLOPs: {res['g_flops_torch']}G | Inf Alloc: {res['inf_peak_alloc_mb']}MB | Train Alloc: {res['train_peak_alloc_mb']}MB")
                
    output_path = Path(r"D:\Study\Projects\Computer_Vision\benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nBenchmark finished! Saved to {output_path}")

if __name__ == "__main__":
    main()
