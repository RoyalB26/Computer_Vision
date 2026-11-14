"""Model summary and resource profiling utilities for BaseModel."""
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml


def _to_plain_dict(val: Any) -> Any:
    """Recursively convert ConfigDict and other custom dict-like objects to standard Python dicts."""
    if hasattr(val, "to_dict"):
        val = val.to_dict()
    if isinstance(val, dict):
        return {k: _to_plain_dict(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [_to_plain_dict(v) for v in val]
    return val


def _get_ram_info() -> Dict[str, float]:
    """Retrieve process and system RAM memory statistics in Megabytes (MB)."""
    info = {
        "process_rss_mb": 0.0,
        "process_pct": 0.0,
        "system_total_mb": 0.0,
        "system_avail_mb": 0.0,
    }
    try:
        import psutil
        proc = psutil.Process()
        mem = proc.memory_info()
        info["process_rss_mb"] = mem.rss / (1024 ** 2)
        info["process_pct"] = proc.memory_percent()
        sys_mem = psutil.virtual_memory()
        info["system_total_mb"] = sys_mem.total / (1024 ** 2)
        info["system_avail_mb"] = sys_mem.available / (1024 ** 2)
    except Exception:
        pass
    return info


def _get_vram_info(device: torch.device) -> Dict[str, Any]:
    """Retrieve GPU VRAM memory statistics in Megabytes (MB)."""
    info = {
        "is_cuda": False,
        "device_name": "N/A",
        "total_vram_mb": 0.0,
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "max_allocated_mb": 0.0,
        "max_reserved_mb": 0.0,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            info["is_cuda"] = True
            info["device_name"] = torch.cuda.get_device_name(device)
            dev_props = torch.cuda.get_device_properties(device)
            info["total_vram_mb"] = dev_props.total_memory / (1024 ** 2)
            info["allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 ** 2)
            info["reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 ** 2)
            info["max_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            info["max_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        except Exception:
            pass
    return info


def _synthesize_dummy_input(model: nn.Module, cfg: Any, device: torch.device) -> Tuple[Any, str]:
    """Synthesize a representative dummy input matching the architecture and dataset config."""
    data_cfg = getattr(cfg, "data", None)
    seq_len = 16
    channels = 3
    height = 224
    width = 224
    is_video = False
    use_uint8 = False

    if data_cfg:
        train_ds = getattr(getattr(data_cfg, "train", None), "dataset", {})
        val_ds = getattr(getattr(data_cfg, "val", None), "dataset", {})
        ds_cfg = train_ds or val_ds
        if isinstance(ds_cfg, dict):
            if "sequence_len" in ds_cfg:
                seq_len = int(ds_cfg["sequence_len"])
                is_video = True
            pipeline = ds_cfg.get("pipeline", [])
            if isinstance(pipeline, (list, tuple)):
                for p in pipeline:
                    if isinstance(p, dict):
                        if "video_size" in p:
                            height, width = p["video_size"]
                            is_video = True
                        elif "img_size" in p:
                            img_s = p["img_size"]
                            if isinstance(img_s, (list, tuple)):
                                height, width = img_s[0], img_s[1]
                            elif isinstance(img_s, int):
                                height = width = img_s
                        elif "Action_Slot_Transform" in p.get("type", ""):
                            height, width = 256, 768
                            is_video = True

    model_cfg = getattr(cfg, "model", {})
    if isinstance(model_cfg, dict):
        if "resolution3d" in model_cfg:
            res = model_cfg["resolution3d"]
            if isinstance(res, (list, tuple)) and len(res) >= 3:
                seq_len = int(res[0])
                height = int(res[1]) * 32
                width = int(res[2]) * 32
                is_video = True
                use_uint8 = True

    # Check model class name hints
    cls_name = model.__class__.__name__.lower()
    if "slot" in cls_name or "video" in cls_name or "x3d" in cls_name or "i3d" in cls_name or "slowfast" in cls_name:
        is_video = True

    if is_video:
        if use_uint8:
            dummy = torch.randint(0, 256, (1, seq_len, channels, height, width), dtype=torch.uint8, device=device)
            desc = f"(1, {seq_len}, {channels}, {height}, {width}) [torch.uint8]"
        else:
            dummy = torch.randn(1, seq_len, channels, height, width, dtype=torch.float32, device=device)
            desc = f"(1, {seq_len}, {channels}, {height}, {width}) [torch.float32]"
    else:
        dummy = torch.randn(1, channels, height, width, dtype=torch.float32, device=device)
        desc = f"(1, {channels}, {height}, {width}) [torch.float32]"

    return dummy, desc


def _run_forward_profiling(
    model: nn.Module,
    cfg: Any,
    device: torch.device,
) -> Dict[str, Any]:
    """Run a forward pass to compute FLOPs, MACs, peak VRAM, peak RAM, input/output shapes, and layer FLOPs table."""
    result = {
        "success": False,
        "input_desc": "N/A",
        "output_desc": "N/A",
        "flops": 0,
        "flops_desc": "N/A",
        "macs_desc": "N/A",
        "flops_table": None,
        "peak_vram_mb": 0.0,
        "peak_ram_mb": 0.0,
        "vram_delta_mb": 0.0,
        "error_msg": None,
    }

    dummy_input, input_desc = _synthesize_dummy_input(model, cfg, device)
    result["input_desc"] = input_desc

    model.eval()

    # Track VRAM and RAM prior to forward
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        vram_start = torch.cuda.memory_allocated(device) / (1024 ** 2)
    else:
        vram_start = 0.0

    ram_start = _get_ram_info()["process_rss_mb"]

    # Compute FLOPs using PyTorch 2.1+ FlopCounterMode
    try:
        from torch.utils.flop_counter import FlopCounterMode
        counter = FlopCounterMode(display=False, depth=3)
        with counter, torch.no_grad():
            out = model(dummy_input)
        total_flops = counter.get_total_flops()
        try:
            result["flops_table"] = counter.get_table(depth=3)
        except Exception:
            result["flops_table"] = None
    except Exception as flop_err:
        total_flops = 0
        try:
            with torch.no_grad():
                out = model(dummy_input)
        except Exception as fwd_err:
            result["error_msg"] = str(fwd_err)
            return result

    # Format output description
    if isinstance(out, torch.Tensor):
        result["output_desc"] = f"Tensor shape: {tuple(out.shape)}, dtype: {out.dtype}"
    elif isinstance(out, dict):
        keys_summary = []
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                keys_summary.append(f"'{k}': {tuple(v.shape)}")
            elif isinstance(v, dict):
                sub_keys = [f"'{sk}': {tuple(sv.shape)}" for sk, sv in v.items() if isinstance(sv, torch.Tensor)]
                keys_summary.append(f"'{k}': {{{', '.join(sub_keys)}}}")
            else:
                keys_summary.append(f"'{k}': {type(v).__name__}")
        result["output_desc"] = f"Dict({', '.join(keys_summary)})"
    elif isinstance(out, (list, tuple)):
        result["output_desc"] = f"{type(out).__name__}(len={len(out)})"
    else:
        result["output_desc"] = f"{type(out).__name__}"

    # Track peak memory
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        vram_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        result["peak_vram_mb"] = vram_peak
        result["vram_delta_mb"] = max(0.0, vram_peak - vram_start)

    ram_end = _get_ram_info()["process_rss_mb"]
    result["peak_ram_mb"] = max(ram_start, ram_end)

    if total_flops > 0:
        result["flops"] = total_flops
        if total_flops >= 1e12:
            result["flops_desc"] = f"{total_flops / 1e12:.3f} TFLOPs ({total_flops:,} FLOPs)"
        else:
            result["flops_desc"] = f"{total_flops / 1e9:.3f} GFLOPs ({total_flops:,} FLOPs)"
        macs = total_flops / 2
        result["macs_desc"] = f"{macs / 1e9:.3f} GMACs ({int(macs):,} MACs)"
    else:
        result["flops_desc"] = "Not available (FlopCounterMode did not register operations)"
        result["macs_desc"] = "N/A"

    result["success"] = True
    return result


def _build_hierarchical_rows(
    module: nn.Module,
    total_model_params: int,
    prefix: str = "",
    depth: int = 1,
    max_depth: int = 3,
) -> List[Dict[str, Any]]:
    """Recursively collect detailed layer hierarchy with parameters, status, size, and proportions."""
    rows: List[Dict[str, Any]] = []
    children = list(module.named_children())
    if not children:
        return rows

    # Detect repetitive indexed children (e.g. 0..31 in ModuleList where all children have identical class and param count)
    all_numeric = all(name.isdigit() for name, _ in children)
    if all_numeric and len(children) > 4:
        first_c = children[0][1]
        same_type = len(set(c.__class__.__name__ for _, c in children)) == 1
        first_p = sum(p.numel() for p in first_c.parameters())
        same_param = all(sum(p.numel() for p in c.parameters()) == first_p for _, c in children)
        if same_type and same_param:
            c_tot = sum(p.numel() for _, c in children for p in c.parameters())
            c_train = sum(p.numel() for _, c in children for p in c.parameters() if p.requires_grad)
            c_size = sum(
                sum(p.numel() * p.element_size() for p in c.parameters())
                + sum(b.numel() * b.element_size() for b in c.buffers())
                for _, c in children
            ) / (1024 ** 2)
            pct_model = (c_tot / total_model_params * 100.0) if total_model_params > 0 else 0.0
            status = "TRAINABLE" if c_train == c_tot and c_tot > 0 else ("FROZEN" if c_train == 0 and c_tot > 0 else "PARTIAL")
            indent = "  " * (depth - 1)
            rows.append({
                "name": f"{indent}[{len(children)}x {first_c.__class__.__name__}]",
                "full_path": f"{prefix}[0..{len(children)-1}]",
                "type": f"{len(children)}x {first_c.__class__.__name__}",
                "total": c_tot,
                "trainable": c_train,
                "size_mb": c_size,
                "pct_parent": 100.0,
                "pct_model": pct_model,
                "status": status,
                "depth": depth,
                "note": f"({len(children)} x {first_p:,} params)",
            })
            return rows

    parent_total = sum(p.numel() for p in module.parameters())
    for name, child in children:
        c_total = sum(p.numel() for p in child.parameters())
        c_train = sum(p.numel() for p in child.parameters() if p.requires_grad)
        c_size = (
            sum(p.numel() * p.element_size() for p in child.parameters())
            + sum(b.numel() * b.element_size() for b in child.buffers())
        ) / (1024 ** 2)
        pct_model = (c_total / total_model_params * 100.0) if total_model_params > 0 else 0.0
        pct_parent = (c_total / parent_total * 100.0) if parent_total > 0 else 0.0

        if c_total == 0:
            status = "NO_PARAM"
        elif c_train == c_total:
            status = "TRAINABLE"
        elif c_train == 0:
            status = "FROZEN"
        else:
            status = f"PARTIAL ({c_train/c_total*100:.1f}%)"

        indent = "  " * (depth - 1)
        full_path = f"{prefix}.{name}" if prefix else name
        display_name = f"{indent}{name}"
        rows.append({
            "name": display_name,
            "full_path": full_path,
            "type": child.__class__.__name__,
            "total": c_total,
            "trainable": c_train,
            "size_mb": c_size,
            "pct_parent": pct_parent,
            "pct_model": pct_model,
            "status": status,
            "depth": depth,
            "note": "",
        })

        if depth < max_depth and list(child.named_children()) and c_total > 0:
            rows.extend(
                _build_hierarchical_rows(
                    child,
                    total_model_params=total_model_params,
                    prefix=full_path,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            )

    return rows


def show_model_summary(
    model: nn.Module,
    cfg: Any,
    device: torch.device,
    work_dir: str,
    logger: Optional[Any] = None,
    save_filename: str = "model_configuration.txt",
    hierarchy_depth: int = 3,
) -> str:
    """Analyze and display model architecture, parameters, size, FLOPs, and RAM/VRAM usage.

    Saves the complete report to `work_dir/model_configuration.txt` and logs it to terminal.

    Args:
        model: PyTorch model instance.
        cfg: Complete configuration object.
        device: Target execution device.
        work_dir: Working directory to save the output text file.
        logger: Optional logger instance.
        save_filename: Name of the report file (default: model_configuration.txt).
        hierarchy_depth: Depth level for hierarchical layer-by-layer parameter breakdown (default: 3).

    Returns:
        Formatted summary report string.
    """
    os.makedirs(work_dir, exist_ok=True)
    out_path = os.path.join(work_dir, save_filename)

    # Move model to target device and sync
    model = model.to(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)

    # 1. Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
    frozen_pct = (frozen_params / total_params * 100.0) if total_params > 0 else 0.0

    # 2. Model memory footprint
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_model_bytes = param_bytes + buffer_bytes
    param_mb = param_bytes / (1024 ** 2)
    buffer_mb = buffer_bytes / (1024 ** 2)
    total_model_mb = total_model_bytes / (1024 ** 2)

    # 3. Macro Submodule breakdown table (Level 1)
    submodule_rows = []
    for name, child in model.named_children():
        c_total = sum(p.numel() for p in child.parameters())
        c_train = sum(p.numel() for p in child.parameters() if p.requires_grad)
        c_bytes = sum(p.numel() * p.element_size() for p in child.parameters()) + sum(
            b.numel() * b.element_size() for b in child.buffers()
        )
        c_pct = (c_total / total_params * 100.0) if total_params > 0 else 0.0
        submodule_rows.append({
            "name": name,
            "type": child.__class__.__name__,
            "total": c_total,
            "trainable": c_train,
            "size_mb": c_bytes / (1024 ** 2),
            "pct": c_pct,
        })

    # 4. Micro Hierarchical Layer-by-layer breakdown (Levels 1 to hierarchy_depth)
    hierarchical_rows = _build_hierarchical_rows(
        model,
        total_model_params=total_params,
        prefix="",
        depth=1,
        max_depth=hierarchy_depth,
    )

    # 5. Memory statistics (RAM and VRAM)
    ram_info = _get_ram_info()
    vram_info = _get_vram_info(device)

    # 6. Dynamic profiling (FLOPs & Forward Memory)
    profile_info = _run_forward_profiling(model, cfg, device)

    # 7. Extract Model Configuration
    model_cfg_dict = _to_plain_dict(getattr(cfg, "model", {}))
    try:
        model_cfg_yaml = yaml.dump(model_cfg_dict, default_flow_style=False, sort_keys=False).strip()
    except Exception:
        model_cfg_yaml = str(model_cfg_dict)

    # Format Report
    lines = []
    sep = "=" * 90
    sub_sep = "-" * 90

    lines.append(sep)
    lines.append("MODEL CONFIGURATION & RESOURCE PROFILING REPORT")
    lines.append(sep)
    lines.append(f"Model Class           : {model.__class__.__name__}")
    lines.append(f"Model Type (Registry) : {model_cfg_dict.get('type', 'N/A')}")
    lines.append(f"Execution Device      : {device}")
    lines.append(f"Timestamp             : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append(sep)
    lines.append("1. PARAMETER STATISTICS & MEMORY FOOTPRINT")
    lines.append(sep)
    lines.append(f"Total Parameters      : {total_params:>15,}  ({total_params / 1e6:>8.3f} M)")
    lines.append(f"Trainable Parameters  : {trainable_params:>15,}  ({trainable_params / 1e6:>8.3f} M)  [{trainable_pct:6.2f}%]")
    lines.append(f"Frozen Parameters     : {frozen_params:>15,}  ({frozen_params / 1e6:>8.3f} M)  [{frozen_pct:6.2f}%]")
    lines.append(sub_sep)
    lines.append(f"Parameters Memory     : {param_mb:>15.2f} MB  ({param_mb / 1024:>6.3f} GB)")
    lines.append(f"Buffers Memory        : {buffer_mb:>15.2f} MB")
    lines.append(f"Total Model Size      : {total_model_mb:>15.2f} MB  ({total_model_mb / 1024:>6.3f} GB)")
    lines.append("")

    lines.append(sep)
    lines.append("2. TOP-LEVEL SUBMODULE BREAKDOWN (MACRO VIEW)")
    lines.append(sep)
    header = f"{'Submodule':<20} {'Class Type':<22} {'Total Params':<14} {'Trainable':<12} {'Size (MB)':<11} {'% Total':<8}"
    lines.append(header)
    lines.append(sub_sep)
    for r in submodule_rows:
        row_str = (
            f"{r['name']:<20} "
            f"{r['type']:<22} "
            f"{r['total']:>12,} "
            f"{r['trainable']:>10,} "
            f"{r['size_mb']:>9.2f} "
            f"{r['pct']:>7.2f}%"
        )
        lines.append(row_str)
    lines.append(sub_sep)
    lines.append("")

    lines.append(sep)
    lines.append(f"3. DETAILED LAYER-BY-LAYER HIERARCHICAL ANALYSIS (DEPTH <= {hierarchy_depth})")
    lines.append(sep)
    h_header = f"{'Layer / Submodule':<34} {'Class Type':<20} {'Params':<12} {'Status':<11} {'Size (MB)':<10} {'% Parent':<9} {'% Total':<8}"
    lines.append(h_header)
    lines.append(sub_sep)
    for hr in hierarchical_rows:
        extra_note = f" {hr['note']}" if hr.get("note") else ""
        row_str = (
            f"{hr['name']:<34} "
            f"{hr['type']:<20} "
            f"{hr['total']:>10,} "
            f"{hr['status']:<11} "
            f"{hr['size_mb']:>8.2f} "
            f"{hr['pct_parent']:>7.2f}% "
            f"{hr['pct_model']:>7.2f}%"
            f"{extra_note}"
        )
        lines.append(row_str)
    lines.append(sub_sep)
    lines.append("")

    lines.append(sep)
    lines.append("4. RESOURCE USAGE (RAM & VRAM)")
    lines.append(sep)
    lines.append("[CPU System Memory (RAM)]")
    lines.append(f"  Process Memory (RSS): {ram_info['process_rss_mb']:>10.2f} MB  (Peak during forward: {profile_info['peak_ram_mb']:>10.2f} MB)")
    lines.append(f"  System Total RAM    : {ram_info['system_total_mb']:>10.2f} MB  ({ram_info['system_total_mb'] / 1024:>6.2f} GB)")
    lines.append(f"  System Available RAM: {ram_info['system_avail_mb']:>10.2f} MB  ({ram_info['system_avail_mb'] / 1024:>6.2f} GB)")
    lines.append("")
    lines.append("[GPU Device Memory (VRAM)]")
    if vram_info["is_cuda"]:
        lines.append(f"  Device Name         : {vram_info['device_name']}")
        lines.append(f"  Total Device VRAM   : {vram_info['total_vram_mb']:>10.2f} MB  ({vram_info['total_vram_mb'] / 1024:>6.2f} GB)")
        lines.append(f"  Model VRAM Allocated: {vram_info['allocated_mb']:>10.2f} MB")
        lines.append(f"  Model VRAM Reserved : {vram_info['reserved_mb']:>10.2f} MB")
        if profile_info["success"]:
            lines.append(f"  Forward Peak Alloc  : {profile_info['peak_vram_mb']:>10.2f} MB")
            lines.append(f"  Activation Delta    : {profile_info['vram_delta_mb']:>10.2f} MB")
    else:
        lines.append(f"  Device '{device}' is not CUDA. GPU VRAM tracking skipped.")
    lines.append("")

    lines.append(sep)
    lines.append("5. COMPUTATIONAL COMPLEXITY (FLOPs & MACs)")
    lines.append(sep)
    if profile_info["success"]:
        lines.append(f"  Profile Input Shape : {profile_info['input_desc']}")
        lines.append(f"  Profile Output      : {profile_info['output_desc']}")
        lines.append(f"  Computational FLOPs : {profile_info['flops_desc']}")
        lines.append(f"  MACs (Multiplier)   : {profile_info['macs_desc']}")
        if profile_info.get("flops_table"):
            lines.append("")
            lines.append("  [Layer-by-Layer FLOPs Breakdown]")
            for f_line in profile_info["flops_table"].strip().split("\n"):
                lines.append(f"    {f_line}")
    else:
        lines.append(f"  Profile Input Shape : {profile_info['input_desc']}")
        lines.append(f"  Dynamic Profiling   : Failed / Skipped ({profile_info.get('error_msg', 'Unknown')})")
    lines.append("")

    lines.append(sep)
    lines.append("6. MODEL CONFIGURATION (YAML)")
    lines.append(sep)
    lines.append(model_cfg_yaml)
    lines.append("")

    lines.append(sep)
    lines.append("7. PYTORCH MODEL ARCHITECTURE")
    lines.append(sep)
    lines.append(str(model))
    lines.append(sep)

    report_text = "\n".join(lines)

    # Write to file
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    # Output to terminal & logger
    print(report_text)
    if logger:
        logger.info(f"Model configuration summary saved to: {out_path}")

    return report_text
