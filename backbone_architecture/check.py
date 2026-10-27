"""Script to inspect and export MobileNet, ResNet, and YOLO architectures (YOLOv1 through YOLOv12) to .txt files."""
import argparse
import os
import sys
from typing import Optional, Tuple
import yaml
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None

try:
    import torchvision.models as tv_models
except ImportError:
    tv_models = None

try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules import Conv
except ImportError:
    YOLO = None
    DetectionModel = None
    Conv = None


def format_model_architecture(
    model: nn.Module,
    model_name: str,
    input_size: Optional[Tuple[int, ...]] = (1, 3, 224, 224),
) -> str:
    """Format full architecture, layer breakdown, and parameter statistics into a clean text string."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    lines = []
    lines.append("=" * 80)
    lines.append(f" MODEL ARCHITECTURE REPORT: {model_name.upper()}")
    lines.append("=" * 80)
    lines.append(f"Model Class          : {model.__class__.__name__}")
    lines.append(f"Total Parameters     : {total_params:,} ({total_params / 1e6:.2f} M)")
    lines.append(f"Trainable Parameters : {trainable_params:,} ({trainable_params / 1e6:.2f} M)")
    lines.append(f"Frozen Parameters    : {frozen_params:,} ({frozen_params / 1e6:.2f} M)")
    lines.append(f"Estimated Model Size : {param_size_mb:.2f} MB")
    if input_size:
        lines.append(f"Standard Input Size  : {list(input_size)}")
    lines.append("-" * 80)

    # 1. Submodules / Top-level Children parameter breakdown table
    lines.append("\n[1. TOP-LEVEL SUBMODULES BREAKDOWN]")
    lines.append(f"{'Submodule Name':<30} {'Type':<30} {'Parameters':>15}")
    lines.append("-" * 80)
    children = list(model.named_children())
    # If the only child is 'model' (e.g. Ultralytics DetectionModel), drill down to its layers
    if len(children) == 1 and children[0][0] == "model" and isinstance(children[0][1], nn.Sequential):
        children = [(f"layer_{i:02d}", layer) for i, layer in enumerate(children[0][1])]

    for name, child in children:
        child_params = sum(p.numel() for p in child.parameters())
        child_type = child.__class__.__name__
        lines.append(f"{name:<30} {child_type:<30} {child_params:>15,}")
    lines.append("-" * 80)

    # 2. Complete hierarchical structure
    lines.append("\n[2. DETAILED PYTORCH MODULE STRUCTURE]")
    lines.append("-" * 80)
    lines.append(str(model))
    lines.append("=" * 80)

    return "\n".join(lines)


def export_model_to_txt(
    model: nn.Module,
    model_name: str,
    output_path: str,
    input_size: Optional[Tuple[int, ...]] = (1, 3, 224, 224),
) -> str:
    """Write model architecture details into a text file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    report_text = format_model_architecture(model, model_name, input_size=input_size)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return os.path.abspath(output_path)


# ==============================================================================
# Model Builders: YOLOv1, YOLOv2, YOLOv4, YOLOv7
# ==============================================================================

class YOLOv1Model(nn.Module):
    """YOLOv1: You Only Look Once (Redmon et al., CVPR 2016).
    Standard architecture: 24 conv layers + 4 maxpools + 1 stride-2 conv + 2 FC layers.
    """
    def __init__(self, nc: int = 80, s: int = 7, b: int = 2):
        super().__init__()
        conv_cls = Conv if Conv is not None else self._fallback_conv
        self.model = nn.Sequential(
            conv_cls(3, 64, 7, 2, 3),            # layer_00
            nn.MaxPool2d(2, 2),                  # layer_01
            conv_cls(64, 192, 3, 1, 1),          # layer_02
            nn.MaxPool2d(2, 2),                  # layer_03
            conv_cls(192, 128, 1, 1),            # layer_04
            conv_cls(128, 256, 3, 1, 1),         # layer_05
            conv_cls(256, 256, 1, 1),            # layer_06
            conv_cls(256, 512, 3, 1, 1),         # layer_07
            nn.MaxPool2d(2, 2),                  # layer_08
            conv_cls(512, 256, 1, 1),            # layer_09
            conv_cls(256, 512, 3, 1, 1),         # layer_10
            conv_cls(512, 256, 1, 1),            # layer_11
            conv_cls(256, 512, 3, 1, 1),         # layer_12
            conv_cls(512, 256, 1, 1),            # layer_13
            conv_cls(256, 512, 3, 1, 1),         # layer_14
            conv_cls(512, 256, 1, 1),            # layer_15
            conv_cls(256, 512, 3, 1, 1),         # layer_16
            conv_cls(512, 512, 1, 1),            # layer_17
            conv_cls(512, 1024, 3, 1, 1),        # layer_18
            nn.MaxPool2d(2, 2),                  # layer_19
            conv_cls(1024, 512, 1, 1),           # layer_20
            conv_cls(512, 1024, 3, 1, 1),        # layer_21
            conv_cls(1024, 512, 1, 1),           # layer_22
            conv_cls(512, 1024, 3, 1, 1),        # layer_23
            conv_cls(1024, 1024, 3, 1, 1),       # layer_24
            conv_cls(1024, 1024, 3, 2, 1),       # layer_25
            conv_cls(1024, 1024, 3, 1, 1),       # layer_26
            conv_cls(1024, 1024, 3, 1, 1),       # layer_27
            nn.Flatten(),                        # layer_28
            nn.Linear(1024 * 7 * 7, 4096),       # layer_29
            nn.LeakyReLU(0.1, inplace=True),     # layer_30
            nn.Dropout(0.5),                     # layer_31
            nn.Linear(4096, s * s * (b * 5 + nc)), # layer_32
        )

    @staticmethod
    def _fallback_conv(c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        p = (k - 1) // 2 if p is None else p
        return nn.Sequential(
            nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity(),
        )


def get_yolov2_model(nc: int = 80) -> nn.Module:
    """YOLOv2 (YOLO9000: Darknet-19 backbone + Passthrough/Reorg layer + Anchor Detection Head)."""
    if DetectionModel is None:
        raise ImportError("ultralytics DetectionModel is required to load YOLOv2.")
    yolov2_yaml = f"""
nc: {nc}
scales:
  m: [1.0, 1.0, 1024]

backbone:
  - [-1, 1, Conv, [32, 3, 1]]       # 0
  - [-1, 1, nn.MaxPool2d, [2, 2]]   # 1
  - [-1, 1, Conv, [64, 3, 1]]       # 2
  - [-1, 1, nn.MaxPool2d, [2, 2]]   # 3
  - [-1, 1, Conv, [128, 3, 1]]      # 4
  - [-1, 1, Conv, [64, 1, 1]]       # 5
  - [-1, 1, Conv, [128, 3, 1]]      # 6
  - [-1, 1, nn.MaxPool2d, [2, 2]]   # 7
  - [-1, 1, Conv, [256, 3, 1]]      # 8
  - [-1, 1, Conv, [128, 1, 1]]      # 9
  - [-1, 1, Conv, [256, 3, 1]]      # 10
  - [-1, 1, nn.MaxPool2d, [2, 2]]   # 11
  - [-1, 1, Conv, [512, 3, 1]]      # 12
  - [-1, 1, Conv, [256, 1, 1]]      # 13
  - [-1, 1, Conv, [512, 3, 1]]      # 14
  - [-1, 1, Conv, [256, 1, 1]]      # 15
  - [-1, 1, Conv, [512, 3, 1]]      # 16 (P4 feature map 26x26)
  - [-1, 1, nn.MaxPool2d, [2, 2]]   # 17
  - [-1, 1, Conv, [1024, 3, 1]]     # 18
  - [-1, 1, Conv, [512, 1, 1]]      # 19
  - [-1, 1, Conv, [1024, 3, 1]]     # 20
  - [-1, 1, Conv, [512, 1, 1]]      # 21
  - [-1, 1, Conv, [1024, 3, 1]]     # 22
  - [-1, 1, Conv, [1024, 3, 1]]     # 23
  - [-1, 1, Conv, [1024, 3, 1]]     # 24

head:
  - [16, 1, Focus, [256, 1, 1]]     # 25 Passthrough route (reorg 26x26 -> 13x13)
  - [[-1, 24], 1, Concat, [1]]      # 26 Concat (256 + 1024 = 1280)
  - [-1, 1, Conv, [1024, 3, 1]]     # 27
  - [[-1], 1, Detect, [nc]]         # 28 Output detection head
"""
    return DetectionModel(yaml.safe_load(yolov2_yaml), nc=nc)


def get_yolov4_model(nc: int = 80) -> nn.Module:
    """YOLOv4 (Bochkovskiy et al., 2020: CSPDarknet53 + SPP + PANet + YOLOv3 Head)."""
    if DetectionModel is None:
        raise ImportError("ultralytics DetectionModel is required to load YOLOv4.")
    yolov4_yaml = f"""
nc: {nc}
scales:
  m: [1.0, 1.0, 1024]

backbone:
  - [-1, 1, Conv, [32, 3, 1]]
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, BottleneckCSP, [64]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, BottleneckCSP, [128]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 8, BottleneckCSP, [256]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 8, BottleneckCSP, [512]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 4, BottleneckCSP, [1024]]

head:
  - [-1, 1, SPP, [1024, [5, 9, 13]]]
  - [-1, 1, Conv, [512, 1, 1]]
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, BottleneckCSP, [512]]
  - [-1, 1, Conv, [256, 1, 1]]
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, BottleneckCSP, [256]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, BottleneckCSP, [512]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 2, BottleneckCSP, [1024]]
  - [[19, 22, 25], 1, Detect, [nc]]
"""
    return DetectionModel(yaml.safe_load(yolov4_yaml), nc=nc)


def get_yolov7_model(nc: int = 80) -> nn.Module:
    """Official YOLOv7 architecture (WongKinYiu/yolov7: E-ELAN + MP + SPPCSPC + RepConv)."""
    hub_dir = torch.hub.get_dir()
    yolov7_repo_dir = os.path.join(hub_dir, "WongKinYiu_yolov7_main")
    if not os.path.exists(yolov7_repo_dir):
        import urllib.request
        import zipfile
        zip_url = "https://github.com/WongKinYiu/yolov7/archive/refs/heads/main.zip"
        zip_dest = os.path.join(hub_dir, "yolov7_main.zip")
        os.makedirs(hub_dir, exist_ok=True)
        urllib.request.urlretrieve(zip_url, zip_dest)
        with zipfile.ZipFile(zip_dest, "r") as zip_ref:
            zip_ref.extractall(hub_dir)
        if os.path.exists(os.path.join(hub_dir, "yolov7-main")) and not os.path.exists(yolov7_repo_dir):
            os.rename(os.path.join(hub_dir, "yolov7-main"), yolov7_repo_dir)

    if yolov7_repo_dir not in sys.path:
        sys.path.insert(0, yolov7_repo_dir)

    # Mock pkg_resources if missing (Python 3.12+ / setuptools >= 70 compatibility)
    if "pkg_resources" not in sys.modules:
        import types
        pr = types.ModuleType("pkg_resources")
        pr.parse_version = lambda v: [int(x) for x in v.split(".") if x.isdigit()]
        sys.modules["pkg_resources"] = pr

    from models.yolo import Model as YOLOv7Model
    cfg_path = os.path.join(yolov7_repo_dir, "cfg", "training", "yolov7.yaml")
    return YOLOv7Model(cfg_path, ch=3, nc=nc)


# ==============================================================================
# Main Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inspect and export neural network architectures.")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Target models to export: 'all', 'yolov1-7', 'yolo', or 'backbones'. Default is 'all'.",
    )
    args = parser.parse_args()
    target = args.models.lower()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    generated_files = []

    run_backbones = target in ["all", "backbones", "mobilenet", "mobilenet_v3_small"]
    run_yolo_1_7 = target in ["all", "yolo", "yolov1-7", "v1-7"]
    run_yolo_8_12 = target in ["all", "yolo"]

    # -------------------------------------------------------------
    # Classification Backbones: MobileNetV4, MobileNetV3 Large, MobileNetV3 Small (1.0x), MobileNetV3 Nano (0.5x), ResNet18
    # -------------------------------------------------------------
    if run_backbones:
        if target != "mobilenet_v3_small":
            print("[1/5] Loading MobileNetV4...")
            if timm is not None:
                mobilenetv4_model = timm.create_model("mobilenetv4_conv_small", pretrained=False)
                mobilenetv4_path = os.path.join(current_dir, "mobilenetv4.txt")
                out_m4 = export_model_to_txt(mobilenetv4_model, "mobilenetv4_conv_small", mobilenetv4_path)
                print(f"  -> Exported MobileNetV4 architecture to: {out_m4}")
                generated_files.append(mobilenetv4_path)
            else:
                print("  [SKIP] 'timm' not installed for MobileNetV4.")

            print("[2/5] Loading MobileNetV3 Large...")
            if tv_models is not None and hasattr(tv_models, "mobilenet_v3_large"):
                mobilenetv3_model = tv_models.mobilenet_v3_large(weights=None)
            elif timm is not None:
                mobilenetv3_model = timm.create_model("mobilenetv3_large_100", pretrained=False)
            else:
                mobilenetv3_model = None

            if mobilenetv3_model is not None:
                mobilenetv3_path = os.path.join(current_dir, "mobilenetv3.txt")
                out_m3 = export_model_to_txt(mobilenetv3_model, "mobilenet_v3_large", mobilenetv3_path)
                print(f"  -> Exported MobileNetV3 Large architecture to: {out_m3}")
                generated_files.append(mobilenetv3_path)

        # MobileNetV3 Small (Standard 1.0x)
        print("[3/5] Loading MobileNetV3 Small...")
        if tv_models is not None and hasattr(tv_models, "mobilenet_v3_small"):
            mobilenetv3_small_model = tv_models.mobilenet_v3_small(weights=None)
            small_variant_name = "mobilenet_v3_small (torchvision 1.0x)"
        elif timm is not None and "mobilenetv3_small_100" in timm.list_models():
            mobilenetv3_small_model = timm.create_model("mobilenetv3_small_100", pretrained=False)
            small_variant_name = "mobilenetv3_small_100 (timm 1.0x)"
        else:
            mobilenetv3_small_model = None
            small_variant_name = "none"

        if mobilenetv3_small_model is not None:
            mobilenetv3_small_path = os.path.join(current_dir, "mobilenet_v3_small.txt")
            out_m3_small = export_model_to_txt(
                mobilenetv3_small_model,
                f"mobilenet_v3_small ({small_variant_name})",
                mobilenetv3_small_path,
            )
            print(f"  -> Exported MobileNetV3 Small architecture to: {out_m3_small}")
            generated_files.append(mobilenetv3_small_path)

        if target != "mobilenet_v3_small":
            print("[4/5] Loading MobileNetV3 Nano...")
            if timm is not None and "mobilenetv3_small_050" in timm.list_models():
                mobilenetv3_nano_model = timm.create_model("mobilenetv3_small_050", pretrained=False)
                nano_variant_name = "mobilenetv3_small_050"
            elif tv_models is not None and hasattr(tv_models, "mobilenet_v3_small"):
                mobilenetv3_nano_model = tv_models.mobilenet_v3_small(weights=None)
                nano_variant_name = "mobilenet_v3_small"
            else:
                mobilenetv3_nano_model = None
                nano_variant_name = "none"

            if mobilenetv3_nano_model is not None:
                mobilenetv3_nano_path = os.path.join(current_dir, "mobilenetv3_nano.txt")
                out_m3_nano = export_model_to_txt(
                    mobilenetv3_nano_model,
                    f"mobilenetv3_nano ({nano_variant_name})",
                    mobilenetv3_nano_path,
                )
                print(f"  -> Exported MobileNetV3 Nano architecture to: {out_m3_nano}")
                generated_files.append(mobilenetv3_nano_path)

            print("[5/5] Loading ResNet18 (r18)...")
            if tv_models is not None and hasattr(tv_models, "resnet18"):
                r18_model = tv_models.resnet18(weights=None)
            elif timm is not None:
                r18_model = timm.create_model("resnet18", pretrained=False)
            else:
                r18_model = None

            if r18_model is not None:
                r18_path = os.path.join(current_dir, "r18.txt")
                out_r18 = export_model_to_txt(r18_model, "resnet18", r18_path)
                print(f"  -> Exported ResNet18 (r18) architecture to: {out_r18}")
                generated_files.append(r18_path)

    # -------------------------------------------------------------
    # YOLO Architectures: YOLOv1 to YOLOv7
    # -------------------------------------------------------------
    if run_yolo_1_7:
        print("\n--- Processing YOLOv1 through YOLOv7 ---")

        # 1. YOLOv1
        print("[YOLO 1/7] Loading YOLOv1...")
        yolov1_model = YOLOv1Model(nc=80)
        yolov1_path = os.path.join(current_dir, "yolov1.txt")
        out_yolo1 = export_model_to_txt(
            yolov1_model,
            "yolov1 (Darknet-24 + Fully Connected Head)",
            yolov1_path,
            input_size=(1, 3, 448, 448),
        )
        print(f"  -> Exported YOLOv1 architecture to: {out_yolo1}")
        generated_files.append(yolov1_path)

        # 2. YOLOv2
        print("[YOLO 2/7] Loading YOLOv2...")
        yolov2_model = get_yolov2_model(nc=80)
        yolov2_path = os.path.join(current_dir, "yolov2.txt")
        out_yolo2 = export_model_to_txt(
            yolov2_model,
            "yolov2 (Darknet-19 + Passthrough Route)",
            yolov2_path,
            input_size=(1, 3, 416, 416),
        )
        print(f"  -> Exported YOLOv2 architecture to: {out_yolo2}")
        generated_files.append(yolov2_path)

        # 3. YOLOv3
        print("[YOLO 3/7] Loading YOLOv3...")
        if YOLO is not None:
            yolo3_obj = YOLO("yolov3.yaml")
            yolov3_path = os.path.join(current_dir, "yolov3.txt")
            out_yolo3 = export_model_to_txt(
                yolo3_obj.model,
                "yolov3 (Darknet-53 + Feature Pyramid Network)",
                yolov3_path,
                input_size=(1, 3, 640, 640),
            )
            print(f"  -> Exported YOLOv3 architecture to: {out_yolo3}")
            generated_files.append(yolov3_path)

        # 4. YOLOv4
        print("[YOLO 4/7] Loading YOLOv4...")
        yolov4_model = get_yolov4_model(nc=80)
        yolov4_path = os.path.join(current_dir, "yolov4.txt")
        out_yolo4 = export_model_to_txt(
            yolov4_model,
            "yolov4 (CSPDarknet53 + SPP + PANet)",
            yolov4_path,
            input_size=(1, 3, 640, 640),
        )
        print(f"  -> Exported YOLOv4 architecture to: {out_yolo4}")
        generated_files.append(yolov4_path)

        # 5. YOLOv5
        print("[YOLO 5/7] Loading YOLOv5...")
        if YOLO is not None:
            yolo5_obj = YOLO("yolov5s.yaml")
            yolov5_path = os.path.join(current_dir, "yolov5.txt")
            out_yolo5 = export_model_to_txt(
                yolo5_obj.model,
                "yolov5s (CSPDarknet + PANet)",
                yolov5_path,
                input_size=(1, 3, 640, 640),
            )
            print(f"  -> Exported YOLOv5 architecture to: {out_yolo5}")
            generated_files.append(yolov5_path)

        # 6. YOLOv6
        print("[YOLO 6/7] Loading YOLOv6...")
        if YOLO is not None:
            yolo6_obj = YOLO("yolov6s.yaml")
            yolov6_path = os.path.join(current_dir, "yolov6.txt")
            out_yolo6 = export_model_to_txt(
                yolo6_obj.model,
                "yolov6s (EfficientRep + Rep-PAN)",
                yolov6_path,
                input_size=(1, 3, 640, 640),
            )
            print(f"  -> Exported YOLOv6 architecture to: {out_yolo6}")
            generated_files.append(yolov6_path)

        # 7. YOLOv7
        print("[YOLO 7/7] Loading YOLOv7...")
        yolov7_model = get_yolov7_model(nc=80)
        yolov7_path = os.path.join(current_dir, "yolov7.txt")
        out_yolo7 = export_model_to_txt(
            yolov7_model,
            "yolov7 (E-ELAN + SPPCSPC + RepConv)",
            yolov7_path,
            input_size=(1, 3, 640, 640),
        )
        print(f"  -> Exported YOLOv7 architecture to: {out_yolo7}")
        generated_files.append(yolov7_path)

    # -------------------------------------------------------------
    # YOLO Modern Architectures: YOLOv8 to YOLOv12
    # -------------------------------------------------------------
    if run_yolo_8_12 and YOLO is not None:
        print("\n--- Processing YOLOv8 through YOLOv12 ---")

        # YOLOv8
        print("[YOLO 8/12] Loading YOLOv8...")
        yolo8_obj = YOLO("yolov8n.yaml")
        yolov8_path = os.path.join(current_dir, "yolov8.txt")
        out_yolo8 = export_model_to_txt(
            yolo8_obj.model,
            "yolov8n (C2f + Anchor-Free Decoupled Head)",
            yolov8_path,
            input_size=(1, 3, 640, 640),
        )
        print(f"  -> Exported YOLOv8 architecture to: {out_yolo8}")
        generated_files.append(yolov8_path)

        # YOLOv9
        print("[YOLO 9/12] Loading YOLOv9...")
        yolo9_obj = YOLO("yolov9c.yaml")
        yolov9_path = os.path.join(current_dir, "yolov9.txt")
        out_yolo9 = export_model_to_txt(
            yolo9_obj.model,
            "yolov9c (GELAN / Programmable Gradient Information)",
            yolov9_path,
            input_size=(1, 3, 640, 640),
        )
        print(f"  -> Exported YOLOv9 architecture to: {out_yolo9}")
        generated_files.append(yolov9_path)

        # YOLOv10
        print("[YOLO 10/12] Loading YOLOv10...")
        yolo10_obj = YOLO("yolov10n.yaml")
        yolov10_path = os.path.join(current_dir, "yolov10.txt")
        out_yolo10 = export_model_to_txt(
            yolo10_obj.model,
            "yolov10n (NMS-free Dual Label Assignment)",
            yolov10_path,
            input_size=(1, 3, 640, 640),
        )
        print(f"  -> Exported YOLOv10 architecture to: {out_yolo10}")
        generated_files.append(yolov10_path)

        # YOLOv11
        print("[YOLO 11/12] Loading YOLOv11...")
        yolo11_obj = YOLO("yolo11n.yaml")
        yolov11_path = os.path.join(current_dir, "yolov11.txt")
        out_yolo11 = export_model_to_txt(
            yolo11_obj.model,
            "yolo11n (C3k2 / C2PSA Enhanced)",
            yolov11_path,
            input_size=(1, 3, 640, 640),
        )
        print(f"  -> Exported YOLOv11 architecture to: {out_yolo11}")
        generated_files.append(yolov11_path)

        # YOLOv12
        print("[YOLO 12/12] Loading YOLOv12...")
        yolo12_obj = YOLO("yolo12n.yaml")
        yolov12_path = os.path.join(current_dir, "yolov12.txt")
        out_yolo12 = export_model_to_txt(
            yolo12_obj.model,
            "yolo12n (Attention-Centric Detection Model)",
            yolov12_path,
            input_size=(1, 3, 640, 640),
        )
        print(f"  -> Exported YOLOv12 architecture to: {out_yolo12}")
        generated_files.append(yolov12_path)

    print("\n" + "=" * 80)
    print(f"Successfully generated {len(generated_files)} architecture reports:")
    for idx, path in enumerate(generated_files, start=1):
        print(f"  {idx:02d}. {path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
