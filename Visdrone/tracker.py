"""VisDrone specialized Excel Tracker Hook and utilities.

Extends ExcelTracker to natively log detection head decomposition:
analyzes how FPN, Class Slots, Deformable Local Splatting, and Global Context
contribute to feat_map and affect the final predictions:
cls_logits, pred_offsets = self.head(feat_map).
"""
from typing import Optional, List
from BaseModel.utils.tracker import ExcelTracker
from BaseModel.core.registrar import HOOKS


@HOOKS.register_module()
class VisDroneDetectionTracker(ExcelTracker):
    """Specialized Excel Tracker Hook for VisDrone.

    Natively tracks 3 sheets:
    1. 'Epoch_Overview': Loss, Val Metrics, Lr, Time.
    2. 'Gradients_Detail': Gradient norms for all submodules.
    3. 'Detection': Detailed analysis of feat_map components and their impact ratios
       on cls_logits (classification) and pred_offsets (box regression).
    """

    def __init__(
        self,
        filename: str = "training_metrics_visdrone.xlsx",
        submodules: Optional[List[str]] = None,
        track_gradients: bool = True,
        grad_interval: int = 1,
        track_detection: bool = True,
        detection_interval: int = 10,
        auto_save: bool = True,
        resume: bool = True,
    ):
        super().__init__(
            filename=filename,
            submodules=submodules,
            track_gradients=track_gradients,
            grad_interval=grad_interval,
            track_detection=track_detection,
            detection_interval=detection_interval,
            auto_save=auto_save,
            resume=resume,
        )


# Aliases
VisDroneExcelTracker = VisDroneDetectionTracker
HOOKS.register_module(module=VisDroneExcelTracker, name="VisDroneExcelTracker")
