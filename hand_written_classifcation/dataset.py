import io
import pickle
import lmdb
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Any, Optional

from BaseModel.core.registrar import DATASETS
from BaseModel.dataset.base_dataset import BaseDataset


@DATASETS.register_module()
class LMDBHandwrittenDataset(BaseDataset):
    """Dataset đọc từ LMDB, chuyển sang RGB, giữ nguyên kích thước gốc của ảnh,

    chuẩn hóa nhãn target one-hot cho BCEWithLogitsLoss.
    Các bước transform / downsample ảnh được chuyển vào Model chạy trực tiếp trên GPU.
    """

    def __init__(
        self,
        lmdb_path: str,
        num_classes: int = 62,
        pipeline=None,
        test_mode: bool = False,
    ):
        self.lmdb_path = lmdb_path
        self.num_classes = num_classes
        self.env = None
        self.txn = None
        super().__init__(
            data_root=None, ann_file=None, pipeline=pipeline, test_mode=test_mode
        )

    def load_annotations(self) -> List[Dict[str, Any]]:
        # Mở LMDB lấy số lượng bản ghi
        env_init = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env_init.begin(write=False) as txn:
            total_samples = int(txn.get(b"__len__").decode("ascii"))
        env_init.close()

        # Tạo danh sách index
        data_infos = [{"index": idx} for idx in range(total_samples)]
        return data_infos

    def _init_db(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self.txn = self.env.begin(write=False)

    def prepare_data(self, idx: int) -> Optional[Dict[str, Any]]:
        self._init_db()
        key = f"{idx:08d}".encode("ascii")
        data_bytes = self.txn.get(key)
        sample = pickle.loads(data_bytes)

        # sample['image'] là raw file bytes của ảnh gốc
        img_raw = sample["image"]
        if isinstance(img_raw, bytes):
            try:
                img = Image.open(io.BytesIO(img_raw)).convert("RGB")
            except Exception:
                w, h = sample.get("size", (28, 28))
                img = Image.frombytes("L", (w, h), img_raw).convert("RGB")
        else:
            w, h = sample.get("size", (28, 28))
            img = Image.frombytes("L", (w, h), img_raw).convert("RGB")

        # Label: tạo target one-hot float tensor cho BCEWithLogitsLoss
        label_idx = int(sample["label"])
        one_hot = F.one_hot(
            torch.tensor(label_idx, dtype=torch.long),
            num_classes=self.num_classes,
        ).float()

        data_info = {
            "image": img,  # PIL Image RGB giữ nguyên kích thước gốc (1200x900)
            "target": one_hot,  # FloatTensor shape [num_classes] cho BCEWithLogitsLoss
            "label_idx": label_idx,  # Integer nhãn cho metric đo accuracy
            "class_name": sample.get("class_name", str(label_idx)),
            "orig_size": sample.get("size", img.size),
        }

        # Áp dụng pipeline (ví dụ ToTensor)
        data = self.pipeline(data_info)
        return data
