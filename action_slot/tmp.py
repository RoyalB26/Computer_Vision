import lmdb
import pickle
from torch.utils.data import Dataset, ConcatDataset, DataLoader

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self.env = None
        
        # Chỉ mở tạm để lấy tổng số lượng mẫu (length), sau đó đóng ngay
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        env.close()

    def _init_db(self):
        # Mở môi trường LMDB bên trong từng worker
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.env is None:
            self._init_db()

        with self.env.begin(write=False) as txn:
            # Giả sử key của bạn được lưu dạng f"{index}".encode()
            key = f"{index}".encode("ascii")
            data = txn.get(key)
            
        # Giải mã data (tùy thuộc vào cách bạn encode lúc tạo LMDB, vd: pickle, cv2, json...)
        sample = pickle.loads(data)
        return sample

# --- GỘP NHIỀU LMDB ---
lmdb_paths = [r"D:/apps/Datasets/TACO/data/ap_Town07_lmdb/rgb_front/", r"D:/apps/Datasets/TACO/data/runner_Town03_lmdb/rgb_front/"]

datasets = [LMDBDataset(path) for path in lmdb_paths]
combined_dataset = ConcatDataset(datasets)

dataloader = DataLoader(
    combined_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

print(len(dataloader))