"""Helper functions and utilities to assist users in building custom datasets."""
import csv
import io
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def load_image(img_input: Union[str, bytes, bytearray, io.BytesIO, Image.Image], mode: str = "RGB") -> Image.Image:
    """Universal image loading helper.

    Accepts file path, raw binary bytes, io.BytesIO, or existing PIL Image,
    and returns a decoded PIL Image in the requested color mode.

    Args:
        img_input: Filepath string, binary bytes, or BytesIO buffer.
        mode: Target color mode ('RGB', 'L', 'RGBA'). Defaults to 'RGB'.
    """
    if isinstance(img_input, Image.Image):
        return img_input.convert(mode) if img_input.mode != mode else img_input

    if isinstance(img_input, str):
        if not os.path.isfile(img_input):
            raise FileNotFoundError(f"Image file does not exist: {img_input}")
        return Image.open(img_input).convert(mode)

    if isinstance(img_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(img_input)).convert(mode)

    if isinstance(img_input, io.BytesIO):
        return Image.open(img_input).convert(mode)

    raise TypeError(f"Unsupported image input type: {type(img_input)}")


def find_images(folder: str, extensions: Tuple[str, ...] = IMG_EXTENSIONS) -> List[str]:
    """Scan a folder recursively and return sorted list of paths to all image files.

    Args:
        folder: Root directory to search.
        extensions: Tuple of valid image file extensions.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Directory not found: {folder}")

    image_paths = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(extensions):
                image_paths.append(os.path.join(root, f))
    return image_paths


def parse_csv(csv_path: str, delimiter: str = ",") -> List[Dict[str, str]]:
    """Parse a CSV or TSV file into a list of row dictionaries.

    Args:
        csv_path: Path to CSV file.
        delimiter: Column delimiter (default: ',').
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items() if k is not None})
    return rows


# =========================================================================
# LMDB Helper Functions for users building custom LMDB-backed datasets
# =========================================================================

# Process-level cache to prevent opening the same LMDB environment twice in the same process
_LMDB_ENV_CACHE: Dict[str, Any] = {}


def open_lmdb(
    db_path: str,
    readonly: bool = True,
    max_readers: int = 128,
):
    """Safely open an LMDB environment and return (env, txn).

    Caches the environment per process so that multiple datasets (e.g. train and val)
    sharing the same LMDB database file in the same process do not encounter:
    `lmdb.Error: The environment is already open in this process.`

    Args:
        db_path: Directory containing LMDB database files (`data.mdb`).
        readonly: Whether to open in read-only mode.
        max_readers: Maximum number of concurrent reader slots.
    """
    try:
        import lmdb
    except ImportError:
        raise ImportError("The 'lmdb' package is required. Install it via: pip install lmdb")

    abs_path = os.path.abspath(db_path)
    if abs_path not in _LMDB_ENV_CACHE:
        _LMDB_ENV_CACHE[abs_path] = lmdb.open(
            abs_path,
            max_readers=max_readers,
            readonly=readonly,
            lock=False,
            readahead=False,
            meminit=False,
        )

    env = _LMDB_ENV_CACHE[abs_path]
    txn = env.begin(write=not readonly)
    return env, txn


def read_lmdb_bytes(txn: Any, key: Union[str, bytes]) -> Optional[bytes]:
    """Fetch raw binary bytes for a given key from an LMDB transaction."""
    if isinstance(key, str):
        key = key.encode("utf-8")
    return txn.get(key)


def create_lmdb(
    samples: List[Tuple[str, Any]],
    output_path: str,
    map_size: int = 10 * 1024 * 1024 * 1024,
) -> None:
    """Helper to convert a list of (image_path, label) into an LMDB database.

    Args:
        samples: List of (image_path, label).
        output_path: Destination folder for LMDB database.
        map_size: Max capacity in bytes (default: 10GB).
    """
    try:
        import lmdb
    except ImportError:
        raise ImportError("The 'lmdb' package is required. Install it via: pip install lmdb")

    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(output_path, map_size=map_size)

    with env.begin(write=True) as txn:
        for idx, (img_path, label) in enumerate(samples):
            with open(img_path, "rb") as f:
                img_bytes = f.read()

            img_key = f"image-{idx:09d}".encode("utf-8")
            lbl_key = f"label-{idx:09d}".encode("utf-8")

            txn.put(img_key, img_bytes)
            txn.put(lbl_key, str(label).encode("utf-8"))

        txn.put(b"num-samples", str(len(samples)).encode("utf-8"))

    env.sync()
    env.close()
