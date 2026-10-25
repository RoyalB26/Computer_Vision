"""Configuration Manager for reading YAML configurations with nested attribute access and CLI overrides."""
import json
import os
from typing import Any, Dict, List, Optional, Union
import yaml


class ConfigDict(dict):
    """Dictionary that allows attribute-style access like a Javascript object."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            self[k] = self._wrap(v)

    def _wrap(self, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            return ConfigDict(value)
        elif isinstance(value, list):
            return [self._wrap(v) for v in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")


class Config:
    """Central configuration class supporting loading from YAML/JSON files and command-line overrides."""

    def __init__(self, cfg_dict: Optional[Dict[str, Any]] = None, filename: Optional[str] = None):
        self._cfg_dict = ConfigDict(cfg_dict or {})
        self.filename = filename

    @classmethod
    def fromfile(cls, filename: str) -> "Config":
        """Load configuration from a YAML or JSON file."""
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Config file not found: {filename}")

        ext = os.path.splitext(filename)[1].lower()
        with open(filename, "r", encoding="utf-8") as f:
            if ext in [".yaml", ".yml"]:
                cfg_dict = yaml.safe_load(f) or {}
            elif ext == ".json":
                cfg_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file extension: {ext}. Expected .yaml or .json")

        return cls(cfg_dict=cfg_dict, filename=filename)

    def merge_from_dict(self, other_dict: Dict[str, Any]) -> None:
        """Recursively merge values from another dictionary."""

        def _merge(source, destination):
            for key, value in source.items():
                if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
                    _merge(value, destination[key])
                else:
                    destination[key] = value

        _merge(other_dict, self._cfg_dict)

    def merge_from_options(self, options: Optional[List[str]]) -> None:
        """Merge list of key-value option strings into config.

        Example:
            options = ["optimizer.lr=0.001", "model.num_classes=10", "amp=True"]
        """
        if not options:
            return

        for opt in options:
            if "=" not in opt:
                raise ValueError(f"Option must be key=value pair, got '{opt}'")
            key, val_str = opt.split("=", 1)
            val = self._parse_val(val_str)

            keys = key.strip().split(".")
            d = self._cfg_dict
            for k in keys[:-1]:
                if k not in d or not isinstance(d[k], dict):
                    d[k] = ConfigDict()
                d = d[k]
            d[keys[-1]] = val

    @staticmethod
    def _parse_val(val_str: str) -> Any:
        """Safely parse string into int, float, bool, None, or str."""
        val_str = val_str.strip()
        if val_str.lower() == "true":
            return True
        elif val_str.lower() == "false":
            return False
        elif val_str.lower() in ["none", "null"]:
            return None
        try:
            return int(val_str)
        except ValueError:
            pass
        try:
            return float(val_str)
        except ValueError:
            pass
        # JSON array/dict check
        if (val_str.startswith("[") and val_str.endswith("]")) or (val_str.startswith("{") and val_str.endswith("}")):
            try:
                return json.loads(val_str)
            except Exception:
                pass
        return val_str

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to standard nested Python dictionary."""

        def _to_native(d):
            if isinstance(d, dict):
                return {k: _to_native(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [_to_native(v) for v in d]
            return d

        return _to_native(self._cfg_dict)

    def dump(self, filepath: str) -> None:
        """Dump the config to a YAML file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name: str) -> Any:
        return self._cfg_dict[name]

    def __contains__(self, name: str) -> bool:
        return name in self._cfg_dict

    def __repr__(self) -> str:
        return f"Config (path: {self.filename}):\n" + yaml.dump(self.to_dict(), default_flow_style=False)
