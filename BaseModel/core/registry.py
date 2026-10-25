"""Registry module for modular component management and dynamic instantiation.

Inspired by MMCV / Detectron2 Registry pattern.
"""
import inspect
from typing import Any, Callable, Dict, Optional, Type, Union


class Registry:
    """A registry to map strings to classes or functions.

    Args:
        name: Registry name (e.g. 'models', 'datasets', 'losses').
    """

    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def module_dict(self) -> Dict[str, Type]:
        return self._module_dict

    def __len__(self) -> int:
        return len(self._module_dict)

    def __contains__(self, key: str) -> bool:
        return key in self._module_dict

    def __repr__(self) -> str:
        format_str = f"{self.__class__.__name__}(name={self._name}, items={list(self._module_dict.keys())})"
        return format_str

    def get(self, key: str) -> Optional[Type]:
        """Get the registry record by name."""
        return self._module_dict.get(key, None)

    def _register_module(self, module_class: Type, module_name: Optional[str] = None, force: bool = False):
        if not inspect.isclass(module_class) and not inspect.isfunction(module_class):
            raise TypeError(f"module must be a class or function, but got {type(module_class)}")

        if module_name is None:
            module_name = module_class.__name__

        if isinstance(module_name, str):
            module_names = [module_name]
        else:
            module_names = list(module_name)

        for name in module_names:
            if not force and name in self._module_dict:
                raise KeyError(f"'{name}' is already registered in '{self.name}' registry.")
            self._module_dict[name] = module_class

    def register_module(
        self,
        name: Optional[Union[str, list]] = None,
        force: bool = False,
        module: Optional[Type] = None
    ) -> Union[Type, Callable]:
        """Register a module. Can be used as a decorator or a normal function.

        Example:
            @MODELS.register_module()
            class ResNet(nn.Module):
                ...

            # Or with custom name:
            @MODELS.register_module(name='CustomResNet')
            class ResNet(nn.Module):
                ...

            # Or as normal function:
            MODELS.register_module(module=ResNet)
        """
        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module

        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register

    def build(self, cfg: Union[Dict[str, Any], Any], *args, **kwargs) -> Any:
        """Build an instance from config dict."""
        return build_from_cfg(cfg, self, default_args=kwargs if kwargs else None)


def build_from_cfg(
    cfg: Union[Dict[str, Any], Any],
    registry: Registry,
    default_args: Optional[Dict[str, Any]] = None
) -> Any:
    """Build a module from config dict using a registry.

    Args:
        cfg: Config dict containing 'type' key. If cfg is already an object, return it.
        registry: The Registry to search for the module class.
        default_args: Default initialization arguments.

    Returns:
        Instantiated object.
    """
    if cfg is None:
        return None

    if not isinstance(cfg, dict):
        # Already an instance
        return cfg

    if "type" not in cfg:
        raise KeyError(f"Config must contain key 'type', but got {cfg}")

    args = cfg.copy()
    obj_type = args.pop("type")

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f"'{obj_type}' is not registered in '{registry.name}'. "
                f"Available types: {list(registry.module_dict.keys())}"
            )
    elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be str or class, but got {type(obj_type)}")

    if default_args is not None:
        for key, value in default_args.items():
            args.setdefault(key, value)

    try:
        return obj_cls(**args)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize '{obj_type}' from registry '{registry.name}': {e}") from e
