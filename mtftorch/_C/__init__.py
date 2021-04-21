import mtftorch
from typing import (
    Any, BinaryIO, Callable, ContextManager, Dict, Iterable, Iterator, List,
    NamedTuple, Optional, overload, Sequence, Tuple, TypeVar, Type, Union,
    Generic, Set, AnyStr)


import mesh_tensorflow as mtf

import tensorflow as tf

from mesh_tensorflow import Tensor
from mesh_tensorflow import Shape
from mesh_tensorflow import VariableDType as dtype

mtftorch.Tensor = Tensor
mtftorch.Size = Shape
mtftorch.dtype = dtype

from mtftorch.types import _int, _float, _bool, _dtype, _device, _qscheme, _size, _layout, Device, Number, Storage

T = TypeVar('T')

# Defined in torch/csrc/Device.cpp
class device:
    type: str  # THPDevice_type
    index: _int  # THPDevice_index

    def __get__(self, instance, owner=None) -> _device: ...

    # THPDevice_pynew
    @overload
    def __init__(self, device: Union[_device, _int, str]) -> None: ...

    @overload
    def __init__(self, type: str, index: _int) -> None: ...

    def __reduce__(self) -> Tuple[Any, ...]: ...  # THPDevice_reduce

_api_usage_seen = set()

def _log_api_usage_once(*args):
    if args not in _api_usage_seen:
        _api_usage_seen.add(args)
        tf.get_logger().info(' '.join(["%s"] * len(args)), *args)

def _get_tracing_state():
    return False