# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations

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

from tensorflow.python.framework.dtypes import bool, uint8, int8, int16, int32, int64, bfloat16, float16, float32, float64, complex64, complex128

float = float32
double = float64
half = float16

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


# Defined in torch/csrc/Generator.cpp
class Generator(object):
    device: _device
    def __init__(self, device: Union[_device, str, None] = None) -> None: ...
    def get_state(self) -> Tensor: ...
    def set_state(self, _new_state: Tensor) -> Generator: ...
    def manual_seed(self, seed: _int) -> Generator: ...
    def seed(self) -> _int: ...
    def initial_seed(self) -> _int: ...


default_generator = Generator()

_api_usage_seen = set()

def _log_api_usage_once(*args):
    if args not in _api_usage_seen:
        _api_usage_seen.add(args)
        tf.get_logger().info(' '.join(["%s"] * len(args)), *args)

def _get_tracing_state():
    return False