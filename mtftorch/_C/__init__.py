# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations

import mtftorch

from contextlib import contextmanager

from typing import (
    Any, BinaryIO, Callable, ContextManager, Dict, Iterable, Iterator, List,
    NamedTuple, Optional, overload, Sequence, Tuple, TypeVar, Type, Union,
    Generic, Set, AnyStr)


import mesh_tensorflow as mtf
import tensorflow as tf2
tf = tf2.compat.v1

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

import contextvars as cv

import mesh_tensorflow.test_utils

class Converter(mesh_tensorflow.test_utils.NumpyConverter):
    def __init__(self, mesh, session=None):
        if session is None:
            session = tf.get_default_session()
        if session is None:
            session = tf.Session()
        self._session = session
        self._mesh = mesh
        self._graph = mesh.graph


class State:
    def __init__(self, name="mtftorch_state", *, graph=None, mesh=None):
        self.graph = graph if graph is not None else mtf.Graph()
        self.mesh = mesh if mesh is not None else mtf.Mesh(self.graph, name)
        self.converter = Converter(self.mesh)

State.GLOBAL = State()
State.current = cv.ContextVar('mtftorch.State.current', default=State.GLOBAL)

def get(attr, *defaults):
    state = State.current.get()
    return getattr(state, attr, *defaults)

def get_graph() -> mtf.Graph:
    return get('graph')

def get_mesh() -> mtf.Mesh:
    return get('mesh')

def get_converter() -> Converter:
    return get('converter')

@contextmanager
def _with_value(cvar, value):
    reset_value = cvar.set(value)
    try:
        yield
    finally:
        cvar.reset(reset_value)

def with_state(state):
    return _with_value(State.current, state)


def numpy(x):
    return get_converter().convert_mtf_tensor_to_np_array(x)

def item(x):
    if x.shape.size > 1:
        raise ValueError("only one element tensors can be converted to Python scalars")
    return numpy(x)

def _to_dims(x):
    if hasattr(x, 'shape'):
        x = x.shape
    if isinstance(x, mtf.Shape):
        return x.dims
    if isinstance(x, mtf.Dimension):
        return [x]
    if isinstance(x, dict):
        return _to_dims(list(x.items()))
    if isinstance(x, str):
        x = x.replace(',', ' ')
        x = x.replace('=', ':')
        if ' ' in x:
            return _to_dims([_to_dims(v) for v in x.split()])
        if ':' in x:
            return mtf.convert_to_shape(x).dims
        else:
            return [mtf.Dimension(x, None)]
    if isinstance(x, (list, tuple)):
        if len(x) == 2 and isinstance(x[0], str) and (isinstance(x[1], int) or x[1] is None):
            name, size = x
            if ':' in name:
                raise ValueError(f"Can't create dimension with a colon in the name: {name!r}")
            return [mtf.Dimension(name, size)]
        else:
            y = []
            for v in x:
                y.extend(_to_dims(v))
            return y
    raise ValueError(f"Can't convert {x!r} to dimensions")

def size(x, dim=None):
    shape = mtf.Shape(_to_dims(x))
    if dim is None:
        return shape
    if isinstance(dim, str):
        return shape.get_dim_by_name(dim)
    if isinstance(dim, int):
        return shape.dims[dim]
    raise ValueError(f'size(): bad dim {dim!r}')


def shapelist(x, *dims):
    old_dims = _to_dims(x)
    if len(dims) <= 0:
        return size(old_dims)
    new_dims = _to_dims(dims)
    for i, dim in enumerate(new_dims):
        if dim.size is None:
            for j, old in enumerate(old_dims):
                if old.name == dim.name:
                    new_dims[i] = old
    return size(new_dims)

def view(x, *dims):
    new_shape = shapelist(x, *dims)
    return mtf.reshape(x, new_shape)

permute = view


def convert_to_dimension(d):
  """Converts input to a Dimension.

  Args:
    d: Dimension, tuple (string, int), or None.

  Returns:
    Dimension or None.

  Raises:
    ValueError: If d cannot be converted to a Dimension.
  """
  if d is None:
    return None
  if isinstance(d, mtf.Dimension):
    if not isinstance(d.name, str) or not (isinstance(d.size, int) or d.size is None):
      raise ValueError("Bad dimension %s" % (d,))
    return d
  name, size = d
  if isinstance(name, str) and (isinstance(size, int) or size is None):
    return mtf.Dimension(name, size)
  else:
    raise ValueError("could not convert %s to Dimension" % (d,))


class TensorMixin:
    item = item
    numpy = numpy
    size = size
    view = view
    permute = permute


class ShapeMixin:
    # support None sizes
    @property
    def to_string(self):
        return "Shape[%s]" % ", ".join(
            ["{}={}".format(d.name, d.size) for d in self.dims])


# https://qastack.vn/programming/9539052/how-to-dynamically-change-base-class-of-instances-at-runtime

def ensure_class_bases_begin_with(namespace, class_name, base_class):
    """ Ensure the named class's bases start with the base class.
        :param namespace: The namespace containing the class name.
        :param class_name: The name of the class to alter.
        :param base_class: The type to be the first base class for the
            newly created type.
        :return: ``None``.
        Call this function after ensuring `base_class` is
        available, before using the class named by `class_name`.
        """
    if hasattr(namespace, '__dict__'):
        namespace = namespace.__dict__
    if hasattr(class_name, '__name__'):
        class_name = class_name.__name__
    existing_class = namespace[class_name]
    assert isinstance(existing_class, type)
    bases = list(existing_class.__bases__)
    if base_class is bases[0]:
        # Already bound to a type with the right bases.
        return existing_class
    bases.insert(0, base_class)
    new_class_namespace = existing_class.__dict__.copy()
    # Type creation will assign the correct ‘__dict__’ attribute.
    del new_class_namespace['__dict__']
    metaclass = getattr(existing_class, '__metaclass__', type)
    new_class = metaclass(class_name, tuple(bases), new_class_namespace)
    namespace[class_name] = new_class
    return new_class

import mesh_tensorflow.ops

for module in [mtf, mesh_tensorflow.ops]:
    Tensor = mtf.Tensor = ensure_class_bases_begin_with(module, mesh_tensorflow.ops.Tensor, TensorMixin)
    Shape = mtf.Shape = ensure_class_bases_begin_with(module, mesh_tensorflow.ops.Shape, ShapeMixin)

mesh_tensorflow.ops.convert_to_dimension = convert_to_dimension

# we extend Shape.to_string to support None sizes
setattr(mtf.Shape, 'to_string', getattr(ShapeMixin, 'to_string'))

# we redefine size to be torch-like
delattr(mtf.Tensor, 'size')
