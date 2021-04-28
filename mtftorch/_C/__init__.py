# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations

import mtftorch
import numpy as np

from contextlib import contextmanager
import functools
import operator
import builtins
from collections import defaultdict

from typing import (
    Any, BinaryIO, Callable, ContextManager, Dict, Iterable, Iterator, List,
    NamedTuple, Optional, overload, Sequence, Tuple, TypeVar, Type, Union,
    Generic, Set, AnyStr)


import mesh_tensorflow as mtf
import tensorflow as tf2
tf = tf2.compat.v1

newaxis = tf.newaxis

from mesh_tensorflow import Tensor
from mesh_tensorflow import Shape
from mesh_tensorflow import VariableDType as dtype

mtftorch.Tensor = Tensor
mtftorch.Size = Shape
mtftorch.dtype = dtype

from tensorflow.python.framework.dtypes import bool, uint8, int8, int16, int32, int64, bfloat16, float16, float32, float64, complex64, complex128, qint8, qint16, qint32, quint8

quint4x2 = quint8 # TODO

float = float32
double = float64
half = float16
cfloat = complex64
cdouble = complex128

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
    """Converter class to convert between mtf.Tensor, tf.Tensor and np.array."""

    def __init__(self, session=None):
        self._cached_session = session

    def convert_np_array_to_mtf_tensor(self, x, dim_names=None, dtype=None):
        """Convert a numpy array to an equivalent mtf.Tensor."""
        if dtype is None:
            dtype = tf.dtypes.as_dtype(x.dtype)
        dim_sizes = x.shape
        if not dim_names:
            dim_names = [f"dim{i}" for i in range(len(dim_sizes))]

        dims = []
        for dim_size, dim_name in zip(dim_sizes, dim_names):
            dims.append(mtf.Dimension(dim_name, dim_size))
        shape = mtf.Shape(dims)
        x_mtf = mtf.constant(self.mesh, x, shape=shape, dtype=dtype)
        return x_mtf

    def convert_mtf_tensor_to_np_array(self, x_mtf, *, session=None):
        """Convert an mtf.Tensor to a numpy array."""
        _, x_tf = self.convert_mtf_tensor_to_tf_tensor(x_mtf)
        if tf.executing_eagerly():
            return x_tf.numpy()
        else:
            session = self.get_session(session)
            #session.run(tf.global_variables_initializer())
            return x_tf.eval(session=session)

    def convert_mtf_tensor_to_tf_tensor(self, mtf_tensor, lowering=None):
        """Convert an mtf.Tensor to a tf.Tensor."""
        if lowering is None:
            lowering = mtf.Lowering(self.graph, {self.mesh: self.mesh_impl})
        return lowering, lowering.export_to_tf_tensor(mtf_tensor)

    @property
    def graph(self):
        return get_graph()

    @property
    def mesh(self):
        return get_mesh()

    @property
    def mesh_impl(self):
        return get_mesh_impl()

    @property
    def lowering(self):
        return get_lowering()

    def get_session(self, session=None):
        if session is None:
            session = tf.get_default_session()
        if session is None:
            if self._cached_session is None:
                self._cached_session = tf.Session()
            session = self._cached_session
        return session


class State:
    def __init__(self, name="mtftorch_state", *,
                 graph=None, mesh=None, mesh_impl=None, lowering=None):
        self.graph = graph if graph is not None else mtf.Graph()
        self.mesh = mesh if mesh is not None else mtf.Mesh(self.graph, name)
        self.mesh_impl = mesh_impl if mesh_impl is not None else mtf.placement_mesh_impl.PlacementMeshImpl(
            shape=[], layout={}, devices=[""])
        self.lowering = lowering if lowering is not None else mtf.Lowering(self.graph, {self.mesh: self.mesh_impl})
        self.converter = Converter()

State.GLOBAL = State()
State.current = cv.ContextVar('mtftorch.State.current', default=State.GLOBAL)

def get(attr, *defaults):
    state = State.current.get()
    return getattr(state, attr, *defaults)

def get_graph(graph=None) -> mtf.Graph:
    if graph is not None:
        return graph
    return get('graph')

def get_mesh(mesh=None) -> mtf.Mesh:
    if mesh is not None:
        return mesh
    return get('mesh')

def get_mesh_impl(mesh_impl=None) -> mtf.MeshImpl:
    if mesh_impl is not None:
        return mesh_impl
    return get('mesh_impl')

def get_lowering(lowering=None) -> mtf.Lowering:
    if lowering is not None:
        return lowering
    return get('lowering')


def get_converter(converter=None) -> Converter:
    if converter is not None:
        return converter
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


def numpy(x) -> np.ndarray:
    return get_converter().convert_mtf_tensor_to_np_array(x)

def to_tf(x):
    _, x_tf = get_converter().convert_mtf_tensor_to_tf_tensor(x)
    return x_tf

def item(x):
    if x.shape.size > 1:
        raise ValueError("only one element tensors can be converted to Python scalars")
    return numpy(x)

def as_shape(x) -> mtf.Shape:
    return mtf.Shape(as_dims(x))

def as_dims(x) -> List[mtf.Dimension]:
    if is_numpy(x):
        return [mtf.Dimension(name=None, size=v) for v in x.shape]
    if hasattr(x, 'shape'):
        x = x.shape
    if isinstance(x, mtf.Shape):
        return x.dims
    if isinstance(x, mtf.Dimension):
        return [x]
    if isinstance(x, dict):
        return as_dims(list(x.items()))
    if isinstance(x, str):
        x = x.replace(',', ' ')
        x = x.replace('=', ':')
        if ' ' in x:
            return as_dims([as_dims(v) for v in x.split()])
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
                y.extend(as_dims(v))
            return y
    raise ValueError(f"Can't convert {x!r} to dimensions")

import warnings

def select_dims(tensor, *dims, ignore_missing=False) -> mtf.Shape:
    shape = size(tensor)
    selected = shapelist(tensor, *dims)
    missing = selected - shape
    if missing:
        if ignore_missing:
            warnings.warn(f"Dimensions {missing} were missing from source shape {shape}")
        else:
            raise ValueError(f"Dimensions {missing} were missing from source shape {shape}")
    return selected

def exclude_dims(tensor, *dims, ignore_missing=True) -> mtf.Shape:
    shape = size(tensor)
    selected = shapelist(tensor, *dims)
    missing = selected - shape
    if missing:
        if ignore_missing:
            warnings.warn(f"Dimensions {missing} were missing from source shape {shape}")
        else:
            raise ValueError(f"Dimensions {missing} were missing from source shape {shape}")
    excluded = shape - selected
    return excluded

def _inc(h, k, by=1):
    v = h.get(k, 0)
    v += by
    h[k] = v
    return v

def unique_dims(x) -> List[mtf.Dimension]:
    dims = as_dims(x)
    seen = defaultdict(int)
    return [dim for dim in dims if _inc(seen, dim) == 1]

def size(x, dim=None) -> mtf.Shape:
    shape = mtf.Shape(as_dims(x))
    if dim is None:
        return shape
    if isinstance(dim, mtf.Dimension):
        dim = dim.name
    if isinstance(dim, str):
        return shape.get_dim_by_name(dim)
    if isinstance(dim, int):
        return shape.dims[dim]
    raise ValueError(f'size(): bad dim {dim!r}')

def as_numpy_list(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, 'numpy'):
        return x.numpy()
    if isinstance(x, (tuple, list)):
        return (tuple if isinstance(x, tuple) else list)([as_numpy_list(v) for v in x])
    return x

def as_numpy(x):
    x = as_numpy_list(x)
    x = np.array(x)
    if x.dtype == np.object:
        raise ValueError("Couldn't infer dtype of tensor")
    return x

def is_numpy(x):
    return isinstance(x, np.ndarray)

def shapelist(x, *dims) -> mtf.Shape:
    old_dims = as_dims(x)
    if len(dims) <= 0:
        return size(old_dims)
    dims = [old_dims[x] if isinstance(x, int) else x for x in dims]
    new_dims = as_dims(dims)
    if is_numpy(x):
        if len(old_dims) != len(new_dims):
            raise ValueError(f"Dimension length mismatch: can't convert numpy shape {x.shape} to dimensions {new_dims}")
        dims = []
        for old, new in zip(list(old_dims), list(new_dims)):
            dim = mtf.Dimension(new.name, old.size)
            if new.size is not None:
                if new.size != dim.size:
                    raise ValueError(f"Can't convert {old} to {dim}")
            dims.append(dim)
        new_dims = dims
    else:
        for i, dim in enumerate(new_dims):
            if dim.size is None:
                for j, old in enumerate(old_dims):
                    if old.name is None:
                        old = mtf.Dimension(dim.name, old.size)
                        old_dims[j] = old
                    if old.name == dim.name:
                        new_dims[i] = old
                        break
    if builtins.any([dim.size == -1 for dim in new_dims]):
        dims = []
        expand_idx = None
        for i, dim in enumerate(new_dims):
            if dim.size is not None and dim.size != -1:
                dims.append(dim)
            if dim.size == -1:
                if expand_idx is not None:
                    raise ValueError(f"Only one dimension's size can be -1, got {new_dims}")
                expand_idx = i
        n = mtf.Shape(dims).size
        k = mtf.Shape(old_dims).size
        assert k % n == 0
        k //= n
        new_dims = [mtf.Dimension(dim.name, k) if dim.size == -1 else dim for dim in new_dims]
    return size(new_dims)

def view(x, *dims) -> mtf.Tensor:
    new_shape = shapelist(x, *dims)
    return mtf.reshape(x, new_shape)

def permute(x, *dims) -> mtf.Tensor:
    new_shape = shapelist(x, *dims)
    return mtf.transpose(x, new_shape)

def zeros(*dims, mesh=None, **kws) -> mtf.Tensor:
    mesh = get_mesh(mesh)
    shape = size(dims)
    return mtf.zeros(mesh, shape, **kws)

def ones(*dims, mesh=None, **kws) -> mtf.Tensor:
    mesh = get_mesh(mesh)
    shape = size(dims)
    return mtf.ones(mesh, shape, **kws)

def tensor(data, shape, *, dtype=None, mesh=None):
    mesh = get_mesh(mesh)
    data = as_numpy(data)
    #shape = size(shape)
    shape = shapelist(data, shape)
    return mtf.constant(mesh, data, shape=shape)

def cat(tensors, dim=0) -> mtf.Tensor:
    dim = size(unique_dims(tensors), dim)
    return mtf.concat(tensors, dim.name)

def stack(tensors, new_dim_name, axis=0) -> mtf.Tensor:
    return mtf.stack(tensors, new_dim_name, axis=axis)

def unbind(tensor, dim=0) -> mtf.Tensor:
    "Eliminates a tensor dimension."
    dim = size(tensor, dim)
    return mtf.unstack(tensor, dim)

def _idx(index, total, default):
    if index is None:
        return default
    if not isinstance(index, (int, float)):
        return index
    index = int(index)
    if index < 0:
        index += total
    return index

def slice(x, slice_dim_name, start=None, stop=None):
    n = size(x, slice_dim_name).size
    start = _idx(start, n, 0)
    stop = _idx(stop, n, n)
    if isinstance(start, int) and isinstance(stop, int):
        if stop < 0: stop = 0
        if stop > n: stop = n
        if start < 0: start = 0
        if start > n: start = n
        if stop < start: stop = start
    else:
        # TODO: verify stop >= start when either stop or start are tensors.
        pass
    return mtf.slice(x, begin=start, size=stop - start, slice_dim_name=slice_dim_name)

def index_args(x, *args):
    args = [v for v in args]
    if isinstance(x, Ellipsis.__class__):
        dims = [x]
    else:
        dims = as_dims(x)
        for i, dim in enumerate(dims):
            if dim.size is None:
                if len(args) <= 0:
                    raise ValueError(f"Not enough args for indexing operation into {dims}")
                dim = dims[i] = mtf.Dimension(dim.name, args[0])
                args = args[1:]
    if len(args) > 0:
        dims += index_args(args[0], *args[1:])
    return dims

def index(x, *more):
    dims = index_args(*more)
    j = 0
    for i, dim in enumerate(dims):
        if isinstance(dim, Ellipsis.__class__):
            j = -1
            continue
        if dim.size is None:
            axis = i if j >= 0 else -1
            x = stack([x], dim.name, axis=axis)
            if j >= 0:
                j += 1
            else:
                j -= 1
            continue
        if isinstance(dim.size, builtins.slice):
            if dim.size.step != 1 and dim.size.step is not None:
                raise ValueError("Slicing with step != 1 not yet supported")
            start = dim.size.start
            stop = dim.size.stop
        else:
            start = dim.size
            stop = (dim.size + 1) if dim.size >= 0 else (dim.size - 1)
        x = slice(x, dim.name, start=start, stop=stop)
        if not isinstance(dim.size, builtins.slice):
            x = unbind(x, dim.name)[0]
    return x

def split(tensor, split_dim, num_or_size_splits):
    shape = shapelist(tensor, split_dim)
    if len(shape) != 1:
        raise ValueError(f"Must only specify one dimension, got {split_dim}")
    split_dim = shape[0]
    return mtf.split(tensor, split_dim, num_or_size_splits)

def squeeze(tensor: mtf.Tensor, squeeze_dim):
    shape = shapelist(tensor, squeeze_dim)
    for dim in shape:
        if tensor.shape.get_dim_by_name(dim.name).size != 1:
            raise ValueError(f"Squeeze dimension size must be 1 for dimension {dim} of shape {tensor.shape}")
    squeeze_dims = [dim.name for dim in shape]
    return view(tensor, [dim for dim in list(tensor.shape) if dim.name not in squeeze_dims])

def rand(*dims, seed, dtype=None):
    if dtype is None:
        dtype = tf.float32
    shape = as_shape(dims)
    tf_shape = [dim.size for dim in shape]
    x = tf.random.stateless_uniform(shape=tf_shape, seed=seed, dtype=dtype)
    return mtf.import_tf_tensor(get_mesh(), x, shape)

def randn(*dims, seed, dtype=None):
    if dtype is None:
        dtype = tf.float32
    shape = as_shape(dims)
    tf_shape = [dim.size for dim in shape]
    x = tf.random.stateless_normal(shape=tf_shape, seed=seed, dtype=dtype)
    return mtf.import_tf_tensor(get_mesh(), x, shape)

def randint(*args, seed, dtype=None):
    if len(args) < 2 or len(args) > 4:
        raise ValueError(f"Expected randint(low=0, high, dims), got {args}")
    if len(args) == 2:
        low = 0
        high, dims = args
    else:
        low, high, dims = args
    if dtype is None:
        dtype = tf.int64
    shape = as_shape(dims)
    tf_shape = [dim.size for dim in shape]
    x = tf.random.stateless_uniform(shape=tf_shape, seed=seed, minval=low, maxval=high, dtype=dtype)
    return mtf.import_tf_tensor(get_mesh(), x, shape)

def range(dim, dtype=None):
    if dtype is None:
        dtype = tf.float32
    elif isinstance(dtype, str):
        dtype = getattr(tf.dtypes, dtype)
    shape = size(dim)
    if len(shape) != 1:
        raise ValueError(f"range() expected a single dimension, got {shape}")
    dim = size(shape, 0)
    return mtf.range(get_mesh(), dim, dtype=dtype)

def arange(dim, dtype=None):
    if dtype is None:
        dtype = tf.int64
    return range(dim, dtype=dtype)

def cumsum(x, dim):
    return mtf.cumsum(x, size(x, dim))

def lerp(x, y, alpha):
    return x * (1 - alpha) + y * alpha

def meshgrid(*args):
    if len(args) <= 1:
        return args[0:1]
    x, y, *zs = args
    shape = size(y) + size(x) + size(zs)
    t = ones(shape)
    dims = size(args)
    # TODO: properly handle non-integer meshgrids.
    return [(cumsum(t, dim) - 1.0 + min(mtf.to_float(args[i]))).to(x.dtype) for i, dim in enumerate(dims)]





def reduction(reduce_fn, tensor, dim=None, keepdim=False) -> mtf.Tensor:
    input_shape = size(tensor)
    if dim is None:
        dim = input_shape
    output_shape = exclude_dims(input_shape, dim)
    result = reduce_fn(tensor, output_shape=output_shape)
    if keepdim:
        result_shape = [(dim.name, dim.size if dim in output_shape else 1)
                        for dim in input_shape]
        result = view(result, result_shape)
    return result

def sum(tensor, dim=None, keepdim=False) -> mtf.Tensor:
    return reduction(mtf.reduce_sum, tensor, dim=dim, keepdim=keepdim)

def mean(tensor, dim=None, keepdim=False) -> mtf.Tensor:
    return reduction(mtf.reduce_mean, tensor, dim=dim, keepdim=keepdim)

def min(tensor, dim=None, keepdim=False) -> mtf.Tensor:
    return reduction(mtf.reduce_min, tensor, dim=dim, keepdim=keepdim)

def max(tensor, dim=None, keepdim=False) -> mtf.Tensor:
    return reduction(mtf.reduce_max, tensor, dim=dim, keepdim=keepdim)

def any(tensor, dim=None, keepdim=False) -> mtf.Tensor:
    return reduction(mtf.reduce_any, tensor, dim=dim, keepdim=keepdim)

def all(tensor, dim=None, keepdim=False) -> mtf.Tensor:
    return reduction(mtf.reduce_all, tensor, dim=dim, keepdim=keepdim)

def cwise(tensor, tf_fn, output_dtype=None, grad_function=None):
    return mtf.cwise(tf_fn, [tensor], output_dtype=output_dtype, grad_function=grad_function)

def binary_op(x, y, tf_fn, output_shape=None):
    output_dtype = x1.dtype if isinstance(x1, Tensor) else x2.dtype
    return mtf.binary_op_with_broadcasting(
        tf_fn, x1, x2, output_dtype=output_dtype, output_shape=output_shape)

abs = mtf.mtf_abs
pow = mtf.mtf_pow
sqrt = mtf.sqrt
rsqrt = mtf.rsqrt
exp = mtf.exp
sin = mtf.sin
cos = mtf.cos
tanh = mtf.tanh

lt = mtf.less
gt = mtf.greater
le = mtf.less_equal
ge = mtf.greater_equal
eq = mtf.equal
ne = mtf.not_equal

def logical_not(tensor):
    return cwise(tensor, tf.math.logical_not)

def logical_xor(x1, x2):
    return binary_op(x1, x2, tf.math.logical_xor)

def logical_or(x1, x2):
    return binary_op(x1, x2, tf.math.logical_or)

def logical_and(x1, x2):
    return binary_op(x1, x2, tf.math.logical_and)

def bitwise_not(tensor):
    return cwise(tensor, tf.bitwise.invert)

def bitwise_xor(x1, x2):
    return binary_op(x1, x2, tf.bitwise.bitwise_xor)

def bitwise_or(x1, x2):
    return binary_op(x1, x2, tf.bitwise.bitwise_or)

def bitwise_and(x1, x2):
    return binary_op(x1, x2, tf.bitwise.bitwise_and)

def bitwise_left_shift(x1, x2):
    return binary_op(x1, x2, tf.bitwise.left_shift)

def bitwise_right_shift(x1, x2):
    return binary_op(x1, x2, tf.bitwise.right_shift)

def promote_types(type1, type2):
    # TODO: ensure proper type promotion.
    assert type1 == type2
    return type1

def allclose(input: Tensor, other: Tensor, rtol: _float=1e-05, atol: _float=1e-08, equal_nan: _bool=False) -> _bool:
    # TODO: This only works for "eager" execution.
    if hasattr(input, 'numpy'):
        input = input.numpy()
    if hasattr(other, 'numpy'):
        other = other.numpy()
    return np.allclose(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan)

def isclose(input: Tensor, other: Tensor, rtol: _float=1e-05, atol: _float=1e-08, equal_nan: _bool=False) -> Tensor:
    # TODO: This only works for "eager" execution.
    if hasattr(input, 'numpy'):
        input = input.numpy()
    if hasattr(other, 'numpy'):
        other = other.numpy()
    return np.isclose(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan)


def convert_to_dimension(d) -> mtf.Dimension:
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

class TensorSizeFunctor(int):
    def __call__(self, *args, **kws):
        return size(self._tensor, *args, **kws)

class TensorMixin:
    item = item
    numpy = numpy
    tf = to_tf

    #size = size

    @property
    def size(self):
        v = TensorSizeFunctor(self.shape.size)
        v._tensor = self
        return v

    view = view
    permute = permute
    select_dims = select_dims
    exclude_dims = exclude_dims
    unbind = unbind
    sum = sum
    mean = mean
    min = min
    max = max
    any = any
    all = all
    slice = slice
    split = split
    squeeze = squeeze
    index = index

    def bfloat16(self):
        return self.to(bfloat16)

    def half(self):
        return self.to(float16)

    def float(self):
        return self.to(float32)

    def double(self):
        return self.to(float64)

    def int(self):
        return self.to(int32)

    def long(self):
        return self.to(int64)

    def bool(self):
        return self.to(tf.bool)

    def range(self, dim, dtype=None):
        return range(self.size(dim), dtype=dtype)

    def arange(self, dim, dtype=None):
        return arange(self.size(dim), dtype=dtype)

    def cat(self, tensors, dim=0):
        tensors = [self if tensor == "_" else tensor for tensor in tensors]
        return cat(tensors, dim=dim)

    def stack(self, tensors, new_dim_name, axis=0):
        tensors = [self if tensor == "_" else tensor for tensor in tensors]
        return stack(tensors, new_dim_name=new_dim_name, axis=axis)

    @property
    def is_sparse(self):
        return False

    @property
    def is_quantized(self):
        return False

    @property
    def device(self):
        return None

    def to(self, dtype):
        return mtf.cast(self, dtype)

    def is_complex(self):
        return self.dtype.is_complex

    def __getitem__(self, item):
        return index(self, *item)

    __hash__ = object.__hash__

    __lt__ = lt = lt
    __gt__ = gt = gt
    __le__ = le = le
    __ge__ = ge = ge
    __eq__ = eq = eq
    __ne__ = ne = ne

    logical_not = logical_not
    logical_xor = logical_xor
    logical_and = logical_and
    logical_or = logical_or

    bitwise_not = bitwise_not
    __invert__ = bitwise_not

    bitwise_xor = bitwise_xor
    __xor__ = bitwise_xor
    __rxor__ = bitwise_xor

    bitwise_and = bitwise_and
    __and__ = bitwise_and
    __rand__ = bitwise_and

    bitwise_or = bitwise_or
    __or__ = bitwise_or
    __ror__ = bitwise_or

    bitwise_left_shift = bitwise_left_shift
    __lshift__ = bitwise_left_shift
    __rlshift__ = bitwise_left_shift

    bitwise_right_shift = bitwise_right_shift
    __rshift__ = bitwise_right_shift
    __rrshift__ = bitwise_right_shift

    cwise = cwise
    abs = abs
    pow = pow
    __pow__ = pow
    sqrt = sqrt
    rsqrt = rsqrt
    exp = exp
    sin = sin
    cos = cos
    tanh = tanh

class ShapeMixin:
    # support None sizes
    @property
    def to_string(self):
        return "Shape[%s]" % ", ".join(
            ["{}={}".format(d.name, d.size) for d in self.dims])

class DTypeMixin:
    @property
    def is_signed(self):
        return not self.is_unsigned

    @property
    def is_floating_point(self):
        return not self.is_floating

    # ensure that str(mtorch.float32).split('.')[-1] == 'float32'
    def __str__(self):
        return repr(self)




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
    if object in bases:
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
    else:
        if base_class in bases:
            # Already bound to a type with the right bases.
            return existing_class
        bases.insert(0, base_class)
        existing_class.__bases__ = tuple(bases)
        return existing_class

import mesh_tensorflow.ops

for module in [mtf, mesh_tensorflow.ops]:
    Tensor = mtf.Tensor = ensure_class_bases_begin_with(module, mesh_tensorflow.ops.Tensor, TensorMixin)
    Shape = mtf.Shape = ensure_class_bases_begin_with(module, mesh_tensorflow.ops.Shape, ShapeMixin)

mesh_tensorflow.ops.convert_to_dimension = convert_to_dimension

# we extend Shape.to_string to support None sizes
setattr(mtf.Shape, 'to_string', getattr(ShapeMixin, 'to_string'))

# we redefine size to be torch-like
delattr(mtf.Tensor, 'size')

# ensure that str(mtorch.float32).split('.')[-1] == 'float32'
#float32.__class__.__str__ = float32.__class__.__repr__
#float32.__class__.is_floating_point = float32.__class__.is_floating

from tensorflow.python.framework import dtypes as _dtypes
ensure_class_bases_begin_with(_dtypes, tf.int32.__class__, DTypeMixin)
assert tf.int32.is_signed

def reset(graph=None):
    if graph is None:
        graph = get_graph()
    graph.operations.clear()