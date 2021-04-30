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

from types import ModuleType

import mesh_tensorflow as mtf
import mesh_tensorflow.ops
import tensorflow as tf2
from tensorflow.python.framework import dtypes as _tf_dtypes

from mesh_tensorflow.ops import Tensor
from mesh_tensorflow.ops import Operation
from mesh_tensorflow.ops import Shape
from mesh_tensorflow.ops import VariableDType as dtype

from tensorflow.python.framework.dtypes import bool, uint8, int8, int16, int32, int64, bfloat16, float16, float32, float64, complex64, complex128, qint8, qint16, qint32, quint8

from mtftorch.types import _none, _ellipsis, _int, _float, _bool, _dtype, _tf_dtype, _tf_dtype_spec, _device, _qscheme, _shape, _layout, Device, Number, Storage

from mtftorch import _jit_internal

tf = tf2.compat.v1

newaxis = tf.newaxis

mtftorch.Tensor = Tensor
mtftorch.Size = Shape
mtftorch.dtype = dtype

quint4x2 = quint8  # TODO

float = float32
double = float64
half = float16
cfloat = complex64
cdouble = complex128

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
        super().__init__()
        self._cached_session = session

    def convert_np_array_to_mtf_tensor(self, x, dim_names=None, dtype=None):
        """Convert a numpy array to an equivalent mtf.Tensor."""
        dtype = get_dtype(dtype, x)
        dim_sizes = x.shape
        if not dim_names:
            dim_names = [f"dim{i}" for i in range(len(dim_sizes))]

        dims = []
        for dim_size, dim_name in zip(dim_sizes, dim_names):
            dims.append(mtf.Dimension(dim_name, dim_size))
        shape = mtf.Shape(dims)
        x_mtf = mtf.constant(self.mesh, x, shape=shape, dtype=dtype)
        return x_mtf

    def convert_to_np_array(self, x, *, session=None):
        if is_tensor(x):
            return self.convert_mtf_tensor_to_np_array(x, session=session)
        if is_tf_tensor(x):
            return self.convert_tf_tensor_to_np_array(x, session=session)
        return x

    def convert_to_tf_tensor(self, x, *, session=None):
        orig = x
        if not is_tensor(x):
            x = tensor(x)
        if is_tensor(x):
            _, x = self.convert_mtf_tensor_to_tf_tensor(x)
        if is_tf_tensor(x):
            return x
        else:
            raise ValueError(f"Couldn't convert {orig} to tf_tensor")

    def convert_mtf_tensor_to_np_array(self, x_mtf, *, session=None):
        """Convert an mtf.Tensor to a numpy array."""
        _, x_tf = self.convert_mtf_tensor_to_tf_tensor(x_mtf)
        return self.convert_tf_tensor_to_np_array(x_tf, session=session)

    def convert_tf_tensor_to_np_array(self, x_tf, *, session=None):
        if tf.executing_eagerly():
            return x_tf.numpy()
        else:
            session = self.get_session(session)
            # session.run(tf.global_variables_initializer())
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
State.requires_grad = cv.ContextVar('mtftorch.State.requires_grad', default=False)


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


def _autograd_init():
    return True


def is_grad_enabled():
    return State.requires_grad.get()


def _set_grad_enabled(value):
    token = State.requires_grad.set(value)

    def reset_grad_enabled():
        nonlocal token
        if token is not None:
            State.requires_grad.reset(token)
            token = None
    return reset_grad_enabled


set_grad_enabled = _set_grad_enabled


def numpy(x: Union[np.ndarray, tf.Tensor, mtf.Tensor, TensorMixin]) -> np.ndarray:
    return get_converter().convert_to_np_array(x)


def to_tf(x: Union[mtf.Tensor, TensorMixin], *, session=None) -> tf.Tensor:
    return get_converter().convert_to_tf_tensor(x, session=session)


def item(x: Union[mtf.Tensor, TensorMixin]) -> np.ndarray:
    if x.shape.size > 1:
        raise ValueError("only one element tensors can be converted to Python scalars")
    return numpy(x)


def as_shape(x) -> Union[mtf.Shape, ShapeMixin]:
    return mtf.Shape(as_dims(x))


def as_dims(x) -> List[mtf.Dimension]:
    if is_numpy(x):
        return [mtf.Dimension(name=None, size=v) for v in x.shape]
    if is_tf_tensor(x):
        return [mtf.Dimension(name=None, size=v) for v in x.shape.as_list()]
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


def select_dims(tensor, *dims, ignore_missing=False) -> Union[mtf.Shape, ShapeMixin]:
    shape = size(tensor)
    selected = shapelist(tensor, *dims)
    missing = selected - shape
    if missing:
        if ignore_missing:
            warnings.warn(f"Dimensions {missing} were missing from source shape {shape}")
        else:
            raise ValueError(f"Dimensions {missing} were missing from source shape {shape}")
    return selected


def exclude_dims(tensor, *dims, ignore_missing=True) -> Union[mtf.Shape, ShapeMixin]:
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


def size(x, dim=None) -> Union[Union[mtf.Shape, ShapeMixin], mtf.Dimension]:
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


def as_numpy_list(x) -> Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]]:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, 'numpy'):
        return x.numpy()
    if isinstance(x, (tuple, list)):
        y = [as_numpy_list(v) for v in x]
        if isinstance(x, tuple):
            y = tuple(y)
        return y
    return x


_np_object = np.dtype('object')


def as_numpy(x) -> np.ndarray:
    x = as_numpy_list(x)
    x = np.array(x)
    if x.dtype == _np_object:
        raise ValueError("Couldn't infer dtype of tensor")
    return x


def is_numpy(x) -> _bool:
    return isinstance(x, np.ndarray)


def is_tf_tensor(x) -> _bool:
    return isinstance(x, tf.Tensor)

def is_tensor(x) -> _bool:
    return isinstance(x, mtf.Tensor)

def shapelist(x, *dims) -> Union[mtf.Shape, ShapeMixin]:
    old_dims = as_dims(x)
    if len(dims) <= 0:
        return size(old_dims)
    dims = [old_dims[x] if isinstance(x, int) else x for x in dims]
    new_dims = as_dims(dims)
    if is_numpy(x) or is_tf_tensor(x):
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


def is_ellipsis(x) -> _bool:
    return isinstance(x, Ellipsis.__class__)


def concrete_dims(dims) -> Union[mtf.Shape, ShapeMixin]:
    dims = as_dims(dims)
    return size([dim for dim in dims if not is_ellipsis(dim) and dim.size is not None and dim.size > 0])


def numel(tensor: Union[mtf.Tensor, TensorMixin]) -> _int:
    return concrete_dims(tensor).size


def view(x: Union[mtf.Tensor, TensorMixin], *dims) -> Union[mtf.Tensor, TensorMixin]:
    new_shape = shapelist(x, *dims)
    return mtf.reshape(x, new_shape)


def permute(x: Union[mtf.Tensor, TensorMixin], *dims) -> Union[mtf.Tensor, TensorMixin]:
    new_shape = shapelist(x, *dims)
    return mtf.transpose(x, new_shape)


def zeros(*dims, dtype=None, requires_grad=False, mesh=None) -> Union[mtf.Tensor, TensorMixin]:
    mesh = get_mesh(mesh)
    shape = size(dims)
    dtype = get_dtype(dtype, tf.float32)
    t = mtf.zeros(mesh, shape, dtype=dtype)
    return _make_tensor(t, requires_grad=requires_grad, dtype=dtype)


def ones(*dims, dtype=None, requires_grad=False, mesh=None) -> Union[mtf.Tensor, TensorMixin]:
    mesh = get_mesh(mesh)
    shape = size(dims)
    dtype = get_dtype(dtype, tf.float32)
    t = mtf.ones(mesh, shape, dtype=dtype)
    return _make_tensor(t, requires_grad=requires_grad, dtype=dtype)


def dtype_of(dtype: Union[_tf_dtype_spec, Tensor]) -> Union[_none, _tf_dtype]:
    if hasattr(dtype, 'dtype'):
        dtype = dtype.dtype
    if isinstance(dtype, str):
        dtype = getattr(tf.dtypes, dtype)
    return dtype


def get_dtype(dtype: Union[_tf_dtype_spec, Tensor], preferred: Union[_tf_dtype_spec, Tensor]) -> _tf_dtype:
    dtype = dtype_of(dtype)
    preferred = dtype_of(preferred)
    if dtype is None:
        if preferred is None:
            raise ValueError(f"get_dtype(dtype={dtype!r}, preferred={preferred!r}): preferred should not be None")
            # return tf.float32
        return preferred
    return dtype


def zeros_like(tensor: Union[mtf.Tensor, TensorMixin], *, dtype=None, requires_grad=False):
    return zeros(*size(tensor), dtype=get_dtype(dtype, tensor), requires_grad=requires_grad)


def ones_like(tensor: Union[mtf.Tensor, TensorMixin], *, dtype=None, requires_grad=False):
    return ones(*size(tensor), dtype=get_dtype(dtype, tensor), requires_grad=requires_grad)


def _make_tensor(tensor: Union[mtf.Tensor, TensorMixin], *, requires_grad, dtype=None) -> Union[mtf.Tensor, TensorMixin]:
    # TODO: verify dtype is compatible with tensor
    if requires_grad is not None:
        tensor.operation.requires_grad = requires_grad
        tensor.requires_grad = requires_grad
    return tensor


def tensor(data, shape=None, *, dtype=None, requires_grad=False, mesh=None, name=None) -> Union[mtf.Tensor, TensorMixin]:
    mesh = get_mesh(mesh)
    if is_tf_tensor(data):
        shape = shapelist(data, shape)
        result = mtf.import_tf_tensor(mesh, data, shape=shape, name=name)
    elif isinstance(data, mtf.Tensor):
        result = mtf.identity(data, name=name)
    else:
        data = as_numpy(data)
        # shape = size(shape)
        shape = shapelist(data, shape)
        result = mtf.constant(mesh, data, shape=shape)
    dtype = get_dtype(dtype, data)
    return _make_tensor(result, requires_grad=requires_grad, dtype=dtype)


def cat(tensors: Sequence[Union[mtf.Tensor, TensorMixin]], dim=0) -> Union[mtf.Tensor, TensorMixin]:
    dim = size(unique_dims(tensors), dim)
    return mtf.concat(tensors, dim.name)


def stack(tensors: Sequence[Union[mtf.Tensor, TensorMixin]], new_dim_name, axis=0) -> Union[mtf.Tensor, TensorMixin]:
    return mtf.stack(tensors, new_dim_name, axis=axis)


def unbind(tensor: Union[mtf.Tensor, TensorMixin], dim=0) -> Sequence[Union[mtf.Tensor, TensorMixin]]:
    """Eliminates a tensor dimension."""
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


def slice(x: Union[mtf.Tensor, TensorMixin], slice_dim_name, start=None, stop=None) -> Union[mtf.Tensor, TensorMixin]:
    n = size(x, slice_dim_name).size
    start = _idx(start, n, 0)
    stop = _idx(stop, n, n)
    if isinstance(start, int) and isinstance(stop, int):
        if stop < 0:
            stop = 0
        if stop > n:
            stop = n
        if start < 0:
            start = 0
        if start > n:
            start = n
        if stop < start:
            stop = start
    else:
        # TODO: verify stop >= start when either stop or start are tensors.
        pass
    return mtf.slice(x, begin=start, size=stop - start, slice_dim_name=slice_dim_name)


def index_args(x, *args):
    args = [v for v in args]
    if is_ellipsis(x):
        dims = [x]
    else:
        dims = as_dims(x)
        for i, dim in enumerate(dims):
            if dim.size is None:
                if len(args) <= 0:
                    raise ValueError(f"Not enough args for indexing operation into {dims}")
                dims[i] = mtf.Dimension(dim.name, args[0])
                args = args[1:]
    if len(args) > 0:
        dims += index_args(args[0], *args[1:])
    return dims


def index(x: Union[mtf.Tensor, TensorMixin], *more) -> Union[mtf.Tensor, TensorMixin]:
    dims = index_args(*more)
    j = 0
    for i, dim in enumerate(dims):
        if is_ellipsis(dim):
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


def split(tensor: Union[mtf.Tensor, TensorMixin], split_dim, num_or_size_splits) -> List[Union[mtf.Tensor, TensorMixin]]:
    shape = shapelist(tensor, split_dim)
    if len(shape) != 1:
        raise ValueError(f"Must only specify one dimension, got {split_dim}")
    split_dim = shape[0]
    return mtf.split(tensor, split_dim, num_or_size_splits)


def squeeze(tensor: Union[mtf.Tensor, TensorMixin], squeeze_dim) -> Union[mtf.Tensor, TensorMixin]:
    shape = shapelist(tensor, squeeze_dim)
    for dim in shape:
        if tensor.shape.get_dim_by_name(dim.name).size != 1:
            raise ValueError(f"Squeeze dimension size must be 1 for dimension {dim} of shape {tensor.shape}")
    squeeze_dims = [dim.name for dim in shape]
    return view(tensor, [dim for dim in list(tensor.shape) if dim.name not in squeeze_dims])


def rand(*dims, seed, dtype=None) -> Union[mtf.Tensor, TensorMixin]:
    dtype = get_dtype(dtype, tf.float32)
    shape = as_shape(dims)
    tf_shape = [dim.size for dim in shape]
    x = tf.random.stateless_uniform(shape=tf_shape, seed=seed, dtype=dtype)
    return mtf.import_tf_tensor(get_mesh(), x, shape)


def randn(*dims, seed, dtype=None) -> Union[mtf.Tensor, TensorMixin]:
    dtype = get_dtype(dtype, tf.float32)
    shape = as_shape(dims)
    tf_shape = [dim.size for dim in shape]
    x = tf.random.stateless_normal(shape=tf_shape, seed=seed, dtype=dtype)
    return mtf.import_tf_tensor(get_mesh(), x, shape)


def randint(*args, seed, dtype=None) -> Union[mtf.Tensor, TensorMixin]:
    if len(args) < 2 or len(args) > 4:
        raise ValueError(f"Expected randint(low=0, high, dims), got {args}")
    if len(args) == 2:
        low = 0
        high, dims = args
    else:
        low, high, dims = args
    dtype = get_dtype(dtype, tf.int64)
    shape = as_shape(dims)
    tf_shape = [dim.size for dim in shape]
    x = tf.random.stateless_uniform(shape=tf_shape, seed=seed, minval=low, maxval=high, dtype=dtype)
    return mtf.import_tf_tensor(get_mesh(), x, shape)


def range(dim, dtype=None) -> Union[mtf.Tensor, TensorMixin]:
    dtype = get_dtype(dtype, tf.float32)
    shape = size(dim)
    if len(shape) != 1:
        raise ValueError(f"range() expected a single dimension, got {shape}")
    dim = size(shape, 0)
    return mtf.range(get_mesh(), dim, dtype=dtype)


def arange(dim, dtype=None) -> Union[mtf.Tensor, TensorMixin]:
    dtype = get_dtype(dtype, tf.int64)
    return range(dim, dtype=dtype)


def cumsum(x: Union[mtf.Tensor, TensorMixin], dim) -> Union[mtf.Tensor, TensorMixin]:
    return mtf.cumsum(x, size(x, dim))


def lerp(x: Union[mtf.Tensor, TensorMixin], y: Union[mtf.Tensor, TensorMixin], alpha) -> Union[mtf.Tensor, TensorMixin]:
    return x * (1 - alpha) + y * alpha


def meshgrid(*args: Union[mtf.Tensor, TensorMixin]) -> List[Union[mtf.Tensor, TensorMixin]]:
    if len(args) <= 1:
        return list(args[0:1])
    x, y, *zs = args
    shape = size(y) + size(x) + size(zs)
    t = ones(shape)
    dims = size(args)
    # TODO: properly handle non-integer meshgrids.
    return [(cumsum(t, dim) - 1.0 + min(mtf.to_float(args[i]))).to(x.dtype) for i, dim in enumerate(dims)]


def reduction(reduce_fn, tensor: Union[mtf.Tensor, TensorMixin], dim=None, keepdim=False) -> Union[mtf.Tensor, TensorMixin]:
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


def sum(tensor: Union[mtf.Tensor, TensorMixin], dim=None, keepdim=False) -> Union[mtf.Tensor, TensorMixin]:
    return reduction(mtf.reduce_sum, tensor, dim=dim, keepdim=keepdim)


def mean(tensor: Union[mtf.Tensor, TensorMixin], dim=None, keepdim=False) -> Union[mtf.Tensor, TensorMixin]:
    return reduction(mtf.reduce_mean, tensor, dim=dim, keepdim=keepdim)


def min(tensor: Union[mtf.Tensor, TensorMixin], dim=None, keepdim=False) -> Union[mtf.Tensor, TensorMixin]:
    return reduction(mtf.reduce_min, tensor, dim=dim, keepdim=keepdim)


def max(tensor: Union[mtf.Tensor, TensorMixin], dim=None, keepdim=False) -> Union[mtf.Tensor, TensorMixin]:
    return reduction(mtf.reduce_max, tensor, dim=dim, keepdim=keepdim)


def any(tensor: Union[mtf.Tensor, TensorMixin], dim=None, keepdim=False) -> Union[mtf.Tensor, TensorMixin]:
    return reduction(mtf.reduce_any, tensor, dim=dim, keepdim=keepdim)


def all(tensor: Union[mtf.Tensor, TensorMixin], dim=None, keepdim=False) -> Union[mtf.Tensor, TensorMixin]:
    return reduction(mtf.reduce_all, tensor, dim=dim, keepdim=keepdim)


def cwise(tensor: Union[mtf.Tensor, TensorMixin], tf_fn, output_dtype=None, grad_function=None) -> Union[mtf.Operation, OperationMixin]:
    return mtf.cwise(tf_fn, [tensor], output_dtype=output_dtype, grad_function=grad_function)


def binary_op(x1: Union[mtf.Tensor, TensorMixin], x2: Union[mtf.Tensor, TensorMixin], tf_fn, output_shape=None) -> Union[mtf.Operation, OperationMixin]:
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


def logical_not(tensor: Union[mtf.Tensor, TensorMixin]):
    return cwise(tensor, tf.math.logical_not)


def logical_xor(x1: Union[mtf.Tensor, TensorMixin], x2: Union[mtf.Tensor, TensorMixin]):
    return binary_op(x1, x2, tf.math.logical_xor)


def logical_or(x1: Union[mtf.Tensor, TensorMixin], x2: Union[mtf.Tensor, TensorMixin]):
    return binary_op(x1, x2, tf.math.logical_or)


def logical_and(x1: Union[mtf.Tensor, TensorMixin], x2: Union[mtf.Tensor, TensorMixin]):
    return binary_op(x1, x2, tf.math.logical_and)


def bitwise_not(tensor: Union[mtf.Tensor, TensorMixin]):
    return cwise(tensor, tf.bitwise.invert)


def bitwise_xor(x1: Union[mtf.Tensor, TensorMixin], x2: Union[mtf.Tensor, TensorMixin]):
    return binary_op(x1, x2, tf.bitwise.bitwise_xor)


def bitwise_or(x1: Union[mtf.Tensor, TensorMixin], x2: Union[mtf.Tensor, TensorMixin]):
    return binary_op(x1, x2, tf.bitwise.bitwise_or)


def bitwise_and(x1: Union[mtf.Tensor, TensorMixin], x2: Union[mtf.Tensor, TensorMixin]):
    return binary_op(x1, x2, tf.bitwise.bitwise_and)


def bitwise_left_shift(x1: Union[mtf.Tensor, TensorMixin], x2: Union[mtf.Tensor, TensorMixin]):
    return binary_op(x1, x2, tf.bitwise.left_shift)


def bitwise_right_shift(x1: Union[mtf.Tensor, TensorMixin], x2: Union[mtf.Tensor, TensorMixin]):
    return binary_op(x1, x2, tf.bitwise.right_shift)


def promote_types(type1, type2):
    # TODO: ensure proper type promotion.
    assert type1 == type2
    return type1


def allclose(input: Union[mtf.Tensor, TensorMixin], other: Tensor, rtol: _float = 1e-05, atol: _float = 1e-08, equal_nan: _bool = False) -> _bool:
    # TODO: This only works for "eager" execution.
    input = as_numpy(input)
    other = as_numpy(other)
    return np.allclose(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan)


def isclose(input: Union[mtf.Tensor, TensorMixin], other: Tensor, rtol: _float = 1e-05, atol: _float = 1e-08, equal_nan: _bool = False) -> Tensor:
    # TODO: This only works for "eager" execution.
    input = as_numpy(input)
    other = as_numpy(other)
    return np.isclose(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan)


def convert_to_dimension(d) -> Union[_none, mtf.Dimension]:
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
    def __call__(self, *args, **kws) -> Shape:
        return size(getattr(self, '_tensor'), *args, **kws)


class MixinBase:
    def __init__(self):
        self.__init__called = getattr(self, '__init__called', 0)
        self.__init__called += 1

    @classmethod
    def construct(cls, self, *args, **kws):
        self.__class__.__init__(self, *args, **kws)
        cls.__init__(self)
        assert self.__init__called == 1
        __init__ = self.__class__.__init__

        def hook__init__(self_, *_args, **_kws):
            assert id(self) == id(self_), "__init__ wasn't trapped!"
            self.__class__.__init__ = __init__

        self.__class__.__init__ = hook__init__


class TensorMixin(MixinBase):
    shape: Union[mtf.Shape, ShapeMixin]
    item = item

    # numpy = numpy

    def numpy(self: Union[Tensor, TensorMixin]) -> np.ndarray:
        return numpy(self)

    tf = to_tf

    numel = numel

    def dim(self: Union[Tensor, TensorMixin]) -> _int:
        return len(self.shape)

    # size = size

    @property
    def size(self: Union[Tensor, TensorMixin]) -> TensorSizeFunctor:
        v = TensorSizeFunctor(self.shape.size)
        v._tensor = self
        return v

    @property
    def to_string(self: Union[Tensor, TensorMixin]) -> str:
        try:
            return "Tensor[%s, %s, %s, %s]" % (self.name, self.shape.to_string, self.dtype, self.numpy())
        except:
            return "Tensor[%s, %s, %s]" % (self.name, self.shape.to_string, self.dtype)

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

    def bfloat16(self: Union[Tensor, TensorMixin]) -> Union[Tensor, TensorMixin]:
        return self.to(bfloat16)

    def half(self: Union[Tensor, TensorMixin]) -> Union[Tensor, TensorMixin]:
        return self.to(float16)

    def float(self: Union[Tensor, TensorMixin]) -> Union[Tensor, TensorMixin]:
        return self.to(float32)

    def double(self: Union[Tensor, TensorMixin]) -> Union[Tensor, TensorMixin]:
        return self.to(float64)

    def int(self: Union[Tensor, TensorMixin]) -> Union[Tensor, TensorMixin]:
        return self.to(int32)

    def long(self: Union[Tensor, TensorMixin]) -> Union[Tensor, TensorMixin]:
        return self.to(int64)

    def bool(self: Union[Tensor, TensorMixin]) -> Union[Tensor, TensorMixin]:
        return self.to(tf.bool)

    def range(self: Union[Tensor, TensorMixin], dim, dtype=None) -> Union[Tensor, TensorMixin]:
        return range(self.size(dim), dtype=dtype)

    def arange(self: Union[Tensor, TensorMixin], dim, dtype=None) -> Union[Tensor, TensorMixin]:
        return arange(self.size(dim), dtype=dtype)

    def cat(self: Union[Tensor, TensorMixin], tensors, dim=0) -> Union[Tensor, TensorMixin]:
        tensors = [self if tensor == "_" else tensor for tensor in tensors]
        return cat(tensors, dim=dim)

    def stack(self: Union[Tensor, TensorMixin], tensors, new_dim_name, axis=0) -> Union[Tensor, TensorMixin]:
        tensors = [self if tensor == "_" else tensor for tensor in tensors]
        return stack(tensors, new_dim_name=new_dim_name, axis=axis)

    @property
    def is_sparse(self: Union[Tensor, TensorMixin]) -> _bool:
        return False

    @property
    def is_quantized(self: Union[Tensor, TensorMixin]) -> _bool:
        return False

    @property
    def device(self: Union[Tensor, TensorMixin]) -> Device:
        return None

    def to(self: Union[Tensor, TensorMixin], dtype) -> Union[Tensor, TensorMixin]:
        return mtf.cast(self, dtype)

    def is_complex(self: Union[Tensor, TensorMixin]) -> _bool:
        return self.dtype.is_complex

    def __getitem__(self: Union[Tensor, TensorMixin], item) -> Union[Tensor, TensorMixin]:
        return index(self, *item)

    # __hash__ = object.__hash__

    __lt__ = lt = lt
    __gt__ = gt = gt
    __le__ = le = le
    __ge__ = ge = ge
    # __eq__ = eq = eq
    # __ne__ = ne = ne

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

    @property
    def is_leaf(self: Union[Tensor, TensorMixin]) -> _bool:
        return self._requires_grad is not None

    @property
    def requires_grad(self: Union[Tensor, TensorMixin]) -> _bool:
        # return self.operation.requires_grad
        if self._requires_grad is not None:
            return self._requires_grad
        return self.operation.requires_grad

    @requires_grad.setter
    def requires_grad(self: Union[Tensor, TensorMixin], value):
        self._requires_grad = value

    def __init__(self: Union[Tensor, TensorMixin]):
        super().__init__()
        self._requires_grad = None
        self.grad = None
        print(f"Tensor {self.shape}")

    def __new__(cls, *args, **kws) -> Union[Tensor, TensorMixin]:
        self: Union[Tensor, TensorMixin] = super().__new__(cls)
        TensorMixin.construct(self, *args, **kws)
        return self


class OperationMixin(MixinBase):
    def __init__(self: Union[Operation, OperationMixin]):
        super().__init__()
        self._requires_grad = is_grad_enabled()
        print(f"Operation {self.graph.operations.index(self)} {type(self)} requires_grad={self.requires_grad}")

    def __new__(cls, *args, **kws) -> Union[Operation, OperationMixin]:
        self: Union[Operation, OperationMixin] = super().__new__(cls)
        OperationMixin.construct(self, *args, **kws)
        return self

    @property
    def requires_grad(self: Union[Operation, OperationMixin]) -> _bool:
        # return getattr(self, '_requires_grad', True)
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self: Union[Operation, OperationMixin], value: _bool):
        self._requires_grad = value

    @property
    def has_gradient(self: Union[Operation, OperationMixin]) -> Union[_bool, List[Union[Operation, OperationMixin]]]:
        if not self.requires_grad:
            return False
        return (
                [t for t in self.inputs if t.dtype.is_floating] and
                [t for t in self.outputs if t.dtype.is_floating])


class ShapeMixin:
    # support None sizes
    @property
    def to_string(self: Union[Shape, ShapeMixin]) -> str:
        return "Shape[%s]" % ", ".join(
            ["{}={}".format(d.name, d.size) for d in self.dims])


class DTypeMixin:
    @property
    def is_signed(self: Union[_tf_dtype, DTypeMixin]) -> _bool:
        return not self.is_unsigned

    @property
    def is_floating_point(self: Union[_tf_dtype, DTypeMixin]) -> _bool:
        return self.is_floating

    # ensure that str(mtorch.float32).split('.')[-1] == 'float32'
    def __str__(self: Union[_tf_dtype, DTypeMixin]) -> str:
        return repr(self)


# https://qastack.vn/programming/9539052/how-to-dynamically-change-base-class-of-instances-at-runtime

def ensure_class_bases_begin_with(namespace: ModuleType, class_name, base_class):
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
        subclasses = existing_class.__subclasses__()
        for subclass in subclasses:
            subclass.__bases__ = tuple([new_class if base is existing_class else base for base in subclass.__bases__])
        return new_class
    else:
        if base_class in bases:
            # Already bound to a type with the right bases.
            return existing_class
        bases.insert(0, base_class)
        existing_class.__bases__ = tuple(bases)
        return existing_class


# we redefine size to be torch-like
delattr(mesh_tensorflow.ops.Tensor, 'size')

# we override has_gradient to add a requires_grad property to tensors
delattr(mesh_tensorflow.ops.Operation, 'has_gradient')

# we print tensor values to the repl
delattr(mesh_tensorflow.ops.Tensor, 'to_string')

for module in [mesh_tensorflow.ops]:
    module: ModuleType
    module.Tensor = ensure_class_bases_begin_with(module, module.Tensor, TensorMixin)
    module.Operation = ensure_class_bases_begin_with(module, module.Operation, OperationMixin)
    module.Shape = ensure_class_bases_begin_with(module, module.Shape, ShapeMixin)

mtf.Tensor = mesh_tensorflow.ops.Tensor
mtf.Operation = mesh_tensorflow.ops.Operation
mtf.Shape = mesh_tensorflow.ops.Shape

globals()['Tensor'] = mtf.Tensor
globals()['Operation'] = mtf.Operation
globals()['Shape'] = mtf.Shape

TensorType = Union[mtf.Tensor, TensorMixin]
OperationType = Union[mtf.Operation, OperationMixin]
ShapeType = Union[mtf.Shape, ShapeMixin]

mesh_tensorflow.ops.convert_to_dimension = convert_to_dimension

# we extend Shape.to_string to support None sizes
setattr(mesh_tensorflow.ops.Shape, 'to_string', getattr(ShapeMixin, 'to_string'))

# ensure that str(mtorch.float32).split('.')[-1] == 'float32'
# float32.__class__.__str__ = float32.__class__.__repr__
# float32.__class__.is_floating_point = float32.__class__.is_floating

for module in [_tf_dtypes]:
    module: ModuleType
    module.DType = ensure_class_bases_begin_with(module, module.DType, DTypeMixin)

assert tf.int32.is_signed


def reset(graph=None):
    if graph is None:
        graph = get_graph()
    graph.operations.clear()
