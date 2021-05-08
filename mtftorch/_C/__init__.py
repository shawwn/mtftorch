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
from tensorflow.python.framework import tensor_util as _tf_tensor_util

import mesh_tensorflow.ops
from mesh_tensorflow.ops import Tensor
from mesh_tensorflow.ops import Operation
from mesh_tensorflow.ops import Shape
from mesh_tensorflow.ops import VariableDType as dtype

from tensorflow.python.framework.dtypes import bool, uint8, int8, int16, int32, int64, bfloat16, float16, float32, float64, complex64, complex128, qint8, qint16, qint32, quint8

from mtftorch.types import _none, _ellipsis, _int, _float, _bool, _dtype, _tf_dtype, _tf_dtype_spec, _device, _qscheme, _shape, _layout, Device, Number, Storage
from mtftorch import return_types

from mtftorch import _jit_internal

tf = tf2.compat.v1

logger = tf.get_logger()

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


# Defined in torch/csrc/Layout.cpp
class layout:
    ...

# # Defined in torch/csrc/utils/disable_torch_function.cpp
# def DisableTorchFunction(): ...

# Defined in torch/csrc/utils/tensor_layouts.cpp
strided : layout = layout()
sparse_coo : layout = layout()
_mkldnn : layout = layout()

# Defined in torch/csrc/MemoryFormat.cpp
class memory_format: ...

# Defined in torch/csrc/utils/tensor_memoryformats.cpp
contiguous_format: memory_format = memory_format()
channels_last: memory_format = memory_format()
channels_last_3d: memory_format = memory_format()
preserve_format: memory_format = memory_format()

_api_usage_seen = set()


def _log_api_usage_once(*args):
    if args not in _api_usage_seen:
        _api_usage_seen.add(args)
        tf.get_logger().info(' '.join(["%s"] * len(args)), *args)


def _get_tracing_state():
    return False


import contextvars as cv


# import mesh_tensorflow.test_utils


class ConverterBase(object):
    pass


class Converter(ConverterBase): # (mesh_tensorflow.test_utils.NumpyConverter):
    """Converter class to convert between mtf.Tensor, tf.Tensor and np.array."""

    def __init__(self, session=None):
        super().__init__()
        if session is None:
            session = tf.get_default_session()
        self._cached_session = session

    def convert_to_np_array(self, x: Union[np.ndarray, tf.Tensor, mtf.Tensor, TensorMixin]):
        if is_tensor(x):
            return self.convert_mtf_tensor_to_np_array(x)
        if is_tf_tensor(x):
            return self.convert_tf_tensor_to_np_array(x)
        return as_numpy(x)

    def convert_to_tf_tensor(self, x: Union[np.ndarray, tf.Tensor, mtf.Tensor, TensorMixin]):
        orig = x
        if is_tf_tensor(x):
            return x
        if not is_tensor(x):
            return tf.constant(x)
        if is_tensor(x):
            return self.convert_mtf_tensor_to_tf_tensor(x)
        raise ValueError(f"Couldn't convert {orig} to tf_tensor")

    def convert_mtf_tensor_to_np_array(self, tensor: Union[mtf.Tensor, TensorMixin]) -> np.ndarray:
        """Convert an mtf.Tensor to a numpy array."""
        assert is_tensor(tensor)
        tensor = self.convert_mtf_tensor_to_tf_tensor(tensor)
        tensor = self.convert_tf_tensor_to_np_array(tensor)
        return tensor

    def convert_tf_tensor_to_np_array(self, tensor: tf.Tensor) -> np.ndarray:
        assert is_tf_tensor(tensor)
        if tf.executing_eagerly():
            return as_numpy(tensor.numpy())
        else:
            session = self.get_session()
            # session.run(tf.global_variables_initializer())
            return as_numpy(tensor.eval(session=session))

    def convert_mtf_tensor_to_tf_tensor(self, tensor: Union[mtf.Tensor, TensorMixin]):
        """Convert an mtf.Tensor to a tf.Tensor."""
        assert is_tensor(tensor)
        # if is_tf_tensor(tensor):
        #     return tensor
        if False:
            try:
                return self.lowering.export_to_tf_tensor(tensor)
            except KeyError:
                state = get_state()
                state.reset_lowering()
                return self.lowering.export_to_tf_tensor(tensor)
        else:
            return self.lowering.export_to_tf_tensor(tensor)

    @property
    def graph(self) -> Union[mtf.Graph, GraphMixin]:
        return get_graph()

    @property
    def mesh(self) -> mtf.Mesh:
        return get_mesh()

    @property
    def mesh_impl(self) -> mtf.MeshImpl:
        return get_mesh_impl()

    @property
    def lowering(self) -> mtf.Lowering:
        return get_lowering()

    def get_session(self, session=None) -> tf.Session:
        if session is None:
            session = tf.get_default_session()
        if session is None:
            if self._cached_session is None:
                self._cached_session = tf.Session()
            session = self._cached_session
        return session


class State:
    def __init__(self, name="mtftorch_state", *,
                 graph=None, mesh=None, mesh_impl=None, lowering=None, session=None):
        if graph is None:
            graph = mtf.Graph()
        if mesh is None:
            mesh = mtf.Mesh(graph, name)
        if mesh_impl is None:
            mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
                shape=[], layout={}, devices=[""])
        if lowering is None:
            lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=False, lazy=True)
        self.graph = graph
        self.mesh = mesh
        self.mesh_impl = mesh_impl
        self.lowering = lowering
        self.converter = Converter(session=session)

    def reset_lowering(self):
        self.lowering = mtf.Lowering(self.graph, {self.mesh: self.mesh_impl}, autostack=False, lazy=True)
        for op in self.graph.operations:
            self.lowering.lower_op(op)


def get_state() -> State:
    return State.current.get()


def get(attr, *defaults):
    state = State.current.get()
    return getattr(state, attr, *defaults)


def get_graph(graph=None) -> Union[mtf.Graph, GraphMixin]:
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


def get_session(session=None) -> tf.Session:
    if session is not None:
        return session
    return get_converter().get_session(session)


@contextmanager
def _with_value(cvar, value):
    reset_value = cvar.set(value)
    try:
        yield
    finally:
        cvar.reset(reset_value)


def with_state(state: State):
    return _with_value(State.current, state)


def _autograd_init() -> _bool:
    return True


def is_grad_enabled() -> _bool:
    return State.requires_grad.get()


def _set_grad_enabled(value: _bool):
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


def from_numpy(x: Union[np.ndarray, tf.Tensor, mtf.Tensor, TensorMixin]) -> np.ndarray:
    return tensor(x)


def to_tf(x: Union[np.ndarray, tf.Tensor, mtf.Tensor, TensorMixin]) -> tf.Tensor:
    return get_converter().convert_to_tf_tensor(x)


def item(x: Union[mtf.Tensor, TensorMixin]) -> np.ndarray:
    if x.shape.size > 1:
        raise ValueError("only one element tensors can be converted to Python scalars")
    return numpy(x)


def as_shape(x) -> Union[mtf.Shape, ShapeMixin]:
    return mtf.Shape(as_dims(x))


def as_dims(x) -> List[mtf.Dimension]:
    if is_numpy(x):
        return [mtf.Dimension(name=str(i), size=v) for i, v in enumerate(x.shape)]
    if is_tf_tensor(x):
        return [mtf.Dimension(name=str(i), size=v) for i, v in enumerate(x.shape.as_list())]
    if hasattr(x, 'shape'):
        x = x.shape
    if isinstance(x, mtf.Shape):
        return x.dims
    if isinstance(x, mtf.Dimension):
        return [x]
    if isinstance(x, int):
        return [mtf.Dimension(None, x)]
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
    if x is None:
        return []
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
    # return isinstance(x, tf.Tensor)
    return _tf_tensor_util.is_tensor(x)


def is_tensor(x) -> _bool:
    return isinstance(x, mtf.Tensor)


def shapelist(x, *dims, selecting=True) -> Union[mtf.Shape, ShapeMixin]:
    if len(dims) == 1 and dims[0] is None:
        return size(x)
    old_dims = as_dims(x)
    if len(dims) <= 0:
        return size(old_dims)
    if builtins.all([isinstance(x, int) for x in dims]):
        if selecting:
            dims = [old_dims[x] for x in dims]
        else:
            dims = [[None, x] for x in dims]
    # dims = [old_dims[x] if isinstance(x, int) else x for x in dims]
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
    new_shape = shapelist(x, *dims, selecting=False)
    return mtf.reshape(x, new_shape)


def permute(x: Union[mtf.Tensor, TensorMixin], *dims) -> Union[mtf.Tensor, TensorMixin]:
    new_shape = shapelist(x, *dims)
    return mtf.transpose(x, new_shape)


def dim_index(x, dim) -> int:
    orig = dim
    shape = size(x)
    if hasattr(dim, 'name'):
        dim = dim.name
    if isinstance(dim, str):
        dim = shape.get_dim_by_name(dim)
        return shape.dims.index(dim)
    if isinstance(dim, int):
        return shape.dims.index(shape[dim])
    raise ValueError(f"Can't get dim_index for {shape} {orig}")


def transpose_shape(x: Union[mtf.Tensor, TensorMixin], dim0, dim1) -> Union[mtf.Shape, ShapeMixin]:
    dim0 = dim_index(x, dim0)
    dim1 = dim_index(x, dim1)
    old_dims = as_dims(x)
    new_dims = as_dims(x)
    new_dims[dim1] = old_dims[dim0]
    new_dims[dim0] = old_dims[dim1]
    return size(new_dims)


def transpose(x: Union[mtf.Tensor, TensorMixin], dim0, dim1) -> Union[mtf.Tensor, TensorMixin]:
    new_shape = transpose_shape(x, dim0, dim1)
    return permute(x, *new_shape)


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
    if isinstance(dtype, (mesh_tensorflow.ops.VariableDType, _tf_dtype)):
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

def _make_tensor_subclass(cls, data: Union[mtf.Tensor, TensorMixin], require_grad: _bool = False) -> Union[mtf.Tensor, TensorMixin]:
    # self, operation, shape, dtype, name = None, index = 0
    tensor = TensorMixin.__new__(cls, data.operation, data.shape, data.dtype)
    #TensorMixin.construct(tensor, data.operation, data.shape, data.dtype)
    tensor.construct(data.operation, data.shape, data.dtype)
    return _make_tensor(tensor, requires_grad=require_grad)


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
    # dtype = get_dtype(dtype, data)
    return _make_tensor(result, requires_grad=requires_grad, dtype=dtype)


def empty(*size, out=None, dtype=None, layout=strided, device=None, requires_grad=False, pin_memory=False, memory_format=contiguous_format) -> Union[mtf.Tensor, TensorMixin]:
    if out is not None:
        raise NotImplementedError()
    if layout != strided:
        raise NotImplementedError()
    if device is not None:
        raise NotImplementedError()
    if pin_memory is not False:
        raise NotImplementedError()
    if memory_format != contiguous_format:
        raise NotImplementedError()
    return zeros(*size, dtype=dtype, requires_grad=requires_grad)


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


def mtf_gather(tensor: mtf.Tensor, indices: mtf.Tensor, dims: List[mtf.Dimension]) -> mtf.Tensor:
  dims = [tensor.shape.get_dim_by_name(dim.name) for dim in dims]
  size = int(np.prod([dim.size for dim in dims]))
  anon = mtf.Dimension("?", size)
  tensor = mtf.replace_dimensions(tensor, dims, anon)
  return mtf.gather(tensor, indices.long(), anon)


def take(input: Union[mtf.Tensor, TensorMixin], index: Union[mtf.Tensor, TensorMixin]) -> Union[mtf.Tensor, TensorMixin]:
    return mtf_gather(input, index, input.shape - index.shape)
    # input_shape = input.size()
    # output_shape = exclude_dims(input, index.shape.dimension_names)
    # dims = as_dims(input_shape - output_shape)
    # assert len(dims) == 1
    # excluded_dim = input_shape - dims
    # return gather2(input, index, dims)
    # import pdb; pdb.set_race()
    # return mtf.gather(input, index.long(), dims[0])#, output_shape=output_shape)
    # #
    # # shape = index.size()
    # #
    # # input = view(input, index.shape)
    # # index = view(index, ['%index', -1])
    # # index = index.long()
    # # dim = size(input, 0)
    # # #import pdb; pdb.set_trace()
    # # result = mtf.gather(input, index, dim)
    # # final = view(result, shape)
    # # return final


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


def reduction2(reduce_fn, tensor: Union[mtf.Tensor, TensorMixin], dim=None, keepdim=False) -> Union[mtf.Tensor, TensorMixin]:
    input_shape = size(tensor)
    if dim is None:
        #dim = input_shape
        tensor = tensor.view(-1)
        dim = tensor.size(0)
    else:
        dim = size(input_shape, dim)
        #dim = shapelist(input_shape, dim)
    output_shape = exclude_dims(input_shape, dim)
    result = reduce_fn(tensor, dim)
    if keepdim:
        result_shape = [(dim.name, dim.size if dim in output_shape else 1)
                        for dim in input_shape]
        result = view(result, result_shape)
    return result


def sum(tensor: Union[mtf.Tensor, TensorMixin], dim=None, keepdim=False) -> Union[mtf.Tensor, TensorMixin]:
    return reduction(mtf.reduce_sum, tensor, dim=dim, keepdim=keepdim)


def mean(tensor: Union[mtf.Tensor, TensorMixin], dim=None, keepdim=False) -> Union[mtf.Tensor, TensorMixin]:
    #import pdb; pdb.set_trace()
    return reduction(mtf.reduce_mean, tensor, dim=dim, keepdim=keepdim)


@overload
def min(tensor: Union[mtf.Tensor, TensorMixin]) -> return_types.Tensor: ...


@overload
def min(tensor: Union[mtf.Tensor, TensorMixin], dim: mtf.Dimension=None, keepdim: _bool=False) -> return_types.min: ...


def min(tensor: Union[mtf.Tensor, TensorMixin], dim: mtf.Dimension=None, keepdim: _bool=False) -> Union[return_types.Tensor, return_types.min]:
    if dim is None:
        return reduction(mtf.reduce_min, tensor, dim=dim, keepdim=keepdim)
    else:
        indices = argmin(tensor, dim=dim, keepdim=keepdim)
        values = take(tensor, indices)
        return return_types.min(indices=indices, values=values)


def minimum(input: Union[mtf.Tensor, TensorMixin], other: Union[mtf.Tensor, TensorMixin]) -> return_types.Tensor:
    return mtf.where(input < other, input, other)


def amin(tensor: Union[mtf.Tensor, TensorMixin], dim=None, keepdim=False) -> Union[mtf.Tensor, TensorMixin]:
    return reduction(mtf.reduce_min, tensor, dim=dim, keepdim=keepdim)


def mm(input: Union[mtf.Tensor, TensorMixin], mat2: Union[mtf.Tensor, TensorMixin]) -> Union[mtf.Tensor, TensorMixin]:
    lhs = input.view(input.size() - input.size(-1), "%mul=-1")
    rhs = mat2.view("%mul=-1", mat2.size() - mat2.size(0))
    return mtf.matmul(lhs, rhs)


def matmul(input: Union[mtf.Tensor, TensorMixin], mat2: Union[mtf.Tensor, TensorMixin]) -> Union[mtf.Tensor, TensorMixin]:
    return mtf.matmul(input, mat2)


def t(input: Union[mtf.Tensor, TensorMixin]) -> Union[mtf.Tensor, TensorMixin]:
    rank = input.dim()
    if rank < 2:
        return input
    if rank > 2:
        raise RuntimeError(f"t() expects a tensor with <= 2 dimensions, but self is {rank}D")
    return transpose(input, 0, 1)


@overload
def max(tensor: Union[mtf.Tensor, TensorMixin]) -> return_types.Tensor: ...


@overload
def max(tensor: Union[mtf.Tensor, TensorMixin], dim: mtf.Dimension=None, keepdim: _bool=False) -> return_types.max: ...


def max(tensor: Union[mtf.Tensor, TensorMixin], dim: mtf.Dimension=None, keepdim: _bool=False) -> Union[return_types.Tensor, return_types.max]:
    if dim is None:
        return reduction(mtf.reduce_max, tensor, dim=dim, keepdim=keepdim)
    else:
        indices = argmax(tensor, dim=dim, keepdim=keepdim)
        values = take(tensor, indices)
        return return_types.max(indices=indices, values=values)


def argmin(tensor: Union[mtf.Tensor, TensorMixin], dim=None, keepdim=False) -> Union[mtf.Tensor, TensorMixin]:
    #raise NotImplementedError("TODO")
    return reduction2(mtf.argmax, -tensor, dim=dim, keepdim=keepdim).long()


def argmax(tensor: Union[mtf.Tensor, TensorMixin], dim=None, keepdim=False) -> Union[mtf.Tensor, TensorMixin]:
    return reduction2(mtf.argmax, tensor, dim=dim, keepdim=keepdim).long()


def any(tensor: Union[mtf.Tensor, TensorMixin], dim=None, keepdim=False) -> Union[mtf.Tensor, TensorMixin]:
    return reduction(mtf.reduce_any, tensor, dim=dim, keepdim=keepdim)


def all(tensor: Union[mtf.Tensor, TensorMixin], dim=None, keepdim=False) -> Union[mtf.Tensor, TensorMixin]:
    return reduction(mtf.reduce_all, tensor, dim=dim, keepdim=keepdim)


def cwise(tensor: Union[mtf.Tensor, TensorMixin], tf_fn, *xs, output_dtype=None, grad_function=None) -> Union[mtf.Operation, OperationMixin]:
    return mtf.cwise(tf_fn, [tensor] + list(xs), output_dtype=output_dtype, grad_function=grad_function)

def slicewise(tensor: Union[mtf.Tensor, TensorMixin], tf_fn, *xs, output_shape=None, output_dtype=None, splittable_dims=None, grad_function=None) -> Union[mtf.Operation, OperationMixin]:
    return mtf.slicewise(tf_fn, [tensor] + list(xs), output_shape=output_shape, output_dtype=output_dtype, splittable_dims=splittable_dims, grad_function=grad_function)

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


def allclose(input: Union[mtf.Tensor, TensorMixin], other: Tensor, rtol: _float = 1e-05, atol: _float = 1e-08, equal_nan: _bool = False) -> Union[_bool, np.ndarray]:
    # TODO: This only works for "eager" execution.
    input = as_numpy(input)
    other = as_numpy(other)
    return np.allclose(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan)


def isclose(input: Union[mtf.Tensor, TensorMixin], other: Tensor, rtol: _float = 1e-05, atol: _float = 1e-08, equal_nan: _bool = False) -> Union[_bool, np.ndarray]:
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
        return d
    if isinstance(d, mtf.Dimension):
        # if not isinstance(d.name, str) or not (isinstance(d.size, int) or d.size is None):
        #     raise ValueError("Bad dimension %s" % (d,))
        # return d
        name, size = d.name, d.size
    else:
        name, size = d
    if isinstance(name, int):
        name = str(name)
    return mtf.Dimension(name, size)
    # if isinstance(name, str) and (isinstance(size, int) or size is None):
    #     return mtf.Dimension(name, size)
    # else:
    #     raise ValueError("could not convert %s to Dimension" % (d,))


class TensorSizeFunctor(int):
    def __call__(self, *args, **kws) -> Shape:
        return size(getattr(self, '_tensor'), *args, **kws)


class MixinBase:
    def __init__(self):
        self.__init__called = getattr(self, '__init__called', 0)
        self.__init__called += 1

    def __hook_init__(self):
        assert self.__init__called == 1
        __init__ = self.__class__.__init__

        def hook__init__(self_, *_args, **_kws):
            assert id(self) == id(self_), "__init__ wasn't trapped!"
            self.__class__.__init__ = __init__

        self.__class__.__init__ = hook__init__
        return self

    @classmethod
    def construct(cls, self, *args, **kws):
        self.__class__.__init__(self, *args, **kws)
        cls.__init__(self)
        self.__hook_init__()


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
            return "Tensor[\nname=\t%s,\ndtype=\t%s,\nshape=\t%s\nvalue=\n%s\n]" % (self.name, self.dtype, self.shape.to_string, self.numpy())
        except:
            return "Tensor[%s, %s, %s]" % (self.name, self.shape.to_string, self.dtype)

    view = view
    permute = permute
    transpose = transpose
    select_dims = select_dims
    exclude_dims = exclude_dims
    unbind = unbind
    sum = sum
    mean = mean
    min = min
    minimum = minimum
    amin = amin
    argmin = argmin
    mm = mm
    matmul = matmul
    max = max
    argmax = argmax
    any = any
    all = all
    slice = slice
    split = split
    squeeze = squeeze
    take = take
    index = index
    t = t

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

    def detach(self):
        t: Union[Tensor, TensorMixin] = mtf.identity(self, self.operation.name)
        t.requires_grad = False
        return t

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
        logger.debug(f"Tensor {self.shape}")

    def __new__(cls, *args, **kws) -> Union[Tensor, TensorMixin]:
        self: Union[Tensor, TensorMixin] = super().__new__(cls)
        TensorMixin.construct(self, *args, **kws)
        return self

    @staticmethod
    def _make_subclass(cls, data: Union[Tensor, TensorMixin], require_grad: _bool = False) -> Union[Tensor, TensorMixin]:
        self = TensorMixin.__new__(cls, data.operation, data.shape, data.dtype)
        return _make_tensor(self, requires_grad=require_grad, dtype=self.dtype)
        # self: Union[Tensor, TensorMixin] = super().__new__(cls)
        # TensorMixin.construct(self, *args, **kws)
        # return self


    # def _make_subclass(self, operation, shape, dtype, name = None, index = 0):
    #     return self


class OperationMixin(MixinBase):
    def __init__(self: Union[Operation, OperationMixin]):
        super().__init__()
        self._requires_grad = is_grad_enabled()
        logger.debug(f"Operation {self.graph.operations.index(self)} {type(self)} requires_grad={self.requires_grad}")

    def __new__(cls, *args, **kws) -> Union[Operation, OperationMixin]:
        self: Union[Operation, OperationMixin] = super().__new__(cls)
        OperationMixin.construct(self, *args, **kws)
        lowering = get_lowering()
        lowering.lower_op(self)
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


class ShapeMixin(MixinBase):
    def __init__(self: Union[Shape, ShapeMixin]):
        super().__init__()
        #logger.debug(f"Shape {self}")

    def __new__(cls, dims) -> Union[Shape, ShapeMixin]:
        self: Union[Shape, ShapeMixin] = super().__new__(cls)
        logger.debug(f"ShapeMixin.construct({dims})")
        new_dims = []
        dims = as_dims(dims)
        for i, dim in enumerate(dims):
            if dim.name is None:
                dim = mtf.Dimension(str(i), dim.size)
            new_dims.append(dim)
        ShapeMixin.construct(self, new_dims)
        return self

    # support None sizes
    @property
    def to_string(self: Union[Shape, ShapeMixin]) -> str:
        return "Shape[%s]" % ", ".join(
            ["{}={}".format(d.name, d.size) for d in self.dims])


class GraphMixin:
    def reset(self: Union[mtf.Graph, GraphMixin]):
        # self._operations = []
        # self._trainable_variables = []
        # self._all_variables = []
        # # Maps a name used in the graph to the next id to use for that name.
        # self._names_in_use = {}
        # self.name_to_variable = {}
        # self.captured_variable_scope = tf.get_variable_scope()
        self._operations.clear()
        self._trainable_variables.clear()
        self._all_variables.clear()
        self._names_in_use.clear()
        self.name_to_variable.clear()


class DTypeMixin:
    @property
    def is_signed(self: Union[_tf_dtype, DTypeMixin]) -> _bool:
        return not self.is_unsigned

    @property
    def is_floating_point(self: Union[_tf_dtype, DTypeMixin]) -> _bool:
        return self.is_floating

    # ensure that str(mtftorch.float32).split('.')[-1] == 'float32'
    def __str__(self: Union[_tf_dtype, DTypeMixin]) -> str:
        return repr(self)


# we override Lowering's constructor.
delattr(mtf.Lowering, "__init__")


class LoweringMixin(MixinBase):
    def __new__(cls, *args, **kws) -> Union[Lowering, LoweringMixin]:
        self: Union[Tensor, LoweringMixin] = super().__new__(cls)
        self.__class__.__init__(self, *args, **kws)
        return MixinBase.__hook_init__(self)

    def __init__(self: Union[Lowering, LoweringMixin], graph, mesh_to_impl, autostack=True, log_file=None, lazy=False):
        """Creates a Lowering of a Graph.

        Args:
          graph: Graph.
          mesh_to_impl: {Mesh: MeshImpl}. Keys are the Mesh's in the graph and
            their values are MeshImpl's, which map Tensor Dimension names to
            Mesh Dimension names.
          autostack: a boolean.  If True, then the graph gets rewritten to
            reduce the number of variables (see rewrite_stack_variables()).
            This is a helpful performance optimization for large meshes.
            For more fine-grained control, you can call
            graph.rewrite_stack_variables() yourself before creating the Lowering.
          log_file: an optional string. If provided, information about the variables
            and operations will also be logged to this file.
          lazy: a boolean.  If true, then don't lower any graph operations.
        """
        super().__init__()
        # tf.logging.info("LOWERING GRAPH:\n%s" % graph.to_string)
        self.mesh_to_impl = mesh_to_impl   # {Mesh: MeshImpl}
        self.graph = graph
        if autostack:
            self.autostack()
        self._counters = []
        self.tensors = {}                  # {Tensor: Mesh.LaidOutTensor}
        self.operations = {}               # {Operation: tf.Operation}
        self.variables = {}                # {Variable: LaidOutVariable}
        self.lowered = []
        if not lazy:
            for op in graph.operations:
                self.lower_op(op)

    def lower_op(self: Union[Tensor, LoweringMixin], op: Union[Operation, OperationMixin]):
        if op in self.lowered:
            tf.logging.warn("Already lowered operation %s" % op.to_string)
            return
        self.lowered.append(op)
        tf.logging.debug("Lowering operation %s" % op.to_string)
        #tf.logging.warn("Lowering operation %s" % type(op).__name__)
        with tf.name_scope(op.name):
            op.lower(self)
        for out in op.outputs:
            self.add_counter(
                "output/%s" % type(op).__name__, self.laid_out_size(out))
            self.add_counter("output_unique/%s" % type(op).__name__, out.size)


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

# let a Dimension be interpreted as an integer
mesh_tensorflow.ops.Dimension.__int__ = lambda self: self.size
mesh_tensorflow.ops.Dimension.__add__ = lambda self, a: int(self) + a
mesh_tensorflow.ops.Dimension.__sub__ = lambda self, a: int(self) - a
mesh_tensorflow.ops.Dimension.__mul__ = lambda self, a: int(self) * a
mesh_tensorflow.ops.Dimension.__floordiv__ = lambda self, a: int(self) // a
mesh_tensorflow.ops.Dimension.__radd__ = lambda self, a: int(self) + a
mesh_tensorflow.ops.Dimension.__rsub__ = lambda self, a: int(self) - a
mesh_tensorflow.ops.Dimension.__rmul__ = lambda self, a: int(self) * a
mesh_tensorflow.ops.Dimension.__rfloordiv__ = lambda self, a: int(self) // a

for module in [mesh_tensorflow.ops]:
    module: ModuleType
    module.Tensor = ensure_class_bases_begin_with(module, module.Tensor, TensorMixin)
    module.Operation = ensure_class_bases_begin_with(module, module.Operation, OperationMixin)
    module.Shape = ensure_class_bases_begin_with(module, module.Shape, ShapeMixin)
    module.Graph = ensure_class_bases_begin_with(module, module.Graph, GraphMixin)
    module.Lowering = ensure_class_bases_begin_with(module, module.Lowering, LoweringMixin)

mtf.Tensor = mesh_tensorflow.ops.Tensor
mtf.Operation = mesh_tensorflow.ops.Operation
mtf.Shape = mesh_tensorflow.ops.Shape
mtf.Graph = mesh_tensorflow.ops.Graph
mtf.Lowering = mesh_tensorflow.ops.Lowering

globals()['Tensor'] = mtf.Tensor
globals()['Operation'] = mtf.Operation
globals()['Shape'] = mtf.Shape
globals()['Graph'] = mtf.Graph

TensorType = Union[mtf.Tensor, TensorMixin]
OperationType = Union[mtf.Operation, OperationMixin]
ShapeType = Union[mtf.Shape, ShapeMixin]
GraphType = Union[mtf.Graph, GraphMixin]
MeshType = Union[mtf.Mesh]
MeshImplType = Union[mtf.MeshImpl]
LoweringType = Union[mtf.Lowering, LoweringMixin]
SessionType = Union[tf.Session]

mesh_tensorflow.ops.convert_to_dimension = convert_to_dimension

# we extend Shape.to_string to support None sizes
setattr(mesh_tensorflow.ops.Shape, 'to_string', getattr(ShapeMixin, 'to_string'))

# ensure that str(mtftorch.float32).split('.')[-1] == 'float32'
# float32.__class__.__str__ = float32.__class__.__repr__
# float32.__class__.is_floating_point = float32.__class__.is_floating

for module in [_tf_dtypes]:
    module: ModuleType
    module.DType = ensure_class_bases_begin_with(module, module.DType, DTypeMixin)

assert tf.int32.is_signed


# support torch.Tensor._make_subclass
mesh_tensorflow.ops.Tensor._make_subclass = TensorMixin._make_subclass

def reset(graph=None):
    if graph is None:
        graph = get_graph()
    #graph.operations.clear()
    graph.reset()


State.GLOBAL = State()
State.current = cv.ContextVar('mtftorch.State.current', default=State.GLOBAL)
State.requires_grad = cv.ContextVar('mtftorch.State.requires_grad', default=True)


def set_state(graph=None, mesh=None, mesh_impl=None, lowering=None, session=None):
    new_state = State(graph=graph, mesh=mesh, mesh_impl=mesh_impl, lowering=lowering, session=session)
    token = State.current.set(new_state)
    def reset_state():
        nonlocal token
        if token is not None:
            State.current.reset(token)
            token = None
    return reset_state
