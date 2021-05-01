
from torch import Tensor, Generator, strided, memory_format, contiguous_format, strided
from typing import Type, List, Tuple, Optional, Union, Any, ContextManager, Callable, overload, Iterator, NamedTuple, Sequence, TypeVar
from typing_extensions import Literal
from torch._six import inf

from torch.types import _int, _float, _bool, Number, _dtype, _device, _qscheme, _size, _layout

import builtins

import collections

# REDUNDANT!
namedtuple_primal_tangent = NamedTuple("namedtuple_primal_tangent", [("primal", Tensor), ("tangent", Tensor)])
namedtuple_values_indices = NamedTuple("namedtuple_values_indices", [("values", Tensor), ("indices", Tensor)])
namedtuple_eigenvalues_eigenvectors = NamedTuple("namedtuple_eigenvalues_eigenvectors", [("eigenvalues", Tensor), ("eigenvectors", Tensor)])
namedtuple_mantissa_exponent = NamedTuple("namedtuple_mantissa_exponent", [("mantissa", Tensor), ("exponent", Tensor)])
namedtuple_a_tau = NamedTuple("namedtuple_a_tau", [("a", Tensor), ("tau", Tensor)])
namedtuple_solution_QR = NamedTuple("namedtuple_solution_QR", [("solution", Tensor), ("QR", Tensor)])
namedtuple_Q_R = NamedTuple("namedtuple_Q_R", [("Q", Tensor), ("R", Tensor)])
namedtuple_sign_logabsdet = NamedTuple("namedtuple_sign_logabsdet", [("sign", Tensor), ("logabsdet", Tensor)])
namedtuple_solution_LU = NamedTuple("namedtuple_solution_LU", [("solution", Tensor), ("LU", Tensor)])
namedtuple_U_S_V = NamedTuple("namedtuple_U_S_V", [("U", Tensor), ("S", Tensor), ("V", Tensor)])
namedtuple_solution_cloned_coefficient = NamedTuple("namedtuple_solution_cloned_coefficient", [("solution", Tensor), ("cloned_coefficient", Tensor)])

min: Type[namedtuple_values_indices] = collections.namedtuple("min", ["values", "indices"])
max: Type[namedtuple_values_indices] = collections.namedtuple("max", ["values", "indices"])