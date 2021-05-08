
import os
import sys
import platform
import textwrap
import ctypes
import warnings

if sys.version_info < (3,):
    raise Exception("Python 2 has reached end-of-life and is no longer supported by PyTorch.")
# TODO(torch_deploy) figure out how to freeze version.py in fbcode build
if sys.executable == 'torch_deploy':
    __version__ = "torch-deploy-1.8"
else:
    from .version import __version__ as __version__

from ._six import string_classes as _string_classes

import mesh_tensorflow as mtf
import mesh_tensorflow.ops
import tensorflow as tf2
tf = tf2.compat.v1

def startup():
    if not globals().get('tensorflow_startup__started', False):
        globals()['tensorflow_startup__started'] = True
        tf.disable_v2_behavior()
        tf.enable_resource_variables()
        # tf2.get_logger().setLevel('DEBUG')
        tf2.get_logger().setLevel('WARN')

startup()

from mesh_tensorflow.ops import Tensor
from mesh_tensorflow.ops import Operation
from mesh_tensorflow.ops import Shape
from mesh_tensorflow.ops import Shape as Size
from mesh_tensorflow.ops import VariableDType as dtype

from mtftorch._C import *
from mtftorch.autograd import *
from mtftorch.autograd.grad_mode import *

# TODO: Clean up this bootstrapping issue more elegantly. (mtftorch.types depends on mtftorch.device, which isn't
#  defined yet.) For now, punt by force-reloading mtftorch.types, since mtftorch.device is now defined at this point.
from importlib import reload
import mtftorch.types
reload(mtftorch.types)


from .random import set_rng_state, get_rng_state, manual_seed, initial_seed, seed

################################################################################
# Define basic utilities
################################################################################


def typename(o):
    if isinstance(o, mtftorch.Tensor):
        return o.type()

    module = ''
    class_name = ''
    if hasattr(o, '__module__') and o.__module__ != 'builtins' \
            and o.__module__ != '__builtin__' and o.__module__ is not None:
        module = o.__module__ + '.'

    if hasattr(o, '__qualname__'):
        class_name = o.__qualname__
    elif hasattr(o, '__name__'):
        class_name = o.__name__
    else:
        class_name = o.__class__.__name__

    return module + class_name
