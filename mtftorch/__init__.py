
import os
import sys
import platform
import textwrap
import ctypes
import warnings

if sys.version_info < (3,):
    raise Exception("Python 2 has reached end-of-life and is no longer supported by PyTorch.")

from ._six import string_classes as _string_classes

from mtftorch._C import *

# TODO: Clean up this bootstrapping issue more elegantly. (mtftorch.types depends on mtftorch.device, which isn't
#  defined yet.) For now, punt by force-reloading mtftorch.types, since mtftorch.device is now defined at this point.
from importlib import reload
import mtftorch.types
reload(mtftorch.types)


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
