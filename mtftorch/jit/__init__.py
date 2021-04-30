import mtftorch.nn

# These are imported so users can access them from the `torch.jit` module
from mtftorch._jit_internal import (
    # Final,
    # Future,
    # _overload,
    # _overload_method,
    # ignore,
    # _isinstance,
    is_scripting,
    # export,
    # unused,
)


_enabled = False

if _enabled:
    raise NotImplementedError
else:
    # TODO MAKE SURE THAT DISABLING WORKS
    class ScriptModule(mtftorch.nn.Module):  # type: ignore
        def __init__(self, arg=None):
            super().__init__()

    class RecursiveScriptModule(ScriptModule):  # type: ignore
        def __init__(self, arg=None):
            super().__init__()