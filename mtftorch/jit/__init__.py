import mtftorch.nn

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