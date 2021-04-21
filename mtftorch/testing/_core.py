from typing import cast, List, Optional, Tuple, Union

# (bool, msg) tuple, where msg is None if and only if bool is True.
_compare_return_type = Tuple[bool, Optional[str]]