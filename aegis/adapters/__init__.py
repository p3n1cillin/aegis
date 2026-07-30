"""Game adapters.

Importing this package registers every built-in adapter into
``aegis.core.registry.ADAPTERS``. Look one up by key::

    from aegis.core.registry import ADAPTERS
    adapter = ADAPTERS.get_or_raise("cs2")()
    session = adapter.parse("match.aegis")

Adding a game: subclass ``GameAdapter``, decorate with ``@ADAPTERS.register``,
declare honest capabilities, and run ``validate_session`` in your tests. See
``aegis/adapters/base.py`` for the full contract and ``generic_fps.py`` for
the lowest-effort path.
"""

from .base import AdapterValidationError, GameAdapter, validate_session  # noqa: F401
from .cs2 import CS2Adapter  # noqa: F401
from .generic_fps import GenericFPSAdapter, example_payload  # noqa: F401

__all__ = [
    "AdapterValidationError",
    "CS2Adapter",
    "GameAdapter",
    "GenericFPSAdapter",
    "example_payload",
    "validate_session",
]
