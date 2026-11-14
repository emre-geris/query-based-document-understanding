"""The demo: route a question about an uploaded page, answer it and report every cost.

`DemoService` (`qdu.demo.service`) is the part the user interface calls; the interface itself
only displays what it returns. `qdu demo ask` runs it once from the command line.
"""

from qdu.demo.examples import (
    Example,
    StoredLayout,
    StoredOcr,
    load_examples,
    load_recorded,
    showcase,
)
from qdu.demo.service import (
    DemoAnswer,
    DemoService,
    PathRun,
    SharedOcr,
    demo_fake_model,
    describe,
)

__all__ = [
    "DemoAnswer",
    "DemoService",
    "Example",
    "PathRun",
    "SharedOcr",
    "StoredLayout",
    "StoredOcr",
    "demo_fake_model",
    "describe",
    "load_examples",
    "load_recorded",
    "showcase",
]
