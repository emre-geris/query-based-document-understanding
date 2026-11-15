"""The demo: route a question about an uploaded page, answer it and report every cost.

`DemoService` (`qdu.demo.service`) routes, answers and costs; `DemoSession`
(`qdu.demo.session`) is one user's page and answers; `qdu.demo.web` serves the design draft
as a web page that draws them (``qdu demo serve``); `qdu.demo.render` holds the shared wording
and the presentation narrative. ``qdu demo ask`` runs the service once from the command line.
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
