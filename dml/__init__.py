__version__ = "0.0.0"

from .core import CompositeLoss, DistillationLink, build_links

__all__ = (
    "__version__",
    "DistillationLink",
    "CompositeLoss",
    "build_links",
)
