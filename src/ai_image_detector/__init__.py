"""Installable library for the local research pipeline (training, inference, limits, checkpoints).

The package intentionally keeps a **small** import surface at the top level; most code lives in
submodules (for example ``ai_image_detector.data``, ``.train_main``). Operator workflows should
prefer ``./local.sh`` / ``scripts/do.sh`` so ``PYTHONPATH`` and the venv match pipeline drivers.
"""

__version__ = "0.1.0"

__all__ = ["model", "__version__"]
