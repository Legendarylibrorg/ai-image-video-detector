"""CLI entry for image training.

``train_main`` defines the CLI and training loop. ``train_support`` holds loss, EMA, and
metric helpers. ``train_run_artifacts`` writes ``config.json`` / manifest / inference spec.
``train_post`` runs optional holdout ``test/`` eval and release export after the loop.
"""

from __future__ import annotations

from .train_main import build_train_argparser, run_image_training
from .train_support import BinaryClassificationLoss

__all__ = ["BinaryClassificationLoss", "main"]


def main() -> None:
    run_image_training(build_train_argparser().parse_args())


if __name__ == "__main__":
    main()
