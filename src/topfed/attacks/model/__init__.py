"""Model-side attack hooks: loss augmenters and update manipulators."""

from topfed.attacks.model.constrain import ConstrainAndScaleAugmenter
from topfed.attacks.model.scale import ModelReplacementManipulator

__all__ = ["ConstrainAndScaleAugmenter", "ModelReplacementManipulator"]
