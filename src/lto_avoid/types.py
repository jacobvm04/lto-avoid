"""Shape-checked array type aliases using jaxtyping."""

from jaxtyping import Bool, Float
import numpy as np

GridArray = Bool[np.ndarray, "height width"]
SDFArray = Float[np.ndarray, "height width"]
Point2D = Float[np.ndarray, "2"]
Points2D = Float[np.ndarray, "n 2"]
Trajectory = Float[np.ndarray, "n 2"]
Coords = Float[np.ndarray, "... 2"]
