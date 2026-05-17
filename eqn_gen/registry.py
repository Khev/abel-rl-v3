from .quadratic import Quadratic
from .cubic import Cubic
from .quartic import Quartic
from .exponential import Exponential

REGISTRY = {
    "quadratic":   Quadratic,
    "cubic":       Cubic,
    "quartic":     Quartic,
    "exponential": Exponential,
}
