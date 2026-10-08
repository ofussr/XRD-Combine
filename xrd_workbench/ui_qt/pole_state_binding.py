"""Small adapters between accepted pole state and local controller values."""
import math


def state_property(name):
    return property(lambda page: getattr(page.state, name),
                    lambda page, value: setattr(page.state, name, value))


def bound_value(value_class, state, name, *, lower=None, upper=None):
    value = value_class(value=getattr(state, name))

    def accept():
        candidate = value.get()
        if lower is not None:
            try:
                candidate = float(candidate)
            except (TypeError, ValueError):
                return
            if not math.isfinite(candidate) or not lower <= candidate <= upper:
                return
        setattr(state, name, candidate)
        layer = getattr(state, 'overlay_layer', None)
        field = {'overlay_colour': 'colour', 'overlay_opacity': 'opacity_percent',
                 'overlay_size': 'size_percent', 'joint_rotation': 'coupled_to_primary'}.get(name)
        if layer is not None and field is not None:
            setattr(layer, field, candidate)

    value.trace_add('write', accept)
    return value
