import numpy as np
from .spline_model import SplineModel


def knot_shift_aggregate(model,
                         duplicates: int = 3, proximity_factor: float = 0.001,
                         **refine_args):
    """Create an aggregate model on the input model by duplicating knots with
    shift. In total, (2 * duplicates) copies are made (on the left and right
    sides).
    """
    d = int(duplicates)
    assert d > 0
    p = float(proximity_factor)
    assert 0.0 < p < 1.0
    q = (1.0 - p) / d
    scale_left = (model.knots[0] - model.t[0]) * q
    scale_right = (model.t[-1] - model.knots[-1]) * q
    shifted_models = []
    for i in range(-d, 0):
        this_model = SplineModel(model.t, model.y,
                                 knots_override=(model.knots +
                                                 i * scale_left),
                                 **model._kwargs_save)
        this_model.refine(**refine_args)
        shifted_models.append(this_model)
    for i in range(1, d + 1):
        this_model = SplineModel(model.t, model.y,
                                 knots_override=(model.knots +
                                                 i * scale_right),
                                 **model._kwargs_save)
        this_model.refine(**refine_args)
        shifted_models.append(this_model)
    w = model.w.copy()
    for m in shifted_models:
        w *= m.w
    model_new = SplineModel(model.t, model.y, w=w, **model._kwargs_save)
    model_agg = model_new.cure_knots()
    model_agg.refine(**refine_args)
    return model_agg
