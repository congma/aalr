"""
Implementation for the underlying regressor/smoother based on cubic splines.
"""
from collections import namedtuple
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from .util import dist, mask_index


class SplineModel:
    """
    Model that serves as the basis for the regression.  This is a thin wrapper
    around SciPy's LSQUnivariateSpline with initialization of internal knots
    evenly distributed along the time domain, as specified by the ``nknots``
    parameter.

    The class is callable; by calling it with a single argument of time (scalar
    or array-like), the predicted value is returned.

    Attributes:
        t, y: Time frames and time-series values respectively

        w: Window values.

        knots: Location in time of the internal knots.

        self_pred: Predicted values at the time frames as specified by ``t``.

        _spr: Underlying LSQUnivariateSpline instance.
    """
    def __init__(self, t, y, nknots: int = 23, knots_override=None, w=None,
                 **spline_args):
        """Initalize a SplineModel instance given the input time-series, number
        of internal knots, and window-mask.

        Input parameters:
            t: Array-like 1d sequence of time-locations

            y: Array-like 1d sequence of the values in the time-series

                t and y must be of the same length

            nknots: Positive integer, number of internal knots used in spline
                    fitting (default: 23). Can be overridden; see below.

            knots_override: Alternative specification of knot positions
                            (optional). If specified (i.e. not None), must be a
                            sequence in the time-domain that specifies the knot
                            positions. The two end-points of the time interval
                            are always knots, so they should be omitted in the
                            knots_override array.

            w: Array-like 1d sequence of boolean mask, must be of the same
               length as t and y (optional). If None, it is filled with True.

            **spline_args: Additional keyword arguments to be passed to
                           scipy.interpolate.LSQUnivariateSpline

            Notice that the parameter ``k`` is fixed to be 3 (cannot be changed
            by spline_args) and ``ext`` re-defaults to "const" (can be changed
            by spline_args).
        """
        assert len(np.shape(t)) == 1
        assert len(np.shape(y)) == 1
        assert len(t) == len(y)
        # Fix the parameter value for "k" but not others, and re-default to
        # "const" for out of bound values unless explicitly overridden.
        kwargs = dict(ext="const")
        kwargs.update(spline_args)
        kwargs["k"] = 3
        self._kwargs_save = kwargs.copy()
        # Knot interval as specified by the number of knots, unless overridden.
        N = len(y) // nknots
        if knots_override is None:
            self.knots = t[max(N // 2, 1)::N]
        else:
            self.knots = np.asarray(knots_override, dtype=float)
        if w is None:
            winit = np.ones(len(t), dtype=float)
        else:
            winit = np.asarray(w, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.t = np.asarray(t, dtype=float)
        self.replace_mask(winit)

    def replace_mask(self, w):
        """Replace the mask in the model with w. Performs recalculation over
        all data.
        """
        assert np.shape(w) == np.shape(self.t)
        assert len(w) == len(self.t)
        self.w = np.asarray(w, dtype=float)
        self._spr = LSQUnivariateSpline(self.t, self.y, self.knots, self.w,
                                        **self._kwargs_save)
        self.self_pred = self(self.t)
        # Flag indicating whether w has just been replaced.
        self._stats_dirty = True

    def inlier_predicate(self, t, y, ub: float = 4, lb: float = -10):
        """Return boolean value or array (same shape as that of t and y) for
        the input values at (t, y) based on the current instance.

        Being a method it has full access to the state of the instance. This
        one is meant to be overridden if necessary. It implements a criterion
        that determines whether a pair of values is inlier.

        The one being implemented here as a default is based on an asymmetrical
        cut: At each iteration the data points that deviate from the preceding
        iteration's prediction by +4 or -10 times the MAD (median absolute
        deviation) -- as computed from the preceding iteration's inliers -- are
        excluded as outliers. This demonstrates what can be done in this
        formalism where the inlier-outlier criterion can be implemented in a
        fairly versatile manner.
        """
        if self._stats_dirty:
            # Calculate the vital statistics
            bm = np.asarray(self.w, dtype=bool)
            residuals = self.y[bm] - self.self_pred[bm]
            self.rmad = np.nanmedian(np.abs(residuals))
            self._stats_dirty = False
        preds = self(t)
        deviations = (y - preds) / self.rmad
        return (lb <= deviations) & (deviations <= ub)

    def __call__(self, t):
        return self._spr(t)

    def refine(self, target_d: int = 1, maxiter: int = 50):
        """Perform the refinement fit that iteratively exclude outliers based
        on the ``inlier_predicate`` method until convergence is achieved.

        Arguments:
            target_d : Tolerance of the Hamming distance for mask convergence
                       (integer, default 1)
            maxiter : Maximum number of iterations (integer, default 50)

        Return values:
            res : A namedtuple with the following records:
                  status: integer, with 0 indicating convergence
                  message: human-readable string indicating status of the final
                           result
                  niter: integer, number of iterations performed
                  dfinal: integer, final distance from the previous iteration
                          before iteration is terminated

        Side effects:
            The state of the instance itself is modified. After the return, the
            instance is in the "refined" state.
        """
        assert target_d >= 0
        assert maxiter > 0
        FitResult = namedtuple("FitResult",
                               ("status", "message", "niter", "dfinal"))
        niter = 0
        while niter <= maxiter:
            w_post_pred = self.inlier_predicate(self.t, self.y)
            dxor = dist(w_post_pred, self.w)
            if dxor <= target_d:
                result = FitResult(0, "Converged", niter, dxor)
                break
            self.replace_mask(w_post_pred)
            niter += 1
        else:
            result = FitResult(1, "Maximum iteration limit exceeded",
                               niter, dxor)
        return result

    def cure_knots(self):
        """Return a new instance with the excessive knots in the masked-out
        intervals removed.
        """
        kt_new = []
        midx = mask_index(self.w)
        for knot in self.knots:
            skip = False
            for mi_lo, mi_hi in midx:
                mloc_lo = self.t[mi_lo]
                mloc_hi = self.t[mi_hi - 1]
                if (mloc_lo < knot < mloc_hi):
                    skip = True
                    break
            if not skip:
                kt_new.append(knot)
        return SplineModel(self.t, self.y,
                           knots_override=kt_new, w=self.w,
                           **self._kwargs_save)
