Synopsis
========
`aalr` -- anomaly-aware local regression

A Python module to perform local regression on noisy data with the awareness
that some of the input data points may contain anomalies.


Illustration
============

The regressor (orange line) converges iteratively to the smooth component in
the input signal (blue). Pink-shaded window or mask in the background indicates
anomalous input found during the current iteration.

![animation](_doc/animation.apng "Animation illustrating the iteration of the AALR method: by iteratively improving the mask on input data, the regressor takes into account possible presence of anomalous input.")


Usage
=====

The `SplineModel` class largely follows the interface of `scipy`'s
[`LSQUnivariateSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQUnivariateSpline.html), which is also the underlying local
regression method used in the default case. The `refine()` method, initiates
the iterative improvement.
