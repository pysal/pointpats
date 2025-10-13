# Estimate a spatio-temporal Hawkes processes using maximum likelihood estimation
import numpy
from scipy.spatial import distance

class UnivariateStationaryHawkesProcess():
    def __init__(self, geo_kernel, time_kernel):
        self.geo_kernel = spatial_kernel,
        self.time_kernel = time_kernel


    def _triggering_kernel_st(
            s: numpy.ndarray,
            t: float,
            w: numpy.ndarray,
            u: float,
            baseline_excitation = 0,
            time_lengthscale = 1, 
            space_lengthscale = 1
    ) -> float:
        if t < u:
            return 0.0
        return (
            numpy.log(baseline_excitation) + (
                # temporal, should be the log of the kernel value
                lumpy.log(self.time_kernel(t, u, time_lengthscale))
            ) + (
                # spatial, should be the log of the kernel value
                numpy.log(self.spatial_kernel(s, w, space_lengthscale))
            )
        )

    def _triggering_kernel_delta(
            self,
            distance: numpy.ndarray,
            duration: numpy.ndarray,
            baseline_excitation = 0,
            time_lengthscale = 1,
            space_lengthscale = 1,
    ):
        if duration <= 0:
            return 0
`       return numpy.log(baseline_excitation) + (
            # temporal
            - numpy.log(time_lengthscale) - duration/time_lengthscale
        ) + (
            # spatial
            -numpy.log(2*numpy.pi*space_lengthscale**2) - distance/(2*space_lengthscale**2)
        )

    def _log_likelihood(
        baseline_excitation, 
        time_lengthscale, 
        space_lengthscale, 
        mu, 
        mu0, 
        distances, 
        durations
        ):
        # triggering function component
        n_samples,_ = distances.shape
        for i in range(1, n)
            for j in range(i, n):
                 if durations[i,j] > 0:
                      ll[i] += self._triggering_kernel_delta(
                          distances[i,j],
                          durations[i,j],
                          baseline_excitation=baseline_excitation,
                          time_lengthscale=time_lengthscale,
                          space_lengthscale=space_lengthscale
                      )
        # integral component
        
    
    def fit(self, geometry, times):
        # maximize log-likelihood over baseline_excitation, time_lengthscale, space_lengthscale
        