# Estimate a spatio-temporal Hawkes processes using maximum likelihood estimation
import numpy
from scipy.spatial import distance
from scipy import optimize

# maybe the better way to do this would be to implement
# the stan code, then allow the user to select various kernels?
# otherwise we'll have a lot of very long classes. 

class UnivariateStationarySeparableSpatialHawkes():
    def __init__(self, space_kernel, time_kernel):
        self.space_kernel = spatial_kernel,
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
        if t <= u:
            return 0.0
        return (
            numpy.log(baseline_excitation) + (
                # temporal, should be the log of the kernel value
                lumpy.log(self.time_kernel(t, u, time_lengthscale))
            ) + (
                # spatial, should be the log of the kernel value
                numpy.log(self.space_kernel(s, w, space_lengthscale))
            )
        )

    def _triggering_kernel_delta(
            self,
            distance: numpy.ndarray,
            duration: numpy.ndarray,
            baseline_excitation = 0,
            time_lengthscale = 1,
            space_lengthscale = 1,
    ) -> float:
        if duration <= 0:
            return 0.0
        return numpy.log(baseline_excitation) + (
            # temporal
            numpy.log(self.time_kernel(duration, time_lengthscale))
        ) + (
            # spatial
            numpy.log(self.space_kernel(distance, space_lengthscale))
        )

    def _log_likelihood(
        baseline_excitation, 
        time_lengthscale, 
        space_lengthscale, 
        intensity, 
        baseline_intensity, 
        distances, 
        durations
        ) -> float:
        # triggering function component
        n_samples,_ = distances.shape
        excitation = numpy.zeros(n_samples)
        for i in range(1, n)
            for j in range(i, n):
                 if durations[i,j] > 0:
                      excitation[i] += self._triggering_kernel_delta(
                          distances[i,j],
                          durations[i,j],
                          baseline_excitation=baseline_excitation,
                          time_lengthscale=time_lengthscale,
                          space_lengthscale=space_lengthscale
                      )
        # integral component
        baseline_all = baseline_intensity * self.space_window * self.time_window + 
        baseline_time_nospace = baseline_excitation * self.time_kernel(self.waits_, time_lengthscale).sum()
        return -(
            excitation.sum() + 
            self.integral_value*intensity + 
            baseline_all + 
            baseline_time_nospace - 
            self.integral_value
        )
    
    def fit(self, geometry: geopandas.GeoSeries, times: pandas.Series[pd.Timestamp]) -> UnivariateStationaryHawkesProcess:
        coords = geometry.get_coordinates()
        durations = numpy.subtract.outer(times, times)
        distances = distance.cdist(coords, coords)
        # we should probably estimate on the unit square+hour and then re-transform?
        self.time_window_ = durations.max() - durations.min()
        
        self.space_window_ = distances.max() - distances.min()
        self.waits_ = self.time_window_ - times # now time is denominated as float timestep since start at 0
        # maximize log-likelihood over mu, mu0, baseline_excitation, time_lengthscale, space_lengthscale
        self._calcluate_integral() # calculate the integral value over space and time

        def score(vars):
            return - self._log_likelihood(
                vars[0], # baseline_excitation
                vars[1], # time_lengthscale
                vars[2], # space_lengthscale
                vars[3], # intensity
                vars[4], # baseline_intensity
                distances,
                durations
            )

        result = optimize.minimize(
            score,
            x0 = numpy.array([0.1, 1.0, 1.0, 0.1, 0.1]), # initial guess
            args = (
                intensity,
                baseline_intensity,
                distances,
                durations
            ),
            bounds = (
                (1e-5, None), # baseline excitation
                (1e-5, None), # time lengthscale
                (1e-5, None), # space lengthscale
                (1e-5, None), # intensity
                (1e-5, None)  # baseline intensity
            ),
            method = 'L-BFGS-B'
        )
        self.baseline_excitation_ = result.x[0]
        self.time_lengthscale_ = result.x[1]
        self.space_lengthscale_ = result.x[2]
        self.intensity_ = result.x[3]
        self.baseline_intensity_ = result.x[4]
        return self

    def _calculate_integral(self):
        # calculate the integral of the triggering function over space and time 
        # I think if proper kernels are used, this integral is n?
        self.integral_value = len(self.waits_) # placeholder, should be the integral value only if proper kernels are used

class MultivariateStationarySeparableSpatialHawkes():
    def __init__(self, space_kernel, time_kernel):
        self.space_kernel = spatial_kernel,
        self.time_kernel = time_kernel

    def fit(self, geometry: geopandas.GeoSeries, times: pandas.Series[pd.Timestamp], types: pandas.Series[int]) -> MultivariateStationaryHawkesProcess:
        # similar to univariate but now we have a matrix of baseline excitations and intensities. 
        # Basically, we do the same as in the univariate, but the triggering loop has to be done
        # across all pairs of event types. Think of the univariate case above as a within-type triggering equation,
        # while doing this from type A (n_samples_A) to type B (n_samples_B) would be 
        # a between-type triggering equation, iterating for i in range(n_samples_A) and j in range(n_samples_B), calculating
        # using the cross-type matrix of distances/durations
        pass