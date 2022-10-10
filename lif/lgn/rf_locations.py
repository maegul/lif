"""Manage the spatial locations of Receptive Fields according to data constraints

Main data constraint is of course `Jin et al 2013`.

This module is more or less like a notebook ... data is hard coded here and functions
are commented out to be run manually whenever needed.

Not great design but I was just trying to be quick and convenient here.
"""

# > Imports
# +
from typing import Union, List, Tuple, Iterable, Literal
from itertools import combinations_with_replacement
from dataclasses import dataclass
import warnings
from textwrap import dedent
from matplotlib.patches import Arc

import numpy as np
import pandas as pd

from scipy import spatial
pdist = spatial.distance.pdist
import scipy.optimize as opt
import scipy.integrate as integrate
import scipy.special as special

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as spl
# -
# +
from ..utils.units.units import val_gen, scalar, ArcLength
from ..receptive_field.filters import filter_functions as ff

from ..utils import (
    data_objects as do,
    settings,
    exceptions as exc)
# -
# > Core Distributions (Jin et al)

# +
# Initial curve
# Derived directly from the data in the paper
# Interpretted as being a discrete cumulative distribution function,
# turned here into a continuous function
def jin_cdf(x):
    return 3.4 * np.exp(x * -2.5)

# Normalising curve to max 1 for use as a probability curve
# Necessary, as the original curve is from discrete data,
# but to be used it must be continuous and have a max of 1
def jin_cdf_re_norm(x):
    jcdf = jin_cdf(x)
    return jcdf / jcdf[0]

# Derivation of a PDF from the CDF data, by finding the differences
def jin_re_norm_pdf_est(x):

    jcdfn = jin_cdf_re_norm(x)

    return jcdfn[:-1] - jcdfn[1:]

# Ensuring that the sum of the PDF is 1 (there are some minor rounding issues)
def jin_pdf_adj(x):

    jin_pdf_est = jin_re_norm_pdf_est(x)

    return jin_pdf_est  + (1 - np.sum(jin_pdf_est)) / jin_pdf_est.size
# -

# > Pair wise distances

# >> Statistical Functions

# >>> Exponential (?)

# Take the exponential curves from Jin et al as distributions and fit to these

# def exponential_pdf(lambda: float): ...
# def exponential_cdf(lambda, upper, lower): ...

# objective function for residuals between bivariate gaussian and exponential
# (rather than raw histogram data)
# ... as we're sticking with bivariate gaussian because it is easy to manipulate the ratio of
# ... this is much like what was done for FENS2018, but this time we have an analytical
# ... equation for the pairwise distances between locations drawn from a bivariate dist.


# >>> Bivariate Gaussian

# +
def bivariate_gauss_radius_pdf(r: val_gen, gauss_params: do.BivariateGaussParams, ) -> val_gen:
    """Analytic PDF of radius magnitude of points drawn from a bivariate Gaussian.

    Args:
        r:
            Radius for which probability is sought.
            Number is "unitless", as the Jin et al data is in dynamic
            units of "Largest RF Diameter".
        gauss_params: Std Dev for bivariate gaussian

    Notes:
        Derived from a Bivariate Gaussian in the complex plane.
        See
            > Schreier, P. J., & Scharf, L. L. (2010).
            > Statistical signal processing of complex-valued data: The theory of improper and noncircular signals.
            > Cambridge University Press.
    """

    # Rayleigh x Bessel
    var_x, var_y = gauss_params.sigma_x**2, gauss_params.sigma_y**2
    R_xx = var_x + var_y
    R_xx_comp = var_x - var_y
    # 0 if sigmas are the same, approaches +/- 1 as one is greater than the other
    # c/special.iv/Bessel below is 1 if rho is 0
    rho = R_xx_comp / R_xx

    exponent = (r**2) / (R_xx*(1-rho**2))

    a = (2*r) / (R_xx*((1-rho**2)**0.5))
    b = np.exp(-1 * exponent)
    # b = np.exp((-r**2) / (R_xx*(1-rho**2)) )
    # modified Bessel function of the first kind of order 0
    # constant 1 if sigmas are the same, else drives the change in the PDF
    c = special.iv(0,
        rho * exponent
        # ((r**2)*rho) / (R_xx*(1-rho**2))
        )

    pdf = (a*b
        # replace infs with large numbers
        # (should only occur where b = 0, from manual inspection that is)
        * np.nan_to_num(c)
        )
    # correct for 0*np.inf=nan ... convert from nan to 0
    # pdf[(b==0) & np.isinf(c)] = 0

    return pdf
# -

# +
def bivariate_gauss_radius_probability(
        lower: float, upper: float,
        gauss_params: do.BivariateGaussParams) -> float:
    """**Probability** of bivariate gaussian vector radius between lower and upper bounds

    Integrates over the `bivariate_gauss_radius_pdf` function using `scipy.integrate.quad`.

    Notes:
        As a `PDF` a probability *Density* function for a continuous variable, values of
        the `PDF` for any specific input radius do not correspond to an actual probability.
        The integral of a `PDF` provides probability.  Thus, the unbounded integral of a PDF
        is `1`, and a bounded integral provides the probability of a value within those
        bounds.
        An analytical histogram can be derived using this.
    """

    probability = integrate.quad(
        bivariate_gauss_radius_pdf,
        lower, upper,
        (gauss_params,))

    return probability[0]
# -
# +
def bivariate_guass_radius_probability_array(
        lower: Union[np.ndarray, List], upper: Union[np.ndarray, List],
        gauss_params: do.BivariateGaussParams
        ) -> np.ndarray:
    """Like `bivariate_gauss_radius_probability` but for multiple bounds.

    `lower` and `upper` must be structured as `[l1, l2,...], [u1, u2,...]`
    such that the bounds will be, in order: `l1-u1, l2-u2, ...`.

    An array of probabilities/integrals will be return for each lower-upper pair.
    """
    if not (len(lower) == len(upper)):
        raise ValueError(f'lower and upper must have same size (lens: {len(lower),len(upper)})')

    n_bins = len(lower)
    probabilities = np.empty_like(lower, dtype=np.floating)

    for i in range(n_bins):
        probability = bivariate_gauss_radius_probability(
                        lower[i], upper[i], gauss_params=gauss_params
                        )
        probabilities[i] = probability

    return probabilities
# -

# >> jin data

# data from tracey mctraceface
# +
@dataclass
class _JinData:
    """Basic container of the data from Jin et al (2011)

    All data captured using Tracey McTraceface

    The on and off arrays have two columns, first, the upper bound of the histogram bins,
    second, the relative probability of values within that bin.

    dist_vals_all contains all the data in shared columns and all_dist_type lists which
    dataset the values come from ('on' or 'off') ... probabily not useful
    """
    dist_vals_on: np.ndarray
    dist_vals_off: np.ndarray
    dist_vals_on_raw: np.ndarray
    "from raw data from Alonso himself"
    dist_vals_off_raw: np.ndarray
    "from raw data from Alonso himself"

    def distance_vals_insert_lower(self, type:str , value: float = 0) -> np.ndarray:
        """Just distances with provided value (default 0r at the beginning

        type must match the suffix of one of the attributes ... "on" / "off" / "on/off_raw"
        Distances presumed to be first column of data
        """

        return np.r_[value, self.__getattribute__(f'dist_vals_{type}')[:,0]]

# -
# +
def _make_jin_data_object():
    "Basic preparation of the jin data otherwise hardcoded in this module"

    dist_vals_on = np.array(
        # dist      ,  normalised freq
        #  V               V
        [[2.00402154, 0.06479475],
        [1.80057392, 0.07516205],
        [1.59971301, 0.12699778],
        [1.40243894, 0.16846659],
        [1.20157816, 0.24362845],
        [0.99713054, 0.27732179],
        [0.79985647, 0.49244064],
        [0.59899569, 0.75939523],
        [0.40172169, 1.        ]])
    dist_vals_off = np.array(
        [[2.00143457, 0.05183592],
        [1.80057392, 0.05183592],
        [1.59971301, 0.08034551],
        [1.40243894, 0.22807769],
        [1.20157816, 0.28250543],
        [1.00071738, 0.44060473],
        [0.79985647, 0.70496758],
        [0.60258254, 0.63498918],
        [0.39813485, 1.        ]])
    # normalise "norm freqs" to probabilities by dividing by sum
    dist_vals_on[:,1] = dist_vals_on[:,1] / dist_vals_on[:,1].sum()
    dist_vals_off[:,1] = dist_vals_off[:,1] / dist_vals_off[:,1].sum()
    # sort in ascending order of distance
    dist_vals_on = dist_vals_on[dist_vals_on[:,0].argsort()]
    dist_vals_off = dist_vals_off[dist_vals_off[:,0].argsort()]

    # snap distances to multiples of `0.2` [0.4, 2]
    dist_vals_on[:,0] = dist_vals_on[:,0].round(2)
    dist_vals_off[:,0] = dist_vals_off[:,0].round(2)

    # >> Raw Jin data from Alonso!
    # +
    raw_dist_vals = np.array([0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8])
    raw_off_count = np.array([320,192,204,126,80,61,18,6,10,3,1,4,2])
    raw_off_norm_freq = np.array([1,0.6,0.6375,0.39375,0.25,0.190625,0.05625,
                                    0.01875,0.03125,0.009375,0.003125,0.0125,0.00625])

    raw_on_counts = np.array([193,156,96,56,51,26,19,12,7,2,2,2,0])
    raw_on_norm_freq = np.array([1,0.808290155440414,0.49740932642487,0.290155440414508,
                                0.264248704663212,0.134715025906736,0.0984455958549223,
                                0.0621761658031088,0.0362694300518135,0.0103626943005181,
                                0.0103626943005181,0.0103626943005181,0])


    # make columns and normalise freqs to probabilities
    dist_vals_off_raw = np.vstack(
        (raw_dist_vals, raw_off_norm_freq/raw_off_norm_freq.sum())).T
    dist_vals_on_raw = np.vstack(
        (raw_dist_vals, raw_on_norm_freq/raw_on_norm_freq.sum())).T
    # -

    jin_data = _JinData(
        dist_vals_on=dist_vals_on,
        dist_vals_off=dist_vals_off,
        dist_vals_off_raw=dist_vals_off_raw,
        dist_vals_on_raw=dist_vals_on_raw,
        )
    return jin_data
# -
# +
# > set as MODULE VARIABLE
jin_data = _make_jin_data_object()
# -

# +
def plot_jin_data_with_raw_data(jin_data: _JinData):
    distance_bins = jin_data.dist_vals_off_raw[:,0]
    fig = (go.Figure()
        .add_scatter(  # type: ignore
            x=distance_bins, y=jin_data.dist_vals_off_raw[:,1],
            name='raw alonso',
            line=go.scatter.Line(color='blue'),  # type: ignore
            legendgroup='off', legendgrouptitle_text='OFF')
        .add_scatter(
            x=distance_bins, y=jin_data.dist_vals_off[:,1],
            name='paper',
            line=go.scatter.Line(color='blue', dash='1 3'),  # type: ignore
            legendgroup='off')
        .add_scatter(
            x=distance_bins, y=jin_data.dist_vals_on_raw[:,1],
            name='raw alonso',
            line=go.scatter.Line(color='red'),  # type: ignore
            legendgroup='on', legendgrouptitle_text='ON')
        .add_scatter(
            x=distance_bins, y=jin_data.dist_vals_on[:,1],
            name='paper',
            line=go.scatter.Line(color='red', dash='1 3'),  # type: ignore
            legendgroup='on')
        .update_layout(
            title='Data from paper and raw data from personal email from Alonso',
            xaxis_title='Distance (largest rf diameter)',
            yaxis_title='Probability (per bin)',
            xaxis_tick0='0', xaxis_dtick=0.2
            )
        )

    return fig
# -
# +
# plot_jin_data_with_raw_data(jin_data).show()
# -


# >> Objective Function and Fitting to Data

# +
def bivariate_gauss_pairwise_distance_probability_residuals(
        x: Union[np.ndarray, List], ratio: float,
        data_bins: np.ndarray, data_prob: np.ndarray
        ) -> np.ndarray:
    """Residuals provided data and the distribution defined by given `sigma_x` value

    Args:
        x:
            * Iterable with only the x axis std dev of the bivariate gaussian
            for the positions of RFs.
            * From this value, `(2^0.5)*sigma` is used to represent the distribution
            of cartesian distances given the provided location `sigma` value.  See notes.
        ratio: the ratio between sigma_y and sigma_x (`y/x = ratio`, `y = x*ratio`)
        data_bins:
            * boundaries between which probabilities will be calculated
            * Must be sorted
        data_prob:
            * Probability data the gaussian is to be fit to.
            * Must have size of `data_bins.size - 1` as will have the probabilities for each
            bin, not each bound of each bin.
            Must also be aligned with the order of `data_bins`.

    Returns:
        residuals: `analytical probabilities - prob_data[:,1]`

    Notes:
        * Variances are linear for gaussian variables under addition/subtraction.
        * Thus, the variance of the gaussian distribution of distances/differences
        along an axis (`x`/`y`) for an initial distribution with **variance** `s^2`
        will be `2s^s`.
        * The standard deviation will then be `(2^0.5)s`.
    """
    gauss_params = do.BivariateGaussParams(sigma_x=x[0], ratio=ratio)

    lower_bounds, upper_bounds = data_bins[0:-1], data_bins[1:]
    # probabilities of distances from prob_data given provided sigma vals
    probabilities = bivariate_guass_radius_probability_array(
                        lower_bounds, upper_bounds,
                        gauss_params=gauss_params.axial_difference_sigma_vals()
                    )

    residuals = probabilities - data_prob  # type: ignore

    return residuals
# -

# >>> Optimisation and Fitting
# +
def mk_optimasation_sigma_x_for_ratio(
        ratio: float,
        data_bins:np.ndarray, data_prob:np.ndarray
        ) -> opt.OptimizeResult:
    """Fit pairwise distance probability to provided data but with provided sigma ratio

    Returns the bare optimisation result
    """

    res: opt.OptimizeResult = opt.least_squares(
        bivariate_gauss_pairwise_distance_probability_residuals, x0=[1],
        bounds=([0], [np.inf]),
        args = (ratio, data_bins, data_prob))

    return res
# -

# +
def mk_sigma_x_ratio_lookup(
        ratios: np.ndarray,
        data_bins: np.ndarray, data_prob: np.ndarray
        ) -> do.RatioSigmaXOptLookUpVals:
    """For all ratios and the data provided, fit a bivaraite gaussian

    Returns an object with 1D numpy arrays (`RatioSigmaXOptLookUpVals` object)
    """

    # sigma values and cost
    sigma_x_values_with_cost = np.empty(shape=(ratios.size, 2), dtype=np.float64)

    for i,r in enumerate(ratios):
        sigma_x: float = np.NaN
        cost: float = np.NaN
        if i % (len(ratios)//20) == 0:
            print(f'{i/len(ratios):<10.2%}', end='\r')

        try:
            res = mk_optimasation_sigma_x_for_ratio(r, data_bins, data_prob)
            sigma_x, cost = res.x[0], res.cost
        except exc.LGNError:
            print(f'Failed to optimise for ratio {r}')
        finally:
            sigma_x_values_with_cost[i] = sigma_x, cost

    lookup_vals = do.RatioSigmaXOptLookUpVals(
        ratios=ratios,
        sigma_x=sigma_x_values_with_cost[:, 0],
        errors=sigma_x_values_with_cost[:, 1]
        )

    return lookup_vals
# -

# +
def plot_sigma_x_ratio_lookup(lookup_vals: do.RatioSigmaXOptLookUpVals):
    """For ratio values, what sigma_x value was optimal and with what error
    """
    fig = (
        px
        .line(
            lookup_vals.to_df(),
            x='ratios', y=['sigma_x', 'error'],
            facet_row='variable')
        .update_layout(
            title='Optimal sigma_x for various sigma ratios',
            showlegend=False,
            xaxis_dtick=2, xaxis2_dtick=2,
            yaxis_title='Error (half sum of squares)',
            yaxis2_title='Sigma_x',
            yaxis2_matches=None)
        )
    fig.layout.annotations = None
    return fig
# -

# >>> Characterise Error function
# +
def characterise_pairwise_distance_distribution_residuals(
        data_bins: np.ndarray, data_prob: np.ndarray,
        ratios: np.ndarray=np.arange(1,10),
        sigma_x_vals: np.ndarray=np.linspace(0.01,1,200)
        ) -> pd.DataFrame:
    """Produce error values for array of ratio and sigma_x values to visualise object function

    Examples:
        >>> pw_errors = characterise_pairwise_distance_distribution_residuals(
        >>>     data_bins=jin_data.distance_vals_zero_insert('off'),
        >>>     data_prob=jin_data.dist_vals_off[:,1],
        >>>     )
    """

    n_residuals = ratios.size * sigma_x_vals.size
    if n_residuals > 1000:
        warnings.warn(dedent(f'''
            Number of values to be calculated ({n_residuals})
            is estimated to take ~{n_residuals*0.007:.2f} seconds (~7ms per calculation).
            '''))


    error_data = [
        {
            'error': np.sum(
                bivariate_gauss_pairwise_distance_probability_residuals(
                    [sigma_x], ratio=ratio, data_bins=data_bins, data_prob=data_prob)
                **2),
            'sigma_x': sigma_x,
            'ratio': ratio
        }
        for ratio in ratios
        for sigma_x in sigma_x_vals
    ]
    error_df = pd.DataFrame(error_data)

    return error_df
# -

# +
def plot_characterisation_pairwise_distance_residuals(pw_errors: pd.DataFrame):
    """Plot characterisation of objective function using different color for each ratio
    """
    fig = (
        px.line(
            pw_errors,
            y='error', x='sigma_x',
            color='ratio'
            # facet_col = 'ratio', facet_col_wrap=3
            )
        .update_layout(yaxis_rangemode='tozero')
        )

    return fig
# -

# +
def plot_profile_rf_locations_pairwise_distances(
        gauss_params: do.BivariateGaussParams,
        data_bins: np.ndarray, data_prob: np.ndarray,
        n_simulated_locs: int = 5000, simulated_pw_dists_n_bins: int = 100):
    """Visualise how well a bivariate gaussian has pairwise distances that fit to data

    data arguments are intended to be the data to which a distribution will be fit

    The data_bins are presumed to start at zero.

    Examples:
        >>> fig = plot_profile_rf_locations_pairwise_distances(
        >>>     gauss_params=do.BivariateGaussParams(sigma_x=0.178,ratio=2.92),
        >>>     data_bins=jin_data.distance_vals_zero_insert('on'),
        >>>     data_prob=jin_data.dist_vals_on[:, 1])
        >>> fig.show()

    """

    # Simulate rf locations and their pairwise distances
    # size = 5000
    x_locs = np.random.normal(size=n_simulated_locs, scale=gauss_params.sigma_x)
    y_locs = np.random.normal(size=n_simulated_locs, scale=gauss_params.sigma_y)
    emp_pwdists = pdist(X=np.vstack((x_locs, y_locs)).T, metric='euclidean')

    # pairwise distance histogram bins
    hist_bins = np.linspace(data_bins.min(), data_bins.max(), simulated_pw_dists_n_bins)
    hist_bin_values = (hist_bins[0:-1] + hist_bins[1:])/2  # central values of bins

    # histogram
    counts, _ = np.histogram(emp_pwdists, bins=hist_bins)
    # normalise to max value so we can plot against the data
    counts_norm = counts / counts.max()

    # histogram with same bins as data
    counts_data_binned, _ = np.histogram(emp_pwdists, bins=data_bins)
    # normalise to sum so that integral is 1
    counts_data_binned_norm = counts_data_binned / counts_data_binned.sum()

    theoretical_pw_dists_data_binned = (
        bivariate_guass_radius_probability_array(
            lower=data_bins[:-1], upper=data_bins[1:],
            gauss_params=gauss_params.axial_difference_sigma_vals())
        )

    dists: np.ndarray = data_bins[1:]  # PRESUMES first value is zero, so take all after

    # prep data for plotting
    # normalise to max so can plot data with simulated histogram
    dists_freq_norm = data_prob / data_prob.max()

    fig = (
        spl.make_subplots(
            rows=2, cols=2,
            subplot_titles=[
            "Simulated RF Locations",
            "Theoretical Probability v Data v Simulation",
            "Simulated Pair-wise distances (with Data)",
            "Residuals of Theoretical v Data v Simulation"]
            )

        .add_trace(
            go.Scatter(
                x=x_locs,y=y_locs,
                mode='markers',
                name='RF Locations',
                marker=go.scatter.Marker(  # type: ignore
                    size=2,
                    # opacity=0.3
                    )
                ),
            row=1, col=1
            )
        .update_xaxes(
            range=[4*max((gauss_params.sigma_x, gauss_params.sigma_y)) * l for l in (-1, 1)],
            constrain='domain',
            row=1, col=1)
        .update_yaxes(
            range=[4*max((gauss_params.sigma_x, gauss_params.sigma_y)) * l for l in (-1, 1)],
            constrain='domain',
            row=1, col=1)
        .update_yaxes(scaleanchor = "x", scaleratio = 1, row=1, col=1)

        .add_trace(
            go.Scatter(
                x=dists, y=theoretical_pw_dists_data_binned,
                mode='lines',
                line=go.scatter.Line(width=5),  # type: ignore
                name='Theoretical probability'
                ),
            row=1, col=2
            )

        .add_trace(
            go.Scatter(
                x=dists, y=data_prob,
                mode='markers',
                marker=go.scatter.Marker(  # type: ignore
                    size=12, opacity=0.7),
                name='Data (probability)'
                ),
            row=1, col=2
            )

        .add_trace(
            go.Scatter(
                x=dists, y=counts_data_binned_norm,
                mode='markers',
                marker=go.scatter.Marker(  # type: ignore
                    size=12, opacity=0.7),
                name='Simulated pw (binned like data)'
                ),
            row=1, col=2
            )

        .add_trace(
            go.Scatter(
                x=hist_bin_values, y= counts_norm,
                name='Empirical pw dists normalised',
                mode='lines'
                ),
            row=2, col=1
            )

        .add_trace(
            go.Scatter(
                x=dists, y=dists_freq_norm,
                mode='markers',
                name='Dist Data Normalised'
                ),
            row=2, col=1
            )

        .add_trace(
            go.Scatter(
                x=dists, y=theoretical_pw_dists_data_binned-data_prob,  # type: ignore
                mode='lines+markers',
                name='Residuals (theory - data)'
                ),
            row=2, col=2
            )

        .add_trace(
            go.Scatter(
                x=dists, y=theoretical_pw_dists_data_binned-counts_data_binned_norm,
                mode='lines+markers',
                name='Residuals (theory - simulated)'
                ),
            row=2, col=2
            )

        .update_layout(
            title=f'sigma_x={gauss_params.sigma_x} sigma_y={gauss_params.sigma_y} (ratio={gauss_params.ratio})'
            )
    )

    return fig
# -

# +
def plot_profile_rf_locations_object(
        rf_locs: do.RFLocationSigmaRatio2SigmaVals, ratio: float,
        n_simulated_locs: int = 5000, simulated_pw_dists_n_bins: int = 100):
    """Wrapper around plot_profile_rf_locations_pairwise_distances by only needing rf_locs obj

    Examples:
        >>> fig = plot_profile_rf_locations_object(rf_locs=rf_loc_OFF, ratio=2.6)
        >>> fig.show()
    """
    fig = plot_profile_rf_locations_pairwise_distances(
            gauss_params=rf_locs.ratio2gauss_params(ratio),
            data_bins=rf_locs.data_bins, data_prob=rf_locs.data_prob,
            n_simulated_locs=n_simulated_locs, simulated_pw_dists_n_bins=simulated_pw_dists_n_bins
        )

    return fig
# -

# > Rf Loc objects


# +
def mk_rf_ratio_loc_sigma_lookup_tables(
        metadata: do.RFLocMetaData,
        data_bins: np.ndarray, data_prob: np.ndarray,
        ratios: np.ndarray = np.linspace(1, 20, 100),
        ) -> do.RFLocationSigmaRatio2SigmaVals:
    """Make a lookup object

    data_bins are presumed to have one more value that data_prob (as they're bins)
    """

    # >>>>> !Important ... presume jin al have the first bin start at 0
    # now affirmed by Alonso through personal communication
    sigma_x_ratio_lookup = mk_sigma_x_ratio_lookup(ratios, data_bins, data_prob)

    rf_locs = do.RFLocationSigmaRatio2SigmaVals(
        lookup_vals = sigma_x_ratio_lookup,
        meta_data = metadata,
        data_bins = data_bins,
        data_prob = data_prob
        )

    return rf_locs

# -

# >> Demo
# +
# off_raw_luv = mk_rf_ratio_loc_sigma_lookup_tables(
#     metadata = do.RFLocMetaData('personal_email_from_alonso', 'regarding jin_et_al'),
#     data_bins=jin_data.distance_vals_insert_lower('off_raw'),
#     data_prob=jin_data.dist_vals_off_raw[:,1]
#     )
# on_raw_luv = mk_rf_ratio_loc_sigma_lookup_tables(
#     metadata = do.RFLocMetaData('personal_email_from_alonso', 'regarding jin_et_al'),
#     data_bins=jin_data.distance_vals_insert_lower('on_raw'),
#     data_prob=jin_data.dist_vals_on_raw[:,1]
#     )
# -

# >> comparing raw with paper data
# +
# off_luv = do.RFLocationSigmaRatio2SigmaVals.load('RfLoc_Generator_jin_etal-fig_5C_OFF.pkl')
# on_luv = do.RFLocationSigmaRatio2SigmaVals.load('RfLoc_Generator_jin_etal-fig_5C_ON.pkl')
# -
# +
# plot_sigma_x_ratio_lookup(off_raw_luv.lookup_vals).update_layout(title='raw_off').show()
# plot_sigma_x_ratio_lookup(off_luv.lookup_vals).update_layout(title='paper_off').show()
# -
# +
# ratio = 2.5

# fig = plot_profile_rf_locations_object(off_raw_luv, ratio=ratio)
# fig.layout.title = f'{fig.layout.title.text} with raw data'
# fig.show()
# fig = plot_profile_rf_locations_object(off_luv, ratio=ratio)
# fig.layout.title = f'{fig.layout.title.text} with paper data'
# fig.show()
# -

# >> Generic make all necessary objects

# +
def mk_all_ratio_rf_loc_objects(jin_data: _JinData, overwrite: bool = False):

    ratios = np.linspace(1, 20, 500)
    data_dir = settings.get_data_dir()

    # OFF Data
    print('Making RF location lookup object for OFF data')
    data_bins = jin_data.distance_vals_insert_lower('off')
    data_prob = jin_data.dist_vals_off[:,1]
    meta_data = do.RFLocMetaData('jin_etal', 'fig_5C_OFF')
    putative_file_name = (
        do.RFLocationSigmaRatio2SigmaVals._filename_template(meta_data.mk_key()))
    if (not (data_dir / putative_file_name).exists() or overwrite):

        rf_locs = mk_rf_ratio_loc_sigma_lookup_tables(
            metadata=meta_data,
            data_bins=data_bins, data_prob=data_prob, ratios=ratios
            )

        try:
            rf_locs.save(overwrite=overwrite)
            print(f'Saved file: {rf_locs._mk_filename()}')
        except FileExistsError:
            print(f"File {rf_locs._mk_filename()} already exists, must set overwrite to True")
    else:
        print(f'{putative_file_name} exists and overwrite not set')

    # OFF Data Raw from Email
    print('Making RF location lookup object for raw (Alonso email) OFF data')
    data_bins = jin_data.distance_vals_insert_lower('off_raw')
    data_prob = jin_data.dist_vals_off_raw[:,1]
    meta_data = do.RFLocMetaData('jin_etal', 'alonso_email_raw_data_off')

    putative_file_name = (
        do.RFLocationSigmaRatio2SigmaVals._filename_template(meta_data.mk_key()))
    if (not (data_dir / putative_file_name).exists() or overwrite):
        rf_locs = mk_rf_ratio_loc_sigma_lookup_tables(
            metadata=meta_data,
            data_bins=data_bins, data_prob=data_prob, ratios=ratios
            )

        try:
            rf_locs.save(overwrite=overwrite)
            print(f'Saved file: {rf_locs._mk_filename()}')
        except FileExistsError:
            print(f"File {rf_locs._mk_filename()} already exists, must set overwrite to True overwirite")

    else:
        print(f'{putative_file_name} already exists and overwrite not set')

    # ON
    print('Making RF location lookup object for ON data')
    data_bins = jin_data.distance_vals_insert_lower('on')
    data_prob = jin_data.dist_vals_on[:,1]
    meta_data = do.RFLocMetaData('jin_etal', 'fig_5C_ON')
    putative_file_name = (
        do.RFLocationSigmaRatio2SigmaVals._filename_template(meta_data.mk_key()))
    if (not (data_dir / putative_file_name).exists() or overwrite):

        rf_locs = mk_rf_ratio_loc_sigma_lookup_tables(
            metadata=meta_data,
            data_bins=data_bins, data_prob=data_prob, ratios=ratios
            )

        try:
            rf_locs.save(overwrite=overwrite)
            print(f'Saved file: {rf_locs._mk_filename()}')
        except FileExistsError:
            print(f"File {rf_locs._mk_filename()} already exists, must set overwrite to True overwirite")

    else:
        print(f'{putative_file_name} exists and overwrite not set')

    # ON Raw data from email
    print('Making RF location lookup object for raw ON data from Alonso email')
    data_bins = jin_data.distance_vals_insert_lower('on_raw')
    data_prob = jin_data.dist_vals_on_raw[:,1]
    meta_data = do.RFLocMetaData('jin_etal', 'alonso_email_raw_data_on')
    putative_file_name = (
        do.RFLocationSigmaRatio2SigmaVals._filename_template(meta_data.mk_key()))
    if (not (data_dir / putative_file_name).exists() or overwrite):

        rf_locs = mk_rf_ratio_loc_sigma_lookup_tables(
            metadata=meta_data,
            data_bins=data_bins, data_prob=data_prob, ratios=ratios
            )

        try:
            rf_locs.save(overwrite=overwrite)
            print(f'Saved file: {rf_locs._mk_filename()}')
        except FileExistsError:
            print(f"File {rf_locs._mk_filename()} already exists, must set overwrite to True overwirite")

    else:
        print(f'{putative_file_name} exists and overwrite not set')

    # AVG of ON and OFF data
    print('Making RF location lookup object for average of ON and OFF data')
    data_prob = (
        np
        .vstack(
            (jin_data.dist_vals_on[:,1], jin_data.dist_vals_off[:,1])
            )
        .mean(axis=0)
        )

    # either would work as they're the same bins
    data_bins = jin_data.distance_vals_insert_lower('on')
    meta_data = do.RFLocMetaData('jin_etal', 'fig_5C_avg_ON_and_OFF')

    putative_file_name = (
        do.RFLocationSigmaRatio2SigmaVals._filename_template(meta_data.mk_key()))
    if (not (data_dir / putative_file_name).exists() or overwrite):


        rf_locs = mk_rf_ratio_loc_sigma_lookup_tables(
            metadata=meta_data,
            data_bins=data_bins, data_prob=data_prob, ratios=ratios
            )

        try:
            rf_locs.save(overwrite=overwrite)
            print(f'Saved file: {rf_locs._mk_filename()}')
        except FileExistsError:
            print(f"File {rf_locs._mk_filename()} already exists, must set overwrite to True overwirite")

    else:
        print(f'{putative_file_name} alread exists and overwrite not set')

    # AVG of raw ON and OFF data from Alonso's email
    print('Making RF location lookup object for average of raw (from email) ON and OFF data')
    data_prob = (
        np
        .vstack(
            (jin_data.dist_vals_on_raw[:,1], jin_data.dist_vals_off_raw[:,1])
            )
        .mean(axis=0)
        )

    # either would work as they're the same bins
    data_bins = jin_data.distance_vals_insert_lower('on_raw')
    meta_data = do.RFLocMetaData('jin_etal', 'alonso_email_raw_data_avg_ON_and_OFF')

    putative_file_name = (
        do.RFLocationSigmaRatio2SigmaVals._filename_template(meta_data.mk_key()))
    if (not (data_dir / putative_file_name).exists() or overwrite):


        rf_locs = mk_rf_ratio_loc_sigma_lookup_tables(
            metadata=meta_data,
            data_bins=data_bins, data_prob=data_prob, ratios=ratios
            )

        try:
            rf_locs.save(overwrite=overwrite)
            print(f'Saved file: {rf_locs._mk_filename()}')
        except FileExistsError:
            print(f"File {rf_locs._mk_filename()} already exists, must set overwrite to True overwirite")

    else:
        print(f'{putative_file_name} alread exists and overwrite not set')
# -

# >>! Making and Saving the objects
# run this to create the necessary objects whenever a change is made to the data
# +
# mk_all_ratio_rf_loc_objects(jin_data=jin_data)
# -

# > RF Loc tools

# >> Scaling Pairwise distance unit

# >>> Diameter of RF (according to jin et al protocol)

# +
def spat_filt_max_magnitude(spat_filt: do.DOGSpatFiltArgs) -> float:

    return ff.mk_dog_sf(ArcLength(0), ArcLength(0), spat_filt)


def spat_filt_magnitude_residual_at_coord(
        x: np.ndarray,
        spat_filt: do.DOGSpatFiltArgs,
        target_mag: float,
        arclength_unit: str = 'mnt',
        ) -> float:
    """Presume circular/radially symmetrical, return mag of DOG at coord
    """
    coord = ArcLength(x[0], arclength_unit)
    mag = ff.mk_dog_sf(coord, ArcLength(0), dog_args=spat_filt)
    return mag - target_mag


def spat_filt_coord_at_magnitude_ratio(
        spat_filt: do.DOGSpatFiltArgs,
        target_ratio: float,
        spat_res: ArcLength[scalar],
        round: bool = True
        ) -> ArcLength[scalar]:
    """Coordinate (1D) where spat_filt has a magnitude with `target_ratio` to maximum

    Uses `spat_filt_max_magnitude` for determining the maximum magnitude of the spatial
    filter, which is presumed to be at the "center" or coords `0,0`.
    Uses optimisation over `spat_filt_magnitude_residual_at_coord`.

    Returned value is the coordinate from the origin `(0,0)` and represents a "radius".
    To get the `diameter` of the spatial filter, must double.

    Though this uses optimisation, it is a trivial example and should run in milliseconds.

    Args:
        spat_filt: The spatial filter in question
        target_ratio: magnitude optimised for is `maximum * target_ratio`
        spat_res:
            spatial resolution of the simulation to ensure correct units are used
            and for rounding the resultant coordinate to the resolution "grid".

    Examples:
        >>> spat_res = ArcLength(30, 'sec')
        >>> r = spat_filt_coord_at_magnitude_ratio(
        ...     spat_filt=sf.parameters, target_ratio=0.2, spat_res=spat_res)
        >>> print(r)
        ArcLength(value=150, unit='sec')

        >>> (
        ...     ff.mk_dog_sf(r, ArcLength(0), sf.parameters)  # target coord and mag
        ...     /
        ...     ff.mk_dog_sf(ArcLength(0), ArcLength(0), sf.parameters)  # max mag
        ...     )
        0.21684611562815662
    """

    arclength_unit = spat_res.unit
    max_mag = spat_filt_max_magnitude(spat_filt)
    # magnitude of spatial filter (ON or OFF) should be encoded in the `max_mag`
    # ... target_magnitude should have the same polarity has `max_mag` and spat filt
    target_magnitude = target_ratio * max_mag
    try:
        res: opt.OptimizeResult = opt.least_squares(
            spat_filt_magnitude_residual_at_coord,
            bounds=[0, np.inf], x0=[1],
            args=(spat_filt, target_magnitude, arclength_unit)
            )
    except Exception as e:
        raise exc.LGNError(
            f'Could not find coords for magnitude with target ratio {target_ratio}'
            ) from e

    if not res.success:
        raise exc.LGNError(
            f'Optimisation to target magnitude with ratio to max of {target_ratio} unseccessful with cost {res.cost}')

    # Idea of rounding to the low is to emulate process in Jin et al:
    # ... 1) took all pixels with magnitude GREATER than the target
    # ... 2) used smoothed binary white noise rev correlation with smallest pixels being
    #        0.1 degrees (6 minutes), often bigger ... which is very likely to be bigger
    #        than the spatial resolution being rounded to here.  So hopefully the rounding
    #        is similar to their process in terms of courseness of granularity.
    target_coord = ArcLength(res.x[0], arclength_unit)
    if round:
        rounded_coord = ff.round_coord_to_res(
            target_coord,
            res=spat_res, low=True
            )
        return rounded_coord
    else:
        return target_coord
# -

# +
def avg_largest_pairwise_value(values: Iterable[float]) -> float:
    """For set of values, average largest value of all pairings

    Uses only unique pairings, which is consistent with an equal probability of
    selecting any value.
    Also presumes selection with replacement which implies that a particular spat_filt
    can occur more than once in an LGN layer ... which is true
    """

    pairings = combinations_with_replacement(values, r=2)
    largest_value_of_each_pair = list(max(p) for p in pairings)
    mean_largest_value: float = np.mean(largest_value_of_each_pair)

    return mean_largest_value
# -
# +
def mk_rf_locations_distance_scale(
        spat_filters: Iterable[do.DOGSpatialFilter],
        spat_res: ArcLength[scalar],
        magnitude_ratio_for_diameter: float = 0.2
        ) -> ArcLength[scalar]:
    """For a list of spatial filters, find average largest pairwise diameter

    First uses `spat_filt_coord_at_magnitude_ratio` to find protocol specific RF diameter
    for each spatial filter.
    Then finds all pairwise combinations (including double-ups with same-same), the largest
    of each pair and the average of these largest values.
    Returns as an ArcLength in the same units as `spat_res` (and snapped to the resolution
    grid too) as a `diameter`.
    """

    coords_for_target_magnitude = [
        spat_filt_coord_at_magnitude_ratio(
            spat_filt=sf.parameters, target_ratio=magnitude_ratio_for_diameter,
            spat_res=spat_res, round=True)
        for sf in spat_filters
    ]

    spat_filt_diameters = [
        # double for diameter
        ArcLength(2*r.value, r.unit)
        for r in coords_for_target_magnitude
    ]

    # use same unit as spat res for easy conversion back to an arclength
    avg_biggest_diameter_value = avg_largest_pairwise_value(
        [d[spat_res.unit] for d in spat_filt_diameters] )
    avg_biggest_diameter = ArcLength(avg_biggest_diameter_value, spat_res.unit )

    return avg_biggest_diameter
# -
def mk_coords_2d(x_locs: np.ndarray, y_locs: np.ndarray) -> np.ndarray:
    return np.vstack((x_locs, y_locs)).T

def mk_coords_xy(coords_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not ((coords_array.shape[1] == 2) and len(coords_array.shape)==2):
        raise ValueError(f'locations array must be columnar (shape: (x,2)), instead {coords_array.shape}')

    return coords_array[:,0], coords_array[:,1]
# +
def rotate_rf_locations(
        x_locs: np.ndarray, y_locs: np.ndarray,
        orientation: ArcLength[scalar]
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate unitless location coordinates to be oriented to `orientation`

    Args:
        locations_array: columnar array with X as first and Y as second column
        orientation:
            The orientation that the rf location elongation will be oriented along.
            As the default is 90 deg, and the rotation matrix operation is counter-clockwise,
            the rotation matrix will rotate by an angle different from the provided `orientation`
            so as to have the desired result.

    Examples:
        >>> x,y = mk_unitless_rf_locations(
        ...     1000,
        ...     do.LGNLocationParams(3, 'jin_etal_on')
        ...     )
        >>> coords = np.vstack((x,y)).T
        >>> px.scatter(x=x, y=y).update_yaxes(scaleanchor = "x", scaleratio = 1).show()
        >>> rot_coords = rotate_rf_locations(coords, ArcLength(0))
        >>> px.scatter(x=rot_coords[:,0], y=rot_coords[:,1]).update_yaxes(scaleanchor = "x", scaleratio = 1).show()
    """

    # only needs to be betwee 0 and 90
    if not (0 <= orientation.deg <= 90):
        raise exc.LGNError(f'RF Location orientation parameter ({orientation.deg}) is out of bounds')

    locations_array = mk_coords_2d(x_locs, y_locs)

    default_ori = ArcLength(90,'deg')
    difference = default_ori.rad - orientation.rad  # convert to rad now for numpy functions
    # as rotation is counter-clockwise, can just directly use `difference`.
    theta = ArcLength(difference,'rad')
    s,c = np.sin(theta.rad), np.cos(theta.rad)
    R = np.array(((c, -s), (s, c)))

    rotated_locations_array = locations_array @ R

    #                     x                             y
    return mk_coords_xy(rotated_locations_array)
# -
# +
def mk_unitless_rf_locations(
        n: int,
        rf_loc_gen: do.RFLocationSigmaRatio2SigmaVals,
        ratio: float,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate `X,Y` coords (unitless)


    """
    gauss_params = rf_loc_gen.ratio2gauss_params(ratio)
    x_locs, y_locs = (
        np.random.normal(scale=s, size=n)
        for s in
            (gauss_params.sigma_x, gauss_params.sigma_y)
        )

    return x_locs, y_locs


def apply_distance_scale_to_rf_locations(
        x_locs: np.ndarray, y_locs: np.ndarray,
        distance_scale: ArcLength[scalar],
        ) -> do.LGNRFLocations:

    if not (x_locs.shape == y_locs.shape):
        raise exc.LGNError(
            f'X and Y location coordinates are not of the same size ({x_locs.shape, y_locs.shape})')

    location_coords = tuple(
        (
            do.RFLocation(
                x=ArcLength(x_locs[i] * distance_scale.value, distance_scale.unit),
                y=ArcLength(y_locs[i] * distance_scale.value, distance_scale.unit)
                )
        )
        for i in range(len(x_locs))
    )

    rf_locations = do.LGNRFLocations(locations=location_coords)

    return rf_locations
# -



# > Plot RF Locations
# ... prototypes here
# +
def plot_unitless_rf_locations(locs: tuple):
    x_locs = locs[0]
    y_locs = locs[1]

    fig = go.Figure()

    for i in range(len(x_locs)):
        # locations are in units of rf diameter
        x0, x1 = x_locs[i]-0.5, x_locs[i]+0.5
        y0, y1 = y_locs[i]-0.5, y_locs[i]+0.5
        fig.add_shape(  # type: ignore
            type="circle",
            xref="x", yref="y",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line_color="#DDD",
            fillcolor="rgba(140, 40, 40, 0.2)",
            # opacity=0.1
        )

    fig = (
        fig
        .update_layout(
            xaxis_range=(x_locs.min()-0.5, x_locs.max()+0.5),
            yaxis_range=(y_locs.min()-0.5, y_locs.max()+0.5),
            template='plotly_dark'
            )
        .update_yaxes(  # type: ignore
            constrain='domain',
            scaleanchor = "x", scaleratio = 1
            )
        )

    return fig
# -
# +
def plot_rf_locations(
        rf_locations: Tuple[Tuple[ArcLength, ArcLength]],
        distance_scale: ArcLength,
        unit: str = 'mnt'
        ):
    """Rudimentary view of RF locations as circles ... needs improvement


    """

    fig = go.Figure()
    putative_rf_radius = distance_scale[unit] / 2

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for x_loc, y_loc in rf_locations:
        # locations are in units of rf diameter
        x0, x1 = x_loc[unit]-putative_rf_radius, x_loc[unit]+putative_rf_radius
        y0, y1 = y_loc[unit]-putative_rf_radius, y_loc[unit]+putative_rf_radius

        xmins.append(x0)
        ymins.append(y0)
        xmaxs.append(x1)
        ymaxs.append(y1)

        fig.add_shape(  # type: ignore
            type="circle",
            xref="x", yref="y",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line_color="#DDD",
            fillcolor="rgba(140, 40, 40, 0.2)",
            # opacity=0.1
        )

    fig = (
        fig
        .update_layout(
            xaxis_range=(min(xmins)-putative_rf_radius, max(xmaxs)+putative_rf_radius),
            yaxis_range=(min(ymins)-putative_rf_radius, max(ymaxs)+putative_rf_radius),
            xaxis_title=f'ArcLength {unit}',
            yaxis_title=f'ArcLength {unit}',
            template='plotly_dark'
            )
        .update_yaxes(  # type: ignore
            constrain='domain',
            scaleanchor = "x", scaleratio = 1
            )
        )

    return fig
# -
