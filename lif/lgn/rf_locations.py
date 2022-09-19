
# > Imports
# +
import pickle
import glob
from functools import partial
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
import warnings
from textwrap import dedent

import numpy as np
import pandas as pd

from scipy import spatial
pdist = spatial.distance.pdist
from scipy.optimize import curve_fit
import scipy.optimize as opt
import scipy.integrate as integrate
import scipy.special as special

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as spl
# -
# +
from lif.utils.units.units import val_gen, ArcLength

from ..receptive_field.filters import cv_von_mises as cvvm
from ..utils import data_objects as do
from ..utils import exceptions as exc
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
# +
def bivariate_gauss_radius_pdf(r: val_gen, sigma_x: float, sigma_y: float) -> val_gen:
	"""Analytic PDF of radius magnitude of points drawn from a bivariate Gaussian.

	Args:
		r:
			Radius for which probability is sought.
			Number is "unitless", as the Jin et al data is in dynamic
			units of "Largest RF Diameter".
		sigma_x: standard deviation of source gaussian along the x axis
		sigma_y: standard deviation of source gaussian along the y axis

	Notes:
		Derived from a Bivariate Gaussian in the complex plane.
		See
			> Schreier, P. J., & Scharf, L. L. (2010).
			> Statistical signal processing of complex-valued data: The theory of improper and noncircular signals.
			> Cambridge University Press.

	"""

	# Rayleigh x Bessel
	var_x, var_y = sigma_x**2, sigma_y**2
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

# >> Fitting values

# >>> Functions

# +
def bivariate_gauss_radius_probability(
		lower: float, upper: float, sigma_x: float, sigma_y: float) -> float:
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
		(sigma_x, sigma_y))

	return probability[0]
# -
# +
def bivariate_guass_radius_probability_array(
		lower: Union[np.ndarray, List], upper: Union[np.ndarray, List],
		sigma_x: float, sigma_y: float
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
						lower[i], upper[i], sigma_x, sigma_y
						)
		probabilities[i] = probability

	return probabilities
# -

# >>> jin data

# data from tracey mctraceface
# +
@dataclass
class _JinData:
	dist_vals_on: np.ndarray
	dist_vals_off: np.ndarray
	dist_vals_all: np.ndarray
	all_dist_type: np.ndarray

	def distance_vals_zero_insert(self, type:str) -> np.ndarray:
		'''Just distances with 0 at the beginning

		type must be either "ON" of "OFF"
		Distances presumed to be first column of data
		'''

		return np.r_[0, self.__getattribute__(f'dist_vals_{type}')[:,0]]

# -
# +
def _make_jin_data_object():

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

	# join all together
	dist_vals_all = np.vstack((dist_vals_on, dist_vals_off))
	# has same length (axis 0) as the dist arrays above
	all_dist_type = np.array(
		['ON', 'ON', 'ON', 'ON', 'ON', 'ON', 'ON', 'ON', 'ON',
		'OFF', 'OFF', 'OFF', 'OFF', 'OFF', 'OFF', 'OFF', 'OFF', 'OFF'],
		dtype=object)
	# sort all together
	sort_args = dist_vals_all[:,0].argsort()
	dist_vals_all = dist_vals_all[sort_args]
	all_dist_type = all_dist_type[sort_args]

	jin_data = _JinData(
		dist_vals_on=dist_vals_on,
		dist_vals_off=dist_vals_off,
		dist_vals_all=dist_vals_all,
		all_dist_type=all_dist_type
		)
	return jin_data
# -
# +
# set module variable
jin_data = _make_jin_data_object()
# -
# check that indexing by `all_dist_type` works
# +
assert (
		all(
		(
		np.all(jin_data.dist_vals_all[jin_data.all_dist_type == 'OFF'] == jin_data.dist_vals_off),
		np.all(jin_data.dist_vals_all[jin_data.all_dist_type == 'ON' ] == jin_data.dist_vals_on)
		)
	)
), 'Jin data is not composed correctly'
# -
# +
def plot_jin_data_probabilities():
	"Plot jin data from this module"
	fig = (
		px
		.line(
			x=jin_data.dist_vals_all[:,0], y=jin_data.dist_vals_all[:,1],
			color=jin_data.all_dist_type,
			labels={
				'x': 'distance (largest RF diameter)',
				'y': 'norm freq as probability',
				'color': 'type'},
			color_discrete_map = {'ON': 'red', 'OFF': 'blue'}
			)
		.update_traces(mode='markers+lines')
		)

	return fig
# -


# >>> Objective Function

# pw_dist func (careful with sigmas)
	# sigma of differences is sqrt(2) * sigma of coordinates
# must take data and make a difference
# use sigma_x with a ratio (to calculate sigma_y) provided as an argument
# +
def bivariate_gauss_pairwise_distance_probability_residuals(
		x: Union[np.ndarray, List], ratio: float,
		data_bins: np.ndarray, data_prob: np.ndarray
		) -> np.ndarray:
	"""Differences between the provided data distribution defined by given `sigma_x` value

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
		prob_data:
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
	sigma_x = x[0]  # only one argument
	sigma_y = sigma_x * ratio
	# sigma values for the bivariate guassian for the differences in the x and y
	sigma_x_dist, sigma_y_dist = ((2**0.5) * s
									for s in (sigma_x, sigma_y)
									)
	# start from zero for distances
	# distances = np.r_[0, prob_data[:, 0]]

	lower_bounds, upper_bounds = data_bins[0:-1], data_bins[1:]
	# probabilities of distances from prob_data given provided sigma vals
	probabilities = bivariate_guass_radius_probability_array(
						lower_bounds, upper_bounds,
						sigma_x_dist, sigma_y_dist
					)

	# pair wise distances for current sigma values
	# pair_wise_dists = bivariate_gauss_radius_pdf(
	# 					prob_data[:,0],  # for all distance data
	# 					sigma_x=sigma_x_dist, sigma_y=sigma_y_dist)


	# normalise to ensure integral is 1
	# pair_wise_dists /= pair_wise_dists.sum()
	residuals = probabilities - data_prob  # type: ignore

	return residuals
# -

# >>> Fit Values
# +
def mk_optimasation_sigma_x_for_ratio(
		ratio: float,
		data_bins:np.ndarray, data_prob:np.ndarray) -> opt.OptimizeResult:

	res: opt.OptimizeResult = opt.least_squares(
		bivariate_gauss_pairwise_distance_probability_residuals, x0=[1],
		bounds=([0], [np.inf]),
		args = (ratio, data_bins, data_prob))

	return res

	# if res.success:
	# 	return res.x[0]
	# else:
	# 	raise exc.LGNError(
	# 		f'Failed to find optimal sigma_x for ratio {ratio} and data {data_bins,data_prob}')
# -

## record optimal and cost (note 0.5 of sum of squares) ... use additional function
## make appropriate data object and save and load methods ...
# +
data_bins = jin_data.distance_vals_zero_insert('off')
data_prob = jin_data.dist_vals_off[:,1]
# -
# +
def mk_sigma_x_ratio_lookup(
		ratios: np.ndarray,
		data_bins: np.ndarray, data_prob: np.ndarray
		) -> pd.DataFrame:
	# sigma values and cost
	sigma_x_values_with_cost = np.empty(shape=(ratios.size, 2), dtype=np.floating)

	for i,r in enumerate(ratios):
		sigma_x: float = np.NaN
		cost: float = np.NaN
		if i % (len(ratios)//20) == 0:
			print(f'{i/len(ratios):<10.2%}', end='\r')

		try:
			res = mk_optimasation_sigma_x_for_ratio(r, data_bins, data_prob)
			sigma_x, cost = res.x[0], res.cost
		except exc.LGNError as e:
			print(f'Failed to optimise for ratio {r}')
		finally:
			sigma_x_values_with_cost[i] = sigma_x, cost

	df = pd.DataFrame({
		'ratios': ratios,
		'sigma_x': sigma_x_values_with_cost[:, 0],
		'error': sigma_x_values_with_cost[:, 1],
		})

	return df
# -

# >>>> Demo with OFF data
# +
# >>>>> !Important ... presume jin al have the first bin start at 0
# which makes an irregular first bin ... but see their comments in the
# supplementary materials
data_bins = jin_data.distance_vals_zero_insert('off')
data_prob = jin_data.dist_vals_off[:,1]
ratios = np.linspace(1, 20, 100)
sigma_x_ratio_lookup = mk_sigma_x_ratio_lookup(ratios, data_bins, data_prob)
# -
# +
def plot_sigma_x_ratio_lookup(lookup_vals: pd.DataFrame):
	fig = (
		px
		.line(
			lookup_vals,
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

# >>>> Demo with ON data

# slightly different fitting ...
# best ratio is slightly bigger (closer to 3), and the error is less with this as well
# seems to be the result of differences in noise between them however
# +
data_bins = jin_data.distance_vals_zero_insert('on')
data_prob = jin_data.dist_vals_on[:,1]
ratios = np.linspace(1, 20, 100)
sigma_x_ratio_lookup = mk_sigma_x_ratio_lookup(ratios, data_bins, data_prob)
# -
# +
plot_sigma_x_ratio_lookup(sigma_x_ratio_lookup).show()
# -

# >>> Characterise Error function
# +
def characterise_pairwise_distance_distribution_residuals(
		data_bins: np.ndarray, data_prob: np.ndarray,
		ratios: np.ndarray=np.arange(1,10), sigma_x_vals: np.ndarray=np.linspace(0.01,1,200)
		) -> pd.DataFrame:

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
pw_errors = characterise_pairwise_distance_distribution_residuals(
	data_bins=jin_data.distance_vals_zero_insert('off'),
	data_prob=jin_data.dist_vals_off[:,1],
	)
# -
# +
def plot_characterisation_pairwise_distance_residuals(pw_errors: pd.DataFrame):
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


# >>> Manual Check
# +
def plot_profile_rf_locations_pairwise_distances(
		sigma_x: float, ratio: float,
		data_bins: np.ndarray, data_prob: np.ndarray,
		n_simulated_locs: int = 5000, simulated_pw_dists_n_bins: int = 100):
	"""Visualise how well a bivariate gaussian has pairwise distances well fit to data

	data arguments are intended to be the data to which a distribution will be fit

	The data_bins are presumed to start at zero.

	Examples:
		>>> fig = plot_profile_rf_locations_pairwise_distances(
		>>> 		sigma_x=0.178,ratio=2.92,
		>>> 		data_bins=jin_data.distance_vals_zero_insert('on'),
		>>> 		data_prob=jin_data.dist_vals_on[:, 1])
		>>> fig.show()

	"""
	sigma_y = sigma_x*ratio

	# Simulate rf locations and their pairwise distances
	# size = 5000
	x_locs = np.random.normal(size=n_simulated_locs, scale=sigma_x)
	y_locs = np.random.normal(size=n_simulated_locs, scale=sigma_y)
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
			sigma_x=(2**0.5)*sigma_x, sigma_y=(2**0.5)*sigma_y)
		)

	dists: np.ndarray = data_bins[1:]  # PRESUMES first value is zero, so take all after
	# dists_pdf = bivariate_gauss_radius_pdf(
	# 	dists, sigma_x=(2**0.5)*sigma_x, sigma_y=(2**0.5)*sigma_y)
	# # scale probabilities so that they sum to 1
	# # as only a PDF (not Prob Mass Function), then these values are not probabilities
	# # only an integral is a probability
	# dists_pdf /= dists_pdf.sum()

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
				marker=go.scatter.Marker(
					size=2,
					# opacity=0.3
					)
				),
			row=1, col=1
			)
		.update_xaxes(
			range=[4*max((sigma_x, sigma_y)) * l for l in (-1, 1)], constrain='domain',
			row=1, col=1)
		.update_yaxes(
			range=[4*max((sigma_x, sigma_y)) * l for l in (-1, 1)], constrain='domain',
			row=1, col=1)

		.add_trace(
			go.Scatter(
				x=dists, y=theoretical_pw_dists_data_binned,
				mode='lines',
				line=go.scatter.Line(width=5),
				name='Theoretical probability'
				),
			row=1, col=2
			)

		.add_trace(
			go.Scatter(
				x=dists, y=data_prob,
				mode='markers',
				marker=go.scatter.Marker(size=12, opacity=0.7),
				name='Data (probability)'
				),
			row=1, col=2
			)

		.add_trace(
			go.Scatter(
				x=dists, y=counts_data_binned_norm,
				mode='markers',
				marker=go.scatter.Marker(size=12, opacity=0.7),
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


		.update_yaxes(scaleanchor = "x", scaleratio = 1, row=1, col=1)
		.update_layout(title=f'{sigma_x=} {sigma_y=} ({ratio=} )')
	)

	return fig
# -
# +
fig = plot_profile_rf_locations_pairwise_distances(
	sigma_x=0.178,ratio=2.92,
	data_bins=jin_data.distance_vals_zero_insert('on'),
	data_prob=jin_data.dist_vals_on[:, 1])
fig.show()
# -

# +
# opt_result = opt.least_squares()
# -


# +
x = np.linspace(0, 10, 100)
jin = jin_cdf(x)
jin_norm = jin_cdf_re_norm(x)
jin_pdf = jin_re_norm_pdf_est(x)
jin_adj_pdf = jin_pdf_adj(x)
# -
# +
fig = px.line(
	x=x,
	y=[jin, jin_norm, np.r_[jin_adj_pdf, 0]],
	)
labels=['jin', 'normalised', 'pdf (adjusted)']
for i,l in enumerate(labels):
	fig.data[i].name = l
fig.show()
# -
# +
print(
	jin.sum(),
	jin_norm.sum(),
	jin_adj_pdf.sum())
# -
# +

proto = jin_norm / jin_norm.sum()
proto2 = jin / jin.sum()
np.allclose(proto[:-1], jin_adj_pdf)
np.allclose(proto2[:-1], jin_adj_pdf)
# -
# +

# -

# +
test = do.LGNJinEtAlLocationDistribution(3.4, 2.5)
# -
# +
jin = test.cumulative(ArcLength(x))
jin_pdf = test.pdf(ArcLength(x))
# -
# +
fig = px.line(
	x=x,
	y=[jin, jin_pdf],
	)
labels=['jin', 'pdf (adjusted)']
for i,l in enumerate(labels):
	fig.data[i].name = l
fig.show()
# -

# > Dispersed RFs
# +
def mk_dispersed_rf_locations(
		sample_size: int,
		base_loc_dist: do.LGNJinEtAlLocationDistribution,
		polar_loc_dist: do.LGNPolarLocationDistribution,
		base_loc_values: ArcLength[np.ndarray],
		polar_loc_values: ArcLength[np.ndarray]
		) -> Tuple[ArcLength[np.ndarray], ArcLength[np.ndarray]]:
	"""Statistically generated cartesian coordinates for multiple Receptive Fields

	Given the provided distributions, generates coordinates for the locations of
	receptive fields.
	The coordinates are generated from polar coordinates.
	The distance from the center is derived from the `bas_loc_dist`.
	The angle from `polar_loc_dist`.
	`base_loc_values` and `polar_loc_values` are the discrete values that are sampled from
	using the above distributions ... which lets the caller control the fineness with which
	coordinates are generated.
	"""

	# polarity (either 0-180deg vector or 180-360deg)
	# -1 to the power of either 0 or 1, as array of size sample_size
	# result: -1 or 1
	polarities = (-1)**np.random.randint(2, size=sample_size)
	magnitudes = ArcLength(
		np.random.choice(
			base_loc_values.deg, p=base_loc_dist.pdf(base_loc_values),
			size=sample_size)
		*
		polarities,  # now either negative or positive
		'deg'  # make sure matches unit chosen from
		)

	angles = ArcLength(
		np.random.choice(
			polar_loc_values.rad, p=polar_loc_dist.pdf(polar_loc_values),
			size=sample_size),
		'rad'
		)


	x_locs = ArcLength(magnitudes.deg * np.cos(angles.rad), 'deg')
	y_locs = ArcLength(magnitudes.deg * np.sin(angles.rad), 'deg')

	return x_locs, y_locs
# -
# +
loc_dist = do.LGNJinEtAlLocationDistribution(3.4, 2.5)
pol_dist = do.LGNPolarLocationDistribution(phi=ArcLength(np.pi/2, 'rad'), k=2)
loc_vals = ArcLength(np.linspace(0, 5, 1000))
pol_vals = ArcLength(np.linspace(0, 180, 1000))
# -
# +
xl, yl = mk_dispersed_rf_locations(10, loc_dist, pol_dist, loc_vals, pol_vals)
# -
# +
(
	px.
	scatter(x=xl.deg, y=yl.deg)
	.update_yaxes(scaleanchor = "x", scaleratio = 1, )
	.show()
)
# -
# +
def mk_pairwise_distances(
		x_loc: ArcLength[np.ndarray], y_loc: ArcLength[np.ndarray]
		) -> ArcLength[np.ndarray]:

	X = np.vstack((x_loc.deg, y_loc.deg)).T
	pwdists = ArcLength(spatial.distance.pdist(X=X, metric='euclidean'), 'deg')
	return pwdists
# -
# +
pol_vals = ArcLength(np.linspace(0, 180, 5000))
pol_samples = np.random.choice(pol_vals.deg, p=pol_dist.pdf(pol_vals), size=pol_vals.deg.shape[0])
px.histogram(pol_samples, nbins=500).show()
# -
px.violin(pol_samples, points='all').show()


# > Gaussian Paiw-Wise experimentation
# +
pdist = spatial.distance.pdist
# -
# +
n = 1000
sigma = 16
sigma2 = 100
x=np.random.normal(size=n, scale=sigma)
y=np.random.normal(size=n, scale=sigma2)
# y=np.random.normal(size=n, scale=sigma)
points=np.vstack((x,y)).T
# -
# +
pw_dists = pdist(points, metric='euclidean')
pw_dists.shape
# -
# +
fig = (
	px
	.histogram(pw_dists, histnorm='probability density')
	.update_layout(title=f'sigma={sigma},{sigma2}')
)
# -
# +
fig.show()
# -
import scipy.stats as sts
# +
x = np.linspace(0, pw_dists.max(), 1000)
rayleigh_scale = 2**0.5 * sigma
rayleigh_scale2 = (sigma**0.5) * (sigma2**0.5)
rp = sts.rayleigh.pdf(x, 0, rayleigh_scale2)
fig_dist = px.line(x=x, y=rp).update_traces(line_color='red')
# -
# +
fig.add_trace(fig_dist.data[0]).show()
# -
# +
(
	px
	.scatter(x=points[:,0], y=points[:,1])
	.update_yaxes(scaleanchor = "x", scaleratio = 1, )
	.update_layout(title=f'{sigma}')
	.show()
	)
# -

# >> Radius in complex plane implementation
# +
import scipy.special as special
# -
# +
# def pwd_pdf(r, sigma_x: float, sigma_y: float):

# 	# Rayleigh x Bessel
# 	var_x, var_y = sigma_x**2, sigma_y**2
# 	R_xx = var_x + var_y
# 	R_xx_comp = var_x - var_y
# 	# 0 if sigmas are the same, approaches +/- 1 as one is greater than the other
# 	# c/special.iv/Bessel below is 1 if rho is 0
# 	rho = R_xx_comp / R_xx

# 	exponent = (r**2) / (R_xx*(1-rho**2))

# 	a = (2*r) / (R_xx*((1-rho**2)**0.5))
# 	b = np.exp(-1 * exponent)
# 	# b = np.exp((-r**2) / (R_xx*(1-rho**2)) )
# 	# modified Bessel function of the first kind of order 0
# 	# constant 1 if sigmas are the same, else drives the change in the PDF
# 	c = special.iv(0,
# 		rho * exponent
# 		# ((r**2)*rho) / (R_xx*(1-rho**2))
# 		)

# 	pdf = (a*b
# 		# replace infs with large numbers
# 		# (should only occur where b = 0, from manual inspection that is)
# 		* np.nan_to_num(c)
# 		)
# 	# correct for 0*np.inf=nan ... convert from nan to 0
# 	# pdf[(b==0) & np.isinf(c)] = 0

# 	return pdf
# -
# >>> Testing
# +
sigma_x, sigma_y = 0.2, 0.2
dists = np.linspace(0, 80, 1000)
dists_pdf = bivariate_gauss_radius_pdf(dists, (2**0.5)*sigma_x, (2**0.5)*sigma_y)
# -
# +
anal_line_fig = (
	px
	.line(x=dists, y=dists_pdf)
	.update_traces(line_color='red')
	.update_layout(title=f'{sigma_x}, {sigma_y}')
	)
# -
anal_line_fig.show()
# +
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
rs = RandomState(MT19937(SeedSequence(123456789)))
# Later, you want to restart the stream
# rs = RandomState(MT19937(SeedSequence(987654321)))
# -
# +
n = 3000
x=rs.normal(size=n, scale=sigma_x)
y=rs.normal(size=n, scale=sigma_y)
points=np.vstack((x,y)).T
# -
# +
pw_dists = pdist(points, metric='euclidean')
# -
# +
fig = (
	px
	.histogram(pw_dists, histnorm='probability density')
	.add_traces(list(anal_line_fig.select_traces()))
	.update_layout(title=f'sigma={sigma_x},{sigma_y}')
)
# -
# +
fig.show()
# -
# Bingo!

# >>> Testing by finding prediction differences
# +
sigma_x, sigma_y = 1, 10
# will also use as bins
dists_bins = np.linspace(0, 80, 1000)
dists = (dists_bins[1:] + dists_bins[0:-1]) / 2
dists_pdf = pwd_pdf(dists, (2**0.5)*sigma_x, (2**0.5)*sigma_y)
# -
# +
n = 3000
x=rs.normal(size=n, scale=sigma_x)
y=rs.normal(size=n, scale=sigma_y)
points=np.vstack((x,y)).T
# -
# +
pw_dists = pdist(points, metric='euclidean')
# -
# +
counts, bins = np.histogram(pw_dists, bins=dists_bins, density=True)
# -
# +
fig = (go
	.Figure()
	.add_traces([
		go.Scatter(
			x=dists, y=dists_pdf,
			name='pdf',
			mode='lines'),
		go.Scatter(
			x=dists, y=counts,
			name='hist',
			mode='lines'),
		])
	)
fig.show()
# -
# +
fig = (go
	.Figure()
	.add_trace(
		go.Scatter(
			x=dists, y=(dists_pdf-counts),
			name='difference',
			mode='lines'
		)
	)
)
fig.show()
# -

# >>>> Multiple Runs
# +
n_runs = 15
sigma_x, sigma_y = 0.1, 0.10
# will also use as bins
dists_bins = np.linspace(0, 80, 1000)
dists = (dists_bins[1:] + dists_bins[0:-1]) / 2
runs_data = []

for r in range(n_runs):
	print(f'Run {r} of {n_runs}')
	dists_pdf = bivariate_gauss_radius_pdf(dists, (2**0.5)*sigma_x, (2**0.5)*sigma_y)
	n = 3000
	x=rs.normal(size=n, scale=sigma_x)
	y=rs.normal(size=n, scale=sigma_y)
	points=np.vstack((x,y)).T
	pw_dists = pdist(points, metric='euclidean')
	counts, bins = np.histogram(pw_dists, bins=dists_bins, density=True)
	run_data = pd.DataFrame({
		'difference': (dists_pdf - counts),
		'pdf': dists_pdf,
		'counts': counts,
		'dists': dists
		})
	run_data['run'] = r
	runs_data.append(run_data)
all_runs_data = pd.concat(runs_data)
# -
# +
(
	px
	.line(
		all_runs_data,
		x='dists',
		y='difference',
		facet_col='run', facet_col_wrap=3 )
	.update_layout(title=f'sigmas = {sigma_x, sigma_y}')
	.show()
)
# -

# >>>> Integrating PDF to get actual probabilities (like a histogram)
# +
# def bivariate_gauss_radius_probability(a: float, b: float, sigma_x: float, sigma_y: float) -> float:

# 	func = lambda r: bivariate_gauss_radius_pdf(r, sigma_x=sigma_x, sigma_y=sigma_y)
# 	probability = integrate.quad(func, a, b)
# 	return probability[0]
# -
# +
# def bivariate_guass_radius_probability_array(
# 		a: np.ndarray, b: np.ndarray, sigma_x: float, sigma_y: float
# 		) -> np.ndarray:
# 	n_bins = len(a)
# 	probabilities = np.empty_like(a)
# 	for i in range(n_bins):
# 		probabilities[i] = integrate.quad(
# 			bivariate_gauss_radius_pdf,
# 			a[i], b[i],
# 			(sigma_x, sigma_y)
# 			)[0]
# 		# probabilities[i] = bivariate_gauss_radius_probability(
# 		# 	a[i], b[i],
# 		# 	sigma_x=sx, sigma_y=sy)

# 	return probabilities
# -
# +
dists_bins = np.linspace(0, 10, 1000)
dists = (dists_bins[1:] + dists_bins[0:-1]) / 2
dist_bins_a, dist_bins_b = dists_bins[0:-1], dists_bins[1:]
sx, sy = 1, 2
n = 5000
# -
# +
probs = bivariate_guass_radius_probability_array(
	dist_bins_a, dist_bins_b, (2**0.5)*sx, (2**0.5)*sy
	)
# -
# +
probs_pdf = bivariate_gauss_radius_pdf(dists, (2**0.5)*sx, (2**0.5)*sy)
# -
# +
x=rs.normal(size=n, scale=sx)
y=rs.normal(size=n, scale=sy)
points=np.vstack((x,y)).T
pw_dists = pdist(points, metric='euclidean')
counts, bins = np.histogram(pw_dists, bins=dists_bins, density=True)
# -
# +
fig = (
	# go.Figure()
	spl.make_subplots(rows=3, cols=1)
	.add_trace(
		go.Scatter(
			x=dists, y = probs,
			name='probs_integrated'
			),
		row=1, col=1)
	.add_trace(
		go.Scatter(
			x=dists, y=probs_pdf,
			name='probs_pdf'),
		row=2, col=1)
	.add_trace(
		go.Scatter(
			x=dists, y=counts,
			name='histogram (density)'),
		row=3, col=1)
	.update_layout(title=f'sigmas: {sx,sy}')
	)
fig.show()
# -
# +
probs.sum(), probs_pdf.sum(), counts.sum()
# (0.9995242370363089, 99.85257588445621, 99.89999999999999)
# -

# the PDF and integrated over bins (to give probabilities) curve ...
# ... they have the same shape! .... just scaled differently.

# The probabilities curve summing to 1 would be the chief property ...
# ... such that normalising the PDF to its sum (the result summing to 1)
# ... will provide the same values.
# +
# -

# >>>>> Numba optimisation!?
# scipy.special.iv is not workable (cuz fortran or cython probably)
# ... need to install numba-scipy

# +
from numba import jit, njit
# -
# +
@jit(nopython=False)
def bivariate_gauss_radius_pdf_jit(r: val_gen, sigma_x: float, sigma_y: float) -> val_gen:
	"""Analytic PDF of radius magnitude of points drawn from bivariate Gaussian

	Args:
		r: Radius for which probability is sought
		sigma_x: standard deviation of source gaussian along the x axis
		sigma_y: standard deviation of source gaussian along the y axis

	Notes:
		From derivation in the complex plane ...
	"""

	# Rayleigh x Bessel
	var_x, var_y = sigma_x**2, sigma_y**2
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
integrate.quad(bivariate_gauss_radius_pdf_jit, 1, 2, (0.1, 0.1))
# -
# +
bivariate_gauss_radius_probability(1, 2, 0.1, 0.1)
# -
# +
@jit(nopython=False)
def bivariate_gauss_radius_probability_jit(a: float, b: float, sigma_x: float, sigma_y: float) -> float:

	func = lambda r: bivariate_gauss_radius_pdf_jit(r, sigma_x=sigma_x, sigma_y=sigma_y)
	probability = integrate.quad(func, a, b)
	return probability[0]
# -
# +
@jit(nopython=False)
def bivariate_guass_radius_probability_array_jit(
		a: np.ndarray, b: np.ndarray, sigma_x: float, sigma_y: float
		) -> np.ndarray:
	n_bins = len(a)
	probabilities = np.empty_like(a)
	for i in range(n_bins):
		probabilities[i] = integrate.quad(
			bivariate_gauss_radius_pdf_jit,
			a[i], b[i],
			(sigma_x, sigma_y)
			)[0]
		# probabilities[i] = bivariate_gauss_radius_probability_jit(
		# 	a[i], b[i],
		# 	sigma_x=sx, sigma_y=sy)

	return probabilities
# -
# +
%%timeit
probs = bivariate_guass_radius_probability_array_jit(
	dist_bins_a, dist_bins_b, sx, sy
	)
# -

# +
sx, sy = 0.1, 0.1
# -
# +
n_runs = 15
# will also use as bins
dists_bins = np.linspace(0, 80, 1000)
dists = (dists_bins[1:] + dists_bins[0:-1]) / 2
runs_data = []
sigma_vals = [
	[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.5, 0.5], [0.8, 0.8], [1, 1], [1.2, 1.2], [1.5, 1.5],
	[2, 2], [3, 3], [5, 5], [10, 10]
]

for r, ss in enumerate(sigma_vals):
	sigma_x, sigma_y = ss
	print(f'Run {r} of {len(sigma_vals)}')
	dists_pdf = bivariate_gauss_radius_pdf(dists, (2**0.5)*sigma_x, (2**0.5)*sigma_y)
	n = 3000
	# x=rs.normal(size=n, scale=sigma_x)
	# y=rs.normal(size=n, scale=sigma_y)
	# points=np.vstack((x,y)).T
	# pw_dists = pdist(points, metric='euclidean')
	# counts, bins = np.histogram(pw_dists, bins=dists_bins, density=True)
	run_data = pd.DataFrame({
		# 'difference': (dists_pdf - counts),
		'pdf': dists_pdf,
		'pdf_norm': dists_pdf / dists_pdf.sum(),
		# 'counts': counts,
		'dists': dists
		})
	run_data['sigma'] = f'{sigma_x},{sigma_y}'
	runs_data.append(run_data)
all_runs_data = pd.concat(runs_data)
# -
# +
(
	px
	.line(
		all_runs_data,
		x='dists',
		y=['pdf', 'pdf_norm'],
		# y='pdf_norm',
		facet_col='sigma', facet_col_wrap=3 )
	.update_layout(title=f'sigmas = {sigma_x, sigma_y}')
	.show()
)
# -

# >>> Multiplication warning for high values
# +
pwd_pdf(70, 1, 5)
# -


# >>> Modified Bessel of first kind and order 0
# +
import scipy as sp
import scipy.special as special
import scipy.integrate as integrate
# -
# +
# >>>> this is the modified bessel function of first kind order 0 (z is input)
z = 3
result = integrate.quad(lambda theta: (1/np.pi)*np.exp(z*np.cos(theta)), 0, np.pi)
# -
# +
# >>>> scipy direct access to a modified bessel
special.iv(0, 3)
# -
# +
# >>>> checking that it's accurate
all(
	np.isclose(
		integrate.quad(lambda theta: (1/np.pi)*np.exp(z*np.cos(theta)), 0, np.pi)[0],
		special.iv(0, z)
	)
	for z in np.linspace(0, 20, 100)
)
# -

# >> Difference between gaussians
# +
v1, v2 = 1, 1
n = 10000
# scale is std deviation
x1 = np.random.normal(0, (v1)**0.5, size=n)
x2 = np.random.normal(0, (v2)**0.5, size=n)

x_diff = x1-x2
# -
# +
x_diff.var()
# -
# +
x2_vars = np.arange(10, 100, 9)
empirical_vars = []
for i in x2_vars:
	v1, v2 = 10, i
	n = 10000
	# scale is std deviation
	x1 = np.random.normal(0, (v1)**0.5, size=n)
	x2 = np.random.normal(0, (v2)**0.5, size=n)

	x_diff = x1-x2
	empirical_vars.append(x_diff.var())
# -
# +
px.scatter(x=x2_vars, y=empirical_vars).show()
# -
# +
px.histogram(x_diff).show()
# -

# > Ratio to Distribution Parameters

# >> Fit jin et al data to pwd_pdf for given ratio

# > Global optimisation
	# viable?
# produce samples of vector angles
# function that takes magnitudes of vectors and applies to provided angles
# returns difference between pairwise distances and what should be
#	Use histogram and pdf of appropriate function then sum of squares of differences
# apply sim annealing to this


#################################
# > Old


# import os


######### second gen #########



def gen_hw_disp_locs(locs, pdf,
	rf_dist_scale=1, trans = 0,
	bipolar=True, disp_k = 2,
	samp_size=16):
	rand_pol = (-1)**np.random.randint(2, size=samp_size)
	rand_pos = np.random.choice(locs, p=pdf, size=samp_size)* rand_pol * rf_dist_scale

	angles = ArcLength(np.linspace(0, np.pi, 100), 'rad')

	angles_prob = cvvm.von_mises(angles, k = disp_k)
	# Make pdf sum to 1
	angles_pdf = angles_prob / angles_prob.sum()

	rand_angle = np.random.choice(angles.value, p=angles_pdf, size=samp_size)

	xs = rand_pos * np.cos(rand_angle) + trans
	ys = rand_pos * np.sin(rand_angle) + trans

	return xs, ys


def pdf_to_pair_wise_dist(locs, pdf, samp_size=1000,
	bipolar=True,
	disp_hw=False, disp_locs=None, disp_pdf=None, disp_k=2,
	locs_return=False
	):

	'''
	From given PDF, generates actual pair-wise distances.

	bipolar parameter allows for whether RFs start at a common center
	and are distributed on either side.

	Parameters
	__________

	disp_k : float
		k value for vonmises behind dispersion of RF locations

	locs_return : Boolean
		whether return object is pair_wise distances array (false) or distances and locs

	'''

	if locs.ndim == 1:

		## Need to Generalise for use in both this and SL ... how RF locations calculated from random sample of dists


		if disp_hw and bipolar:

			## Angle prob approach

			rand_pol = (-1)**np.random.randint(2, size=samp_size)
			rand_pos = np.random.choice(locs, p=pdf, size=samp_size)* rand_pol

			angles = np.linspace(0, np.pi, 100)

			angles_prob = cvvm.von_mises(angles, k = disp_k)
			# Make pdf sum to 1
			angles_pdf = angles_prob / angles_prob.sum()

			rand_angle = np.random.choice(angles, p=angles_pdf, size=samp_size)

			xs = rand_pos * np.cos(rand_angle)
			ys = rand_pos * np.sin(rand_angle)


			## Jitter (2D Gaussian like)
			# assert (disp_locs is not None) and (disp_pdf is not None), 'Provide dispersion prob and dists for dispersion'


			# ori = np.radians(90)
			# orth_ori = ori + np.pi/2.

			# rand_pol = (-1)**np.random.randint(2, size=samp_size)
			# rand_pol_disp = (-1)**np.random.randint(2, size=samp_size)

			# rand_pos = np.random.choice(locs, p=pdf, size=samp_size) * rand_pol
			# rand_disp = np.random.choice(disp_locs, p=disp_pdf, size=samp_size) * rand_pol_disp

			# x_disp = rand_disp * np.cos(orth_ori)
			# y_disp = rand_disp * np.sin(orth_ori)

			# xs = rand_pos * np.cos(ori) + x_disp
			# ys = rand_pos * np.sin(ori) + y_disp

			locs = np.vstack((xs, ys)).T




		elif bipolar:
			rand_pol = (-1)**np.random.randint(2, size=samp_size)
			rand_pos = np.random.choice(locs, p=pdf, size=samp_size)* rand_pol

			locs= np.vstack((np.zeros(samp_size), rand_pos)).T

		else:
			locs= np.vstack((np.zeros(samp_size), np.random.choice(locs, p = pdf, size = samp_size))).T

	pw_dists = spatial.distance.pdist(locs, 'euclidean')

	if locs_return: # return locs and distances
		return pw_dists, locs
	else:
		return pw_dists


def pdf_sq_diff(b, bipolar = True, **pw_dist_kwargs):

	'''
	Generates error between actual RF distances and theoretical

	Used for optimisation function

	Error is between actual pair-wise distances, converted to
	an empirical PDF, and
	the theoretical PDF (ie from Jin et al)

	Error is sum of absolute differences

	Parameter to be optmised is b, the exponent of the exponential function
	in jin_rf_dist()


	'''

	## Creating PDF for use in simulating Pair-Wise
	dist, prob_dist = jin_rf_dist(a=1, b=b, pdf=True) # a can be one as normalisation occurring
	# dist_cdf, cum_dist = jin_rf_dist(3.4, 2.5, norm=True)

	## Simulation with Histogram ##
	# Pair wise distances from PDF ... BIPOLAR
	pw_dist = pdf_to_pair_wise_dist(dist, prob_dist, samp_size=3000, bipolar=bipolar, **pw_dist_kwargs)

	# Hist of pair-wise distances ... BIPOLAR
	counts, bins = np.histogram(pw_dist, bins='doane')

	# PDF of Pair-wise
	# normalise counts to total counts to make PDF
	sim_pdf = counts / counts.sum().astype('float')


	# Centering Bin locations
	bin_locs = bins[:-1] + ((bins[1] - bins[0])/2.)

	# Fitting curve to the actual PDF
	# Using simple_exp2, with both a and b as free variables,
	# as emulates function trying to optimise toward (?)
	popt, pcov = curve_fit(simple_exp2, bin_locs, sim_pdf)

	# For comparing to simulated Pair-Wise
	# creating new PDf on the basis of the histograms bin locations
	d, p = jin_rf_dist(a=3.4, b=2.5, x=bin_locs, pdf=True)



	sum_sq_diffs = np.sum(np.abs(p - simple_exp2(d, *popt)))


	return sum_sq_diffs



def minimise_exponent(n, **sq_diff_kwargs):
	'''
	Minimises the error between the desired RF Dist PDF and the actual

	Relies of pdf_sq_diff as the calculator of error

	Returns
	---------
	opt_x : list
		list of optimised values
	opt_nfev : list
		number of iterations for each optimised value in opt_x
	'''
	opt_x = []
	opt_nfev = []

	sq_diff_func = partial(pdf_sq_diff, **sq_diff_kwargs)

	for i in range(n):
		## below zero yields negative probabilities ... max 50 is arbitrary and probably way greater than anything reasonably
		res = opt.minimize_scalar(sq_diff_func, method='Bounded', bounds=[0, 50])
		opt_x.append(res.x)
		opt_nfev.append(res.nfev)

	return opt_x, opt_nfev






def show_pdfs_adj(pdf_b_val=2.5, bipolar=True, **pw_dist_kwargs):

	'''
	For graphing and comparing RF distance PDFs
	Graphs:
		* The actual generated PDFs for the random sampling process
		* The uncorrected PDF

	Parameters
	----------
	pdf_b_val : float
		Custom b value for the underlying PDF exponential
		default is 2.5, from jin et al
	bipolar : boolean
		Whether RFs are located in a bipolar fashion from a center
		default is True
	'''

	##
	# Generate actual distance PDF from random sampling and pairwise calculation
	##

	dist, prob_dist = jin_rf_dist(b = pdf_b_val, pdf=True)

	# Pair wise distances from PDF ... BIPOLAR
	pw_dist = pdf_to_pair_wise_dist(dist, prob_dist, samp_size=3000,
		bipolar=bipolar, **pw_dist_kwargs)

	# Hist of pair-wise distances ... BIPOLAR
	counts, bins = np.histogram(pw_dist, bins='doane')

	#make PDF
	# normalise counts to total counts to make PDF
	sim_pdf = counts / counts.sum().astype('float')

	# Centering Bin locations
	bin_locs = bins[:-1] + ((bins[1] - bins[0])/2.)

	##
	# Generate Theoretical PDF on basis of Jin et al
	##

	# uses bin locations from the actual sampled set of distances for equivalence
	d,p = jin_rf_dist(3.4, 2.5, x=bin_locs, pdf=True)


	with sns.plotting_context('poster', rc={'figure.figsize': [16., 8.]}):
		plt.subplot(121)

		plt.plot(dist, prob_dist, label='pdf')

		plt.plot(d, p, label='jinPwPDF')

		# plotting simulated PDf as histogram, normalised to toal counts to make PDF
		plt.plot(bin_locs, sim_pdf, label='pw_dist')

		plt.legend(loc='upper right', fontsize='xx-small')
		plt.xlim(xmax=2.5)
		plt.xlabel('RF distance')

		sns.despine(offset=20)

		plt.subplot(122)

		plt.plot(d, sim_pdf[:-1] - p, label='PW dist - theoretical')
		plt.legend(loc='upper right', fontsize='xx-small')
		plt.xlim(xmax=2.5)
		plt.xlabel('RF distance')

		sns.despine(offset=20)

		plt.tight_layout()


def make_rf_pdf(n=10, load=True, d_num=None, file_desc='', **pw_dist_kwargs):


	# curr_dir = os.getcwd()
	file_name_base = 'optimise_data'+file_desc

	existing_file_names = glob.glob(file_name_base+'*')
	# number_existing_opt_dats = str(len(existing_file_names)).zfill(2)
	number_existing_opt_dats = len(existing_file_names)
	# file_name = '%s_%s.pkl'\

	if load and number_existing_opt_dats > 0:

		assert (d_num is None) or (d_num <= number_existing_opt_dats), 'd_num inappropriate'

		if d_num is None:
			d_num = str(number_existing_opt_dats - 1).zfill(2); # as index is -1 from length
		else:
			d_num = str(d_num).zfill(2);


		with open('%s_%s.pkl'%(file_name_base, d_num), 'r') as f:
			opt_data = pickle.load(f)

	else:

		if load:
			print('no files to load!  \n ** Running optimisation **')


		opt_x, opt_n = minimise_exponent(n)

		opt_data = dict(opt_var = dict(b = np.mean(opt_x)),
						opt_results = dict(opt_x = opt_x, opt_n = opt_n),
			)

		saved_file_name = '%s_%s.pkl'%(file_name_base, str(number_existing_opt_dats).zfill(2))

		with open(saved_file_name, 'w') as f:
			pickle.dump(opt_data, f)

			print('Opt Dat File saved as %s'%saved_file_name)

	return opt_data
