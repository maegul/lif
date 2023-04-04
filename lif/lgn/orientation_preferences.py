# # Imports
from typing import List, cast, Protocol, Any, Optional, overload, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.optimize import least_squares

import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as psp

from ..utils import data_objects as do
from ..utils.units.units import (
    val_gen, scalar,
    ArcLength
    )
from ..receptive_field.filters import cv_von_mises as cvvm


# # Orientations

# +
def _mk_angle_probs(vm_params: do.VonMisesParams, n_angle_increments: int = 1000):

    angles = ArcLength(
        np.linspace(0, 180, n_angle_increments, endpoint=False))
    probs = cvvm.von_mises(
        angles,
        k=vm_params.k,
        phi=vm_params.phi)
    probs /= probs.sum()

    return angles, probs

# -

# ### Demo
# +
# mk_orientations(
#     20,
#     *_mk_angle_probs(do.VonMisesParams.from_circ_var(
#         phi=ArcLength(10, 'deg'),
#         cv=0.5  # converts circ_var to kappa value
#         ))
#     )
# -


# # Circular Variance Distribution

# ## Core functions and dataclasses

#

# +
def pdf_probabilities(
        dist: do.DistProtocol,
        bins: np.ndarray
        ) -> np.ndarray:

    bin_density_vals = dist.cdf(bins)
    hist_vals = bin_density_vals[1:] - bin_density_vals[:-1]
    return hist_vals


def gamma_probabilities(
        bins: np.ndarray, a: float, beta: float
        ) -> np.ndarray:
    """Integrates between bin values to calculate probabilities from gamma PDF

    Args:
        bins: array of lower and upper bounds that share all inner boundaries.
        a: chief parameter to `gamma` distribution
        beta: scale parameter passed in as `scale = 1/beta`.

    Notes:
        * Uses the `cdf` command to calculate probabilities
        * Done by subtracting the lower bound values of the CDF from the upper bound vals
    """
    dist = stats.gamma(a=a, scale=1/beta)

    probabilities = pdf_probabilities(dist, bins)
    return probabilities
# -
# +
def cv_hist_gamma_residuals(x, circvar_hist_data: do.CircVarHistData):

    a, beta = x

    probabilities = gamma_probabilities(circvar_hist_data.hist_bins, a=a, beta=beta)
    residuals = probabilities - circvar_hist_data.probs  # type: ignore
    return residuals
# -


# Get Distributions fron Naito, Shou and anywhere else a definition of circ_var
# ... has been taken from

# ## Data Objects


# ## Raw Data and fits to Distribution

# +
_naito_lg_highsf = do.CircVarHistData(
    hist_mp = np.array([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75]),
    count=np.array([11.330471657795853,36.16595133510011,14.306153877245706,19.34192185265844,
        3.204583430949195,2.288986967462225,1.2589440894807837,1.2589440894807837])
    )
# -
# +
_naito_opt_highsf = do.CircVarHistData(
    hist_mp = np.array([0.05,0.15,0.25,0.35,0.45,0.55]),
    count = np.array([36.165952756169524,21.173107229077253,20.257510729613735,8.240343347639485,
        1.2589441389484979,1.2589441389484979])
    )
# -
# +
_naito_lg_optsf = do.CircVarHistData(
    hist_mp = np.array([0.05,0.15,0.25,0.35,0.45,0.55,0.65]),
    count = np.array([67.35664375206306,12.30769498680405,5.4825147274736965,1.4545427166938294,
        1.3426573369266224, 0, 1.4545427166938294]),
    )
# -
# +
_shou_x_cells = do.CircVarHistData(
    hist_mp= np.array([0.025,0.075,0.125, 0.175,0.225,0.275, 0.325,0.375,0.425, 0.475,0.525]),
    count = np.array([49.63351155757312,117.17277509497397,124.71204150837673,
        90.78533689606057,59.68586294077425,24.81675577878656,11.62305105933935,
        8.79582615431332,9.42408301675343,3.141376344261262,1.570688172130631])
    )
# -
# +
# fig = (
#     go.Figure()
#     .add_bar(
#         x=_naito_lg_optsf.hist_mp, y=_naito_lg_optsf.probs, name='large',
#         width=0.1,
#         offset=-0.05
#         )
#     )
# fig.show()
# -
# +
# fig = (
#     go.Figure()
#     .add_bar(
#         x=_naito_lg_highsf.hist_mp, y=_naito_lg_highsf.probs, name='large',
#         width=0.1,
#         offset=-0.05
#         )
#     )
# fig.show()
# fig = (
#     go.Figure()
#     .add_bar(
#         x=_naito_opt_highsf.hist_mp, y=_naito_opt_highsf.probs, name='opt',
#         width=0.1, offset=0.05
#         )
#     )
# fig.show()
# # -

# ## Final Distribution Objects

# +
circ_var_distributions = do.AllCircVarianceDistributions(
    naito_lg_highsf=do.CircVarianceDistribution(
        name='naito_large_stim_highsf', source='Naito_et_al_2013', specific='fig2.4',
        distribution=stats.gamma(a=2.97082993, scale= 1/13.50320706),
        raw_data = _naito_lg_highsf
        ),
    naito_opt_highsf=do.CircVarianceDistribution(
        name='naito_opt_size_highsf', source='Naito_et_al_2013', specific='fig2.3',
        distribution=stats.gamma(a=1.1910443, scale=1/6.6238885),
        raw_data = _naito_opt_highsf
        ),
    naito_lg_optsf=do.CircVarianceDistribution(
        name='naito_large_size_optsf', source='Naito_et_al_2013', specific='fig2.2',
        distribution=stats.gamma(a=0.45080026, scale=1/5.95344415),
        raw_data = _naito_lg_optsf
        ),
    shou_xcells=do.CircVarianceDistribution(
        name='shou_x_cells', source='shou_leventhal_1989', specific='fig3A',
        distribution=stats.gamma(a=2.77538677, scale=1/18.48129056),
        raw_data = _shou_x_cells
        )
    )
# -



# ## Plotting
# +
def plot_circ_var_distribution(cvdist: do.CircVarianceDistribution):

    bins = cvdist.raw_data.hist_bins
    probabilities = pdf_probabilities(cvdist.distribution, bins=bins)
    bins_width = bins[1]-bins[0]

    x = np.linspace(bins[0], bins[-1], 1000)
    y = cvdist.distribution.pdf(x)

    n_samples = 3000
    samples = cvdist.distribution.rvs(size=n_samples)
    sample_hist, sample_bins = np.histogram(samples, density=True)
    sample_bin_width = sample_bins[1]-sample_bins[0]

    fig = (
        psp.make_subplots(rows=2, cols=1, shared_xaxes=True)

        .add_bar(
            x=cvdist.raw_data.hist_mp, y=cvdist.raw_data.probs, name='raw_data',
            width=bins_width, offset=-(bins_width/2),
            row=1, col=1
            )
        .add_scatter(
            x=cvdist.raw_data.hist_mp, y=probabilities, name='fit gamma dist',
            row=1, col=1
            )

        .add_bar(
            x=sample_bins, y=sample_hist, name='random samples (gamma)',
            width=(sample_bin_width),
            offset=(0),  # use lower bound values as lower edge of bar
            row=2, col=1
            )
        .add_scatter(
            x=x, y=y, name='pdf',
            row=2, col=1
            )
        .update_layout(title=cvdist.name)

        )

    return fig
# -
# +
def plot_all_circ_var_distributions(only_return_figs: bool = False):

    figs = []
    for cvd_name in circ_var_distributions.__dataclass_fields__.keys():
        cvd = circ_var_distributions.get_distribution(cvd_name)
        fig = plot_circ_var_distribution(cvd)
        if only_return_figs:
            figs.append(fig)
            continue
        else:
            fig.show()

    if only_return_figs:
        return figs
# -
# +
# plot_circ_var_distribution(circ_var_distributions.naito_lg_highsf).show()
# plot_circ_var_distribution(circ_var_distributions.naito_opt_highsf).show()
# plot_circ_var_distribution(circ_var_distributions.naito_lg_optsf).show()
# plot_circ_var_distribution(circ_var_distributions.shou_xcells).show()
# -

# ## Manual optimisation of Gamma to distributions

# should only need to be done once with results encoded in objects above

# ### naito_lg_highsf
# +
# opt = least_squares(
#         fun=cv_hist_gamma_residuals, x0=[1, 4], bounds=([0,0], [np.inf,np.inf]),
#         args=(_naito_lg_highsf,))
# -
# +
# opt
 # active_mask: array([0, 0])
 #        cost: 0.0106511351138501
 #         fun: array([ 0.03235112, -0.05137971,  0.10017752, -0.08330356,  0.02154867,
 #       -0.00337715, -0.00604571, -0.01133495])
 #        grad: array([ 1.14901802e-07, -2.07871399e-08])
 #         jac: array([[-0.16683533,  0.02405576],
 #       [-0.07946383,  0.02481974],
 #       [ 0.07306071, -0.00662892],
 #       [ 0.08285192, -0.01651144],
 #       [ 0.05027538, -0.01279339],
 #       [ 0.02402919, -0.00717687],
 #       [ 0.01006747, -0.00340309],
 #       [ 0.00387617, -0.00145169]])
 #     message: '`ftol` termination condition is satisfied.'
 #        nfev: 10
 #        njev: 10
 #  optimality: 3.4135371175654864e-07
 #      status: 2
 #     success: True
 #           x: array([ 2.97082993, 13.50320706])
# -

# ### naito_opt_highsf
# +
# opt = least_squares(
#         fun=cv_hist_gamma_residuals, x0=[1, 4], bounds=([0,0], [np.inf,np.inf]),
#         args=(_naito_opt_highsf,))
# -
# +
# opt
#  active_mask: array([0, 0])
#         cost: 0.0039718580278766645
#          fun: array([-0.01303485,  0.02859402, -0.07604946, -0.0088665 ,  0.031449  ,
#         0.01024776])
#         grad: array([-1.38965113e-07,  3.51131913e-08])
#          jac: array([[-0.43390499,  0.05177069],
#        [ 0.06563662,  0.00917628],
#        [ 0.11928282, -0.01001237],
#        [ 0.09481395, -0.01393912],
#        [ 0.0630219 , -0.01211258],
#        [ 0.03876414, -0.00894109]])
#      message: '`ftol` termination condition is satisfied.'
#         nfev: 9
#         njev: 9
#   optimality: 2.325858640845238e-07
#       status: 2
#      success: True
           # x: array([1.1910443, 6.6238885])
# -

# ### naito_lg_optsf
# +
# opt = least_squares(
#         fun=cv_hist_gamma_residuals, x0=[1, 4], bounds=([0,0], [np.inf,np.inf]),
#         args=(_naito_lg_optsf,))
# -
# +
# opt
 # active_mask: array([0, 0])
 #        cost: 0.00016636011320908477
 #         fun: array([ 0.00040864,  0.00109219, -0.00458465,  0.00958108, -0.00263859,
 #        0.00610347, -0.0132034 ])
 #        grad: array([7.13352601e-10, 3.23354748e-10])
 #         jac: array([[-0.59012648,  0.03731321],
 #       [ 0.28320999, -0.00919341],
 #       [ 0.14709441, -0.00950581],
 #       [ 0.07604468, -0.00692957],
 #       [ 0.03960359, -0.00456016],
 #       [ 0.02077051, -0.00285964],
 #       [ 0.010956  , -0.00174401]])
 #     message: '`gtol` termination condition is satisfied.'
 #        nfev: 9
 #        njev: 9
 #  optimality: 1.925074432787063e-09
 #      status: 1
 #     success: True
 #           x: array([0.45080026, 5.95344415])
# -


# ### Shou Leventhal X Cells
# +
# opt = least_squares(
#         fun=cv_hist_gamma_residuals, x0=[1, 4], bounds=([0,0], [np.inf,np.inf]),
#         args=(_shou_x_cells,))
# -
# +
# opt
#  active_mask: array([0, 0])
#         cost: 0.0003339328119984593
#          fun: array([-0.00802029,  0.0099978 , -0.00432316, -0.00278926, -0.00773125,
#         0.01388293,  0.01077012, -0.00012706, -0.01014752, -0.00207683,
#        -0.0011445 ])
#         grad: array([-1.85818153e-08,  1.20112288e-09])
#          jac: array([[-0.12198572,  0.01050242],
#        [-0.12339083,  0.01803718],
#        [-0.00152683,  0.00636262],
#        [ 0.05893798, -0.00412069],
#        [ 0.06493506, -0.00808619],
#        [ 0.04977907, -0.00775436],
#        [ 0.03237762, -0.00584466],
#        [ 0.01911765, -0.00386635],
#        [ 0.01058355, -0.00235158],
#        [ 0.0055942 , -0.00134792],
#        [ 0.00285553, -0.00073909]])
#      message: '`ftol` termination condition is satisfied.'
#         nfev: 8
#         njev: 8
#   optimality: 2.2198300900314285e-08
#       status: 2
#      success: True
#            x: array([ 2.77538677, 18.48129056])
# -
