"""Making temporal and spatial filters"""

# %%%%%%%%%%%
from typing import Union, Optional, Sequence, Iterable

from dataclasses import dataclass
# -----------
import numpy as np
from scipy.optimize import least_squares, curve_fit
# -----------

# %%%%%%%%%%%
from . import filter_functions as ff
# -----------
# %%%%%%%%%%%
# -----------
# %%%%%%%%%%%
import datetime as dt
from pathlib import Path

# -----------
# %%%%%%%%%%%

seq_num = Union[np.ndarray, Sequence[Union[int, float]]]
# seq_num = Iterable[Union[int, float, np.number]]

@dataclass
class RespParams:
    dc: float
    sf: float
    mean_lum: float
    contrast: float


@dataclass
class MetaData:
    author: Optional[str] = None
    year: Optional[int] = None
    title: Optional[str] = None
    doi: Optional[str] = None


@dataclass
class TempFiltData:
    freq: seq_num
    amplitude: seq_num
    resp_params: RespParams
    meta_data: MetaData

# -----------
# %%%%%%%%%%%
import receptive_field.filters.filter_functions as ff
import matplotlib.pyplot as plt
import numpy as np
# -----------
# %%%%%%%%%%%
def tq_ft_wrapper(
        x: np.ndarray, 
        freqs: np.ndarray, amplitude_real: np.ndarray
        ) -> np.float64:

    A, tau, w, phi = x[0], x[1], x[2], x[3]
    fit_values = A * ff.mk_tq_ft(freqs*2*np.pi, tau=tau, w=w, phi=phi)

    # return np.sum((fit_values - amplitude_real)**2)
    return fit_values - amplitude_real
# -----------
# %%%%%%%%%%%
def tq_ft_wrapper2(
        freqs: np.ndarray,
        A, tau, w, phi
        ) -> np.float64:

    fit_values = A * ff.mk_tq_ft(freqs*2*np.pi, tau=tau, w=w, phi=phi)

    # return np.sum((fit_values - amplitude_real)**2)
    return fit_values
# -----------
# %%%%%%%%%%%
fs = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64])
amps = np.array([32, 30, 34, 40, 48, 48, 28, 20, 3])
# -----------
# %%%%%%%%%%%
opt_res = least_squares(tq_ft_wrapper, np.array([20, 16, 4*2*np.pi, 0.24]), 
    bounds = (
        np.array([0, 1, np.pi, 0]), 
        np.array([amps.max()*2*3, 150, 20*np.pi, 2])),
    # guesses: 
    # amp: double to 1 then 3 times max data
    # tau: 150ms ... no visual system temp filt should be longer than this ?!
    # w: 
    # max_nfev=2000,
    kwargs = dict(freqs=fs, amplitude_real=amps))
# -----------
# %%%%%%%%%%%
plt.clf()
# -----------
# %%%%%%%%%%%
plt.plot(fs, amps)
# -----------
# %%%%%%%%%%%
plt.plot(fs, opt_res.x[0]*ff.mk_tq_ft(fs*2*np.pi, *opt_res.x[1:]))
# -----------

def gen_tq_temp_filt(data: TempFiltData):
    """Generate a tq temp filt object using data

    """

    def tq_ft_wrapper(
            x: np.ndarray, 
            freqs: np.ndarray, amplitude_real: np.ndarray
            ) -> float:

        tau, w, phi = x[0], x[1], x[2]
        fit_values = ff.mk_tq_ft(freqs, tau=tau, w=w, phi=phi)

        return np.sum((fit_values - amplitude_real)**2)

    opt_result = least_squares(tq_ft_wrapper, np.array([16, 4*2*np.pi, 0.24]))

    return opt_result



np.linspace()