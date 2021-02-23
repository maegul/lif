import lif.receptive_field.filters.filters as filters
import numpy as np


t = filters.TQTempFiltParams(
    amplitude=44, 
    arguments=filters.TQTempFiltArgs(
        tau=15, w=3, phi=0.3))


def test_tq_params_dict_conversion():
    putative_dict = {
        'amplitude': t.amplitude,
        'tau': t.arguments.tau,
        'w': t.arguments.w,
        'phi': t.arguments.phi
    }

    assert putative_dict == t.to_flat_dict()


def test_tq_params_array_round_trip():
    assert t == t.from_iter(t.to_array())
    

def test_fit_tq_temp_filt():

    # mock data known to lead to a fit
    fs = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64])
    amps = np.array([32, 30, 34, 40, 48, 48, 28, 20, 3])

    data = filters.TempFiltData(frequencies=fs, amplitudes=amps)
    opt_res = filters._fit_tq_temp_filt(data)
    
    assert opt_res.success == True  # noqa: E712
