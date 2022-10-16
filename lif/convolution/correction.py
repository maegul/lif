"""
Adjust and correct the results of convolving Spatial and Temporal Filters (from [the receptive field module][receptive_field.filters.filter_functions]) with a stimulus (from [the stimulus module][stimulus.stimulus]) to correct for resolution artefacts and F1 data recording issues from empirical data
"""

from ..utils.units.units import (
    SpatFrequency, TempFrequency, ArcLength, Time, val_gen, val_gen_flt)
from ..utils import data_objects as do, settings, exceptions as exc
from ..receptive_field.filters.filter_functions import (
    mk_tq_tf_ft, mk_dog_sf_ft, mk_dog_sf_conv_amp, mk_tq_tf_conv_amp
    )
from . import estimate_real_amp_from_f1 as est_amp

def mk_tq_tf_conv_amp(
        freqs: TempFrequency[val_gen],
        tf_params: do.TQTempFiltParams,
        temp_res: Time[float]) -> val_gen:
    """Amplitude of convolution with given filter

    If convolving a sinusoid of a given frequency with this tq temp filter
    as defined by the provided parameters, this will return the amplitude
    of the resultant sinusoid if the input sinusoid has amplitude 1

    Essentially same as mk_tq_tf_ft but divided by temp_res


    Parameters
    ----


    Returns
    ----

    """

    tf_amp = mk_tq_tf_ft(freqs, tf_params)

    return tf_amp / temp_res.s  # use s as this unit used by mk_tq_tf (needs to be made cleaner!!)


# >> Joining Separable Spat and Temp

def joint_spat_temp_conv_amp(
        temp_freqs: TempFrequency[val_gen],
        spat_freqs_x: SpatFrequency[val_gen], spat_freqs_y: SpatFrequency[val_gen],
        tf: do.TQTempFilter, sf: do.DOGSpatialFilter, collapse_symmetry: bool = False
        ) -> val_gen:
    """Joint amplitude of separate TF and SF treated as separable

    Presumes TF and SF are sparable components of a single Spatia-Temporal
    filter.  TF and SF are defined by the tf and sf parameters.

    temp_res and spat_res represent the stimulus used for convolution.

    Parameters
    ----


    Returns
    ----

    """

    # sf of temp_filt
    tf_sf = tf.source_data.resp_params.sf
    sf_tf = sf.source_data.resp_params.tf

    norm_tf = mk_tq_tf_ft(sf_tf, tf.parameters)
    norm_sf = mk_dog_sf_ft(tf_sf, SpatFrequency(0), sf.parameters)
    # The amplitude that is what all amps are normlised to
    # norm_factor will normalise all amps to 1
    # this 1 will represent norm_amp which is the average of the spat and temp
    # responses
    norm_amp = (norm_tf + norm_sf) / 2
    norm_factor = norm_tf * norm_sf

    joint_amp: val_gen = (  # type: ignore
        mk_tq_tf_ft(temp_freqs, tf.parameters) *
        mk_dog_sf_ft(
            spat_freqs_x, spat_freqs_y, sf.parameters,
            collapse_symmetry=collapse_symmetry) /
        norm_factor *  # normalise to intersection
        norm_amp  # bring to normalised amplitude - avg of intersection
        )

    return joint_amp


def joint_dc(tf: do.TQTempFilter, sf: do.DOGSpatialFilter) -> float:
    """Calculate joint DC of temp and spat filters by averaging
    (must be within 30% of eachother)

    """

    # Joint DC Amplitude
    tf_dc = tf.source_data.resp_params.dc
    sf_dc = sf.source_data.resp_params.dc

    if abs(tf_dc - sf_dc) > 0.3*min([tf_dc, sf_dc]):
        tf_desc = (
            tf.source_data.meta_data.make_key()
            if tf.source_data.meta_data is not None
            else tf.parameters)
        sf_desc = (
            sf.source_data.meta_data.make_key()
            if sf.source_data.meta_data is not None
            else sf.parameters)
        raise ValueError(
            f'DC amplitudes of Temp Filt and Spat Filt are too differente\n'
            f'filters: {tf_desc}, {sf_desc}'
            f'DC amps: TF: {tf_dc}, SF: {sf_dc}'
            )
    # Just take the average
    joint_dc = (tf_dc + sf_dc) / 2

    return joint_dc


def mk_joint_sf_tf_resp_params(
        grating_stim_params: do.GratingStimulusParams,
        sf: do.DOGSpatialFilter, tf: do.TQTempFilter
        ) -> do.JointSpatTempResp:

    amplitude = joint_spat_temp_conv_amp(
        spat_freqs_x=grating_stim_params.spat_freq_x,
        spat_freqs_y=grating_stim_params.spat_freq_y,
        temp_freqs=grating_stim_params.temp_freq,
        sf=sf, tf=tf
        )

    DC = joint_dc(tf, sf)

    resp_estimate = do.JointSpatTempResp(amplitude, DC)

    return resp_estimate


def mk_estimate_sf_tf_conv_params(
        spacetime_params: do.SpaceTimeParams,
        grating_stim_params: do.GratingStimulusParams,
        sf: do.DOGSpatialFilter,
        tf: do.TQTempFilter
        ) -> do.EstSpatTempConvResp:
    """Produce estimate/analytical amplitude of response after convolving stim with tf and sf

    Presumes that convolving both a spatial (sf) and temporal (tf) filter with a grating
    grating stimulus defined by grating_stim_params
    """

    # should be convolution_amplitude after full sf and tf convolution with stim!!
    sf_conv_amp = mk_dog_sf_conv_amp(
        freqs_x=grating_stim_params.spat_freq_x,
        freqs_y=grating_stim_params.spat_freq_y,
        dog_args=sf.parameters, spat_res=spacetime_params.spat_res
        )
    tf_conv_amp = mk_tq_tf_conv_amp(
        freqs=grating_stim_params.temp_freq,
        temp_res=spacetime_params.temp_res,
        tf_params=tf.parameters
        )

    convolution_amp = grating_stim_params.amplitude * sf_conv_amp * tf_conv_amp

    sf_dc = mk_dog_sf_conv_amp(
                SpatFrequency(0), SpatFrequency(0), sf.parameters, spacetime_params.spat_res
                )
    tf_dc = mk_tq_tf_conv_amp(TempFrequency(0), tf.parameters, spacetime_params.temp_res)

    # >>> DC is the amplitude at frequency `0` ... so multiply by this,
    # ... not amplitude, which is the amplitude at frequency `F1`.
    estimated_dc = grating_stim_params.DC * sf_dc * tf_dc

    conv_params = do.EstSpatTempConvResp(
        amplitude=convolution_amp,
        DC=estimated_dc
        )

    return conv_params


def mk_conv_resp_adjustment_params(
        spacetime_params: do.SpaceTimeParams,
        grating_stim_params: do.GratingStimulusParams,
        sf: do.DOGSpatialFilter,
        tf: do.TQTempFilter
        ) -> do.ConvRespAdjParams:
    """Factor to adjust amplitude of convolution to what filters dictate

    joint_spat_temp_conv_amp used to unify sf and tf
    stim attributes used to find appropriate factor for the expected output
    of convolution.

    Returned amplitude also adjusted for rectification effects.

    Args:
        spacetime_params: Parameters about space and time resolution and extent


    Returns:
        params: Adjustment parameters


    **Notes**

    Steps:

    1. Derive the amplitude of response of convolving TF.SF with stimulus.
        This is based on the parameters of the TF and SF and the amp of the stim.
    2. Derive the theoretical amplitude of a cell with the TF and SF at the freqs of the stimulus
    3. Derive the amplitude of an unrectified sin wave that would produce the above theoretical
        amplitude when rectified
    4. Produce factor that will normalise convolved amplitude to 1 and multiply by
        real unrectified amplitude necessary to produce theoretical.

    Thus, the factor returned is the theoretical amplitude / convolutional amplitude.

    """

    # should be convolution_amplitude after full sf and tf convolution with stim!!
    conv_resp_params = mk_estimate_sf_tf_conv_params(
        spacetime_params, grating_stim_params, sf, tf)

    # Joint response
    joint_resp_params = mk_joint_sf_tf_resp_params(grating_stim_params, sf, tf)

    # derive real unrectified amplitude
    real_unrect_joint_amp_opt = est_amp.find_real_f1(
        DC_amp=joint_resp_params.DC, f1_target=joint_resp_params.ampitude)

    real_unrect_joint_amp = real_unrect_joint_amp_opt.x[0]

    # factor to adjust convolution result by to provide realistic amplitude and firing rate
    # after convolution with a stimulus
    amp_adjustment_factor = real_unrect_joint_amp / conv_resp_params.amplitude

    # have to shift conv_resp estimate to same scale as joint_resp
    # means that must apply DC shift after (re-)scaling the response
    dc_shift_factor = joint_resp_params.DC - (conv_resp_params.DC * amp_adjustment_factor)

    adjustment_params = do.ConvRespAdjParams(
        amplitude=amp_adjustment_factor,
        DC=dc_shift_factor
        )

    return adjustment_params


def adjust_conv_resp(conv_resp: val_gen, adjustment: do.ConvRespAdjParams) -> val_gen:

    adjusted_resp = (conv_resp * adjustment.amplitude) + adjustment.DC

    return adjusted_resp
