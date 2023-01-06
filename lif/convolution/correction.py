"""
Adjust and correct the results of convolving Spatial and Temporal Filters
(from [the receptive field module][receptive_field.filters.filter_functions])
with a stimulus (from [the stimulus module][stimulus.stimulus])
to correct for resolution artefacts and F1 data recording issues from empirical data
"""

from textwrap import dedent
from ..utils.units.units import (
    SpatFrequency, TempFrequency, ArcLength, Time, val_gen)
from ..utils import data_objects as do, settings, exceptions as exc
from ..receptive_field.filters.filter_functions import (
    mk_tq_tf_ft, mk_dog_sf_ft, mk_dog_sf_conv_amp, mk_tq_tf_conv_amp
    )
from . import estimate_real_amp_from_f1 as est_amp


# >> Joining Separable Spat and Temp

def joint_spat_temp_f1_magnitude(
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

    # static spat freq at which temp_filt measured
    tf_sf = tf.source_data.resp_params.sf
    # static temp freq at which spat_filt measured
    sf_tf = sf.source_data.resp_params.tf

    # find "intersection responses"
    # response of both filters at the other filter's static frequency
    norm_tf = mk_tq_tf_ft(sf_tf, tf.parameters)
    norm_sf = mk_dog_sf_ft(tf_sf, SpatFrequency(0), sf.parameters)

    # norm amplitude ... average of the two intersection responses
    # The amplitude that is what all amps are normlised to: the mid-point or average
    norm_amp = (norm_tf + norm_sf) / 2

    # factors (filters' responses relative to their norm response)
    # these factors represent how much the actual response (in each domain, time/space)
    # varies from the intersection or norm response
    sf_factor = (
        mk_dog_sf_ft(
            spat_freqs_x, spat_freqs_y, sf.parameters,
            collapse_symmetry=False)
        /
        norm_sf
    )
    tf_factor = (
        mk_tq_tf_ft(temp_freqs, tf.parameters)
        /
        norm_tf
    )

    # Now use both factor multiplicatively, as presuming that both filters are linearly separable
    # Each factor "moves" the norm amplitude by however much the actual spatial or temporal freq
    # elicits a response greater/lesser than that of the intersection point
    joint_amp = norm_amp * sf_factor * tf_factor

    return joint_amp


def joint_dc(tf: do.TQTempFilter, sf: do.DOGSpatialFilter) -> float:
    """Calculate joint DC of temp and spat filters by averaging

    The DC Values must be within 30% of eachother

    """

    # Joint DC Amplitude
    tf_dc = tf.source_data.resp_params.dc
    sf_dc = sf.source_data.resp_params.dc

    if abs(tf_dc - sf_dc) > (0.3 * min([tf_dc, sf_dc])):
        tf_desc = (
            tf.source_data.meta_data.make_key()
            if tf.source_data.meta_data is not None
            else tf.parameters)
        sf_desc = (
            sf.source_data.meta_data.make_key()
            if sf.source_data.meta_data is not None
            else sf.parameters)
        raise ValueError(dedent(f'''
            DC amplitudes of Temp Filt and Spat Filt are too differente\n
            filters: {tf_desc}, {sf_desc}
            DC amps: TF: {tf_dc}, SF: {sf_dc}
            ''')
            )
    # Just take the average
    joint_dc = (tf_dc + sf_dc) / 2

    return joint_dc


def mk_joint_sf_tf_resp_params(
        grating_stim_params: do.GratingStimulusParams,
        sf: do.DOGSpatialFilter, tf: do.TQTempFilter
        ) -> do.JointSpatTempResp:
    """Joint responses of the spatial and temporal filters

    Returns F1 amplitude and DC values of "join" or "fusion" of filters (as object).
    """

    amplitude = joint_spat_temp_f1_magnitude(
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

    Presumes that convolving both a spatial (sf) and temporal (tf) filter with a
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

    # Estimate of the convolution_amplitude after full sf and tf convolution with stim!!
    conv_resp_params = mk_estimate_sf_tf_conv_params(
        spacetime_params, grating_stim_params, sf, tf)

    # Target joint response of the filters (ie, theoretical, or target, response)
    joint_resp_params = mk_joint_sf_tf_resp_params(grating_stim_params, sf, tf)

    # derive real unrectified amplitude
    # That is, amplitude that would produce the target F1 after rectification and FFT
    # ... allow any exception to bubble up
    real_unrect_joint_amp_opt = est_amp.find_real_f1(
        DC_amp=joint_resp_params.DC, f1_target=joint_resp_params.amplitude)

    real_unrect_joint_amp: float = real_unrect_joint_amp_opt.x[0]

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
