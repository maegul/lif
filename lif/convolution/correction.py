"""
Adjust and correct the results of convolving Spatial and Temporal Filters
(from [the receptive field module][receptive_field.filters.filter_functions])
with a stimulus (from [the stimulus module][stimulus.stimulus])
to correct for resolution artefacts and F1 data recording issues from empirical data
"""

from textwrap import dedent
from typing import Optional, Dict, Tuple

import scipy.optimize as opt

from ..utils.units.units import (
    SpatFrequency, TempFrequency, ArcLength, Time, val_gen, scalar)
from ..utils import data_objects as do, settings, exceptions as exc
from ..receptive_field.filters.filter_functions import (
    mk_tq_tf_ft, mk_dog_sf_ft, mk_dog_sf_conv_amp, mk_tq_tf_conv_amp
    )
from ..receptive_field.filters import contrast_correction as cont_corr
from . import estimate_real_amp_from_f1 as est_amp


# >> Joining Separable Spat and Temp

def joint_spat_temp_f1_magnitude(
        temp_freqs: TempFrequency[scalar],
        spat_freqs_x: SpatFrequency[scalar], spat_freqs_y: SpatFrequency[scalar],
        tf: do.TQTempFilter, sf: do.DOGSpatialFilter,
        contrast: Optional[do.ContrastValue]=None,
        contrast_params: Optional[do.ContrastParams]=None,
        collapse_symmetry: bool = False,
        ) -> scalar:
    """Joint amplitude of separate TF and SF treated as separable

    Presumes TF and SF are sparable components of a single Spatia-Temporal
    filter.  TF and SF are defined by the tf and sf parameters.

    temp_res and spat_res represent the stimulus used for convolution.

    contrast controls contrast correction.  If not provided, temporal filter is adjusted
    to spatial filter's contrast (if they're different).
    If a value is provided, both filters are adjusted to this contrast.
    This correcting for contrast will affect the targetted F1 amplitude of the responses
    of the filters.

    `contrast_params` define the contrast curve to be used.
    If not provided, a default is taken from the `contrast_correction` module (typically `ON`)

    """

    # create default contrast value if necessary
    # default to using spatial filter's contrast
    if not contrast:
        contrast = do.ContrastValue(contrast=sf.source_data.resp_params.contrast)
    if not contrast_params:
        contrast_params = settings.simulation_params.contrast_params

    tf_contrast = do.ContrastValue(tf.source_data.resp_params.contrast)
    sf_contrast = do.ContrastValue(sf.source_data.resp_params.contrast)

    # static spat freq at which temp_filt measured
    tf_sf = tf.source_data.resp_params.sf
    # static temp freq at which spat_filt measured
    sf_tf = sf.source_data.resp_params.tf

    # find "intersection responses"
    # response of both filters at the other filter's static frequency
    norm_tf = mk_tq_tf_ft(sf_tf, tf.parameters)
    norm_sf = mk_dog_sf_ft(tf_sf, SpatFrequency(0), sf.parameters)

    # contrast corrected norm responses
    norm_tf_cc = cont_corr.correct_contrast_response_amplitude(
        response_amplitude=norm_tf,
        base_contrast=tf_contrast, target_contrast=contrast,
        contrast_params=contrast_params
        )
    norm_sf_cc = cont_corr.correct_contrast_response_amplitude(
        response_amplitude=norm_sf,
        base_contrast=sf_contrast, target_contrast=contrast,
        contrast_params=contrast_params
        )

    # norm amplitude ... average of the two intersection responses
    # The amplitude that is what all amps are normlised to: the mid-point or average
    norm_amp = (norm_tf_cc + norm_sf_cc) / 2

    # factors (filters' responses relative to their norm response)
    # these factors represent how much the actual response (in each domain, time/space)
    # varies from the intersection or norm response

    # responses
    actual_sf_conv_amp = mk_dog_sf_ft(
            spat_freqs_x, spat_freqs_y, sf.parameters,
            collapse_symmetry=collapse_symmetry)
    actual_tf_conv_amp = mk_tq_tf_ft(temp_freqs, tf.parameters)

    # contrast corrected responses
    actual_sf_conv_amp_cc = cont_corr.correct_contrast_response_amplitude(
        response_amplitude=actual_sf_conv_amp,
        base_contrast=sf_contrast, target_contrast=contrast,
        contrast_params=contrast_params
        )
    actual_tf_conv_amp_cc = cont_corr.correct_contrast_response_amplitude(
        response_amplitude=actual_tf_conv_amp,
        base_contrast=tf_contrast, target_contrast=contrast,
        contrast_params=contrast_params
        )

    # factors
    sf_factor = actual_sf_conv_amp_cc / norm_sf
    tf_factor = actual_tf_conv_amp_cc / norm_tf

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

    if (tf_dc is None):
        raise ValueError(f'Temp Filter has no DC value in source data ... source from dist?')
    if (sf_dc is None):
        raise ValueError(f'Temp Filter has no DC value in source data ... source from dist?')


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
        sf: do.DOGSpatialFilter, tf: do.TQTempFilter,
        contrast: Optional[do.ContrastValue]=None,
        contrast_params: Optional[do.ContrastParams]=None,
        ) -> do.JointSpatTempResp:
    """Joint responses of the spatial and temporal filters

    Returns F1 amplitude and DC values of "join" or "fusion" of filters (as object).
    """

    amplitude = joint_spat_temp_f1_magnitude(
        spat_freqs_x=grating_stim_params.spat_freq_x,
        spat_freqs_y=grating_stim_params.spat_freq_y,
        temp_freqs=grating_stim_params.temp_freq,
        sf=sf, tf=tf,
        contrast=contrast, contrast_params=contrast_params,
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
        tf: do.TQTempFilter,
        filter_actual_max_f1_amp: Optional[do.LGNF1AmpMaxValue] = None,
        target_max_f1_amp: Optional[do.LGNF1AmpMaxValue] = None,
        contrast_params: Optional[do.ContrastParams] = None,
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
    # Handle if max_f1 passed in or not
    if (
            (target_max_f1_amp or filter_actual_max_f1_amp) # at least one
            and not
            (target_max_f1_amp and filter_actual_max_f1_amp) # but not both
            ):
        raise ValueError('Need to pass BOTH target and actual max_f1_amp')

    if (target_max_f1_amp and filter_actual_max_f1_amp):
        max_f1_adj_factor = mk_max_f1_resp_adjustment_factor(
            target_max_f1_amp, filter_actual_max_f1_amp)
    else:  # if max f1 parameters not provided, just set to 1
        max_f1_adj_factor = None

    # Estimate of the convolution_amplitude after full sf and tf convolution with stim!!
    conv_resp_params = mk_estimate_sf_tf_conv_params(
        spacetime_params, grating_stim_params, sf, tf)

    # Target joint response of the filters (ie, theoretical, or target, response)
    joint_resp_params = mk_joint_sf_tf_resp_params(
        grating_stim_params, sf, tf,
        contrast=grating_stim_params.contrast,
        contrast_params=contrast_params,
        )

    # derive real unrectified amplitude
    # That is, amplitude that would produce the target F1 after rectification and FFT
    # ... allow any exception to bubble up
    # Max f1 adj factor is used here to ensure that the rectified FFT F1 amplitude compensation
    # accounts for this scaling.  If this scaling were done later, linearity could not be presumed
    # Default to 1 if not being used
    real_unrect_joint_amp_opt = est_amp.find_real_f1(
        DC_amp=joint_resp_params.DC,
        f1_target=(
            joint_resp_params.amplitude * (
                max_f1_adj_factor
                    if max_f1_adj_factor
                    else 1
                    )
            )
        )

    real_unrect_joint_amp: float = real_unrect_joint_amp_opt.x[0]

    # factor to adjust convolution result by to provide realistic amplitude and firing rate
    # after convolution with a stimulus
    amp_adjustment_factor = real_unrect_joint_amp / conv_resp_params.amplitude

    # have to shift conv_resp estimate to same scale as joint_resp
    # means that must apply DC shift after (re-)scaling the response
    dc_shift_factor = joint_resp_params.DC - (conv_resp_params.DC * amp_adjustment_factor)

    adjustment_params = do.ConvRespAdjParams(
        amplitude=amp_adjustment_factor,
        DC=dc_shift_factor,
        max_f1_adj_factor=max_f1_adj_factor,
        joint_response=joint_resp_params
        )

    return adjustment_params


# Best used outside of this module ahead of time but kept here for consistency
def mk_actual_filter_max_amp(
        sf: do.DOGSpatialFilter,
        tf: do.TQTempFilter,
        contrast: do.ContrastValue,
        contrast_params: Optional[do.ContrastParams] = None
        ) -> do.LGNActualF1AmpMax:

    # x: temp_freq (Hz), spat_freq (cpd)
    joint_f1_wrapper = lambda x: (
        # make negative to "minimize" to the maximal response
         -1 * joint_spat_temp_f1_magnitude(
            TempFrequency(x[0]), SpatFrequency(x[1]), SpatFrequency(0),
            tf, sf,
            contrast=contrast, contrast_params=contrast_params
            )
         # prevent negative values
         if all(v >= 0 for v in x)
            else 1e9
        )

    opt_result: opt.OptimizeResult = opt.minimize(joint_f1_wrapper, x0=[4, 1])
    if not opt_result['success']:
        # try a different method just in case
        opt_result: opt.OptimizeResult = opt.minimize(
            joint_f1_wrapper, x0=[4, 1], method='Nelder-Mead')
    if not opt_result['success']:
        raise exc.LGNError('Could not find maximal response of joint sf and tf filters')

    # value of function is the maximal response, but negative, so make positive
    max_response = do.LGNF1AmpMaxValue(max_amp = -1 * opt_result['fun'], contrast=contrast)
    temp_freq, spat_freq = do.TempFrequency(opt_result['x'][0]), do.SpatFrequency(opt_result['x'][1])
    actual_max = do.LGNActualF1AmpMax(max_response, temp_freq, spat_freq)

    return actual_max


def mk_max_f1_resp_adjustment_factor(
        target_f1_max_amp: do.LGNF1AmpMaxValue,
        actual_f1_max_amp: do.LGNF1AmpMaxValue
        ) -> float:
    """Generate ratio for scaling of F1 amplitude to that of target

    Checks that both amplitudes are scaled to the same contrast
    """

    # Check that contrasts are the same for both values
    # to ensure that they've been scaled for contrast to the same extent
    actual_cont, target_cont = (
        actual_f1_max_amp.contrast.contrast, target_f1_max_amp.contrast.contrast)
    if not (actual_f1_max_amp.contrast == target_f1_max_amp.contrast):
        raise exc.LGNError(
            f'F1 max amplitudes not scaled to the same contrast: {actual_cont=}, {target_cont=}')

    factor = target_f1_max_amp.max_amp / actual_f1_max_amp.max_amp

    return factor


def adjust_conv_resp(conv_resp: val_gen, adjustment: do.ConvRespAdjParams ) -> val_gen:

    adjusted_resp = (
        (conv_resp * (adjustment.amplitude))
        + adjustment.DC
        )

    return adjusted_resp
