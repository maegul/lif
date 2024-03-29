
# ## Imports

# +
from typing import Optional
import copy

import numpy as np
from numpy import fft

from scipy.signal import convolve as signal_convolve

# from lif import *
from lif.convolution import (
    correction,
    convolve,
    estimate_real_amp_from_f1 as est_amp
    )
from lif.utils import data_objects as do
from lif.utils import settings
from lif.utils.units.units import ArcLength, SpatFrequency, Time, TempFrequency, scalar
from lif.receptive_field.filters import (
    filters,
    contrast_correction as cont_corr,
    filter_functions as ff
    )
from lif.convolution import estimate_real_amp_from_f1 as est_f1

from lif.stimulus import stimulus

from lif.plot import plot

import plotly.express as px
import plotly.graph_objects as go
# -

# ## Boilerplate for NBs

# +
import os

GLOBAL_ENV_VARS = {
    'WRITE_FIG': True,  # whether to write new figures
    'SHOW_FIG': False,  # whether to show new figures
    'RUN_LONG': False,  # whether to run long tasks
}

print('***\nSetting Env Variables\n***\n')
for GEV, DEFAULT_VALUE in GLOBAL_ENV_VARS.items():
    runtime_value = os.environ.get(GEV)  # defaults to None
    # parse strings into booleans, but only if actual value provided
    if runtime_value:
        new_value = (
                True
                    if runtime_value == "True"
                    else False
                )
    # if none provided, just take default value
    else:
        new_value = DEFAULT_VALUE
    print(f'Setting {GEV:<10} ... from {str(DEFAULT_VALUE):<5} to {str(new_value):<5} (runtime value: {runtime_value})')
    GLOBAL_ENV_VARS[GEV] = new_value

def show_fig(fig):
    if GLOBAL_ENV_VARS['SHOW_FIG']:
        fig.show()

def write_fig(fig, file_name: str, **kwargs):
    if GLOBAL_ENV_VARS['WRITE_FIG']:
        fig.write_image(file_name, **kwargs)
# -
# ## Load filters

# * These are loaded from file, having been previously fit to data

# +
# tf = do.TQTempFilter.load(do.TQTempFilter.get_saved_filters()[0])
# sf = do.DOGSpatialFilter.load(do.DOGSpatialFilter.get_saved_filters()[0])

tf = filters.temporal_filters['kaplan87']
sf = filters.spatial_filters['maffei73_2right']
# -

# ## Space, time and stimulus parameters

# +
stim_amp=17
stim_DC=-11
spat_res=ArcLength(1, 'mnt')
spat_ext=ArcLength(120, 'mnt')
temp_res=Time(1, 'ms')
temp_ext=Time(1000, 'ms')

orientation = ArcLength(90, 'deg')
temp_freq = TempFrequency(8)
spat_freq_x = SpatFrequency(2)
spat_freq_y = SpatFrequency(0)
# -
# +
st_params = do.SpaceTimeParams(spat_ext, spat_res, temp_ext, temp_res)
stim_params = do.GratingStimulusParams(
    spat_freq_x, temp_freq,
    orientation=orientation,
    amplitude=stim_amp, DC=stim_DC
)
# -

# ## Process of Correcting convolution amplitude
#
# ### The Problem
#
# * The problem in need of correction is that the responses of spatial and temporal filters taken from the literature are the magnitudes of the `F1` frequencies in the actual response (or PSTH)
# * The `F1` magnitude of a neuronal response is not simply the `amplitude` of a sinusoid at the temporal frequency of the stimulus.
# * It is in many ways an artefact of the (half-wave) rectification of said sinusoid and the `DC` shift in the response where the "energy" of the sinusoid leaks from the `F1` `amplitude` into the `DC` as more of the sinusoid is below 0 and removed by rectification.
# * Essentially, **rectification conflates `F1` and `DC`**.
# * Additionally, the actual `F1` and `DC` shift that the spatial and temporal filters we have derived from the literature will produce when convolved with a sinusoidal (drifting grating) and then rectified **will be different from what the filters are prescribing**.  This is for the same reason that rectification with `DC` shifts conflate `DC` and `F1`?
# * *Fortunately*, Sinusoidal `amplitude` (ie `F1`) can be corrected by multiplying by a factor and `DC` shifts can be corrected by addition/subtraction.
# * Thus, the process for ensuring that the result of convolution with our spatial and temporal filters is a sinusoid with the correct `amplitude` and DC shift:
#     1. Determine the `amplitude` and `DC` that the filters *will actually generate*.
#         - This depends on the fourier of the filter and the resolution with which the filter and the stimulus are rendered.
#     2. Determine what `amplitude` and `DC` *they should generate*.  
#         - These values must be obtained by "reverse engineering" the process of rectifying a sinusoid.  That is, the required values must produce the `amplitude` and `DC` values prescribed by the filters **once the sinusoid has been rectified**.
#     3. Provide correction values and apply them after actual convolution.


# ### What will convolution with filters actually produce

# * The `amplitude` of the sinusoid will be the amplitude of filter's fourier transform at the frequency of the signal being convolved then divided by the resolution of the rendering of the filter and the signal.
#     - Where for 2D spatial convolution must be divided by the square of the resolution.
# * This is presuming an input signal sinusoid of amplitude 1.  For other amplitudes multiply the above by the amplitude of the input signal.
# * The `DC` is similar.  The fourier as above but at a frequency of `0` (the frequency of a `DC` shift), divided by the resolution, *multiplied by the signal's `DC` shift*.

# Will use `scipy` convolution

# +
from scipy.signal import convolve as signal_convolve
# -

# #### Temporal Demonstration and Test

# Prepare time coords, temp filter and sinusoid

# +
time_coords = ff.mk_temp_coords(st_params.temp_res, st_params.temp_ext)
signal = est_amp.gen_sin(
    amplitude=stim_params.amplitude,
    DC_amp=stim_params.DC,
    time=time_coords,
    freq=stim_params.temp_freq
    )
temp_filter = ff.mk_tq_tf(time_coords, tf.parameters)
signal_conv = signal_convolve(signal, temp_filter)[:time_coords.value.size]
# -

# Plotting if desired

# +
px.line(x=time_coords.ms, y=signal).show()
px.line(x=time_coords.ms, y=temp_filter).show()
px.line(signal_conv).show()
# -

# Make estimations to test amplitude of convolution can be estimated
# Essential processes are

# * obtain fourier values at the frequency of the signal
# * to divide by the resolution used for the filter and stimuli
#     * Using the same unit used in the fourier calculation is important here (unfortunately not made into a clean interface in this code base)
# * Multiply by the *amplitude of the stimulus at the frequency for which the fourier value has been obtained.*
#     * This is *crucial* to getting the `DC` value correct.
#     * Even for a pure sinusoid, there are two frequencies present in the signal with potentially non-zero amplitudes: the primary or `F1` frequency, and the `DC` or `0` frequency.

# +
estimated_amplitude = (
                      stim_params.amplitude *
                      ff.mk_tq_tf_ft(stim_params.temp_freq, tf.parameters) /
                      st_params.temp_res.s
                      )
estimated_dc = (
               # #### ! Multiply by the DC ... not the `F1` amplitude
               stim_params.DC *
               ff.mk_tq_tf_ft(TempFrequency(0), tf.parameters) /
               st_params.temp_res.s
               )

# remove artefacts from the time constant and "ramping-up" at the beginning of convolution
stable_conv = signal_conv[signal_conv.size//2:]

# amplitude is half of total min to maximum
actual_amplitude = (stable_conv.max()-stable_conv.min())/2
# DC is halfway point between min and max ... or max minus amplitude
actual_DC = stable_conv.max() - actual_amplitude

print('~~~~~\nActual values and estimated values with percentage errors ...\n')
print(f'est_amp: {estimated_amplitude:.3f}, actual: {actual_amplitude:.3f}')
print(f'Error amplitude: {(abs(estimated_amplitude- actual_amplitude)/actual_amplitude):.3%}')

print(f'est_DC: {estimated_dc:.3f}, actual: {actual_DC:.3f}')
print(f'Error DC: {(abs(estimated_dc- actual_DC)/actual_DC):.3%}')
# -

# est_amp: 920658.219, actual: 919911.616
# Error: 0.081%
# est_amp: -365434.224, actual: -364963.134
# Error: -0.129%

# * simplified functions

# +
estimated_amplitude = (
    stim_params.amplitude *
    ff.mk_tq_tf_conv_amp(stim_params.temp_freq, tf.parameters, st_params.temp_res)
    )
estimated_dc = (
    stim_params.DC *
    ff.mk_tq_tf_conv_amp(TempFrequency(0), tf.parameters, st_params.temp_res)
    )

print(f'est_amp: {estimated_amplitude:.3f}, actual: {actual_amplitude:.3f}')
print(f'Error: {(abs(estimated_amplitude- actual_amplitude)/actual_amplitude):.3%}')

print(f'est_amp: {estimated_dc:.3f}, actual: {actual_DC:.3f}')
print(f'Error: {(abs(estimated_dc- actual_DC)/actual_DC):.3%}')
# -

# * Demonstration with actual stimulus

# +
stim_array = stimulus.mk_sinstim(st_params, stim_params)
signal = stim_array[0,0,:]
# -
# +
signal_conv = signal_convolve(signal, temp_filter)[:time_coords.value.size]
# -

# +
estimated_amplitude = (
                      stim_params.amplitude *
                      ff.mk_tq_tf_ft(stim_params.temp_freq, tf.parameters) /
                      st_params.temp_res.s
                      )
estimated_dc = (
               # #### ! Multiply by the DC ... not the `F1` amplitude
               stim_params.DC *
               ff.mk_tq_tf_ft(TempFrequency(0), tf.parameters) /
               st_params.temp_res.s
               )

# remove artefacts from the time constant and "ramping-up" at the beginning of convolution
stable_conv = signal_conv[signal_conv.size//2:]

# amplitude is half of total min to maximum
actual_amplitude = (stable_conv.max()-stable_conv.min())/2
# DC is halfway point between min and max ... or max minus amplitude
actual_DC = stable_conv.max() - actual_amplitude

print('~~~~~\nActual values and estimated values with percentage errors ...\n')
print(f'est_amp: {estimated_amplitude:.3f}, actual: {actual_amplitude:.3f}')
print(f'Error amplitude: {(abs(estimated_amplitude- actual_amplitude)/actual_amplitude):.3%}')

print(f'est_DC: {estimated_dc:.3f}, actual: {actual_DC:.3f}')
print(f'Error DC: {(abs(estimated_dc- actual_DC)/actual_DC):.3%}')
# -


# #### Spatial Demonstration

# +
stim_amp=17
stim_DC=-11
spat_res=ArcLength(1, 'mnt')
# trying to fit a small spat filt in
spat_ext=ArcLength(250, 'mnt')
temp_res=Time(1, 'ms')
temp_ext=Time(1000, 'ms')

orientation = ArcLength(90, 'deg')
temp_freq = TempFrequency(8)
spat_freq_x = SpatFrequency(2)
spat_freq_y = SpatFrequency(0)
# -
# +
st_params = do.SpaceTimeParams(
    spat_ext, spat_res, temp_ext, temp_res,
    array_dtype='float32')
stim_params = do.GratingStimulusParams(
    spat_freq_x, temp_freq,
    orientation=orientation,
    amplitude=stim_amp, DC=stim_DC
)
# -


# +
xc, yc = ff.mk_spat_coords(spat_res=st_params.spat_res, spat_ext=st_params.spat_ext)
spat_filt = ff.mk_dog_sf(xc, yc, dog_args=sf)
# -
# +
stim_array = stimulus.mk_sinstim(st_params, stim_params)
# -
# +
# convolution for a single spatial filter location ... done manually
signal_conv = (spat_filt[..., np.newaxis] * stim_array).sum(axis=(0, 1))
# signal_conv = signal_convolve(signal, temp_filter)[:time_coords.value.size]
# -
show_fig(px.line(signal_conv))

# +
estimated_amplitude = (
                      stim_params.amplitude *
                      ff.mk_dog_sf_ft(
                        stim_params.spat_freq_x, stim_params.spat_freq_y, sf.parameters) /
                      (st_params.spat_res.mnt**2)
                      )
estimated_dc = (
               # #### ! Multiply by the DC ... not the `F1` amplitude
               stim_params.DC *
               ff.mk_dog_sf_ft(SpatFrequency(0), SpatFrequency(0), sf.parameters) /
               (st_params.spat_res.mnt**2)
               )

# remove artefacts from the time constant and "ramping-up" at the beginning of convolution
# slicing not necessary for spatial ... ?
stable_conv = signal_conv

# amplitude is half of total min to maximum
actual_amplitude = (stable_conv.max()-stable_conv.min())/2
# DC is halfway point between min and max ... or max minus amplitude
actual_DC = stable_conv.max() - actual_amplitude

print('~~~~~\nActual values and estimated values with percentage errors ...\n')
print(f'est_amp: {estimated_amplitude:.3f}, actual: {actual_amplitude:.3f}')
print(f'Error amplitude: {(abs(estimated_amplitude- actual_amplitude)/actual_amplitude):.3%}')

print(f'est_DC: {estimated_dc:.3f}, actual: {actual_DC:.3f}')
print(f'Error DC: {(abs(estimated_dc- actual_DC)/actual_DC):.3%}')
# -

# Check convolution estimation functions

# +
estimated_amplitude = (
                      stim_params.amplitude *
                      ff.mk_dog_sf_conv_amp(
                        stim_params.spat_freq_x, stim_params.spat_freq_y, sf.parameters,
                        spat_res=st_params.spat_res)
                      )
estimated_dc = (
               # #### ! Multiply by the DC ... not the `F1` amplitude
               stim_params.DC *
               ff.mk_dog_sf_conv_amp(
                SpatFrequency(0), SpatFrequency(0), sf.parameters, st_params.spat_res)
               )

print('~~~~~\nActual values and estimated values with percentage errors ...\n')
print(f'est_amp: {estimated_amplitude:.3f}, actual: {actual_amplitude:.3f}')
print(f'Error amplitude: {(abs(estimated_amplitude- actual_amplitude)/actual_amplitude):.3%}')

print(f'est_DC: {estimated_dc:.3f}, actual: {actual_DC:.3f}')
print(f'Error DC: {(abs(estimated_dc- actual_DC)/actual_DC):.3%}')

# -



# ### Joint Amplitude

# How join a Spatial Filter and a Temporal Filter?

# 1. Determine the "intersection" of the two filters
# 2. Use the response at the "intersection" as the "norm response"
# 3. For specific spatial or temporal frequencies, find the factor by which the filters' responses change relative to their norm/intersection responses
# 4. Multiply the norm response by these two factors

# Done by `correction.joint_spat_temp_conv_amp()`

# Demonstration follows ...

# #### Norm or Intersection Response

# +
# static spat freq at which temp_filt measured
tf_sf = tf.source_data.resp_params.sf
# static temp freq at which spat_filt measured
sf_tf = sf.source_data.resp_params.tf

# find "intersection response"
# response of both filters at the other filter's static frequency
norm_tf = ff.mk_tq_tf_ft(sf_tf, tf.parameters)
norm_sf = ff.mk_dog_sf_ft(tf_sf, SpatFrequency(0), sf.parameters)
# -
# +
# norm amplitude
# The amplitude that is what all amps are normlised to
norm_amp = (norm_tf + norm_sf) / 2
# norm_factor will normalise all amps to 1
# this 1 will represent norm_amp which is the average of the spat and temp
# responses
norm_factor = norm_tf * norm_sf

print(norm_amp, norm_factor)
# -

# #### Deriving factors and joint response amplitude

# +
# stimulus frequencies
temp_freq = TempFrequency(7)
spat_freq_x = SpatFrequency(0.1)
spat_freq_y = SpatFrequency(0)

# factors (filters' responses relative to norm response)
sf_factor = (
    ff.mk_dog_sf_ft(
        spat_freq_x, spat_freq_y, sf.parameters,
        collapse_symmetry=False)
    /
    norm_sf
)
tf_factor = ff.mk_tq_tf_ft(temp_freq, tf.parameters) / norm_tf

joint_amp = norm_amp * sf_factor * tf_factor
# -

# ### Joint DC

# * The DC of a spatial or temporal filter is the response of the cell to a uniform stimulus
#   of the same average luminance (or DC luminance) as as the sinusoidal stimuli.
# * To fuse two filters, their DCs can just be averaged.
# * There is a potential issue around mean luminance being too difference between stimuli
# * Just find average of the two?  Raise error if they're too different?

# ### Correcting actual amplitude and DC convolution to produce appropriate F1 when rectified

# * explain what estimate_real... does
# * how fits into process of correction

# Create Stimulus

# +
stim_amp=17
stim_DC=-11
spat_res=ArcLength(1, 'mnt')
# trying to fit a small spat filt in
spat_ext=ArcLength(250, 'mnt')
temp_res=Time(1, 'ms')
temp_ext=Time(1000, 'ms')

orientation = ArcLength(90, 'deg')
temp_freq = TempFrequency(8)
spat_freq_x = SpatFrequency(2)
spat_freq_y = SpatFrequency(0)
# -
# +
st_params = do.SpaceTimeParams(
    spat_ext, spat_res, temp_ext, temp_res,
    array_dtype='float32')
stim_params = do.GratingStimulusParams(
    spat_freq_x, temp_freq,
    orientation=orientation,
    amplitude=stim_amp, DC=stim_DC
)
# -


# +
stim_array = stimulus.mk_sinstim(st_params, stim_params)
# -
# +
xc, yc = ff.mk_spat_coords(spat_res=st_params.spat_res, spat_ext=st_params.spat_ext)
spat_filt = ff.mk_dog_sf(xc, yc, dog_args=sf)

time_coords = ff.mk_temp_coords(st_params.temp_res, st_params.temp_ext)
temp_filt = ff.mk_tq_tf(time_coords, tf.parameters)
# -

# +
spat_signal_conv = (spat_filt[..., np.newaxis] * stim_array).sum(axis=(0, 1))
temp_signal_conv = signal_convolve(spat_signal_conv, temp_filt)[:stim_array.shape[2]]
signal_conv = temp_signal_conv
# -

# +
show_fig(px.line(temp_signal_conv))
# -

# +
# remove artefacts from the time constant and "ramping-up" at the beginning of convolution
stable_conv = signal_conv[signal_conv.size//2:]

# amplitude is half of total min to maximum
actual_amplitude = (stable_conv.max()-stable_conv.min())/2
# DC is halfway point between min and max ... or max minus amplitude
actual_DC = stable_conv.max() - actual_amplitude
# -
# show_fig(px.line(stable_conv))



# #### Going through the whole correction pipeline

# Get an estimate of what amplitudes the convolution process will create

# +
conv_resp_params = correction.mk_estimate_sf_tf_conv_params(
    st_params, stim_params, sf, tf)
estimated_amplitude, estimated_dc = conv_resp_params.amplitude, conv_resp_params.DC
# -

# Get the joint response of the spatial and temporal filters.
# **This is the target of the process, to produce these values in the final analysis.**

# +
joint_resp_params = correction.mk_joint_sf_tf_resp_params(stim_params, sf, tf)
print(joint_resp_params)
# -
# +
# show_fig(plot.joint_sf_tf_amp(tf, sf))
# plot.joint_sf_tf_amp(tf, sf).show()
# -

# Find the necessary amplitude for the above target to be reached by a rectified signal
# when analysed by an FFT.
# *In a way*, this is the actual target for the amplitude of the signal, as it's necessary
# to get the above target after rectification and FFT analysis.

# +
real_unrect_joint_amp_opt = est_amp.find_real_f1(
    DC_amp=joint_resp_params.DC,
    f1_target=joint_resp_params.amplitude
    )
real_unrect_joint_amp = real_unrect_joint_amp_opt.x[0]
print(real_unrect_joint_amp)
# -

# Find the ratio between the target above and the amplitude of the result of convolution.
# Similarly, find the DC shift necessary.

# +
amp_adjustment_factor = real_unrect_joint_amp / conv_resp_params.amplitude
print(amp_adjustment_factor)
# -
# +
dc_shift_factor = joint_resp_params.DC - (conv_resp_params.DC * amp_adjustment_factor)

adjustment_params = do.ConvRespAdjParams(
    amplitude=amp_adjustment_factor,
    DC=dc_shift_factor,
    max_f1_adj_factor=None,
    joint_response=joint_resp_params
    )
print(adjustment_params)
# -

# Now, to get the actual amplitude and check its accuracy,
# we have to measure what our actual target is.
# That is, the F1 from an FFT of the rectified signal!
# Looking at the min and max values of the actual RECTIFIED signal would be incorrect,
# as the F1 amplitude after FFT is the target.

# +
def check_actual_signal_against_target(
        time_coords, stim_params,
        signal_conv, adjustment_params, correct_rectify: bool = False):

    if correct_rectify:
        # Take the whole signal, as necessary for the FFT to produce an F1 at the temp_freq
        full_corrected_conv = correction.adjust_conv_resp(signal_conv, adjustment_params)
        # Rectify
        full_corrected_conv[full_corrected_conv < 0] = 0
    else:
        full_corrected_conv = signal_conv

    # Get FFT ... note the align_freqs argument
    spec, freq = est_f1.gen_fft(full_corrected_conv, time=time_coords, align_freqs=True)
    spec = abs(spec)

    # indices and integer temp_freqs should align (as using "align_freqs=True")
    # this is the F1 amplitude
    actual_amplitude = spec[int(stim_params.temp_freq.hz)]

    # Get the DC (need unrectified signal)
    # actual_DC = spec[0]
    spec_unrect, freq_unrect = est_f1.gen_fft(
        correction.adjust_conv_resp(signal_conv, adjustment_params),
        time=time_coords,
        align_freqs=True)
    actual_DC = abs(spec_unrect)[0]

    # Pull out the initial targets from above (so not quite "estimated")
    estimated_amplitude, estimated_dc = joint_resp_params.amplitude, joint_resp_params.DC


    print('~~~~~\nActual values and estimated values with percentage errors ...\n')
    # print(f'{actual_amplitude/1e6:.3g}M ... {actual_DC/1e6:.3g}M')
    print(f'est_amp: {estimated_amplitude:.3f}, actual: {actual_amplitude:.3f}')
    print(f'Error amplitude: {(abs(estimated_amplitude- actual_amplitude)/actual_amplitude):.3%}')

    print(f'est_DC: {estimated_dc:.3f}, actual: {actual_DC:.3f}')
    print(f'Error DC: {(abs(estimated_dc- actual_DC)/actual_DC):.3%}')
# -
# +
check_actual_signal_against_target(
    time_coords, stim_params, signal_conv, adjustment_params, correct_rectify=True)
# -

# #### Testing the full convolution function


# +
def actual_f1_amplitude(
        full_corrected_conv, time_coords):

    spec, _ = est_f1.gen_fft(full_corrected_conv, time=time_coords, align_freqs=True)
    spec = abs(spec)

    # indices and integer temp_freqs should align (as using "align_freqs=True")
    # this is the F1 amplitude
    actual_amplitude = spec[int(stim_params.temp_freq.hz)]

    return actual_amplitude


def check_full_convolution_results(
        st_params, sf, tf, spat_filt, temp_filt, stim_params, stim_array,
        filter_actual_max_f1_amp: Optional[do.LGNF1AmpMaxValue] = None,
        target_max_f1_amp: Optional[do.LGNF1AmpMaxValue] = None,
        ):

    convolution_response = convolve.mk_single_sf_tf_response(
        st_params, sf, tf, spat_filt, temp_filt, stim_params, stim_array,
        rectified = False,
        filter_actual_max_f1_amp=filter_actual_max_f1_amp,
        target_max_f1_amp=target_max_f1_amp)
    full_corrected_conv = convolution_response.response.copy()
    full_corrected_conv[full_corrected_conv<0] = 0

    spec, _ = est_f1.gen_fft(full_corrected_conv, time=time_coords, align_freqs=True)
    spec = abs(spec)

    # indices and integer temp_freqs should align (as using "align_freqs=True")
    # this is the F1 amplitude
    actual_amplitude = spec[int(stim_params.temp_freq.hz)]

    # Get the DC (need unrectified signal)
    # actual_DC = spec[0]
    spec_unrect, _ = est_f1.gen_fft(
        convolution_response.response,
        time=time_coords,
        align_freqs=True)
    actual_DC = abs(spec_unrect)[0]

    # Pull out the initial targets from above (so not quite "estimated")
    estimated_amplitude, estimated_dc = (
        convolution_response.adjustment_params.joint_response.amplitude,
        convolution_response.adjustment_params.joint_response.DC
        )

    if (max_f1_factor := convolution_response.adjustment_params.max_f1_adj_factor):
        estimated_amplitude *= max_f1_factor


    print('~~~~~\nActual values and estimated values with percentage errors ...\n')
    # print(f'{actual_amplitude/1e6:.3g}M ... {actual_DC/1e6:.3g}M')
    print(f'est_amp: {estimated_amplitude:.3f}, actual: {actual_amplitude:.3f}')
    print(f'Error amplitude: {(abs(estimated_amplitude- actual_amplitude)/actual_amplitude):.3%}')

    print(f'est_DC: {estimated_dc:.3f}, actual: {actual_DC:.3f}')
    print(f'Error DC: {(abs(estimated_dc- actual_DC)/actual_DC):.3%}')

    return convolution_response
# -

# +
convolution_response = convolve.mk_single_sf_tf_response(
    st_params, sf, tf, spat_filt, temp_filt, stim_params, stim_array, rectified=False)
print(convolution_response.adjustment_params)
# -
# +
conv_resp_obj = check_full_convolution_results(
    st_params, sf, tf, spat_filt, temp_filt, stim_params, stim_array)
# -

# Trying different controls over the amplitude of the response

# +
stim_params = do.GratingStimulusParams(
    spat_freq_x, temp_freq,
    orientation=orientation,
    amplitude=stim_amp, DC=stim_DC,
    contrast=do.ContrastValue(contrast=0.3)
)
# -
# +
conv_resp_obj = check_full_convolution_results(
    st_params, sf, tf, spat_filt, temp_filt, stim_params, stim_array)
# -

# Max F1 Value scaling


# +
conv_resp_obj = check_full_convolution_results(
    st_params, sf, tf, spat_filt, temp_filt, stim_params, stim_array)
# -
# +
cell_max_f1 = correction.mk_actual_filter_max_amp(sf, tf, stim_params.contrast)
cell_max_f1_val = cell_max_f1.value
print(cell_max_f1_val)
# -

# Try differnt target max_f1 values
# There's likely to always be a bit of error in the F1 ressponse
# as the first part of the signal is a little wobbly
# but it is kept because the whole signal is needed for the FFT to estimate the F1

# +
target_max_f1_amp = do.LGNF1AmpMaxValue(95, contrast=stim_params.contrast)
conv_resp_obj = check_full_convolution_results(
    st_params, sf, tf, spat_filt, temp_filt, stim_params, stim_array,
    filter_actual_max_f1_amp=cell_max_f1_val,
    target_max_f1_amp=target_max_f1_amp
    )
# -
# +
for f1_max_val in [10, 30, 50, 70, 90]:
    print(f'\n\n******* {f1_max_val=}\n')
    target_max_f1_amp = do.LGNF1AmpMaxValue(f1_max_val, contrast=stim_params.contrast)
    conv_resp_obj = check_full_convolution_results(
        st_params, sf, tf, spat_filt, temp_filt, stim_params, stim_array,
        filter_actual_max_f1_amp=cell_max_f1_val,
        target_max_f1_amp=target_max_f1_amp
        )
# -

# #### Distribution of Max Responses



# +
test = do.LGNF1AmpDistParams().draw_f1_amp_vals(1000)
px.histogram([t.max_amp for t in test]).show()
np.mean([t.max_amp for t in test])
# 45

# -


# ### But how account for differences in contrast?

# * rescale the F1 values that the filter should prescribes before correcting for rectification
# * which contrast to target?

# ## Contrast Correction

# * For any filter (spatial or temporal), bring the response up to what it *should*
#   be for a given contrast and given the contrast that filter was originally recorded at.
# * Basically, scale the response up/down according to a conventional contrast curve.

# ### Contrast Module Basics

# +
cont_corr.ON
print(f'{cont_corr.ON}')
# -

# `ContrastParams(max_resp=53, contrast_50=0.133, exponent=1.2)`

# +
contrasts = np.linspace(0, 1, 50)
resp = cont_corr.contrast_response(contrasts, cont_corr.ON)
fig = (
    px
    .line(x=contrasts, y=resp)
    .update_layout(
        title=f'Contrast curve ({cont_corr.ON=})',
        xaxis_title='Contrast (0-1)',
        yaxis_title='Response'
        )
    )
# -
# +
show_fig(fig)
write_fig(
    fig.update_layout(template='plotly_dark'),
    'contrast_curve_basic.svg')
# -

# ![see plot here](./contrast_curve_basic.svg)


# +
base, target = do.ContrastValue(0.3), do.ContrastValue(0.8)
scaling_factor = cont_corr.mk_contrast_resp_amplitude_adjustment_factor(
    base_contrast=base, target_contrast=target, params=cont_corr.ON)
print(f'{scaling_factor=}')
# -

# ```python
# scaling_factor=1.233530295968726
# ```

# +
if scaling_factor == (
        cont_corr.contrast_response(target.contrast, cont_corr.ON)
        /
        cont_corr.contrast_response(base.contrast, cont_corr.ON)
    ):
    print('True')

if np.isclose(
            cont_corr.contrast_response(target.contrast, cont_corr.ON),
            scaling_factor * cont_corr.contrast_response(base.contrast, cont_corr.ON)
        ):
    print("True")
# -


# ### Correcting Contrast in Joint Responses

# +
base_response = 45
base_contrast, target_contrast = do.ContrastValue(0.3), do.ContrastValue(0.8)

new_response = cont_corr.correct_contrast_response_amplitude(
    base_response, base_contrast, target_contrast, cont_corr.ON)
print(base_response, new_response)
# -

# `45 55.50886331859267`

# +
print(sf.source_data.resp_params.contrast, tf.source_data.resp_params.contrast, )
# -

# `0.5 0.4`

# +
temp_freq = TempFrequency(1)
spat_freq_x = SpatFrequency(1)
spat_freq_y = SpatFrequency(0)

correction.joint_spat_temp_f1_magnitude(
    temp_freq,
    spat_freq_x, spat_freq_y,
    tf, sf,
    contrast=do.ContrastValue(0.2),
    contrast_params=cont_corr.ON)
# -


# ### Contrast Correction in full convolution code

# Just need to pass contrast params and stimulus contrast to `mk_conv_resp_adjust_params()` which
# uses `joint_spat_temp_f1_magnitude()` under the hood.

# Actually, now `contrast` is a parameter of `GratingStimulusParams` with default value of `0.3`


# ## Max F1 Amplitude Distribution

# +
import lif.lgn.cells as lgn_cells
# -

# +
f1_amps_params = do.LGNF1AmpDistParams()

f1_amps_params.draw_f1_amp_vals(n=10)
# -


# +
stparams = do.SpaceTimeParams(
    spat_ext=ArcLength(5), spat_res=ArcLength(1, 'mnt'), temp_ext=Time(1),
    temp_res=Time(1, 'ms'))

lgnparams = do.LGNParams(
    n_cells=10,
    orientation = do.LGNOrientationParams(ArcLength(30), 0.5),
    circ_var = do.LGNCircVarParams('naito_lg_highsf', 'naito'),
    spread = do.LGNLocationParams(2, 'jin_etal_on'),
    filters = do.LGNFilterParams(spat_filters='all', temp_filters='all'),
    F1_amps = do.LGNF1AmpDistParams()
    )
# -
# +
lgn = lgn_cells.mk_lgn_layer(lgnparams, spat_res=stparams.spat_res)
# -
# +
len(lgn.cells)
# -
# +
for i in range(len(lgn.cells)):
    print(lgn.cells[i].max_f1_amplitude)
# -
# +
for i in range(len(lgn.cells)):
    print(
        lgn.cells[i].location
        )
# -
# +
for i in range(len(lgn.cells)):
    print(
        lgn.cells[i].location.round_to_spat_res(stparams.spat_res)
        )
# -

# +
lgnparams.F1_amps
# -

# +
base, target = lgnparams.F1_amps.contrast, do.ContrastValue(0.8)
scaling_factor = cont_corr.mk_contrast_resp_amplitude_adjustment_factor(
    base_contrast=base, target_contrast=target, params=cont_corr.ON)
print(f'{scaling_factor=}')

# -
# +
test = lgnparams.F1_amps.contrast
# -
# +
test.contrast
# -


# +
f1_max_amp = f1_amps_params.draw_f1_amp_vals(n=1)[0]
# -
# +
contrast_params = settings.simulation_params.contrast_params

max_f1_amp = f1_max_amp.max_amp
max_f1_amp_contrast = f1_max_amp.contrast.contrast
stim_contrast = do.ContrastValue(0.4)

contrast_adjusted_f1_max_amp = cont_corr.correct_contrast_response_amplitude(
        response_amplitude=max_f1_amp,
        base_contrast=f1_max_amp.contrast,
        target_contrast=stim_contrast,
        contrast_params=contrast_params
    )

print(f1_max_amp.max_amp, contrast_adjusted_f1_max_amp)
# -


# ### How find actual max response at opt temp and spat freq?

# +
tf = do.TQTempFilter.load(do.TQTempFilter.get_saved_filters()[0])
sf = do.DOGSpatialFilter.load(do.DOGSpatialFilter.get_saved_filters()[0])
# -

# +
temp_freq = TempFrequency(4)
spat_freq_x = SpatFrequency(1)
spat_freq_y = SpatFrequency(0)

correction.joint_spat_temp_f1_magnitude(
    temp_freq,
    spat_freq_x, spat_freq_y,
    tf, sf,
    contrast=do.ContrastValue(0.2),
    contrast_params=cont_corr.ON)
# -
# +
correction.mk_actual_filter_max_amp(sf, tf, contrast=do.ContrastValue(0.1))
# -
# +
import scipy.optimize as opt
# -
# +
def mk_joint_f1_wrapper(
    tf: do.TQTempFilter, sf: do.DOGSpatialFilter,
    contrast: Optional[do.ContrastValue]=None,
    ):

    joint_f1_wrapper = lambda x: (
         -1 * correction.joint_spat_temp_f1_magnitude(
            TempFrequency(x[0]), SpatFrequency(x[1]), SpatFrequency(0),
            tf, sf,
            contrast=contrast
            )
        )

    return joint_f1_wrapper
# -
# +
wrapper = mk_joint_f1_wrapper(tf, sf, contrast=do.ContrastValue(0.8))
wrapper([4, 1])
# -
# +
opt_result = opt.minimize(wrapper, x0=[4,1])
opt_result['fun']
# -
# +
correction.joint_spat_temp_f1_magnitude(
    TempFrequency(opt_result.x[0]), SpatFrequency(opt_result.x[1]), SpatFrequency(0),
    tf, sf, contrast=do.ContrastValue(0.4)
    )
# -
# +
def mk_simple_f1_wrapper(
    tf: do.TQTempFilter, sf: do.DOGSpatialFilter,
    contrast: Optional[do.ContrastValue]=None,
    ):

    joint_f1_wrapper = lambda x: (
                            -1 *
                            (
                                ff.mk_tq_tf_ft(TempFrequency(x[0]), tf.parameters) +
                                ff.mk_dog_sf_ft(
                                    SpatFrequency(float(x[1])), SpatFrequency(0), sf.parameters)
                            )
                        )

    return joint_f1_wrapper
# -
# +
wrapper = mk_simple_f1_wrapper(tf, sf)
# -
# +
wrapper([4,1])
# -
# +
opt.minimize(wrapper, x0=[4,1])
# %timeit opt.minimize(wrapper, x0=[4,1])
# -

# ## Make Lookup for all permutations

# +
from itertools import product
# -
# +
tfs = do.TQTempFilter.get_saved_filters()
sfs = do.DOGSpatialFilter.get_saved_filters()
# -
# +
all_filters = tuple(product(tfs, sfs))
# -
# +
for f in all_filters:
    print(f[0].stem, f[1].stem)
# -
