
# ## Imports

# +
from lif import *
from lif.convolution import correction

import plotly.express as px
import plotly.graph_objects as go
# -

# ## Load filters

# * These are loaded from file, having been previously fit to data

# +
tf = TQTempFilter.load(TQTempFilter.get_saved_filters()[0])
sf = DOGSpatialFilter.load(DOGSpatialFilter.get_saved_filters()[0])
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
from scipy.signal import convolve
# -

# Prepare time coords, temp filter and sinusoid

# +
time_coords = mk_temp_coords(st_params.temp_res, st_params.temp_ext)
signal = est_amp.gen_sin(
    amplitude=stim_params.amplitude,
    DC_amp=stim_params.DC,
    time=time_coords,
    freq=stim_params.temp_freq
    )
temp_filter = mk_tq_tf(time_coords, tf.parameters)
signal_conv = convolve(signal, temp_filter)[:time_coords.value.size]
# -

# Plotting if desired

# +
px.line(x=time_coords.ms, y=signal).show()
# px.line(x=time_coords.ms, y=temp_filter).show()
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
                      mk_tq_tf_ft(stim_params.temp_freq, tf.parameters) /
                      st_params.temp_res.s
                      )
estimated_dc = (
               # ### ! Multiply by the DC ... not the `F1` amplitude
               stim_params.DC *
               mk_tq_tf_ft(TempFrequency(0), tf.parameters) /
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
    mk_tq_tf_conv_amp(stim_params.temp_freq, tf.parameters, st_params.temp_res)
    )
estimated_dc = (
    stim_params.DC *
    mk_tq_tf_conv_amp(TempFrequency(0), tf.parameters, st_params.temp_res)
    )

print(f'est_amp: {estimated_amplitude:.3f}, actual: {actual_amplitude:.3f}')
print(f'Error: {(abs(estimated_amplitude- actual_amplitude)/actual_amplitude):.3%}')

print(f'est_amp: {estimated_dc:.3f}, actual: {actual_DC:.3f}')
print(f'Error: {(abs(estimated_dc- actual_DC)/actual_DC):.3%}')
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

# static spat freq at which temp_filt measured
tf_sf = tf.source_data.resp_params.sf
# static temp freq at which spat_filt measured
sf_tf = sf.source_data.resp_params.tf

# find "intersection response"
# response of both filters at the other filter's static frequency
norm_tf = mk_tq_tf_ft(sf_tf, tf.parameters)
norm_sf = mk_dog_sf_ft(tf_sf, SpatFrequency(0), sf.parameters)

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
    mk_dog_sf_ft(
        spat_freq_x, spat_freq_y, sf.parameters,
        collapse_symmetry=False)
    /
    norm_sf
)
tf_factor = mk_tq_tf_ft(temp_freq, tf.parameters) / norm_tf

joint_amp = norm_amp * sf_factor * tf_factor
# -

# ### Joint DC

# * The DC of a spatial or temporal filter is the response of the cell to a uniform stimulus of the same average luminance (or DC luminance) as as the sinusoidal stimuli.
# * To fuse two filters, their DCs can just be averaged.
# * There is a potential issue around mean luminance being too difference between stimuli
# * Just find average of the two?  Raise error if they're too different?

# ### Correcting actual amplitude and DC convolution to produce appropriate F1 when rectified

# * explain what estimate_real... does
# * how fits into process of correction

# ### But how account for differences in contrast?

# * rescale the F1 values that the filter should prescribes before correcting for rectification
# * which contrast to target?

# ##
