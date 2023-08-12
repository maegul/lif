
# # Imports

# +
import numpy as np

import lif.convolution.soodak_rf as srf
import lif.convolution.correction as correction

import plotly.express as px
import plotly.graph_objects as go

from lif.utils.units.units import (
		ArcLength, Time, SpatFrequency, TempFrequency
	)
import lif.utils.data_objects as do
import lif.utils.settings as settings

from lif.receptive_field.filters import filters
import lif.lgn as lgn
import lif.stimulus.stimulus as stimulus
import lif.convolution.convolve as convolve
import lif.convolution.correction as correction
import lif.convolution.estimate_real_amp_from_f1 as est_f1
import lif.receptive_field.filters.filter_functions as ff

from lif.lgn import cells

import lif.simulation.all_filter_actual_max_f1_amp as all_max_f1
import lif.simulation.leaky_int_fire as lifv1
from lif.simulation import run

from lif.plot import plot
# -


# # Basic Soodak RF Demo

# +
r, ph = srf.mk_cent_surr_comp_resp(
	f=0.1, theta=0, surr_w1=3, surr_w2=3, surr_h=0.5,
	cent_x=2, surr_x=2)
# -
# +
print(r, ph)
# -
# +
sr = srf.sum_resps(r, ph)
print(sr)
# -
# +
amp = np.absolute(sr)
phs = np.degrees(np.angle(sr))

print(amp, phs)
# -


# # Gauss Convolution

# ## Stimulus

# Needs to be only big enough for a single RF

# ### smallest rf
# +
for k, sf in lgn.cells.filters.spatial_filters.items():
	print(k, sf.parameters.surr.arguments.h_sd.mnt)
# -
# +
for k in (
	sorted(
			lgn.cells.filters.spatial_filters.keys(),
			key=lambda k: lgn.cells.filters.spatial_filters[k].parameters.surr.arguments.h_sd.mnt
			)
		):
	print(k, lgn.cells.filters.spatial_filters[k].parameters.surr.arguments.h_sd.mnt)
# -
# +
16 * 5 * 2
# -
# +
spat_filt_key = 'so81_2bottom'
# spat_filt_key = 'maffei73_2right'
spat_filt = lgn.cells.filters.spatial_filters[spat_filt_key]

spat_filt = ff.mk_ori_biased_spatfilt_params_from_spat_filt(spat_filt, circ_var=0.8)
rf_theta = ArcLength(0, 'deg')
# -

# ### Params
# +
# should cover full size of sf and some movement
spat_ext = ArcLength((16 * 5 * 2) + 60, 'mnt')
# -

# +
spat_res=ArcLength(1, 'mnt')
temp_res=Time(1, 'ms')
temp_ext=Time(1000, 'ms')

st_params = do.SpaceTimeParams(
	spat_ext, spat_res, temp_ext, temp_res,
	array_dtype='float16'
	# array_dtype='float32'
	)

orientation = ArcLength(0, 'deg')
temp_freq = TempFrequency(4)
spat_freq_x = SpatFrequency(2.5)
spat_freq_y = SpatFrequency(0)

stim_params = do.GratingStimulusParams(
	spat_freq_x, temp_freq,
	orientation=orientation,
	amplitude=1, DC=1,
	contrast=do.ContrastValue(0.4)
)
# -

# ### Make stimulus
# +
stim_array = stimulus.mk_sinstim(st_params, stim_params)
print(stim_array.nbytes / 10**6)
# -

# ## Gauss

# +
# spatial filter array
xc, yc = ff.mk_spat_coords(
	st_params.spat_res,
	sd=spat_filt.parameters.max_sd()
	)

tc = ff.mk_temp_coords(st_params.temp_res, st_params.temp_ext)

spat_filt_array = ff.mk_dog_sf(
	x_coords=xc, y_coords=yc,
	dog_args=spat_filt.parameters  # use oriented params
	)

# Rotate array
spat_filt_array = ff.mk_oriented_sf(spat_filt_array, rf_theta)
# -

# Location
# +
rf_loc = do.RFLocation(
	x=ArcLength(13.5, 'mnt'),
	y=ArcLength(-12.5, 'mnt')
	)
# -
# +
px.imshow(spat_filt_array).show()
# -
# +
px.imshow(stim_array[...,0]).show()
# -



# ## Convolution

# ### Stim Slicing

# +
spat_slice_idxs = stimulus.mk_rf_stim_spatial_slice_idxs(
	st_params, spat_filt, rf_loc)
stim_slice = stimulus.mk_stimulus_slice_array(
	st_params, stim_array, spat_slice_idxs)
# -

# ### Temporal Response
# +
spatial_product = np.einsum('ij,ij...', spat_filt_array, stim_slice)
# -
# +
px.line(x=tc.ms, y=spatial_product).show()
# -

# # Soodak Response
# +
a, ph = srf.mk_spat_filt_response(stim_params, spat_filt, rf_loc, rf_theta=rf_theta)
# -
# +
r = srf.mk_spat_filt_temp_response(st_params, stim_params, spat_filt, rf_loc, rf_theta, tc)
r.max()
# -
# +
sf_dc = correction.mk_dog_sf_conv_amp(SpatFrequency(0), SpatFrequency(0), spat_filt.parameters, st_params.spat_res )
# -
# +
px.line(x=tc.ms, y=r).show()
# -

# +
amp_ratio = spatial_product.max() / r.max()
print(amp_ratio, spatial_product.max(), r.max())
# -

# +
%%timeit
tc = ff.mk_temp_coords(st_params.temp_res, st_params.temp_ext)
a, ph = srf.mk_spat_filt_response(st_params, stim_params, spat_filt, rf_loc, rf_theta)
r = mk_temp_resp(a, ph, stim_params.temp_freq, tc=tc)
# -













