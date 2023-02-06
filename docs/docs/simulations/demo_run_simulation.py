

# # Imports
# +
from pathlib import Path
import numpy as np

from lif.utils.units.units import (
		ArcLength, Time, SpatFrequency, TempFrequency
	)
import lif.utils.data_objects as do
import lif.utils.settings as settings

import lif.lgn as lgn
import lif.stimulus.stimulus as stimulus
import lif.receptive_field.filters.filter_functions as ff
# -

# # Stimulus generation and managements

# ## Estimating Required Width

# +
spat_res = ArcLength(1, 'mnt')
lgn_params = lgn.demo_lgnparams
max_spat_ext = stimulus.estimate_max_stimulus_spatial_ext_for_lgn(
	spat_res, lgn_params, n_cells=1000, safety_margin_increment=0.1)
print(max_spat_ext.deg)
# -
# +
print(f'Number of pixels in width: {max_spat_ext.base / spat_res.base}')
# -

# ## size of stimulus this would require
# +
size_factor = 1.1
# spat_ext=ArcLength(120, 'mnt')
spat_res=ArcLength(1, 'mnt')
spat_ext = ff.round_coord_to_res(ArcLength(max_spat_ext.base * size_factor), spat_res, high=True)
# spat_ext = ff.round_coord_to_res(ArcLength(max_spat_ext.base * 1.1), spat_res, high=True)
temp_res=Time(1, 'ms')
temp_ext=Time(500, 'ms')

orientation = ArcLength(90, 'deg')
temp_freq = TempFrequency(4)
spat_freq_x = SpatFrequency(2)
spat_freq_y = SpatFrequency(0)

st_params = do.SpaceTimeParams(
	spat_ext, spat_res, temp_ext, temp_res,
	array_dtype='float32'
	)

stim_params = do.GratingStimulusParams(
    spat_freq_x, temp_freq,
    orientation=orientation,
    contrast=do.ContrastValue(0.4)
)
# -
# +
stim = stimulus.mk_sinstim(st_params, stim_params)
print(f'predicted size (MB): {stim.nbytes / (1000*1000) * (1.1/size_factor)**2}')
# -



# +
# stim.nbytes / (1000*1000) * (1.1/0.5)**2
# -
# +
stim_test_path = Path('~/Downloads/stim_test_file.npy').expanduser()
print(stim_test_path)
np.save(stim_test_path.expanduser(), stim)
# -
# +
print(f'Actual size: {stim_test_path.stat().st_size / (1000*1000)}')
# -

# +
demo_lgn = lgn.mk_lgn_layer(lgn_params, spat_res)
# -
# +
from collections import Counter
# sorted((c.spat_filt.key for c in demo_lgn.cells))
Counter((c.spat_filt.key for c in demo_lgn.cells))
# -


# ## Saving and Caching

# +
signature = stimulus.mk_stim_signature(st_params, stim_params)
print(signature)
# -
# +
new_st_params, new_stim_params = stimulus.mk_params_from_stim_signature(signature)

print(st_params)
print(new_st_params)
print(new_st_params == st_params)
# -

# +
from dataclasses import replace

mod_st_params = replace(st_params, temp_ext=Time(534.5))

stimulus.mk_stim_signature(mod_st_params, stim_params)
# -
st_params.asdict_() == st_params.asdict_()
stim_params.asdict_()

hash(stim_params)

# +
multi_stim_params = stimulus.mk_multi_stimulus_params(
	spat_freqs=[2,4], temp_freqs=[1,2], orientations=[0, 90],
    spat_freq_unit='cpd', temp_freq_unit='hz', ori_arc_unit='deg',
	)
# -
# +
len(multi_stim_params)
# -
# +
for sp in multi_stim_params:
	print(sp.spat_freq.cpd, sp.temp_freq.hz, sp.orientation.deg)
# -

# Reworking the signature code to preserve ints a floats

# +
st_params.spat_ext.value
# -
# +
signature = stimulus.mk_stim_signature(st_params, stim_params)
print(signature)
# -
# +
new_st_params, new_stim_params = stimulus.mk_params_from_stim_signature(signature)

print(st_params)
print(new_st_params)
print(new_st_params == st_params)
# -

# Basics

# +
a, b, c = 1, 1., 1.1
f'a={a}, b={b}, c={c}'
# -

# f-string preserves integers as being without a decimal point ... can use for parsing


# +
signature = stimulus.mk_stim_signature(st_params, stim_params)
print(signature, len(signature), sep='\n\n')
# -
stim_params.contrast

# +
new_st_params, new_stim_params = stimulus.mk_params_from_stim_signature(signature)

print(st_params)
print(new_st_params)
print(new_st_params == st_params)
print(stim_params)
print(new_stim_params)
print(stim_params == new_stim_params)
# -


# ### Testing the caching


# +
data_dir = settings.get_data_dir()
# -

# List all stimulus cache

# +
pkl_files = data_dir.glob('STIMULUS*')
for f in pkl_files:
	print(f.name)
# -

# Make some stim caches ... but keep it small

# +
spat_ext=ArcLength(500, 'mnt')
spat_res=ArcLength(1, 'mnt')
# spat_ext = ff.round_coord_to_res(ArcLength(max_spat_ext.base * 1.1), spat_res, high=True)
temp_res=Time(1, 'ms')
temp_ext=Time(500, 'ms')

st_params = do.SpaceTimeParams(
	spat_ext, spat_res, temp_ext, temp_res,
	array_dtype='float32'
	)

multi_stim_params = stimulus.mk_multi_stimulus_params(
	spat_freqs=[4], temp_freqs=[2], orientations=[0],
	)
# -

# Add to stimulus cache

# +
stimulus.mk_stimulus_cache(st_params, multi_stim_params)
# -

# Get a dict of or print out all cached stimuli

# +
stimulus.get_params_for_all_saved_stimuli()
# -
# +
stimulus.print_params_for_all_saved_stimuli()
# -

# Load a stimulus from the cache

# +
stim_array = stimulus.load_stimulus_from_params(st_params, multi_stim_params[0])

stim_array.shape
# -

# ## Slicing Stimulus

# +
sf = lgn.cells.spatial_filters['maffei73_2right']
# -


# +
sf_loc = do.RFLocation(
	ArcLength(5.342, 'mnt'),
	ArcLength(0, 'mnt')
	)

spat_slice = stimulus.mk_rf_stim_spatial_slice_idxs(
	st_params, sf.parameters.parameters, sf_loc)
print(spat_slice)
# -

# +
slice_range = spat_slice.x2 - spat_slice.x1
# -

# +
xc, yc = ff.mk_spat_coords(st_params.spat_res, sd=sf.parameters.max_sd())
spat_filt = ff.mk_dog_sf(xc, yc, sf.parameters)
# -
# +
print(spat_filt.shape[0], slice_range)
# -

# +
spat_slice.is_within_extent(st_params)
# -
# +
sliced_array = stim_array[spat_slice.y1:spat_slice.y2, spat_slice.x1:spat_slice.x2]
# -
# +
sliced_array.shape
# -

# +
spat_slice.x2 *= 10
# -
# +
# Should be, and is False
spat_slice.is_within_extent(st_params)
# -

# +
print(spat_filt.shape[0] == (spat_slice.y2-spat_slice.y1))
# -

# +
stim_slice_array = stimulus.slice_stimulus_array(st_params, stim_array, spat_filt, spat_slice)
# -
# +
stim_slice_array.shape
# -






