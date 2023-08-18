
# # Imports

# +
from typing import DefaultDict, Tuple, Dict
import itertools as it
import random

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
import lif.convolution.soodak_rf as srf
import lif.convolution.estimate_real_amp_from_f1 as est_f1
import lif.receptive_field.filters.filter_functions as ff

from lif.lgn import cells
import lif.lgn.spat_filt_overlap as sfo

import lif.simulation.all_filter_actual_max_f1_amp as all_max_f1
import lif.simulation.leaky_int_fire as lifv1
from lif.simulation import run

from lif.plot import plot
# -

# # Generate LGN Response

# ## Params

# +
stimulus.print_params_for_all_saved_stimuli()
# -
# +
spat_res=ArcLength(1, 'mnt')
spat_ext=ArcLength(660, 'mnt')
"Good high value that should include any/all worst case scenarios"
temp_res=Time(1, 'ms')
temp_ext=Time(1000, 'ms')

st_params = do.SpaceTimeParams(
	spat_ext, spat_res, temp_ext, temp_res,
	array_dtype='float32'
	)
# -
# +
# good subset of spat filts that are all in the middle in terms of size
subset_spat_filts = [
	'berardi84_5a', 'berardi84_5b', 'berardi84_6', 'maffei73_2mid',
	'maffei73_2right', 'so81_2bottom', 'so81_5', 'soodak87_1'
]
# -
# +
multi_stim_params = do.MultiStimulusGeneratorParams(
	spat_freqs=[0.8], temp_freqs=[4], orientations=[90], contrasts=[0.3]
	)

stim_params = stimulus.mk_multi_stimulus_params(multi_stim_params)[0]

lgn_params = do.LGNParams(
	n_cells=30,
	orientation = do.LGNOrientationParams(ArcLength(0), circ_var=0.5),
	circ_var = do.LGNCircVarParams('naito_lg_highsf', 'naito'),
	spread = do.LGNLocationParams(2, 'jin_etal_on'),
	filters = do.LGNFilterParams(spat_filters='all', temp_filters='all'),
	F1_amps = do.LGNF1AmpDistParams()
	)
lif_params = do.LIFParams(total_EPSC=3.5)
# -

# +
sim_params = do.SimulationParams(
	n_simulations=1,
	space_time_params=st_params,
	multi_stim_params=multi_stim_params,
	lgn_params=lgn_params,
	lif_params = lif_params,
	n_trials = 3
	# n_trials = 10
	)
# -


# ## LGN RFs


# +
lgn_layer = cells.mk_lgn_layer(lgn_params, st_params.spat_res)
all_xc, all_yc = ff.mk_spat_coords(spat_res=st_params.spat_res, spat_ext=st_params.spat_ext)
all_rfs = np.zeros(shape=(*all_xc.value.shape, len(lgn_layer.cells)))

for i, cell in enumerate(lgn_layer.cells):

	xc, yc = ff.mk_spat_coords(st_params.spat_res, sd=cell.spat_filt.parameters.max_sd() )

	spat_filt = ff.mk_dog_sf(x_coords=xc, y_coords=yc, dog_args=cell.oriented_spat_filt_params)
	spat_filt = ff.mk_oriented_sf(spat_filt, cell.orientation)

	rf_slice_idxs = stimulus.mk_rf_stim_spatial_slice_idxs(
		st_params, cell.spat_filt, cell.location)


	all_rfs[rf_slice_idxs.y1:rf_slice_idxs.y2, rf_slice_idxs.x1:rf_slice_idxs.x2, i] = spat_filt
# -
# +
def mk_all_lgn_rfs_array(
		lgn_layer: do.LGNLayer,
		st_params: do.SpaceTimeParams
		) -> np.ndarray:

	all_xc, _ = ff.mk_spat_coords(spat_res=st_params.spat_res, spat_ext=st_params.spat_ext)
	all_rfs = np.zeros(shape=(*all_xc.value.shape, len(lgn_layer.cells)))

	for i, cell in enumerate(lgn_layer.cells):
		i, cell = 0, lgn_layer.cells[0]

		xc, yc = ff.mk_spat_coords(st_params.spat_res, sd=cell.spat_filt.parameters.max_sd() )

		spat_filt = ff.mk_dog_sf(x_coords=xc, y_coords=yc, dog_args=cell.oriented_spat_filt_params)
		spat_filt = ff.mk_oriented_sf(spat_filt, cell.orientation)

		rf_slice_idxs = stimulus.mk_rf_stim_spatial_slice_idxs(
			st_params, cell.spat_filt, cell.location)

		all_rfs[rf_slice_idxs.y1:rf_slice_idxs.y2, rf_slice_idxs.x1:rf_slice_idxs.x2, i] = spat_filt

	return all_rfs
# -
# +
test = mk_all_lgn_rfs_array(lgn_layer, st_params)
# -



# # Create Synchronous Reponse Units and Factors

# +
from collections import defaultdict
# -
# +
all_rfs[all_rfs<0]=0
all_rf_sums = all_rfs.sum(axis=(0,1))
rf_pos_bool = all_rfs>0

# cell_overlap_map = dict()
cell_overlap_map = defaultdict(lambda : defaultdict(float))
for i in range(all_rfs.shape[0]):
	for j in range(all_rfs.shape[1]):
		cell_overlap_idxs = tuple(np.nonzero(rf_pos_bool[i,j,:])[0])
		for coi in cell_overlap_idxs:
			cell_overlap_map[cell_overlap_idxs][coi] += (all_rfs[i, j, coi] / all_rf_sums[coi])
# -
# +
for k, vs in cell_overlap_map.items():
	print(k, vs)
# -
# +
for k, vs in cell_overlap_map.items():
	if len(k) == 1:
		print(k, vs)
# -
# +
for i in range(len(lgn_layer.cells)):
	print(i, sum(
			ovs[i]
			for k, ovs in cell_overlap_map.items()
			if i in k
		)
	)
# -

# +
def mk_spat_filt_overlapping_weights(
		all_spat_filt_arrays: np.ndarray,
		) -> DefaultDict[Tuple[int,...], DefaultDict[int, float]]:

	# Trim spat filts to floor all negative values at 0
	all_spat_filt_arrays[all_spat_filt_arrays<0]=0
	# calculate ahead of time total volume of each spat filt, and boolean array of postive values
	all_rf_sums = all_spat_filt_arrays.sum(axis=(0,1))
	rf_pos_bool = all_spat_filt_arrays>0

	cell_overlap_map = defaultdict(lambda : defaultdict(float))

	for i in range(all_spat_filt_arrays.shape[0]):
		for j in range(all_spat_filt_arrays.shape[1]):
			# find all RFs with postive (IE nonzero) value at these coordinates
			cell_overlap_idxs = tuple(np.nonzero(rf_pos_bool[i,j,:])[0])

			# for each such cell, find and add to the aggregate of its value at these coords
			# ... relative to the cell's total RF volume
			for coi in cell_overlap_idxs:
				cell_overlap_map[cell_overlap_idxs][coi] += (
						all_spat_filt_arrays[i, j, coi] / all_rf_sums[coi]
					)

	return cell_overlap_map
# -

# +
import numba
numba.typed.Dict
# -
# +
np.all(
	all_spat_filt_arrays.sum(axis=(0,1))
	==
	all_spat_filt_arrays
	.reshape(
		all_spat_filt_arrays.shape[0]*all_spat_filt_arrays.shape[1], all_spat_filt_arrays.shape[2]
		)
	.sum(axis=0)
	)
# -
# +
@numba.njit
def mk_spat_filt_overlapping_weights_proto(
		all_spat_filt_arrays: np.ndarray,
		) -> Dict[Tuple[int,...], Dict[int, float]]:

	# Trim spat filts to floor all negative values at 0
	# all_spat_filt_arrays[all_spat_filt_arrays<0]=0
	# calculate ahead of time total volume of each spat filt, and boolean array of postive values
	all_rf_sums = (
		all_spat_filt_arrays
		.reshape(
			all_spat_filt_arrays.shape[0]*all_spat_filt_arrays.shape[1], all_spat_filt_arrays.shape[2]
			)
		.sum(axis=0)
		)
	rf_pos_bool = all_spat_filt_arrays>0
	all_rf_portions = all_spat_filt_arrays / all_rf_sums  # portion of each pixel of total


	idx_vals = 2**np.arange(all_spat_filt_arrays.shape[2])

	# cell_overlap_map = numba.typed.Dict.empty(
	# 		key_type=numba.types.int64,
	# 		value_type=numba.types.float64[:]
	# 	)
	cell_overlap_map = dict()

	for i in range(all_spat_filt_arrays.shape[0]):
		for j in range(all_spat_filt_arrays.shape[1]):
			# find all RFs with postive (IE nonzero) value at these coordinates
			cell_overlap_idxs = np.nonzero(rf_pos_bool[i,j,:])[0]
			cell_overlap_idxs_int = np.sum(cell_overlap_idxs * idx_vals)
			# cell_overlap_idxs = tuple(np.nonzero(rf_pos_bool[i,j,:])[0])

			if not cell_overlap_idxs_int in cell_overlap_map:
				cell_overlap_map[cell_overlap_idxs_int] = np.zeros(all_spat_filt_arrays.shape[2])

			cell_overlap_map[cell_overlap_idxs_int] = (
				cell_overlap_map[cell_overlap_idxs_int] +
				all_rf_portions[i, j, :]
				)

	return cell_overlap_map
# -
# +
all_spat_filt_arrays = mk_all_lgn_rfs_array(lgn_layer, st_params)
all_spat_filt_arrays[all_spat_filt_arrays<0] = 0
# -
# +
com = mk_spat_filt_overlapping_weights_proto(all_spat_filt_arrays)
# -

# ---

# +
# calculate ahead of time total volume of each spat filt, and boolean array of postive values
all_rf_sums = all_spat_filt_arrays.sum(axis=(0,1))
rf_pos_bool = all_spat_filt_arrays>0
# -
# +
rf_pos_bool[300, 300]
# -
# +
test = np.argwhere(rf_pos_bool)
# -
# +
test_uniq = np.unique(test, axis=1)
# -
# +
test[-30:]
test_uniq.size
test.size
# -
# +
test_uniq = np.unique(rf_pos_bool, axis=2)
test_uniq.shape
# -
# +
rf_pos_bool_flat = rf_pos_bool.reshape(
	(rf_pos_bool.shape[0]*rf_pos_bool.shape[1], 30)
	)
# -
# +
rf_pos_bool_uniq = np.unique(rf_pos_bool_flat, axis=0)
# -
# +
rf_pos_bool_uniq.shape
# -

# +
idx_vals = 2**np.arange(len(lgn_layer.cells))
# -

# ## Faster Method Prototype
# +
lgn_layer = cells.mk_lgn_layer(lgn_params, st_params.spat_res)
idx_vals = 2**np.arange(len(lgn_layer.cells))
# -

# +
# prep spat filt arrays
all_spat_filt_arrays = mk_all_lgn_rfs_array(lgn_layer, st_params)
# -
# +
all_spat_filt_arrays[all_spat_filt_arrays<0] = 0

all_rf_sums = all_spat_filt_arrays.sum(axis=(0,1))  # sum of each spat filt
all_rf_portions = all_spat_filt_arrays / all_rf_sums  # portion of each pixel of total

rf_pos_bool = all_spat_filt_arrays>0  # where each spat filt has a positive value

rf_pos_bool_flat = rf_pos_bool.reshape(  # flatten to 1d-coords + spat_filts with positive values
		(rf_pos_bool.shape[0]*rf_pos_bool.shape[1], rf_pos_bool.shape[2])
	)

rf_idxs_bin = np.sum(rf_pos_bool_flat * idx_vals, axis=1)  # convert to integer by treating binary

portion_rf_idxs_sort_args = np.argsort(rf_idxs_bin)  # get idxs of flat arrays when sorted by rf_groups

rf_idxs_bin_sort = rf_idxs_bin[portion_rf_idxs_sort_args]

all_rf_portions_flat = all_rf_portions.reshape(  # flatten to 1d-coords + spat_filt portion values
		(all_rf_portions.shape[0]*all_rf_portions.shape[1], all_rf_portions.shape[2])
	)

# sort portions into rf groups
all_rf_portions_flat_sort = all_rf_portions_flat[portion_rf_idxs_sort_args, :]

# get idxs of when each group starts and ends
rf_idxs_bin_uniq, rf_portions_group_idxs = np.unique(rf_idxs_bin_sort, return_index=True)

all_intersection_portions = np.add.reduceat(all_rf_portions_flat_sort, rf_portions_group_idxs)
# -


# ### Make Function

# +
def mk_spat_filt_overlapping_rates_vect(
		all_spat_filt_arrays: np.ndarray
		) -> Dict[Tuple[int,...], Dict[int, float]]:

	all_spat_filt_arrays[all_spat_filt_arrays<0] = 0  # trim negative values

	# portion of each pixel of total
	all_rf_portions = all_spat_filt_arrays / all_spat_filt_arrays.sum(axis=(0,1))

	rf_pos_bool = all_spat_filt_arrays>0  # where each spat filt has a positive value

	# flatten to 1d-coords + spat_filts with positive values
	rf_pos_bool_flat = rf_pos_bool.reshape(
			(rf_pos_bool.shape[0]*rf_pos_bool.shape[1], rf_pos_bool.shape[2])
		)

	# convert all rf positive value booleans to integer by treating as binary
	# binary base values for creating ints from booleans
	idx_vals = 2**np.arange(all_spat_filt_arrays.shape[2])
	rf_pos_val_idxs_bin = np.sum(rf_pos_bool_flat * idx_vals, axis=1)

	# get indices of flat arrays when sorted by positive value keys
	portion_rf_idxs_sort_args = np.argsort(rf_pos_val_idxs_bin)

	# sort the integer indices
	rf_idxs_bin_sort = rf_pos_val_idxs_bin[portion_rf_idxs_sort_args]

	# reshape all spat_filt portion values to 1D-coords + RF portion values
	all_rf_portions_flat = all_rf_portions.reshape(  # flatten to 1d-coords + spat_filt portion values
			(all_rf_portions.shape[0]*all_rf_portions.shape[1], all_rf_portions.shape[2])
		)

	# sort portions by rf groups so that in same order as the integer indices above
	all_rf_portions_flat_sort = all_rf_portions_flat[portion_rf_idxs_sort_args, :]

	# get idxs of when each group starts and ends ... for a group-wise reduce next
	rf_idxs_bin_uniq, rf_portions_group_idxs = np.unique(rf_idxs_bin_sort, return_index=True)

	# The Main Show!!
	# add.reduceat performs addition for between each pair of indices ((0,1), (1,2), (2,3), ...)
	# as the group idxs from above are the starts+ends of each group of coords/pixels that belong
	# ... to a unique set of spat_filts with positive values at those coords ...
	# ... each of these sums will have the sum of all the spat_filt portions for each unique region
	# ... of overlapping spat_filts

	all_intersection_portions = np.add.reduceat(all_rf_portions_flat_sort, rf_portions_group_idxs)

	# construct dictionary with set of spat_filts as keys and dict of cell:val as values
	zfill_n = all_spat_filt_arrays.shape[2]

	# reconstruct boolean arrays of which spat_filts have positive values
	# done for all uniq rf_idx binary keys
	rf_idx_bools_sort = np.array([
		np.array([
			bool(int(d))  # boolean from integer of either '0' or '1' for each digit of the binary
				for d in
				# 1: binary
				# 2: remove '0b' from front
				# 3: pad (on left) to number of spat_filts
				#		must go here, as binary is right->left (2**n, 2**(n-1), ... 2**1, 2**0)
				#		so if bin(i) is less than n_spat_filts, it will be the large bases missing
				# 4: reverse to make left->right as I treat the indices as the bases in making
				#		the integer keys above.
				#   1   2  3             4
				bin(i)[2:].zfill(zfill_n)[::-1]
		])
		for i in rf_idxs_bin_uniq
	])

	# dictionary: tuple[cell_idxs] : dict[cell_idx: value, cell_idx: value, ...]
	cell_overlap_map = {}

	for i in range(rf_idx_bools_sort.shape[0]):
		cell_idxs = tuple(np.nonzero(rf_idx_bools_sort[i])[0])
		if cell_idxs:  # discard pixels where no cell has a positive value
			cell_overlap_map[cell_idxs] = (
				dict(
					zip(
						cell_idxs,
						all_intersection_portions[i, cell_idxs]
						)
					)
				)

	return cell_overlap_map
# -
# +
all_spat_filt_arrays = mk_all_lgn_rfs_array(lgn_layer, st_params)
cell_overlap_map = mk_spat_filt_overlapping_rates_vect(all_spat_filt_arrays)
# -


# ### Testing Accurate

# +
zfill_n = all_spat_filt_arrays.shape[2]

rf_idx_bools_sort = np.array([
	np.array([
		bool(int(d))
			for d in
			bin(i)[2:].zfill(zfill_n)[::-1]  # reverse as bin is right->left
	])
	for i in rf_idxs_bin_uniq
])
# -
# +
proto_cell_map = {}

for i in range(rf_idx_bools_sort.shape[0]):
	cell_idxs = tuple(np.nonzero(rf_idx_bools_sort[i])[0])
	if cell_idxs:
		proto_cell_map[cell_idxs] = dict(zip(cell_idxs, all_intersection_portions[i, cell_idxs]))
# -
# +
test_cell_map = mk_spat_filt_overlapping_weights(all_spat_filt_arrays)
# -
# +
if not set(test_cell_map.keys()).symmetric_difference(proto_cell_map.keys()):
	print('Keys or cell idxs are the same!!')
else:
	print('Not the same')

# sum(k in test_cell_map for k in proto_cell_map)
# print(list(k for k in proto_cell_map if k not in test_cell_map))
# len(test_cell_map)
# -

# +
all(
		(
			dict(v).keys() == proto_cell_map[k].keys(),
			np.isclose(
				np.array(sorted(dict(v).values())),
				np.array(sorted(proto_cell_map[k].values()))
				)
		)
	for k, v in test_cell_map.items()
	)
# -
# +
# val_mismatch = [
# 		(v, proto_cell_map[k])
# 		for k, v in test_cell_map.items()
# 		if dict(v) != proto_cell_map[k]
# 	]
# -



# +
for k in proto_cell_map:
	if k in test_cell_map:
		print(k)
print('NOT IN ')
for k in proto_cell_map:
	if not k in test_cell_map:
		print(k)
# -
# +
for k in test_cell_map:
	if k in proto_cell_map:
		print(k)
print('NOT IN ')
for k in test_cell_map:
	if not k in proto_cell_map:
		print(k)
# -


# ### OK, lets just get the overlapping keys right


# +
# prep spat filt arrays
all_spat_filt_arrays = mk_all_lgn_rfs_array(lgn_layer, st_params)
all_spat_filt_arrays[all_spat_filt_arrays<0] = 0
# -
# +
cell_overlap_map = dict()

for i in range(all_spat_filt_arrays.shape[0]):
	for j in range(all_spat_filt_arrays.shape[1]):
		# find all RFs with postive (IE nonzero) value at these coordinates
		cell_overlap_idxs = tuple(np.nonzero(rf_pos_bool[i,j,:])[0])
		cell_overlap_map[cell_overlap_idxs] = None
# -
# +
len(cell_overlap_map.keys())
sorted(cell_overlap_map.keys())
# -

# +
idx_vals = 2**np.arange(30)

rf_pos_bool = all_spat_filt_arrays>0  # where each spat filt has a positive value

rf_pos_bool_flat = rf_pos_bool.reshape(  # flatten to 1d-coords + spat_filts with positive values
		(rf_pos_bool.shape[0]*rf_pos_bool.shape[1], rf_pos_bool.shape[2])
	)

rf_idxs_bin = np.sum(rf_pos_bool_flat * idx_vals, axis=1).astype('uint')  # convert to integer by treating binary

portion_rf_idxs_sort_args = np.argsort(rf_idxs_bin)  # get idxs of flat arrays when sorted by rf_groups

rf_idxs_bin_sort = rf_idxs_bin[portion_rf_idxs_sort_args]

# get idxs of when each group starts and ends
rf_idxs_bin_uniq, rf_portions_group_idxs = np.unique(rf_idxs_bin_sort, return_index=True)

# all_intersection_portions = np.add.reduceat(all_rf_portions_flat_sort, rf_portions_group_idxs)
# -


# #### Getting bool[:] -> int -> bool[:] right

# +
rf_idxs_bool_uniq = np.unique(rf_pos_bool_flat, axis=0)
# -
# +
zfill_n = rf_idxs_bool_uniq.shape[1]

return_conv_match = {}
for i, r in enumerate(rf_idxs_bool_uniq):
	int_key = np.sum(r * idx_vals).astype('uint')
	new_bool = np.array([
					bool(int(d))
						for d in
						# bin(int_key)[2:].zfill(zfill_n)  # reverse as bin is right->left
						bin(int_key)[2:].zfill(zfill_n)[::-1]  # reverse as bin is right->left
				])
	is_match = (np.all(r == new_bool))

	return_conv_match[i] = {
		'match': is_match,
		'orig':r,
		'new': new_bool,
		'int': int_key
		}
# -
# +
sum(v['match'] for v in return_conv_match.values())
# -
# +
for k, v in return_conv_match.items():
	if not v['match']:
		print(i)
		print(v['orig'])
		print(v['new'])
# -
# +
non_match_idxs = [k for k, v in return_conv_match.items() if not v['match']]
# -
# +
non_match_idxs[:3]
# -
# +
idxs = rf_idxs_bool_uniq[2]

idxs_int = np.sum(idxs * idx_vals).astype('uint')
idxs_int_bin = bin(idxs_int)[2:].zfill(zfill_n)[::-1]
# len(bin(idxs_int)[2:])


new_idxs = np.array([
				bool(int(d))
					for d in
					# bin(int_key)[2:].zfill(zfill_n)  # reverse as bin is right->left
					idxs_int_bin  # reverse as bin is right->left
			])

print(idxs_int, idxs_int_bin)
print(idxs)
print(new_idxs)
# -



# +
rf_idxs_bool = np.array([
	np.array([
			bool(int(d))
				for d in
				bin(i)[2:][::-1].zfill(zfill_n)  # reverse as bin is right->left
		])
	for i in rf_idxs_bin_uniq
])
# -
# +
rf_idxs_bool[10]
# -
# +
rf_idxs_bool = {
	tuple(
		np.nonzero(
			np.array([
				bool(int(d))
					for d in
					bin(i)[2:][::-1].zfill(zfill_n)  # reverse as bin is right->left
			])
			)[0]
		)
	: None
	for i in rf_idxs_bin_uniq
}
# -
# +
len(rf_idxs_bool.keys())
test = rf_idxs_bin_uniq.astype('uint')
# -

# weights per number of overlapping cells.
# +
com_lens = [len(k) for k in cell_overlap_map.keys()]
com_len_wts = {}
for k, ovm in cell_overlap_map.items():
	lk = len(k)
	if lk in com_len_wts:
		com_len_wts[lk] += sum(ovm.values())
	else:
		com_len_wts[lk] = 0
		com_len_wts[lk] += sum(ovm.values())
# -
# +
px.scatter(x=list(com_len_wts.keys()), y=list(com_len_wts.values())).show()
# -
# +
px.histogram(com_lens).show()
# -


# Pair with most overlap

# +
from itertools import combinations
# -
# +
list(combinations((1, 2, 3, 4), 2))
# -
# +
cell_pair_wts = dict()
for k, ovm in cell_overlap_map.items():
	cell_pairs = list(combinations(k, 2))
	for cp in cell_pairs:
		if cp not in cell_pair_wts:
			cell_pair_wts[cp] = 0
		cell_pair_wts[cp] += (ovm[cp[0]] + ovm[cp[1]])
# -
# +
for cp in sorted(cell_pair_wts, key=lambda k: cell_pair_wts[k], reverse=True)[:10]:
	print(cp, cell_pair_wts[cp])
# -
# +
for cp in sorted(cell_pair_wts, key=lambda k: cell_pair_wts[k], reverse=True)[200:210]:
	print(cp, cell_pair_wts[cp])
# -
# +
for cp in sorted(cell_pair_wts, key=lambda k: cell_pair_wts[k], reverse=True)[300:310]:
	print(cp, cell_pair_wts[cp])
# -


# ## Test

# +
import lif.lgn.spat_filt_overlap as sfo
# -
# +
%timeit wts = sfo.mk_lgn_overlapping_weights(lgn_layer, st_params)
# -
# +
%timeit wts = sfo.mk_all_lgn_rfs_array(lgn_layer, st_params)
# -
# +
1000 * 2.14 / 3600
# -


# # Apply factors to LGN Response

# ## Create response

# +
temp_coords = ff.mk_temp_coords(st_params.temp_res, st_params.temp_ext)
# -
# +
all_resps = np.zeros(shape=(len(lgn_layer.cells), temp_coords.value.size))
for i, cell in enumerate(lgn_layer.cells):
	spatial_product = srf.mk_spat_filt_temp_response(
		st_params, stim_params, cell.spat_filt, cell.location, cell.orientation, temp_coords)
	all_resps[i, :] = spatial_product

all_resps[all_resps<0] = 0
# -
# +
# px.line(all_resps).show()
px.line(all_resps.sum(axis=0)).show()
# -
# +
fig = go.Figure()
for r in all_resps:
	fig = fig.add_scatter(y=r, mode='lines')
fig.add_scatter(
	y=all_resps.mean(axis=0), mode='lines', name='Avg',
	line=go.scatter.Line(color='red', width=5)
	)
fig.show()
# -

# +
len(cell_overlap_map.keys())
# -

# +
all_new_resps = np.zeros(shape=(len(cell_overlap_map.keys()), temp_coords.value.size))

for i, (com, ovs) in enumerate(cell_overlap_map.items()):
	test = (all_resps[com, :] * np.array(tuple(ovs.values()))[:, np.newaxis]).sum(axis=0)

	no_resp_idx = np.any((test == 0), axis=0)
	test[no_resp_idx] = 0

	all_new_resps[i, :] = test
# -
# +
np.allclose(
		all_new_resps.sum(axis=0),
		all_resps.sum(axis=0)
	)
# -
# +
fig = (
	go.Figure()
	.add_scatter(
		y=all_new_resps.sum(axis=0),
		mode='lines'
		)
	.add_scatter(
		y=all_resps.sum(axis=0),
		mode='lines'
		)
	)
fig.show()
# -


# # Create Poisson Spikes

# +
all_new_resps.shape
# -

# proto
# +
spikes = convolve.mk_response_poisson_spikes(st_params, tuple(resp for resp in all_new_resps))
# -
# +
spike_times = spikes.spike_trains()
len(spike_times.keys())
# -

# (20, 26) 1.9835365824967284
# (16, 25) 1.055576628458735
# (13, 21) 0.30600067540443193
# +
target_idxs = (13, 21)
# target_idxs = (16, 25)
# target_idxs = (20, 26)
# -
# +
cell_1_overlap = sum(
		ovm[target_idxs[0]]
		for k, ovm in cell_overlap_map.items()
		if ((target_idxs[0] in k) and (target_idxs[1] in k))
	)

cell_2_overlap = sum(
		ovm[target_idxs[1]]
		for k, ovm in cell_overlap_map.items()
		if ((target_idxs[0] in k) and (target_idxs[1] in k))
	)
# -
# +
cell_overlap_idxs_1 = [i for i, k in enumerate(cell_overlap_map) if target_idxs[0] in k]
cell_overlap_idxs_2 = [i for i, k in enumerate(cell_overlap_map) if target_idxs[1] in k]

cell_1_spikes = np.sort(np.r_[
	tuple(
			spike_times[i]
			for i in cell_overlap_idxs_1
		)
])

cell_2_spikes = np.sort(np.r_[
	tuple(
			spike_times[i]
			for i in cell_overlap_idxs_2
		)
])
# -
# +
fig = (
		go.Figure()
		.add_scatter(
			y=np.ones_like(cell_1_spikes),
			x=cell_1_spikes,
			mode='markers'
			)
		.add_scatter(
			y=np.ones_like(cell_2_spikes)*2,
			x=cell_2_spikes,
			mode='markers'
			)
	)
fig.show()
# -
# +
px.imshow(all_rfs[..., target_idxs[0]]).show()
px.imshow(all_rfs[..., target_idxs[1]]).show()
# -
# +
px.line(all_resps[target_idxs[0], ...]).show()
px.line(all_resps[target_idxs[1], ...]).show()
# -


# # Adding jitter to synchrony

# ## Generating Jitter

# +
gaussian_jitter = np.random.normal(0, 2, 10)
# -

# ## Organising Spikes

# +
# -
# +
jitter_sd = Time(10, 'ms')  # ms
# simulation temp res (could be determined earlier than here)
sim_temp_res = lifv1.defaultclock.dt / lifv1.bnun.msecond

%%timeit
# remove unit
all_synch_spikes = {
	k: spks / lifv1.bnun.msecond
	for k, spks in spike_times.items()
}

# add jitter
all_synch_spikes_w_jitter = {
	k: ((spks ) + np.random.normal(0, jitter_sd.ms, spks.size))
	for k, spks in all_synch_spikes.items()
}

# remove
all_synch_spikes_no_dbl = {
	k: (
		np.r_[spks[0], spks[1:][~(np.abs( spks[1:] - spks[0:-1] ) <= (sim_temp_res))]]
			if spks.size > 1
			else
				spks
		)
	for k, spks in all_synch_spikes_w_jitter.items()
}

# # check if any spikes within resolution
# all_synch_spikes_dbl_intvl = {
# 	k: (np.abs( spks[1:] - spks[0:-1] ) <= (sim_temp_res))
# 	for k, spks in all_synch_spikes_w_jitter.items()
# }

# # remove
# all_synch_spikes_no_dbl = {
# 	k: (
# 		np.r_[spks[0], spks[1:][~all_synch_spikes_dbl_intvl[k]]]
# 			if spks.size > 1
# 			else
# 				spks
# 		)
# 	for k, spks in all_synch_spikes_w_jitter.items()
# }
# -
# +
%%timeit
sim_temp_res = lifv1.defaultclock.dt / lifv1.bnun.msecond

all_synch_spikes = {}
for k, spks in spike_times.items():
	spks = (spks / lifv1.bnun.msecond) + np.random.normal(0, jitter_sd.ms, spks.size)
	spk_dup_idxs = (np.abs( spks[1:] - spks[0:-1] ) <= (sim_temp_res))


	de_dup_spks = (
		np.r_[spks[0], spks[1:][~spk_dup_idxs]]
		if spks.size > 1
		else
			spks
			)

	all_synch_spikes[k] = de_dup_spks
# -
# +
from numba import njit
# -
# +
@njit
def mk_jittered_spikes(spks, jitter_sd, sim_temp_res):
	spks = spks + np.random.normal(0, jitter_sd, spks.size)

	if spks.size <= 1:
		return spks
	else:
		spk_dup_idxs = ~(np.abs( spks[1:] - spks[0:-1] ) <= (sim_temp_res))
		de_dup_spks = np.empty(shape=(1+sum(spk_dup_idxs)))
		de_dup_spks[0] = spks[0]
		de_dup_spks[1:] = spks[1:][spk_dup_idxs]

	# de_dup_spks = (
	# 	np.r_[spks[0], spks[1:][spk_dup_idxs]]
	# 	if spks.size > 1
	# 	else
	# 		spks
	# 		)

	return de_dup_spks
# -
# +
%%timeit
all_synch_spikes = {
	k: mk_jittered_spikes(spks/lifv1.bnun.msecond, jitter_sd.ms, sim_temp_res)
	for k, spks in spike_times.items()
}
# -
# +
print(
	sum(int(np.any(dbl_intv)) for dbl_intv in all_synch_spikes_dbl_intvl.values()),
	sum(int(
		np.any(
			np.abs( spks[1:] - spks[0:-1] ) <= (sim_temp_res)
			)
		)
		for spks in all_synch_spikes_no_dbl.values())
	)
# -


# # Dealing with parallel simulations and indices etc


# ## LGN Poisson pocesses

# ### Indices for tiling response arrays into trials

# Maybe use the `n_inputs` argument as an optional tuple type with the number of lengths per layer


# +
random_lens = (3, 2, 4)
n_trials = 3
# random_lens = tuple(random.randint(450, 550) for _ in range(1000))
# n_trials = 30
offsets = np.r_[0, np.cumsum(random_lens)[:-1]]
repeated_cell_idxs = np.r_[
	tuple(
			np.tile(np.arange(rl)+cum_offset, n_trials)
			for rl, cum_offset in zip(random_lens, offsets)
		)
]
# -

# #### Test
# +
cells.mk_repeated_lgn_cell_idxs(3, (0, 2, 3))
repeated_cell_idxs
# -

# +
random_lens = (3, 5, 9)
n_trials = 4


# response rates
# numbers only a guide to represent where the elements actually come from
# while the array represents what would be a homogenous sequence of elements each being
# the response of a single cell/overlapping region from a single layer
resp_rates = np.r_[tuple(np.arange(rl) for rl in random_lens)]

all_cell_idxs = (np.arange(rl) for rl in random_lens)

# same as above but now with repeated trials
all_idxs = np.r_[tuple(
	np.tile(idxs, n_trials)
		for idxs in all_cell_idxs
	)]
# -
# +
resp_rates
# -
# +
all_idxs
# -
len(repeated_cell_idxs)
repeated_cell_idxs.nbytes / 10**6
# +
# offsets = it.islice(it.accumulate(random_lens, initial=0), len(random_lens))
# list(offsets)
# -
# +
offsets = it.islice(it.accumulate(random_lens, initial=0), len(random_lens))
repeated_cell_idxs2 = list(
	it.chain.from_iterable(
			it.chain.from_iterable(
				it.repeat(
					tuple((idx + cum_offset) for idx in range(rl)),
					n_trials
					)
				)
			for rl, cum_offset in zip(random_lens, offsets)
	)
)
# -
# +
np.all(repeated_cell_idxs == np.array(repeated_cell_idxs2))
# -



# +
# all_cell_idxs = (np.arange(rl) for rl in random_lens)
all_cell_idxs = it.chain.from_iterable(range(rl) for rl in random_lens)
# list(all_cell_idxs)

layer_offset_vals = (
	it.chain.from_iterable(
		it.repeat(offset, offset) for offset in it.accumulate(random_lens)
		)
	)
list(layer_offset_vals)

for i, rl in enumerate(random_lens):
	for t in range(n_trials):
		for c in range(rl):
			print(c + (i * n_trials * rl))

# same as above but now with repeated trials
all_idxs = np.r_[tuple(
	np.tile(idxs, n_trials)
		for i, idxs in enumerate(all_cell_idxs)
	)]
# -

# +
random_lens = (5, 7, 3)
n_trials = 2

# model of the lists of responses that need to be processes
# each number representing the cell number within its layer of each cell for all layers

# cell number for each cell in all layers
all_cell_idxs = tuple(np.arange(rl) for rl in random_lens)

# same as above but now with repeated trials
all_idxs = np.r_[tuple(
	np.tile(idxs, n_trials)
		for idxs in all_cell_idxs
	)]
# -
# +
# model of the response_array data structure with each number representing a single cell
# enumerated by its position in its specific layer
resp_rates = np.r_[tuple(np.arange(rl) for rl in random_lens)]
# -
# +
# random_lens = tuple(random.randint(450, 550) for _ in range(1000))
# n_trials = 30

# each pair is the start and end of the slices that will provide each trial-layer
# works by taking the n_cells of the layer, repeating those lengths n_trials times
# ... then doing a cumulative sum over these lengths.
# The result is each number is the amount of elements/indices between each trial-layer
# ... where the jumps in cell numbers for different trials are accounted for.

trial_idxs = tuple(
	it.accumulate(
		it.chain.from_iterable(
				it.repeat(rl, n_trials)
				for rl in random_lens
			),
		initial=0
		)
	)
# -
# +
all_trial_idxs = [
	all_idxs[a : b]
	for a,b in zip(trial_idxs[:-1], trial_idxs[1:])
]
# -
# +

# indices for the response rates to pair up appropriately with the all_trial_idxs
# works similar to that above for trial-layers, except, as each trial has the same response rates
# ... only the data for first trial is needed

# indices for each layer when all cells are flattened *without trials*.
# each pair of indices (eg, [0, 1], [1, 2], [2, 3], ...) provide the slicing indices to get all
# ... of the cells of a single layer
resp_rate_idxs_base = tuple(
		it.accumulate(
			random_lens,
			initial=0)
	)

# take all start indices (of the slice for each layer) and repeat for all trials
starts = tuple(it.chain.from_iterable(
		it.repeat(rr_idx, n_trials)
		for rr_idx in resp_rate_idxs_base[:-1]
	))

# take all end indices (of the slice for each layer) and repeat for all trials
ends = tuple(it.chain.from_iterable(
		it.repeat(rr_idx, n_trials)
		for rr_idx in resp_rate_idxs_base[1:]
	))

resp_rate_idxs = tuple(zip(starts, ends))

for a,b in resp_rate_idxs:
	print(a,b)
# -

# +
len(resp_rate_idxs) == len(all_trial_idxs)
# -
# +
for rr, allt in zip(resp_rate_idxs, all_trial_idxs):
	print(resp_rates[rr[0]:rr[1]], allt)
# -
# +

# -






# +
test = np.arange(25).reshape(5,5)

test_l = [r for r in test]
# -
# +
test_l
# -
# +
test = np.arange(25).reshape(5,5)
# -
# +
test *= 10
test_l
# -



# # V1 Indexing and Synaptic Weight Normalising

# ## Synaptic Weight Normalising
# +
bn = lifv1.bn
bnun = lifv1.bnun
# -
# +
eqs = '''
dv/dt = (v_rest - v + (I/g_EPSC))/tau_m : volt
dI/dt = -I/tau_EPSC : amp
'''

# on_pre =    'I += EPSC'
threshold = 'v>v_thres'
reset =     'v = v_reset'
# -
# +
n_synapses = 30
lif_params = do.LIFParams()
lif_params_w_units = lif_params.mk_dict_with_units(n_inputs=False)
# -
# +
G = bn.NeuronGroup(
	N=2,
	# N=1,
	model=eqs,
	threshold=threshold, reset=reset,
	namespace=lif_params_w_units,
	method='euler')


G.v = lif_params_w_units['v_rest']

dummy_spk_times_1 = np.random.randint(0, 100, 20)
dummy_spk_times_2 = np.random.randint(0, 100, 50)
dummy_spk_times = np.r_[dummy_spk_times_1, dummy_spk_times_2] * bnun.msecond
dummy_spk_idxs = np.arange(dummy_spk_times.size)
v1_idxs = np.r_[np.zeros(dummy_spk_times_1.size), np.ones(dummy_spk_times_2.size)].astype(int)
# dummy_spk_idxs = np.arange(n_synapses)
# dummy_spk_times = dummy_spk_idxs * 2 * bnun.msecond

PS = bn.SpikeGeneratorGroup(
	N=dummy_spk_times.size,
	indices=dummy_spk_idxs,
	times=dummy_spk_times,
	sorted=False
	)

on_pre =    'I += EPSC/N_incoming'
S = bn.Synapses(PS, G, on_pre=on_pre, namespace=lif_params_w_units)

S.connect(i=dummy_spk_idxs, j=v1_idxs)
M = bn.StateMonitor(G, 'I', record=True)
# -
# +
bn.run(500*bnun.msecond)
# -
# +
px.line(M[0].I/bnun.namp).show()
px.line(M[1].I/bnun.namp).show()
# -
# +
S.N_incoming_post
# -
# +
M = bn.StateMonitor(G, 'I', record=True)
# -


# # Test and Review

# ## Testing Basic Overlap Region Weights Calculation

# +
lgn_layer = cells.mk_lgn_layer(lgn_params, st_params.spat_res)
# -
# +
all_rfs = sfo.mk_all_lgn_rfs_array(lgn_layer, st_params)
overlapping_wts = sfo.mk_lgn_overlapping_weights_vect(lgn_layer, st_params)
# -

# +
multi_stim_params = do.MultiStimulusGeneratorParams(
	spat_freqs=[1.2], temp_freqs=[4], orientations=[90], contrasts=[0.3]
	# spat_freqs=[0.8], temp_freqs=[4], orientations=[90], contrasts=[0.3]
	)

stim_params = stimulus.mk_multi_stimulus_params(multi_stim_params)[0]
# -
# +
temp_coords = ff.mk_temp_coords(st_params.temp_res, st_params.temp_ext)
# -
# +
all_resps = np.zeros(shape=(len(lgn_layer.cells), temp_coords.value.size))
for i, cell in enumerate(lgn_layer.cells):
	spatial_product = srf.mk_spat_filt_temp_response(
		st_params, stim_params, cell.spat_filt, cell.location, cell.orientation, temp_coords)
	all_resps[i, :] = spatial_product

all_resps[all_resps<0] = 0
# -
# +
# px.line(all_resps).show()
px.line(all_resps.sum(axis=0)).show()
# -
# +
fig = go.Figure()
for r in all_resps:
	fig = fig.add_scatter(y=r, mode='lines')
fig.add_scatter(
	y=all_resps.mean(axis=0), mode='lines', name='Avg',
	line=go.scatter.Line(color='red', width=5)
	)
fig.show()
# -


# +
all_new_resps = np.zeros(shape=(len(overlapping_wts.keys()), temp_coords.value.size))
for i, (com, ovs) in enumerate(overlapping_wts.items()):
	test = (all_resps[com, :] * np.array(tuple(ovs.values()))[:, np.newaxis]).sum(axis=0)

	# no_resp_idx = np.any((all_resps[com, :] == 0), axis=0)
	# test[no_resp_idx] = 0

	all_new_resps[i, :] = test
# -
# +
np.allclose(
		all_new_resps.sum(axis=0),
		all_resps.sum(axis=0)
	)
# -
# +
px.line(
	y = all_new_resps.sum(axis=0) - all_resps.sum(axis=0)
).show()

# -

# +
fig = (
	go.Figure()
	.add_scatter(
		y=all_new_resps.sum(axis=0),
		mode='lines', name='new'
		)
	.add_scatter(
		y=all_resps.sum(axis=0),
		mode='lines', name='original cells'
		)
	)
fig.show()
# -

# +
fig = go.Figure()
for r in all_new_resps:
	fig = fig.add_scatter(y=r, mode='lines')
fig.add_scatter(
	y=all_new_resps.mean(axis=0), mode='lines', name='Avg',
	line=go.scatter.Line(color='red', width=5)
	)
fig.show()
# -

# +
olr_idx = 102
olr_keys = tuple(overlapping_wts.keys())
key = olr_keys[olr_idx]

fig = (go.Figure())
for r_idx in key:
	fig = fig.add_scatter(
		y=all_resps[r_idx, :] * overlapping_wts[key][r_idx],
		mode='lines', name=f'cell_{r_idx}')

fig = fig.add_scatter(
	y=all_new_resps[olr_idx, :],
	mode='lines', name='region',
	line=go.scatter.Line(color='red', width=5))
fig.show()
# -



# ## Testing New `lgn_response_spikes` funciton

# +
spat_res=ArcLength(1, 'mnt')
spat_ext=ArcLength(660, 'mnt')
"Good high value that should include any/all worst case scenarios"
temp_res=Time(1, 'ms')
temp_ext=Time(1000, 'ms')

st_params = do.SpaceTimeParams(
	spat_ext, spat_res, temp_ext, temp_res,
	array_dtype='float32'
	)
# -

# +

# -

# +
response_pulse = np.ones(400)*50
response_arrays = (
		np.r_[np.zeros(100), response_pulse, np.zeros(500)],
		np.r_[np.zeros(300), response_pulse, np.zeros(300)],
		np.r_[np.zeros(500), response_pulse, np.zeros(100)],
		np.r_[np.zeros(600), response_pulse],
	)
# -
# +
lgn_resp = convolve.mk_lgn_response_spikes(
		st_params,
		response_arrays, n_trials = None
	)
# -
# +
fig = go.Figure()
for i, c in enumerate(lgn_resp.cell_spike_times):
	fig.add_scatter(
		y=i * np.ones_like(c.value),
		x=c.ms,
		mode='markers'
		)

fig.show()
# -


# +
response_pulse = np.ones(400)*50
# 3 layers, n_cells = (3, 1, 3)
response_arrays = (
		np.r_[np.zeros(600), response_pulse],
		np.r_[np.zeros(100), response_pulse, np.zeros(500)],
		np.r_[np.zeros(300), response_pulse, np.zeros(300)],
		np.r_[np.zeros(150), response_pulse, np.zeros(450)],
		np.r_[np.zeros(450), response_pulse, np.zeros(150)],
		np.r_[np.zeros(500), response_pulse, np.zeros(100)],
		np.r_[np.zeros(500), response_pulse, np.zeros(100)],
	)
# -
# +
n_trials = 9
lgn_resp = convolve.mk_lgn_response_spikes(
		st_params,
		(response_arrays), n_trials = n_trials, n_lgn_layers=2, n_inputs=(2, 2, 3)
	)

# lgn_resp = convolve.mk_lgn_response_spikes(
# 		st_params,
# 		(response_arrays), n_trials = n_trials, n_lgn_layers=2
# 	)

# lgn_resp = convolve.mk_lgn_response_spikes(
# 		st_params,
# 		(response_arrays[0],), n_trials = n_trials
# 	)
# -

# +
colors = ['red', 'green', 'blue', 'magenta']
fig = go.Figure()
for t, l in enumerate(lgn_resp):  # layers
	for i, c in enumerate(l.cell_spike_times):  # cells in layer
		fig.add_scatter(
			y=(i * np.ones_like(c.value)) + (len(response_arrays) * t),
			x=c.ms,
			mode='markers',
			legendgroup = f'cell {i}',
			marker_color=colors[i]
			)

fig.show()
# -


# ## Creating V1 input information


# +
sim_params = do.SimulationParams(
	n_simulations=5,
	space_time_params=st_params,
	multi_stim_params=multi_stim_params,
	lgn_params=lgn_params,
	lif_params = lif_params,
	n_trials = 2
	# n_trials = 10
	)
# -
# +
# Create new
all_lgn_layers = run.create_all_lgn_layers(sim_params, force_central_rf_locations=False)

# Load old layer
# -

# ### Create overlap maps
# +
# create lgn layer overlapping regions data
all_lgn_overlap_maps = run.create_all_lgn_layer_overlapping_regions(all_lgn_layers, sim_params)
# -
# +
overlap_map = all_lgn_overlap_maps[stim_params.contrast]
# -


# ### Prototype v1 index code and test

# +
n_overlap_regions = tuple(len(layer) for layer in overlap_map)
# -
# +
cells.mk_repeated_v1_indices_for_inputs_for_all_lgn_and_trial_synapses(n_trials=5, n_inputs=10)
# -
# +
total_v1_cells = 3 * 2
# total_v1_cells = sim_params.n_simulations * sim_params.n_trials

v1_idxs = np.array(cells.mk_repeated_v1_indices_for_inputs_for_all_lgn_and_trial_synapses(
	n_trials=total_v1_cells,
	n_inputs=2
	)
)
# -
# +
v1_idxs
# -
# +
n_simulations = 10
n_trials = 10
n_inputs = [3, 5, 2, 3, 4, 4, 5, 8, 3, 4]
# n_inputs = np.random.randint(450, 550, n_simulations)
total_v1_cells = n_simulations * n_trials

assert total_v1_cells == len(n_inputs) * n_trials

v1_synapse_indices = tuple(
		n_trial  # each trial has just one V1 cell, so n_trial = n_v1_cell
		for n_trial, n_inputs in enumerate(
					it.chain.from_iterable(
						# repeat each layer n_trials times (as trials x layers)
						it.repeat(n_input, n_trials) for n_input in n_inputs
					)
				)
			for _ in range(n_inputs)  # dummy to get repeats
			# repeat the same v1 cell index for each lgn cell/input
	)

assert len(v1_synapse_indices) == sum(n_inputs) * n_trials

#       V- last is same as max                   V- minus 1 as idxs start 0
assert v1_synapse_indices[-1] == (total_v1_cells - 1)

np.array(v1_synapse_indices)
# np.array(v1_synapse_indices).size
# -
# +
np.array(
		cells.mk_repeated_v1_indices_for_inputs_for_all_lgn_and_trial_synapses(
			n_trials = 3, n_inputs = [2, 3, 1, 3]
		)
	)
# -

# ## Creating LGN Spike time and index information for synchrony

# Gotta make sure the function does what it's supposed to

# ### Creating LGN responses

# #### Params

# +
spat_res=ArcLength(1, 'mnt')
spat_ext=ArcLength(660, 'mnt')
"Good high value that should include any/all worst case scenarios"
temp_res=Time(1, 'ms')
temp_ext=Time(1000, 'ms')

st_params = do.SpaceTimeParams(
	spat_ext, spat_res, temp_ext, temp_res,
	array_dtype='float32'
	)
# -
# +
multi_stim_params = do.MultiStimulusGeneratorParams(
	spat_freqs=[0.8], temp_freqs=[4], orientations=[90], contrasts=[0.3]
	# spat_freqs=[0.8], temp_freqs=[4], orientations=[90], contrasts=[0.3]
	)

stim_params = stimulus.mk_multi_stimulus_params(multi_stim_params)[0]
# -
# +
lgn_params = do.LGNParams(
	n_cells=30,
	orientation = do.LGNOrientationParams(ArcLength(0), circ_var=0.5),
	circ_var = do.LGNCircVarParams('naito_lg_highsf', 'naito'),
	spread = do.LGNLocationParams(2, 'jin_etal_on'),
	filters = do.LGNFilterParams(spat_filters='all', temp_filters='all'),
	F1_amps = do.LGNF1AmpDistParams()
	)
# -
# +
lif_params = do.LIFParams(total_EPSC=3.5)
sim_params = do.SimulationParams(
	n_simulations=4,
	space_time_params=st_params,
	multi_stim_params=multi_stim_params,
	lgn_params=lgn_params,
	lif_params = lif_params,
	n_trials = 3
	# n_trials = 10
	)
# -
# +
# Create new
all_lgn_layers = run.create_all_lgn_layers(sim_params, force_central_rf_locations=False)
# -



# ### Create overlap maps
# +
# create lgn layer overlapping regions data
all_lgn_overlap_maps = run.create_all_lgn_layer_overlapping_regions(all_lgn_layers, sim_params)
# -
# +
overlap_maps = all_lgn_overlap_maps[stim_params.contrast]
# -

# +
test = overlap_maps[0]
tk = list(test.keys())[100]
test[tk]
# -


# +
n_inputs = [len(overlap_map) for overlap_map in overlap_maps]
# -

# ### Create Response Arrays
# +
actual_max_f1_amps = all_max_f1.mk_actual_max_f1_amps(stim_params=stim_params)
# -
# +
lgn_layers = all_lgn_layers[stim_params.contrast]
all_responses = list()
for i, lgn in enumerate(lgn_layers):
	lgn_resp = run.loop_lgn_cells_mk_response(
					lgn,
					sim_params, None, actual_max_f1_amps, stim_params,
					analytical=True
				)
	all_responses.append(lgn_resp)
# -

# #### Synchronous Reponse Arrays

# +
from typing import Sequence
response_arrays_collector: Sequence[np.ndarray] = list()

# get temporal dimension size
temp_array_size = all_responses[0][0].response.size
# for each layer
for i, layer_responses in enumerate(all_responses):
	overlap_map = overlap_maps[i]
	# n_regions = len(overlap_map)

	# put all original cell response arrays into a 2D array for better processing
	layer_responses_array = np.zeros(shape=(len(layer_responses), temp_array_size))
	for i, cell_response in enumerate(layer_responses):
		layer_responses_array[i] = cell_response.response

	# construct new response array based on number of overlapping regions
	# all_new_resps = np.zeros(shape=(n_regions, temp_array_size))

	# multiply out new responses
	# For each overlapping region ...
	for i, (cell_idxs, overlap_wts) in enumerate(overlap_map.items()):
		# cell_idxs: indices of lgn_layer.cells of the cells overlapping in this region
		# overlap_wts: contributions of each cell to this region, relative to cell's magnitude

		# multiply each cell's response by its weighting for the overlapping region
		# then sum over all the cells to create a single response array
		# 1: aggregate into single response (axis=0 means over the cells)
		# 2: get response arrays of all overlapping cells for this region
		# 3: multiply response arrays by wts
		#       - convert values to array (via tuple) which will be in same order as cell_idxs
		#       - slice with newaxis so that the values will be broadcast onto the cell resonses
		overlapping_region_response: np.ndarray = np.sum(           # 1
			layer_responses_array[cell_idxs, :]                     # 2
			*
			np.array(tuple(overlap_wts.values()))[:, np.newaxis]    # 3
			,
			axis=0
			)

		# add response array to giant flattened list of response arrays
		response_arrays_collector.append(overlapping_region_response)

# Flattened tuple of response arrays: (r11, r12, ... r1n, ... r21, r22, ... r_mn)
#   where m = number of layers, n = number of OVERLAPPING REGIONS per layer
#     /- Temporal response of a single OVERLAPPING REGION
#     |                   /- Response arrays of each REGION from a single LGN layer
#     |                   V     of length `len(overlap_map)` FOR EACH LAYER
#     V               |--------------------|
#  ( array[], array[], ... array[], array[] )
response_arrays = tuple(response_arrays_collector)
# -

# +
lgn_layer_responses = convolve.mk_lgn_response_spikes(
		sim_params.space_time_params,
		response_arrays = response_arrays,
		n_trials = sim_params.n_trials,
		n_lgn_layers=len(lgn_layers),
		# n_lgn_layers=len(all_lgn_layers),
		n_inputs=n_inputs  # either int or list of ints if synchrony and variable n cells per layer
	)
# -
len(lgn_layer_responses)


# #### Not Synchronous

# +
# Flattened tuple of response arrays: (r11, r12, ... r1n, ... r21, r22, ... r_mn)
#   where m = number of layers, n = number of cells per layer
#     |- Temporal response of a single cell
#     |                   |- Response arrays of each cell from a single LGN layer
#     |                   V     of length lgn_params.n_cells (ie, `n` input cells)
#     V               |--------------------|
#  ( array[], array[], ... array[], array[] )
response_arrays_not_synch = tuple(
		single_response.response
		for responses in all_responses  # first layer is responses for a single layer
			for single_response in responses # second layer is cellular responses within a layer
	)
# -
# +
lgn_layer_responses_not_synch = convolve.mk_lgn_response_spikes(
		sim_params.space_time_params,
		response_arrays = response_arrays_not_synch,
		n_trials = sim_params.n_trials,
		n_lgn_layers=len(lgn_layers),
		n_inputs=sim_params.lgn_params.n_cells  # either int or list of ints if synchrony and variable n cells per layer
	)
# -
# +
len(lgn_layer_responses_not_synch)
# -

# ### LGN Spike idxs and times

# +
spike_idxs, spike_times = lifv1.mk_input_spike_indexed_arrays(lgn_response=lgn_layer_responses)
# -
# +
spike_idxs[-10:]
# -
# +
px.scatter(y=spike_idxs, x=spike_times.value).show()
# -
# +
all_spike_times = tuple(
		spike_times
		for response in lgn_layer_responses
			for spike_times in response.cell_spike_times
	)
# -
# +
len(all_spike_times)
# -
# +
for llr in lgn_layer_responses:
	print(len(llr.cell_spike_times))
sum(len(llr.cell_spike_times) for llr in lgn_layer_responses)
# -

# ### Test V1 synapse alignment

# maybe use fewer cells in the lgn layers to have fewer overlapping regions?
# Then create v1 object, by passing in the number of overlapping regions
# Then check that the number of synapses for each v1 cell / layer match the number of overlapping
# regions for that layer
# Check anything else?

# +
v1_model = run.create_multi_v1_lif_network(sim_params, overlap_maps)
# -
# +
v1_model.input_spike_generator
v1_model.network.objects
S = list(v1_model.network.objects)[-1]
G = list(v1_model.network.objects)[2]
# -
# +
S.N_incoming_post
n_inputs
sim_params.n_trials
# -
# +
G.N
# -


# ### Testing a run (??)
# +
v1_model.reset_spikes(spike_idxs, spike_times, spikes_sorted=False)
v1_model.run(sim_params.space_time_params)
# -
# +
v1_model.membrane_monitor.v.shape
# -
# +
for i in range(sim_params.n_trials):
	px.line(y=v1_model.membrane_monitor.v[i]).show()
# -
# +
px.line(y=v1_model.membrane_monitor.v[0]).show()
# -
# +
st = v1_model.spike_monitor.spike_trains()
# -
# +
# -


# ### Comparing to no synchrony
# +
spike_idxs_not_synch, spike_times_not_synch = (
	lifv1.mk_input_spike_indexed_arrays(lgn_response=lgn_layer_responses_not_synch)
	)
# -
spike_idxs_not_synch.shape
# +
px.scatter(y=spike_idxs_not_synch, x=spike_times_not_synch.value).show()
# -

# +
v1_model_not_synch = run.create_multi_v1_lif_network(sim_params, overlap_map=None)
# -
# +
v1_model_not_synch.input_spike_generator
v1_model_not_synch.network.objects
ntwk_objs = list(v1_model_not_synch.network.objects)
S = ntwk_objs[1]
G = ntwk_objs[3]
# -
# +
S.N_incoming_post
n_inputs
sim_params.n_trials
# -
# +
G.N
# -

# +
v1_model_not_synch.reset_spikes(spike_idxs_not_synch, spike_times_not_synch, spikes_sorted=False)
v1_model_not_synch.run(sim_params.space_time_params)
# -
# +
v1_model_not_synch.membrane_monitor.v.shape
# -
# +
for i in range(sim_params.n_trials):
	px.line(y=v1_model_not_synch.membrane_monitor.v[i]).show()
# -


# +
px.line(y=v1_model.membrane_monitor.v[0]).show()

px.line(y=v1_model_not_synch.membrane_monitor.v[0]).show()
# -

# +
spike_times_not_synch.value.shape, spike_times.value.shape
# -


# OK ... so this approach to synchrony actually creates less synchrony as all of the spikes are
# spread out amongst **more** not **fewer** poisson processes.


# ## Fixing Synchrony

# Duplicate the spikes to all source cells!
# * First divide the rates by the number of duplications that will be made
# * Then copy the spikes to each source cell
# * And, if wanted, add jitter

# ### LGN Rates and Spikes

# #### Adjusted Rates

# +
# create lgn layer overlapping regions data
all_lgn_overlap_maps = run.create_all_lgn_layer_overlapping_regions(all_lgn_layers, sim_params)
# -
# +
overlap_maps = all_lgn_overlap_maps[stim_params.contrast]
# -
# +
eg_om = overlap_maps[0]
omks = list(eg_om.keys())

# These ... need to be divided by their number (ie, length of the dict)
eg_om[omks[100]]
len(eg_om[omks[100]])
# -
# +
adjusted_om = {
	cell_idxs: {
		cell_idx: cell_wt/len(cell_idxs)
		for cell_idx, cell_wt in cell_wts.items()
	}
	for cell_idxs, cell_wts in eg_om.items()
}
# -
# +
eg_om[omks[100]], adjusted_om[omks[100]]
# -


# #### Making Adjusted Rates

# +
adjusted_overlap_maps = tuple(
			{
			cell_idxs: {
				cell_idx: cell_wt/len(cell_idxs)
				for cell_idx, cell_wt in cell_wts.items()
			}
			for cell_idxs, cell_wts in eg_om.items()
		}
		for eg_om in overlap_maps
	)
# -
len(adjusted_overlap_maps)

# +
def mk_adjusted_overlapping_regions_wts(
		all_lgn_overlapping_maps: Dict[do.ContrastValue, Tuple[sfo.LGNOverlapMap, ...]]
		) -> Dict[do.ContrastValue, Tuple[sfo.LGNOverlapMap, ...]]:

	all_adjusted_overlap_maps = dict()

	for contrast_value, overlap_maps in all_lgn_overlapping_maps.items():
		adjusted_overlap_maps = tuple(
					{
					cell_idxs: {
						cell_idx: cell_wt / len(cell_idxs)
						for cell_idx, cell_wt in cell_wts.items()
					}
					for cell_idxs, cell_wts in overlap_map.items()
				}
				for overlap_map in overlap_maps
			)

		all_adjusted_overlap_maps[contrast_value] = adjusted_overlap_maps

	return all_adjusted_overlap_maps
# -



# #### Generate Spikes as before


# +
n_inputs = [len(overlap_map) for overlap_map in overlap_maps]
# -
# +
# not necessary if done above for ordinary (not synchrony) run (?)
lgn_layers = all_lgn_layers[stim_params.contrast]
all_responses = list()
for i, lgn in enumerate(lgn_layers):
	lgn_resp = run.loop_lgn_cells_mk_response(
					lgn,
					sim_params, None, actual_max_f1_amps, stim_params,
					analytical=True
				)
	all_responses.append(lgn_resp)
# -
# +
from typing import Sequence
response_arrays_collector: Sequence[np.ndarray] = list()

# get temporal dimension size
temp_array_size = all_responses[0][0].response.size
# for each layer
for i, layer_responses in enumerate(all_responses):
	overlap_map = adjusted_overlap_maps[i]
	# n_regions = len(overlap_map)

	# put all original cell response arrays into a 2D array for better processing
	layer_responses_array = np.zeros(shape=(len(layer_responses), temp_array_size))
	for i, cell_response in enumerate(layer_responses):
		layer_responses_array[i] = cell_response.response

	# construct new response array based on number of overlapping regions
	# all_new_resps = np.zeros(shape=(n_regions, temp_array_size))

	# multiply out new responses
	# For each overlapping region ...
	for i, (cell_idxs, overlap_wts) in enumerate(overlap_map.items()):
		# cell_idxs: indices of lgn_layer.cells of the cells overlapping in this region
		# overlap_wts: contributions of each cell to this region, relative to cell's magnitude

		# multiply each cell's response by its weighting for the overlapping region
		# then sum over all the cells to create a single response array
		# 1: aggregate into single response (axis=0 means over the cells)
		# 2: get response arrays of all overlapping cells for this region
		# 3: multiply response arrays by wts
		#       - convert values to array (via tuple) which will be in same order as cell_idxs
		#       - slice with newaxis so that the values will be broadcast onto the cell resonses
		overlapping_region_response: np.ndarray = np.sum(           # 1
			layer_responses_array[cell_idxs, :]                     # 2
			*
			np.array(tuple(overlap_wts.values()))[:, np.newaxis]    # 3
			,
			axis=0
			)

		# add response array to giant flattened list of response arrays
		response_arrays_collector.append(overlapping_region_response)

# Flattened tuple of response arrays: (r11, r12, ... r1n, ... r21, r22, ... r_mn)
#   where m = number of layers, n = number of OVERLAPPING REGIONS per layer
#     /- Temporal response of a single OVERLAPPING REGION
#     |                   /- Response arrays of each REGION from a single LGN layer
#     |                   V     of length `len(overlap_map)` FOR EACH LAYER
#     V               |--------------------|
#  ( array[], array[], ... array[], array[] )
response_arrays = tuple(response_arrays_collector)
# -
# +
lgn_layer_responses = convolve.mk_lgn_response_spikes(
		sim_params.space_time_params,
		response_arrays = response_arrays,
		n_trials = sim_params.n_trials,
		n_lgn_layers=len(lgn_layers),
		# n_lgn_layers=len(all_lgn_layers),
		n_inputs=n_inputs  # either int or list of ints if synchrony and variable n cells per layer
	)
# -
len(lgn_layer_responses)
len(lgn_layer_responses[0].cell_spike_times)
len(response_arrays)

lgn_layer_responses[0]

# +
spike_idxs, spike_times = lifv1.mk_input_spike_indexed_arrays(lgn_response=lgn_layer_responses)
# -
# +
# spike_idxs[-10:]
# -
# +
spike_times.value.shape

# should be about 10 times less than normal ...
# overlap_map = all_lgn_overlap_maps[stim_params.contrast]
# np.mean([len(k) for k in overlap_map[0]])
# -
# +
px.scatter(y=spike_idxs, x=spike_times.value).show()
# -
spike_idxs[-100:]



# ### Copying Spikes to create synchrony

# Copy spikes to cells that are the source of the overlapping region

# +
n_layers = sim_params.n_simulations
n_trials = sim_params.n_trials
n_cells = sim_params.lgn_params.n_cells  # IE, true LGN cells ... we're getting back to this here
# -
# +
overlapping_region_to_lgn_cell_idxs_map = list()
spike_idx_cursor = 0

for n_layer in range(n_layers):
	n_overlapping_regions = len(adjusted_overlap_maps[n_layer])

	for n_trial in range(n_trials):

		spike_idx_start, spike_idx_end = (
			spike_idx_cursor,
			(spike_idx_cursor + (n_overlapping_regions))
			)

		spike_idx_range = (spike_idx_start, spike_idx_end)

		print(n_layer, n_trial, spike_idx_range)
		overlapping_region_to_lgn_cell_idxs_map.append(
				{
					'n_layer': n_layer,
					'n_trial': n_trial,
					'idx_range': spike_idx_range
				}
			)

		spike_idx_cursor += n_overlapping_regions
# -
# +
overlapping_map_layer_cell_idxs = tuple(
		tuple(
				set(keys) for keys in layer_overlapping_map
			)
		for layer_overlapping_map in adjusted_overlap_maps
	)
# -
# +
overlapping_cell_idxs = overlapping_region_to_lgn_cell_idxs_map[0]

start_idx, end_idx = overlapping_cell_idxs['idx_range']
n_layer = overlapping_cell_idxs['n_layer']
n_trial = overlapping_cell_idxs['n_trial']

# for each spike (within range), which overlapping region (by idx)
trial_overlapping_region_idxs = spike_idxs[(spike_idxs>=start_idx) & (spike_idxs<end_idx)]





overlapping_region_keys = tuple(adjusted_overlap_maps[0].keys())
trial_overlapping_region_lgn_cell_idxs = tuple(
		tuple(adjusted_overlap_maps[0][overlapping_region_keys[region_idx]].keys())
		for region_idx in trial_overlapping_region_idxs
	)
# -
# +
%timeit 26 in trial_overlapping_region_lgn_cell_idxs[43]
test = set(trial_overlapping_region_lgn_cell_idxs[43])
%timeit 26 in test

# -


# #### Try doing before spike idxs even created

len(adjusted_overlap_maps)
# +
n_layers = sim_params.n_simulations
n_trials = sim_params.n_trials
n_cells = sim_params.lgn_params.n_cells  # IE, true LGN cells ... we're getting back to this here

trial_layer_idxs = tuple(
	{'n_layer': n_layer, 'n_trial': n_trial}
	for n_layer in range(n_layers)
		for n_trial in range(n_trials)
	)
# -
# +
# cell idxs for each overlapping region
overlapping_map_layer_cell_idxs = tuple(
		tuple(
				tuple(keys) for keys in layer_overlapping_map
			)
		for layer_overlapping_map in adjusted_overlap_maps
	)
# -
len(overlapping_map_layer_cell_idxs)
# +
# np.random.normal(loc=0, scale=3, size=1000)
jitter = Time(1, 'ms')

mk_jitter = lambda jitter, size: np.random.normal(loc=0, scale=jitter, size=size)
# -
# +
from collections import deque
# -
# +
true_all_spike_times_with_synchrony = deque()
# true_all_spike_times_with_synchrony = list()
for trial_layer_idx, trial_layer in zip(trial_layer_idxs, lgn_layer_responses):
	layer_cell_spikes = [deque() for _ in range(sim_params.lgn_params.n_cells)]
	# layer_cell_spikes = [list() for _ in range(sim_params.lgn_params.n_cells)]

	for overlapping_region_idx, spike_times in enumerate(trial_layer.cell_spike_times):
		true_cell_idxs = (
			overlapping_map_layer_cell_idxs
				[trial_layer_idx['n_layer']]
					[overlapping_region_idx]
			)
		for cell_idx in true_cell_idxs:
			layer_cell_spikes[cell_idx].append(
							spike_times.ms + mk_jitter(jitter.ms, spike_times.value.size),
					# Time(
					# 		spike_times.ms + mk_jitter(jitter.ms, spike_times.value.size),
					# 		'ms'
					# 	)
				)
	for cell_spikes in layer_cell_spikes:
		# concatenate all spike_times
		concatenated_spike_times = Time(
				np.r_[tuple(spike_times for spike_times in cell_spikes)],
				# np.r_[tuple(spike_times.ms for spike_times in cell_spikes)],
				'ms'
			)
		# deduplicate here??
		true_all_spike_times_with_synchrony.append(concatenated_spike_times)
# -

# ##### Managing Negative and colliding spike times

# +
from typing import Optional
# -
# +
def clean_sycnchrony_spike_times(
		spike_times: Time[np.ndarray],
		simulation_temp_res: Optional[Time[float]] = None
		) -> Time[np.ndarray]:

	"""Remove spikes that are negative, past simulation time or too close to each other

	"""


	if not simulation_temp_res:
		sim_temp_res = Time(lifv1.defaultclock.dt / lifv1.bnun.msecond, 'ms')
	else:
		sim_temp_res = simulation_temp_res

	# sort spikes (necessary as likely to be unsorted)
	spks = np.sort(spike_times.ms)
	de_dup_spks = None  # placeholder should no de-duplication need to occur

	# jitter pushed spikes below 0?
	if np.any(spk_negative := spks<0):
		spks[spk_negative] *= -1  # rotate jitter around 0 (ie, make it positive)

	# jitter pushed spikes beyond simulation time?
	if np.any(spk_late := spks > sim_params.space_time_params.temp_ext.ms):
		# rotate around simulation time as with negatives above
		spks[spk_late] -= (spks[spk_late] - sim_params.space_time_params.temp_ext.ms)

	# any two spikes closer than the simulation resolution?
	if np.any(spk_dup_idxs := (np.abs( spks[1:] - spks[0:-1] ) <= (sim_temp_res.ms)) ):

		# do not include in final array
		de_dup_spks = np.r_[spks[0], spks[1:][~spk_dup_idxs]]


	managed_spike_times = Time(
			de_dup_spks
				if (de_dup_spks is not None) else
			spks,
			'ms'  # make sure using same unit throughout as above
		)

	return managed_spike_times

# -

# +
sim_temp_res = Time(lifv1.defaultclock.dt / lifv1.bnun.msecond, 'ms')
# -
# +
incorrect_spike_count = 0
for i, input_spikes in enumerate(true_all_spike_times_with_synchrony):
	test_spikes = np.sort(input_spikes.ms)
	if (
			np.any(np.abs( test_spikes[1:] - test_spikes[0:-1] ) <= (sim_temp_res.ms)).astype(int)
			or
			np.any(test_spikes < 0).astype(int)
			or
			np.any(test_spikes > sim_params.space_time_params.temp_ext.ms)
		):
		# print(i)
		incorrect_spike_count += 1
print(incorrect_spike_count)
# -

# +

managed_all_spike_times_with_synchrony = deque()

for i, spike_times_array in enumerate(true_all_spike_times_with_synchrony):

	managed_all_spike_times_with_synchrony.append(clean_sycnchrony_spike_times(spike_times_array))

	# spks = np.sort(spike_times_array.ms)
	# de_dup_spks = None
	# if np.any(spk_negative := spks<0):
	# 	# print(i, 'negative')
	# 	spks[spk_negative] *= -1

	# if np.any(spk_late := spks > sim_params.space_time_params.temp_ext.ms):
	# 	# print(i, 'negative')
	# 	spks[spk_late] -= (spks[spk_late] - sim_params.space_time_params.temp_ext.ms)

	# if np.any(spk_dup_idxs := (np.abs( spks[1:] - spks[0:-1] ) <= (sim_temp_res.ms)) ):
	# 	# print(i, 'duplicates')

	# 	de_dup_spks = np.r_[spks[0], spks[1:][~spk_dup_idxs]]


	# managed_all_spike_times_with_synchrony.append(
	# 	Time(
	# 			de_dup_spks
	# 				if (de_dup_spks is not None) else
	# 			spks,
	# 			'ms'
	# 		)
	# 	)
# -
# +
incorrect_spike_count = 0
for i, input_spikes in enumerate(managed_all_spike_times_with_synchrony):
	test_spikes = np.sort(input_spikes.ms)
	if (
			np.any(np.abs( test_spikes[1:] - test_spikes[0:-1] ) <= (sim_temp_res.ms)).astype(int)
			or
			np.any(test_spikes < 0).astype(int)
			or
			np.any(test_spikes > sim_params.space_time_params.temp_ext.ms)
		):
		# print(i)
		incorrect_spike_count += 1
print(incorrect_spike_count)
# -

# ##### SPike time and idx arrays

# +
n_inputs = len(managed_all_spike_times_with_synchrony)

spike_idxs = np.r_[
		tuple(
			# for each input, array of cell number same length as number of spikes
			(
				i * np.ones(shape=managed_all_spike_times_with_synchrony[i].value.size)
			).astype(int)
			for i in range(n_inputs)
		)
	]

spike_times: Time[np.ndarray] = Time(
	np.r_[
		tuple(spike_times.ms for spike_times in managed_all_spike_times_with_synchrony)
		],
	'ms'
	)
# -
# +
px.scatter(x=spike_times.value, y=spike_idxs, title=f'Jitter = {jitter.ms}ms').show()
# -
# +
spike_idxs.size
# -



# #### Comparing synchrony to without synchrony (both normal lgn cells)

# Using sections above
# +
px.scatter(y=spike_idxs_not_synch, x=spike_times_not_synch.value).show()
# -
# +
px.scatter(x=spike_times.value, y=spike_idxs, title=f'Jitter = {jitter.ms}ms').show()
# -
# +
px.histogram(x=spike_times_not_synch.ms).show()
# -
# +
px.histogram(x=spike_times.ms).show()
# -




# #### Test Run with V1

# ##### Baseline ... without synchrony

# +
lif_params.mk_dict_with_units(n_inputs=False)
# -
# +
v1_model = run.create_multi_v1_lif_network(sim_params, overlap_map=None)
# -
# +
v1_model.reset_spikes(spike_idxs_not_synch, spike_times_not_synch, spikes_sorted=False)
v1_model.run(sim_params.space_time_params)
# -
# +
px.line(y=v1_model.membrane_monitor.v[0]).show()
# -


# ##### With synchrony

# +
v1_model_synch = run.create_multi_v1_lif_network(sim_params, overlap_map=None)
# -
# +
v1_model_synch.reset_spikes(spike_idxs, spike_times, spikes_sorted=False)
v1_model_synch.run(sim_params.space_time_params)
# -
# +
px.line(y=v1_model_synch.membrane_monitor.v[0], title=f'Jitter: {jitter.ms}ms').show()
# -


# +
import copy
# -
# +
print(sim_params.n_simulations)
new_sim_params = copy.deepcopy(sim_params)
new_sim_params.n_simulations = 100
print(new_sim_params.n_simulations)
v1_model_test = run.create_multi_v1_lif_network(new_sim_params, overlap_map=None)
print(v1_model_test.membrane_monitor.v.shape)
# -


# +
pos_spk_idx = spike_times.value<0
# -
# +
v1_model.reset_spikes(
	spike_idxs[~pos_spk_idx], Time(spike_times.ms[~pos_spk_idx], 'ms'),
	spikes_sorted=False)
v1_model.run(sim_params.space_time_params)
# -




# ### Testing Synchrony Implentation in main code


# +
spat_res=ArcLength(1, 'mnt')
spat_ext=ArcLength(660, 'mnt')
"Good high value that should include any/all worst case scenarios"
temp_res=Time(1, 'ms')
temp_ext=Time(1000, 'ms')

st_params = do.SpaceTimeParams(
	spat_ext, spat_res, temp_ext, temp_res,
	array_dtype='float32'
	)
# -
# +
# good subset of spat filts that are all in the middle in terms of size
subset_spat_filts = [
	'berardi84_5a', 'berardi84_5b', 'berardi84_6', 'maffei73_2mid',
	'maffei73_2right', 'so81_2bottom', 'so81_5', 'soodak87_1'
]
# -
# +
multi_stim_params = do.MultiStimulusGeneratorParams(
	spat_freqs=[0.8], temp_freqs=[4], orientations=[90], contrasts=[0.3]
	)

stim_params = stimulus.mk_multi_stimulus_params(multi_stim_params)[0]

lgn_params = do.LGNParams(
	n_cells=30,
	orientation = do.LGNOrientationParams(ArcLength(0), circ_var=0.5),
	circ_var = do.LGNCircVarParams('naito_lg_highsf', 'naito'),
	spread = do.LGNLocationParams(2, 'jin_etal_on'),
	filters = do.LGNFilterParams(spat_filters='all', temp_filters='all'),
	F1_amps = do.LGNF1AmpDistParams()
	)
lif_params = do.LIFParams(total_EPSC=3.5)
# -

# +
sim_params = do.SimulationParams(
	n_simulations=10,
	space_time_params=st_params,
	multi_stim_params=multi_stim_params,
	lgn_params=lgn_params,
	lif_params = lif_params,
	n_trials = 3,
	analytical_convolution=True
	# n_trials = 10
	)

synch_params = do.SynchronyParams(
	True,
	jitter=Time(3, 'ms')
	)
# -
# +
all_lgn_layers = run.create_all_lgn_layers(sim_params, force_central_rf_locations=False)

# if synch_params.lgn_has_synchrony:

all_lgn_overlap_maps = run.create_all_lgn_layer_overlapping_regions(all_lgn_layers, sim_params)
# the main one, with weights reduced so that after duplication to target LGN cells rates are accurate
all_adjusted_lgn_overlap_maps = run.mk_adjusted_overlapping_regions_wts(all_lgn_overlap_maps)

lgn_layers = all_lgn_layers[stim_params.contrast]
overlap_regions = all_adjusted_lgn_overlap_maps[stim_params.contrast]
# -
# +
partitioned_results = run.run_single_stim_multi_layer_simulation(
		sim_params, stim_params,
		synch_params = synch_params,
		lgn_layers = lgn_layers,
		lgn_overlap_maps = overlap_regions,
		log_print=True, log_info='Test',
		save_membrane_data=True
	)
# -
# +
len(partitioned_results)
test = partitioned_results[0]
test_lgn = test.get_lgn_response(0)
test_lgn.cell_spike_times



len(test.lgn_responses)
test.lgn_responses
# -

# Check that LGN spikes match v1 membrane potential
# +
test = partitioned_results[6]
# -
# +
px.line(test.get_mem_pot(0)).show()
# -
# +
trial_idx = 1
v1_mem_pot = test.get_mem_pot(trial_idx)
# -
# +
test_spike_times = np.r_[
	tuple(
		spikes.s * 10_000
		for spikes in test.get_lgn_response(trial_idx).cell_spike_times
		)
]
# -
# +
fig = (
	px
	.line(v1_mem_pot)
	.add_scatter(
			x=test_spike_times,
			y=np.ones_like(test_spike_times) * -0.07,
			mode='markers'
		)
	)
fig.show()
# -

