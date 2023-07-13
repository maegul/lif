"""Calculate the overlapping regions of an LGN layer's spatial filters


"""

# # Imports
# +
from collections import defaultdict

from typing import DefaultDict, Tuple, Dict

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

import lif.simulation.all_filter_actual_max_f1_amp as all_max_f1
import lif.simulation.leaky_int_fire as lifv1
from lif.simulation import run

from lif.plot import plot
# -

LGNOverlapMap = Dict[Tuple[int,...], Dict[int, float]]

def mk_all_lgn_rfs_array(
		lgn_layer: do.LGNLayer,
		st_params: do.SpaceTimeParams
		) -> np.ndarray:

	all_xc, _ = ff.mk_spat_coords(spat_res=st_params.spat_res, spat_ext=st_params.spat_ext)
	all_spat_filt_arrays = np.zeros(shape=(*all_xc.value.shape, len(lgn_layer.cells)))

	for i, cell in enumerate(lgn_layer.cells):

		xc, yc = ff.mk_spat_coords(st_params.spat_res, sd=cell.spat_filt.parameters.max_sd() )

		spat_filt = ff.mk_dog_sf(x_coords=xc, y_coords=yc, dog_args=cell.oriented_spat_filt_params)
		spat_filt = ff.mk_oriented_sf(spat_filt, cell.orientation)

		rf_slice_idxs = stimulus.mk_rf_stim_spatial_slice_idxs(
			st_params, cell.spat_filt, cell.location)

		all_spat_filt_arrays[
				rf_slice_idxs.y1:rf_slice_idxs.y2,
				rf_slice_idxs.x1:rf_slice_idxs.x2,
				i
			] = spat_filt

	return all_spat_filt_arrays


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
def mk_spat_filt_overlapping_weights_vect(
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

def mk_lgn_overlapping_weights(
		lgn_layer: do.LGNLayer,
		st_params: do.SpaceTimeParams
		) -> DefaultDict[Tuple[int,...], DefaultDict[int, float]]:

	all_spat_filt_arrays = mk_all_lgn_rfs_array(lgn_layer, st_params)
	overlapping_weights = mk_spat_filt_overlapping_weights(all_spat_filt_arrays)
	return overlapping_weights


def mk_lgn_overlapping_weights_vect(
		lgn_layer: do.LGNLayer,
		st_params: do.SpaceTimeParams
		) -> Dict[Tuple[int,...], Dict[int, float]]:

	all_spat_filt_arrays = mk_all_lgn_rfs_array(lgn_layer, st_params)
	overlapping_weights = mk_spat_filt_overlapping_weights_vect(all_spat_filt_arrays)
	return overlapping_weights

