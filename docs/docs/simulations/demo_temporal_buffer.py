
# # Imports
# +
from pathlib import Path
import re
import numpy as np

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

# # Adding a buffer at start of temporal convolution

# Should reduce any temporal transience from dropping straight into stimulation

# ## Load filters

# * These are loaded from file, having been previously fit to data

# +
sf = filters.spatial_filters['berardi84_5b']
tf = filters.temporal_filters['kaplan87']
# -

# ## Space, time and stimulus parameters

# +
# stim_amp=0.5
spat_res=ArcLength(1, 'mnt')
spat_ext=ArcLength(120, 'mnt')
temp_res=Time(1, 'ms')
temp_ext=Time(1000, 'ms')

orientation = ArcLength(90, 'deg')
temp_freq = TempFrequency(8)
spat_freq_x = SpatFrequency(0.8)
spat_freq_y = SpatFrequency(0)

st_params = do.SpaceTimeParams(spat_ext, spat_res, temp_ext, temp_res)
stim_params = do.GratingStimulusParams(
    spat_freq_x, temp_freq,
    orientation=orientation,
    amplitude=1, DC=1
)
# -


# ## Simulate response
# +
xc, yc = ff.mk_spat_coords(st_params.spat_res, spat_ext=spat_ext)
tc = ff.mk_temp_coords(temp_res, temp_ext=temp_ext )

stim_array = stimulus.mk_sinstim(st_params, stim_params)
# -

# ### Continuous Rate response
# +
spat_filt = ff.mk_dog_sf(x_coords=xc, y_coords=yc, dog_args=sf.parameters )
temp_filt = ff.mk_tq_tf(tc, tf.parameters)

resp = convolve.mk_single_sf_tf_response(
		st_params,
		sf, tf,
		spat_filt, temp_filt,
		stim_params,
		stim_array
		)
# -
# +
px.line(resp.response).show()
# -

# ### Continuous Rate response ... with buffer!
# +
spat_filt = ff.mk_dog_sf(x_coords=xc, y_coords=yc, dog_args=sf.parameters )
temp_filt = ff.mk_tq_tf(tc, tf.parameters)

resp = convolve.mk_single_sf_tf_response(
		st_params,
		sf, tf,
		spat_filt, temp_filt,
		stim_params,
		stim_array
		)
# -
# +
px.line(resp.response).show()
# px.line(resp).show()
# -

