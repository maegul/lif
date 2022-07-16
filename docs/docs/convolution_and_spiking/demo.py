#
# ## Imports

# +
from lif import *

from lif.plot import plot

from scipy.ndimage import gaussian_filter1d
import plotly.express as px
# -

# ## Load filters

# * These are loaded from file, having been previously fit to data

tf = TQTempFilter.load(TQTempFilter.get_saved_filters()[0])
sf = DOGSpatialFilter.load(DOGSpatialFilter.get_saved_filters()[0])

# ## Space, time and stimulus parameters

# +
# stim_amp=0.5
spat_res=ArcLength(1, 'mnt')
spat_ext=ArcLength(120, 'mnt')
temp_res=Time(1, 'ms')
temp_ext=Time(1000, 'ms')

orientation = ArcLength(90, 'deg')
temp_freq = TempFrequency(8)
spat_freq_x = SpatFrequency(2)
spat_freq_y = SpatFrequency(0)
# -

st_params = do.SpaceTimeParams(spat_ext, spat_res, temp_ext, temp_res)
stim_params = do.GratingStimulusParams(
    spat_freq_x, temp_freq,
    orientation=orientation,
    amplitude=1, DC=1
)

# ## Simulate response
#
# ### Continuous Rate response

resp = conv.mk_single_sf_tf_response(sf, tf, st_params, stim_params)

# ### Poisson Spiking Response

n_trials = 20
s, pop_s = conv.mk_sf_tf_poisson(st_params, resp, n_trials=n_trials)

# ### Aggregate trials and plot

all_spikes = conv.aggregate_poisson_trials(s)

plot.poisson_trials_rug(s).show()

plot.psth(st_params, s, 20).show()
