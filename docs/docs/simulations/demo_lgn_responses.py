
# # Imports

# +

from lif.utils import data_objects as do
from lif.utils import settings
from lif.utils.units.units import ArcLength, SpatFrequency, Time, TempFrequency, scalar

import lif.simulation.all_filter_actual_max_f1_amp as all_max_f1

import lif.plot.plot as plot
# -


# # All filter Max f1 amplitudes

# Depends on the filter_index.json in the data directory


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
    amplitude=stim_amp, DC=stim_DC,
    contrast=do.ContrastValue(0.4)
)
# -
# +
actual_max_f1_amps = all_max_f1.mk_actual_max_f1_amps(stim_params=stim_params)
# -
# +
for k, max_f1 in sorted(
		actual_max_f1_amps.items(), key=lambda x: x[1].value.max_amp
		):
	print(
		f'{k[0] + " - " + k[1]:<30}',
		f'{max_f1.value.max_amp:<8.3f}',
		f'{max_f1.spat_freq.cpd:<5.3f}',
		f'{max_f1.temp_freq.hz:<5.3f}'
		)
# -

# +
fig = plot.joint_sf_tf_amp(
	sf=all_max_f1.spatial_filters['cheng81'],
	tf=all_max_f1.temporal_filters['berardi85'],
	contrast=stim_params.contrast)
fig.show()
# -
# +
all_max_f1.spatial_filters['cheng81'].key
# -





