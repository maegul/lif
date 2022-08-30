# ===========
from lif import *
# -----------
# ===========
from lif.plot import plot
# -----------
# ===========
import plotly.express as px
from scipy.ndimage import gaussian_filter1d
# -----------

# > Poisson Response and Plots
# ===========
tf = TQTempFilter.load(TQTempFilter.get_saved_filters()[0])
sf = DOGSpatialFilter.load(DOGSpatialFilter.get_saved_filters()[0])
# -----------
# ===========
# stim_amp=0.5
spat_res=ArcLength(1, 'mnt')
spat_ext=ArcLength(120, 'mnt')
temp_res=Time(1, 'ms')
temp_ext=Time(1000, 'ms')

orientation = ArcLength(90, 'deg')
temp_freq = TempFrequency(8)
spat_freq_x = SpatFrequency(2)
spat_freq_y = SpatFrequency(0)
# -----------
# ===========
st_params = do.SpaceTimeParams(spat_ext, spat_res, temp_ext, temp_res)
stim_params = do.GratingStimulusParams(
    spat_freq_x, temp_freq,
    orientation=orientation,
    amplitude=1, DC=1
)
# -----------
# ===========
resp = conv.mk_single_sf_tf_response(sf, tf, st_params, stim_params)
# -----------
# ===========
n_trials = 20
s, pop_s = conv.mk_sf_tf_poisson(st_params, resp, n_trials=n_trials)
# -----------
# ===========
all_spikes = conv.aggregate_poisson_trials(s)
# -----------
# ===========
plot.poisson_trials_rug(s).show()
# -----------
# ===========
plot.psth(st_params, s, 20).show()
# -----------


# > ori bias
# ===========
def cv(r, theta):
    """
    Circular Variance (Ringach (2000))
    """
    x = np.sum((r * np.exp(1j * 2 * theta))) / np.sum(r)
    # return (1 - np.abs(x))
    return (np.abs(x))

def vm(x, a=1, k=0.5, phi=np.pi / 2):
    """
    Single von mises function (Swindale (1998))
    """

    return a * np.exp(k * (np.cos(phi - x)**2 - 1))

# -----------
# ===========
angles = ArcLength(np.linspace(0, 180, 8, False), 'deg')
k_vals = np.linspace(0, 30, 1000)

cv_vals = [
    cv(vm(angles.rad, k=k_val), angles.rad)
    for k_val in k_vals
]
# -----------
# ===========
px.line(x=k_vals, y=cv_vals).show()
# -----------
# ===========
k_val = 0.6
a, b = vm(0, k=k_val), vm(np.pi/2, k=k_val)
a,b
# -----------
# ===========
target_amp = (a + b)/2
ori_ratio = b/a
target_amp, ori_ratio
# -----------
# ===========
ff.mk_dog_sf_ft(SpatFrequency(10), SpatFrequency(0), sf.parameters)
# -----------
# ===========
sf.parameters.cent.arguments.asdict_()
# -----------
# ===========
amps = vm(angles.rad, k=0.6)
# -----------
# ===========
px.line(x=angles.deg, y=amps).show()
# -----------




# ===========
angles = ArcLength(np.arange(0, 180, 10), 'deg')

vm_amp = vm(angles.rad, k=2)
# -----------
# ===========
px.line(x=angles.base, y=vm_amp).show()
# -----------
# ===========
from scipy.stats import vonmises
# -----------
# ===========
angles = np.arange(0, 180, 10)
angles
# -----------
# ===========
cv(
    np.array([2, 1, 3, 2]),
        np.deg2rad(np.array([0, 45, 90, 135]))
    )
# -----------

# ===========
# preferred is this much greater response than non-preferred
bias_factor = 1.5
# -----------
# ===========
# -----------
