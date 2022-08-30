# > Imports and setup
# +
from lif import *
# -
# +
import plotly.express as px
import plotly.graph_objects as go
# -
# +
from scipy.ndimage import interpolation
# -

# >> Spat filt and ori biased duplicate
# +
sf = DOGSpatialFilter.load(DOGSpatialFilter.get_saved_filters()[0])
ori_sf_params = ff.mk_ori_biased_spatfilt_params_from_spat_filt(sf, 0.6)
# -

# >> coords and RF
# +
spat_res = ArcLength(1, 'mnt')
xc, yc = ff.mk_spat_coords(sd=ori_sf_params.max_sd(), spat_res=spat_res)
spat_filt = ff.mk_dog_sf(xc, yc, ori_sf_params)
# -

# > Rotating RF
# +
# tf = TQTempFilter.load(TQTempFilter.get_saved_filters()[0])
sf = DOGSpatialFilter.load(DOGSpatialFilter.get_saved_filters()[0])
# -
# +
ori_sf_params = ff.mk_ori_biased_spatfilt_params_from_spat_filt(sf, 0.5)
ori_sf_params.max_sd().mnt
# -
# +
spat_res = ArcLength(1, 'mnt')
xc, yc = ff.mk_spat_coords(sd=ori_sf_params.max_sd(), spat_res=spat_res)
# -
# +
spat_filt = ff.mk_dog_sf(xc, yc, ori_sf_params)
spat_filt.shape
# -
# +
px.imshow(
    spat_filt,
    color_continuous_scale=px.colors.diverging.Portland,
    color_continuous_midpoint=0
    ).show()
# -
# +
rot_spat_filt = interpolation.rotate(spat_filt, 70, reshape=False)
spat_filt.shape, rot_spat_filt.shape
# -
# +
px.imshow(
    rot_spat_filt,
    color_continuous_scale=px.colors.diverging.Portland,
    color_continuous_midpoint=0
    ).show()
# -

# > New Attempt
# >> Setup
# +
# 3 times extent to experiment with slicing
spat_ext = ArcLength(3 * ff.mk_spat_ext_from_sd_limit(ori_sf_params.max_sd()).base)
spat_ext, spat_res
# -
# +
# predicted size of
ff.mk_rounded_spat_radius(spat_res, spat_ext).value*2 + 1
# -
# +
st_params = do.SpaceTimeParams(
    spat_ext, spat_res,
    Time(0.2), Time(1, 'ms')
    )
stim_params = do.GratingStimulusParams(
    spat_freq=SpatFrequency(4), temp_freq=TempFrequency(2),
    orientation=ArcLength(90)
    )
# -
# +
grating = stim.mk_sinstim(st_params, stim_params)
grating.shape, spat_filt.shape[0] // 2
# -
# >>> Predicting size of grating
# +
print(
    ff.spatial_extent_in_res_units(st_params.spat_res, spat_ext=st_params.spat_ext),
    grating.shape[0]
    )
# -
# +
grating_size_n_res = ff.spatial_extent_in_res_units(
    st_params.spat_res, spat_ext=st_params.spat_ext)
# -


# >> RF Location
# +
st_params.spat_ext.mnt, st_params.spat_res.mnt

# -
# +
@dataclass
class RFLocation:
    x: ArcLength[scalar]
    y: ArcLength[scalar]

    @classmethod
    def from_polar(
        cls,
        theta: ArcLength[scalar], mag: ArcLength[scalar],
        unit: str = 'mnt') -> 'RFLocation':
        x = ArcLength(mag[unit] * np.cos(theta.rad), unit)
        y = ArcLength(mag[unit] * np.sin(theta.rad), unit)

        return cls(x=x, y=y)

    def round_to_spat_res(self, spat_res: ArcLength[int]) -> 'RFLocation':
        x = ff.round_coord_to_res(coord=self.x, res=spat_res)
        y = ff.round_coord_to_res(coord=self.x, res=spat_res)

        return self.__class__(x=x, y=y)
# -
# +
rf_loc = RFLocation.from_polar(ArcLength(115), ArcLength(50, 'mnt'))
rf_loc.round_to_spat_res(st_params.spat_res)
# -
# +
rf_loc = (
    RFLocation
    .from_polar(ArcLength(115), ArcLength(50, 'mnt'))
    .round_to_spat_res(st_params.spat_res)
    )
print(rf_loc)
# -

# >> Determining Spat Filt Spatial Extent
# +
spat_res = ArcLength(1, 'mnt')
xc, yc = ff.mk_spat_coords(sd=ori_sf_params.max_sd(), spat_res=spat_res)
spat_filt = ff.mk_dog_sf(xc, yc, ori_sf_params)
spat_filt.shape
# -
# +
predicted_n_spat_res_coords = ff.spatial_extent_in_res_units(
    spat_res=st_params.spat_res,
    sf=ori_sf_params)
print(f'''
    predicted = {predicted_n_spat_res_coords}
    actual = {spat_filt.shape}
    ''')
# -
# >>>> Old basic approach (for posterity)
# +
# (
#     ff.mk_rounded_spat_radius(
#         st_params.spat_res,
#         ArcLength(
#             ori_sf_params.max_sd().value
#             * 2*settings.simulation_params.spat_filt_sd_factor,
#             ori_sf_params.max_sd().unit
#             )
#             )
#     .value * 2 + 1
# )
# -


# > Slicing Stimulus
# Need to know how big ot make stimulus to fit all RFs ...
# +
# 3 times extent to experiment with slicing
spat_ext = ArcLength(3 * ff.mk_spat_ext_from_sd_limit(ori_sf_params.max_sd()).base)
spat_ext, spat_res
# -
# +
st_params = do.SpaceTimeParams(
    spat_ext, spat_res,
    Time(0.2), Time(1, 'ms')
    )
stim_params = do.GratingStimulusParams(
    spat_freq=SpatFrequency(4), temp_freq=TempFrequency(2),
    orientation=ArcLength(90)
    )
# -
# +
grating = stim.mk_sinstim(st_params, stim_params)
grating.shape, spat_filt.shape[0] // 2
# -

# >> Spatial Extent of spatial filter
# +
# predictable now with appropriate function
predictable = (
    ff.spatial_extent_in_res_units(st_params.spat_res, ori_sf_params)
    ==
    spat_filt.shape[0]
    )
print(predictable)
# -


# >> Calculate Slice Indices
# +
grating.shape, spat_filt.shape
# -
# +
# spat filt radius will be
spat_filt_radius = ff.mk_spat_radius(
    ff.mk_spat_ext_from_sd_limit(ori_sf_params.max_sd())
    )
spat_filt_diam = (spat_filt_radius.mnt * 2) + 1
spat_filt_radius, spat_filt_diam, spat_filt.shape
# -
# +
def mk_putative_spat_coords_ext(sf_params: DOGSpatFiltArgs) -> ArcLength[float]:

    sf_radius = ff.mk_spat_radius(
        ff.mk_spat_ext_from_sd_limit(sf_params.max_sd())
        )

    return ArcLength((sf_radius.value * 2) + 1, sf_radius.unit)
# -
# +
mk_putative_spat_coords_ext(ori_sf_params)
# -
# +
sf_pos = (0, 0)  # x, y coords in mnts
# -
# >>> Rounding Spat to Res
# +
# ... now part of filter functions ...
def round_coord_to_res(
        coord: ArcLength[float], res: ArcLength[int]) -> ArcLength[int]:
    """Rounds a Spatial coord to a whole number multiple of the resolution

    coords returned in units of resolution and with value as int (provided res is int)
    """
    res_unit = res.unit
    coord_val = coord[res_unit]
    res_val = res.value

    low = int(coord_val // res_val) * res_val
    high = low + res_val

    low_diff = abs(low - coord_val)
    high_diff = abs(high - coord_val)

    rounded_val = (
        high if low_diff > high_diff
        else low
        )

    return ArcLength(rounded_val, res_unit)

# -
# +
def round_spat_coords_to_resolution(
        x: ArcLength[float], y: ArcLength[float],
        res: ArcLength[int]
        ) -> Tuple[ArcLength[int], ArcLength[int]]:

    """Round both x and y coords to resolution using round_coord_to_res

    Returned in units of res and ints (if res is in ints)
    """

    new_x = round_coord_to_res(x, res)
    new_y = round_coord_to_res(y, res)

    return new_x, new_y
# -
# +
round_test = round_coord_to_res(ArcLength(3.34, 'mnt'), ArcLength(1, 'sec'))
round_test.value, round_test.unit
# -
# +
sf_pos = (ArcLength(3.5, 'mnt'), ArcLength(5.3, 'mnt'))
# -
# +
sf_pos_rounded = round_spat_coords_to_resolution(*sf_pos, res=spat_res)
sf_pos_rounded
# -

# >> how make slice from pos to match size of spat_filt

# X Know extent of sclice in spatial terms
# +
sf_ext = ff.spatial_extent_in_res_units(st_params.spat_res, ori_sf_params)
print(f'Spat FIlt Extent in number of resolution units: {sf_ext}\nResolution: {st_params.spat_res}')
# -

# X translate to desired position
# +
sf_loc = do.RFLocation(ArcLength(3.5, 'mnt'), ArcLength(5.3, 'mnt'))
sf_loc_snapped = sf_loc.round_to_spat_res(st_params.spat_res)
print(sf_loc_snapped)
# -

# Now in units of resolution ... but how translate to indices?

# account for the resolution to specify appropriate indices

# spat_filt and stimulus will be made with the same resolution
# so ... from extent to extent will be the same number of pixels
# +
st_params.asdict_()
# -
# +
test_res, test_ext = (
    ArcLength(1, 'mnt'),
    ArcLength(100, 'mnt')
    )
test_coords = ff.mk_spat_coords_1d(test_res, test_ext)
# -
# +
ff.mk_spat_radius(test_ext)
# -
# +
# test_coords.base.shape
test_coords.value
# -
# +
# testing ... should add to tests
# ... some of this may now be redundant with shift to
# using rounding to res approach in filter functions
spat_coord_cent_idx = test_coords.value.shape[0] // 2
spat_coord_max = ff.mk_spat_radius(test_ext)
spat_coord_stride = np.max(np.diff(test_coords.value))
# center is shape // 2 or (ext // 2) / res (dividing by res necessary for when res is not 1)
spat_coord_cent_idx2 = int((test_ext.value // 2) / test_res.value)
# -
# +
test_coords.value[spat_coord_cent_idx] == 0
test_coords.value[spat_coord_cent_idx2] == 0
test_coords.value[0] == -spat_coord_max.value
test_coords.value[-1] == spat_coord_max.value
spat_coord_stride == test_res.value
test_coords.value[spat_coord_cent_idx+1] == test_res.value
# -
# +
test_coords
# -
# +
sf_pos = (ArcLength(3.5, 'mnt'), ArcLength(5.3, 'mnt'))
sf_pos_rounded = round_spat_coords_to_resolution(*sf_pos, res=spat_res)
test_sf_pos = sf_pos_rounded[0]
# -
# +
test_rf_ext = ArcLength(20, 'mnt')
test_rf_coords = ff.mk_spat_coords_1d(test_res, test_rf_ext)

# ext of rf to units of res
test_rf_rad = ff.mk_spat_radius(test_rf_ext)

# is this guaranteed by the mk_spat_radius function?
# yes ...if the coords have been made without an exception
assert test_rf_rad.value % test_res.value == 0

# negative, as left/down from zero
test_rf_ext_idx = -int(test_rf_rad.value / test_res.value)

# adjust rf ext by pos (in units of res)
pos_adj = int(test_sf_pos.value // test_res.value)
test_rf_pos_idx = test_rf_ext_idx + pos_adj

# adjust by center_idx
test_cent_idx = int((test_ext.value // 2) / test_res.value)
test_rf_left_idx = test_rf_pos_idx + test_cent_idx

test_coord_patch = test_coords.value[
    test_rf_left_idx :
    test_rf_left_idx+(2*test_rf_rad.value + 1)
    ]

test_coord_patch.shape, test_rf_coords.value.shape
# -
# +
# >>> Final function: mk_rf_stim_index
def mk_rf_stim_slice_idxs(
        st_params: SpaceTimeParams,
        spat_filt_params: DOGSpatFiltArgs,
        spat_filt_location: do.RFLocation
        # pos_cent_x: ArcLength[float], pos_cent_y: ArcLength[float]
        ):

    spat_res = st_params.spat_res
    sf_loc = spat_filt_location.round_to_spat_res(spat_res)

    sf_ext_n_res = ff.spatial_extent_in_res_units(spat_res, sf=spat_filt_params)
    sf_radius_n_res = sf_ext_n_res//2
    stim_ext_n_res = ff.spatial_extent_in_res_units(spat_res, spat_ext=st_params.spat_ext)
    stim_cent_idx = int(stim_ext_n_res//2)  # as square, center is same for x and y
    # as snapped to res, should be whole number quotients
    x_pos_n_res = sf_loc.x.value // spat_res.value
    y_pos_n_res = sf_loc.y.value // spat_res.value

    slice_x_idxs = (
        int(stim_cent_idx - sf_radius_n_res + x_pos_n_res),
        int(stim_cent_idx + sf_radius_n_res + x_pos_n_res + 1),
        )
    slice_y_idxs = (
        int(stim_cent_idx - sf_radius_n_res - y_pos_n_res),
        int(stim_cent_idx + sf_radius_n_res - y_pos_n_res + 1),
        )

    rf_idxs = do.RFStimSpatIndices(
        x1=slice_x_idxs[0], x2=slice_x_idxs[1],
        y1=slice_y_idxs[0], y2=slice_y_idxs[1])

    return rf_idxs
# -
# +
spat_filt_location = do.RFLocation(ArcLength(3.5, 'mnt'), ArcLength(5.3, 'mnt'))
rf_idxs = mk_rf_stim_slice_idxs(st_params, ori_sf_params, spat_filt_location)
rf_idxs.is_within_extent(st_params)
# -
# +
stim_coords_x, stim_coords_y = ff.mk_spat_coords(st_params.spat_res, spat_ext=st_params.spat_ext)
# -
# +
print(ff.spatial_extent_in_res_units(st_params.spat_res, sf=ori_sf_params))
spat_filt_location = do.RFLocation(ArcLength(7, 'mnt'), ArcLength(0, 'mnt'))
rf_idxs = mk_rf_stim_slice_idxs(st_params, ori_sf_params, spat_filt_location)
print(rf_idxs)
#                                      V: y goes first, as first axis are rows!
rf_slice = stim_coords_x.value[rf_idxs.y1:rf_idxs.y2, rf_idxs.x1:rf_idxs.x2 ]
print(rf_slice.shape)
print(rf_slice[0,:])
print(rf_slice[0, rf_slice.shape[0]//2])
# -
# +
stim_sz = ff.spatial_extent_in_res_units(
    st_params.spat_res, spat_ext=st_params.spat_ext)
stim_cent_idx = int(np.floor(stim_sz/2))

rf_pos = 5
# why +1 necessary ... because endpoint is not inclusive (eg, [3:4] has size 1!!)
pos_idxs = (5-rad + stim_cent_idx, 5+rad + stim_cent_idx + 1)
pos_idxs

grating[pos_idxs[0]:pos_idxs[1], 0, 0].shape
# -
# +
pos = -5
rad = 3
ti = (33 - rad + pos, 33 + rad + pos + 1)
print(
    xc.value[33, ti[0]:ti[1]].shape[0] == (2*rad + 1),
    xc.value[33, ti[0]:ti[1]][rad] == pos
    )
# -

    # rounding to resolution
    # pos_x, pos_y = round_spat_coords_to_resolution(
    #         pos_cent_x, pos_cent_y, st_params.spat_res
    #     )

    # should all be in same unit as spat_res
    sf_loc_snapped = spat_filt_location.round_to_spat_res(spat_res)
    pos_x, pos_y = sf_loc_snapped.x, sf_loc_snapped.y

    rf_cent_idx_x =  1 * int(pos_x.value / spat_res.value)
    rf_cent_idx_y = -1 * int(pos_y.value / spat_res.value)

    ###
    # translating continuous values to discrete indices (of the coords arrays)
    ###
    # negative for y axis too!!!

    # presuming that these are in the same units!!
    # assert spat_filt_radius.unit == st_params.spat_res.unit, (
    #     f'spat_filt_radius unit ({spat_filt_radius.unit}) '
    #     f'not same as spat_res unit ({st_params.spat_res.unit})'
    #     )

    spat_filt_start_idx = -1 * int(  # negative as start slice below zero, then to above zero
            spat_filt_radius.value
            /
            spat_res.value  # how many pixels does radius span
        )

    # adjust by pos
    pos_x_idx_adj = int(pos_x.value / st_params.spat_res.value)
    pos_y_idx_adj = int(pos_y.value / spat_res.value)
    spat_filt_start_idx_x = spat_filt_start_idx + pos_x_idx_adj
    spat_filt_start_idx_y = spat_filt_start_idx + pos_y_idx_adj

    # adjust by cent
    ## If coords have been created by this point for both the spat_filt
    ## and the stimulus, then the extents for both are whole number multiples
    ## to have passed the check

# -

# +
spat_filt_diam, spat_filt_radius
# -
# +
# how take account of resolution?  Divide radius by resolution (into pixels)?
# resolution is not independent in my implementation
# it seems it really must be a whole number factor of spat coords and extent!!
len(range(3-33, 3 + 33 + 1))
# -
np.arange(3-33, 3 + 33 + 1)[33]

# +
res = 5
ext = 33
cds = np.arange(-ext, ext+res, res)
cds.shape, cds[cds.shape[0]//2]
# -





# >> THeta Trans Func
# unnecessary ... rotation works just fine
# from 90 degs ... rotates anti-clockwise in degrees
# +
def img_theta_trans(theta):
    '''
    Transforms theta to operate in a way that works intuitively for scipy.ndimage.interpolation.rotate
    '''
    return -1*(theta%180) - 90
# -
# +
theta = np.linspace(0, 360, 360, False)
px.line(x=theta, y=img_theta_trans(theta)).show()
# -
# +
rot_spat_filt = interpolation.rotate(spat_filt, img_theta_trans(135), reshape=False)

px.imshow(
    rot_spat_filt,
    color_continuous_scale=px.colors.diverging.Portland,
    color_continuous_midpoint=0
    ).show()
# -

# +
test_img = np.zeros((101, 101))
test_img[0:50, 50] = 1
# -
# +
angle = -75
px.imshow(
    interpolation.rotate(test_img, angle, reshape=False),
    title = f'{angle}'
    ).show()
# -
