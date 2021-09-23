# ===========
from lif import *
# -----------
# ===========
import plotly.express as px
import plotly.graph_objects as go
# -----------
# ===========
from scipy.ndimage import interpolation
# -----------
# ===========
sf = DOGSpatialFilter.load(DOGSpatialFilter.get_saved_filters()[0])
ori_sf_params = sf.mk_ori_biased_parameters(0.5)
# -----------
# ===========
spat_res = ArcLength(1, 'mnt')
xc, yc = ff.mk_spat_coords(sd=ori_sf_params.max_sd(), spat_res=spat_res)
spat_filt = ff.mk_dog_sf(xc, yc, ori_sf_params)
# -----------

# > Rotating RF
# ===========
# tf = TQTempFilter.load(TQTempFilter.get_saved_filters()[0])
sf = DOGSpatialFilter.load(DOGSpatialFilter.get_saved_filters()[0])
# -----------
# ===========
sf.parameters.max_sd().mnt
# -----------
# ===========
ori_sf_params = sf.mk_ori_biased_parameters(0.5)
ori_sf_params.max_sd().mnt
# -----------
# ===========
ff.mk_spat_ext_from_sd_limit(ori_sf_params.max_sd()).mnt / 0.5
# -----------
# ===========
spat_res = ArcLength(1, 'mnt')
xc, yc = ff.mk_spat_coords(sd=ori_sf_params.max_sd(), spat_res=spat_res)
# -----------
# ===========
xc.mnt.max() / ori_sf_params.max_sd().mnt
# -----------
# ===========
spat_filt = ff.mk_dog_sf(xc, yc, ori_sf_params)
# -----------
# ===========
spat_filt.shape
# -----------
# ===========
px.imshow(
    spat_filt,
    color_continuous_scale=px.colors.diverging.Portland,
    color_continuous_midpoint=0
    ).show()
# -----------
# ===========
rot_spat_filt = interpolation.rotate(spat_filt, 45, reshape=False)
# -----------
# ===========
spat_filt.shape, rot_spat_filt.shape
# -----------
# ===========
px.imshow(
    rot_spat_filt,
    color_continuous_scale=px.colors.diverging.Portland,
    color_continuous_midpoint=0
    ).show()
# -----------


# > Slicing Stimulus
# Need to know how big ot make stimulus to fit all RFs ...
# ===========
spat_res
# -----------
# ===========
# 3 times extent to experiment with slicing
spat_ext = ArcLength(3 * ff.mk_spat_ext_from_sd_limit(ori_sf_params.max_sd()).base)
spat_ext
# -----------
# ===========
st_params = do.SpaceTimeParams(
    spat_ext, spat_res,
    Time(0.2), Time(1, 'ms')
    )
stim_params = do.GratingStimulusParams(
    spat_freq=SpatFrequency(4), temp_freq=TempFrequency(2),
    orientation=ArcLength(90)
    )
# -----------
# ===========
grating = stim.mk_sinstim(st_params, stim_params)
grating.shape
spat_filt.shape[0] // 2
# -----------

# >> Spatial Extent Play
# testing whether size of stimulus is predictable
# and whether size needed is predictable too
st_params.spat_ext.mnt
np.ceil(st_params.spat_ext.mnt/2) * 2
# ===========
xc.mnt.min()
center_idx = xc.base.shape[0]//2
xc.base[center_idx, center_idx]
# -----------
# ===========
test_coords = ff.mk_spat_coords_1d(st_params.spat_res, st_params.spat_ext)
test_coords.mnt.shape
test_coords.mnt.max()
# -----------
st_params.spat_ext.mnt
# ===========
ff.mk_spat_radius(st_params.spat_ext)
# -----------

# >> Calculate Slice Indices
# ===========
grating.shape, spat_filt.shape
# -----------
# ===========
# spat filt radius will be
spat_filt_radius = ff.mk_spat_radius(
    ff.mk_spat_ext_from_sd_limit(ori_sf_params.max_sd())
    )
spat_filt_diam = (spat_filt_radius.mnt * 2) + 1
spat_filt_radius, spat_filt_diam, spat_filt.shape
# -----------
# ===========
def mk_putative_spat_coords_ext(sf_params: DOGSpatFiltArgs) -> ArcLength[float]:

    sf_radius = ff.mk_spat_radius(
        ff.mk_spat_ext_from_sd_limit(sf_params.max_sd())
        )

    return ArcLength((sf_radius.value * 2) + 1, sf_radius.unit)
# -----------
# ===========
mk_putative_spat_coords_ext(ori_sf_params)
# -----------
# ===========
sf_pos = (0, 0)  # x, y coords in mnts
# -----------
# >>> Rounding Spat to Res
# ===========
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
# -----------
# ===========
round_test = round_coord_to_res(ArcLength(3.5, 'mnt'), ArcLength(1, 'sec'))
round_test.value, round_test.unit
# -----------
# ===========
sf_pos = (ArcLength(3.5, 'mnt'), ArcLength(5.3, 'mnt'))
# -----------
# ===========
sf_pos_rounded = round_spat_coords_to_resolution(*sf_pos, res=spat_res)
sf_pos_rounded
# -----------

# >> how make slice from pos to match size of spat_filt

# X Know extent of sclice in spatial terms
# ===========
put_ext = mk_putative_spat_coords_ext(ori_sf_params)
print(f'{put_ext.value}:{put_ext.unit}, radius: {put_ext.value//2}')
# -----------

# X translate to desired position
# ===========
sf_pos = (ArcLength(3.5, 'mnt'), ArcLength(5.3, 'mnt'))
sf_pos_rounded = round_spat_coords_to_resolution(*sf_pos, res=spat_res)
sf_pos_rounded
# -----------

# Now in units of resolution ... but how translate to indices?

# account for the resolution to specify appropriate indices

# spat_filt and stimulus will be made with the same resolution
# so ... from extent to extent will be the same number of pixels
# ===========
st_params
# -----------
# ===========
test_res, test_ext = (
    ArcLength(1, 'mnt'),
    ArcLength(100, 'mnt')
    )
test_coords = ff.mk_spat_coords_1d(test_res, test_ext)
# -----------
# ===========
ff.mk_spat_radius(test_ext)
# -----------
# ===========
# test_coords.base.shape
test_coords.value
# -----------
# ===========
# testing ... should add to tests
spat_coord_cent_idx = test_coords.value.shape[0] // 2
spat_coord_max = ff.mk_spat_radius(test_ext)
spat_coord_stride = np.max(np.diff(test_coords.value))
# center is shape // 2 or (ext // 2) / res (dividing by res necessary for when res is not 1)
spat_coord_cent_idx2 = int((test_ext.value // 2) / test_res.value)
# -----------
# ===========
test_coords.value[spat_coord_cent_idx] == 0
test_coords.value[spat_coord_cent_idx2] == 0
test_coords.value[0] == -spat_coord_max.value
test_coords.value[-1] == spat_coord_max.value
spat_coord_stride == test_res.value
test_coords.value[spat_coord_cent_idx+1] == test_res.value
# -----------
# ===========
test_coords
# -----------
# ===========
sf_pos = (ArcLength(3.5, 'mnt'), ArcLength(5.3, 'mnt'))
sf_pos_rounded = round_spat_coords_to_resolution(*sf_pos, res=spat_res)
test_sf_pos = sf_pos_rounded[0]
# -----------
# ===========
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
# -----------
# ===========
def mk_rf_stim_index(
        st_params: SpaceTimeParams,
        spat_filt_params: DOGSpatFiltArgs,
        pos_cent_x: ArcLength[float], pos_cent_y: ArcLength[float]
        ):

    # in same units as spat_res and value is int
    # spat_filt_radius = ff.mk_spat_radius(
    #     ff.mk_spat_ext_from_sd_limit(
    #         # unit is preserved, so convert to spat_res now and allow ceil and int
    #         # conversions to occur after conversion to appropriate unit
    #         spat_filt_params.max_sd().in_same_units_as(
    #             st_params.spat_res  # type: ignore  float int problems :(
    #             )
    #         ))

    spat_filt_radius = ff.mk_rounded_spat_radius(
            st_params.spat_res,
            ff.mk_spat_ext_from_sd_limit(
                spat_filt_params.max_sd().in_same_units_as(
                    st_params.spat_res  # type: ignore
                    )
                )
        )
    spat_filt_diam = (2 * spat_filt_radius.value) + 1

    pos_x, pos_y = round_spat_coords_to_resolution(
            pos_cent_x, pos_cent_y, st_params.spat_res
        )

    # presuming that these are in the same units!!
    assert spat_filt_radius.unit == st_params.spat_res.unit, (
        f'spat_filt_radius unit ({spat_filt_radius.unit}) '
        f'not same as spat_res unit ({st_params.spat_res.unit})'
        )
    spat_filt_start_idx = -1 * int(
            spat_filt_radius.value
            /
            st_params.spat_res.value
        )

    # adjust by pos
    pos_x_idx_adj = int(pos_x.value / st_params.spat_res.value)
    pos_y_idx_adj = int(pos_y.value / st_params.spat_res.value)
    spat_filt_start_idx_x = spat_filt_start_idx + pos_x_idx_adj
    spat_filt_start_idx_y = spat_filt_start_idx + pos_y_idx_adj

    # adjust by cent
    ## If coords have been created by this point for both the spat_filt
    ## and the stimulus, then the extents for both are whole number multiples
    ## to have passed the check

# -----------

# ===========
spat_filt_diam, spat_filt_radius
# -----------
# ===========
# how take account of resolution?  Divide radius by resolution (into pixels)?
# resolution is not independent in my implementation
# it seems it really must be a whole number factor of spat coords and extent!!
len(range(3-33, 3 + 33 + 1))
# -----------
np.arange(3-33, 3 + 33 + 1)[33]

# ===========
res = 5
ext = 33
cds = np.arange(-ext, ext+res, res)
cds.shape, cds[cds.shape[0]//2]
# -----------





# >> THeta Trans Func
# unnecessary ... rotation works just fine
# from 90 degs ... rotates anti-clockwise in degrees
# ===========
def img_theta_trans(theta):
    '''
    Transforms theta to operate in a way that works intuitively for scipy.ndimage.interpolation.rotate
    '''
    return -1*(theta%180) - 90
# -----------
# ===========
theta = np.linspace(0, 360, 360, False)
px.line(x=theta, y=img_theta_trans(theta)).show()
# -----------
# ===========
rot_spat_filt = interpolation.rotate(spat_filt, img_theta_trans(135), reshape=False)

px.imshow(
    rot_spat_filt,
    color_continuous_scale=px.colors.diverging.Portland,
    color_continuous_midpoint=0
    ).show()
# -----------

# ===========
test_img = np.zeros((101, 101))
test_img[0:50, 50] = 1
# -----------
# ===========
angle = -75
px.imshow(
    interpolation.rotate(test_img, angle, reshape=False),
    title = f'{angle}'
    ).show()
# -----------
