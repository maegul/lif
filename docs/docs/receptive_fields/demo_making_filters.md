## Creating Filters from empirical data


Imports

```python
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as psp
```
```python
from scipy.interpolate.interpolate import interp1d
from lif.utils.units.units import ArcLength, SpatFrequency, TempFrequency, Time
from lif.utils import data_objects as do
from lif.receptive_field.filters import (
    filter_functions as ff, filters)
from lif.plot import plot
```

Get the `data directory` from `settings`

* Spatial filters **as well as the raw data they're derived from** are to be stored
in the `data directory`.

```python
import lif.utils.settings as settings
data_dir = settings.get_data_dir()
```


### Temp Filter


Get the raw data (captured with `tracey`)

```python
tf_raw_path = data_dir / 'Kaplan_et_al_1987_fig6A.csv'
tf_raw_data = pd.read_csv(tf_raw_path, header=None)
tf_raw_data
```

<!-- #region -->
```python
          0         1         2         3         4         5         6         7
0 -1.955984 -0.921596  0.079780  1.081155  2.082531  3.083907  4.679505  6.088033
1  5.018553  4.907236  5.163265  5.497217  5.653061  5.653061  4.461966  1.333952
```
<!-- #endregion -->

* **Processing** raw `temp_filt` data
    * data originally represented in octaves ... *make linear*
    * rename columns for convenience

```python
clean_tf_raw_data: pd.DataFrame = 2**tf_raw_data.T
clean_tf_raw_data.rename(columns={0: 'tf_hz', 1: 'resp_imp_s'}, inplace=True)
clean_tf_raw_data
```

<!-- #region -->
```python
       tf_hz  resp_imp_s
0   0.257745   32.414175
1   0.527925   30.007175
2   1.056857   35.834197
3   2.115730   45.167623
4   4.235496   50.320038
5   8.479074   50.320038
6  25.625439   22.038688
7  68.026892    2.520922
```
<!-- #endregion -->

save clean data (if necessary)

```python
clean_tf_raw_data.to_csv(data_dir/'Kaplan_et_al_1987_fig6A_clean.csv', index=False)
```

#### Loading *cleaned* data from file in data dir

```python
clean_tf_raw_data = pd.read_csv(data_dir/'Kaplan_et_al_1987_fig6A_clean.csv')
```

#### Preparing data for tq tf fit


Extract values from dataframe and
**ENSURE frequency are the appropriate unit and scale**, which depends on the original data

```python
amps = clean_tf_raw_data.resp_imp_s.values
#  User must know that original data is in hertz ------V
fs = do.TempFrequency(clean_tf_raw_data.tf_hz.values, 'hz')
```

Define a `TempFiltParams` object with all requisite data and metadata

```python
# basic data
data = do.TempFiltData(fs, amps)
# metadata about experimental conditions under which data was created
resp_params = do.TFRespMetaData(
    dc=12, contrast=0.4, mean_lum=100,
    sf=do.SpatFrequency(0.8, 'cpd')
    )
# bibliographic metadata
meta_data = do.CitationMetaData(
    'Kaplan et al', 1987,
    'contrast affects transmission', 'fig 6a open circles')
```
```python
tf_params = do.TempFiltParams(
    data=data,
    resp_params=resp_params,
    meta_data=meta_data
    )
```

#### Fitting a temp filter to the raw data


* The `filters` module contains "simple" to use functions for fitting data to the specified filter type.
* These functions will accept the full `empirical-data-object` as their input, and
return a `filter` object by performing the necessary optimisation.
* This `filter` object will contain:
    * Raw data (as provided by to the fitting function)
    * Optimisation object as returned by the optimisation function
      (with information about the goodness of fit)
    * Parameters of the actual filter that will be used by the simulation code

```python
tqtf = filters.make_tq_temp_filt(parameters=tf_params)
```

EG, this object now contains the filter parameters and optimisation results

```python
tqtf.parameters.asdict_()
```

<!-- #region -->
```python
{'amplitude': 301.920227430036,
 'arguments': {'tau': {'value': 14.902653977227773,
   'unit': 'ms',
   '_base_unit': 's',
   '_s': 1,
   '_ms': 0.001,
   '_us': 1e-06},
  'w': 11.420030537940058,
  'phi': 1.1201854258947725}}
```
<!-- #endregion -->

```python
tqtf.optimisation_result.success
```

<!-- #region -->
```python
True
```
<!-- #endregion -->

#### Graphing results


Use the `plot` module (which should be convenient)

```python
fig = plot.tq_temp_filt_fit(tqtf)
fig.show()
```
```python
fig.write_image('tq_temp_filt_fit.svg')
    # -

# write to html file like so ...
```

```python
# fig.write_html('tq_temp_filt_fit.html', include_plotlyjs='cdn')
# writing static images requires kaleido
fig.update_layout(template='plotly_dark').write_image('tq_temp_filt_fit.svg')
```

![See plot here](./tq_temp_filt_fit.svg)


#### Saving Filter


save and load methods are built into the temporal filter object itself.

```python
tqtf.save()
```

Overwrite the previous file **if necessary**

```python
tqtf.save(overwrite=True)
```

#### Loading the filter


Filter object has utility functions for managing saved filters

```python
for filter in do.TQTempFilter.get_saved_filters():
    print(f'{filter.parent} ...')
    print(filter.name)
    print('')
```
```
/Users/errollloyd/.lif_hws_data ...
Kaplan_et_al_1987_contrast_affects_transmission_fig_6a_open_circles-TQTempFilter.pkl
```

```python
loaded_tqtf = do.TQTempFilter.load(
    do.TQTempFilter.get_saved_filters()[0])
```


### Spat Filt


#### Pre-Cleaned CSV Data


This has already been cleaned

```python
sf_data_raw = data_dir / 'Kaplan_et_al_1987_fig8A.csv'
sf_data_raw_df = pd.read_csv(sf_data_raw)
```

```
        freq        amp
0   0.102946  15.648086
1   0.256909  16.727744
2   0.515686  15.764523
3   1.035121  18.014953
4   2.062743  27.488355
5   4.140486  28.952478
6   8.311079  16.119054
7  16.205212   1.355197
8  33.003893   1.537217
```


#### Preparing data for DOG spat filt Fitting


Extract values from dataframe and
**ENSURE frequency are the appropriate unit and scale**, which depends on the original data

```python
data = do.SpatFiltData(
    amplitudes=sf_data_raw_df['amp'].values,
    #  User must know that the original data is in CPD -------V
    frequencies=SpatFrequency(sf_data_raw_df['freq'].values, 'cpd')
    )
```
```python
resp_params = do.SFRespMetaData(
    dc=15, tf=TempFrequency(4, 'hz'),
    mean_lum=100, contrast=0.5
    )
meta_data = do.CitationMetaData(
    author='Kaplan_et_al',
    year=1987,
    title='contrast affects transmission',
    reference='fig8A_open_circle',
    doi=None)
```
```python
sf_params = do.SpatFiltParams(
    data = data, resp_params = resp_params, meta_data = meta_data
    )
```

#### Fitting a DOG Spatial Filter

```python
sf = filters.make_dog_spat_filt(sf_params)
```
```python
sf.parameters.asdict_()
```

```
{'cent': {'amplitude': 36.426593766279,
  'arguments': {'h_sd': {'value': 1.4763319227563279,
    'unit': 'mnt',
     ...},
   'v_sd': {'value': 1.4763319227563279,
    'unit': 'mnt',
     ...}}},
 'surr': {'amplitude': 21.123002549917594,
  'arguments': {'h_sd': {'value': 6.455530620581974,
    'unit': 'mnt',
     ...},
   'v_sd': {'value': 6.455530620581974,
    'unit': 'mnt',
     ...}}}}
```


#### Saving and Loading


Same saving and loading and "*getter*" as with temp filters above.

```python
sf.save()
# sf.save(overwrite=True)
```


```python
# do.DOGSpatialFilter.load()
# do.DOGSpatialFilter.save()
# do.DOGSpatialFilter.get_saved_filters()
loaded_sf = do.DOGSpatialFilter.load(do.DOGSpatialFilter.get_saved_filters()[0])
```

## Graphing Filters


### Filters


Temporal filter

```python
from importlib import reload
```

```python
tqtf = do.TQTempFilter.load(do.TQTempFilter.get_saved_filters()[0])
fig = plot.tq_temp_filt(tqtf)
# fig.update_layout(template='plotly_dark').write_image('tq_temp_filt.svg')
```

![](./tq_temp_filt.svg)


Spatial Filter

```python
sf = do.DOGSpatialFilter.load(do.DOGSpatialFilter.get_saved_filters()[0])
fig = plot.spat_filt(sf, spat_res=ArcLength(20, 'sec'))
# fig.update_layout(template='plotly_dark').write_image('spat_filt.svg')
```

![](./spat_filt.svg)


### The fits of filters to their raw data


temporal filter ...

```python
tqtf = do.TQTempFilter.load(do.TQTempFilter.get_saved_filters()[0])
fig = plot.tq_temp_filt_fit(tqtf)
# fig.update_layout(template='plotly_dark').write_image('tq_temp_filt_fit.svg')
```

![](./tq_temp_filt_fit.svg)

```python
tqtf = do.TQTempFilter.load(do.TQTempFilter.get_saved_filters()[0])
fig = plot.tq_temp_filt_profile(tqtf)
# fig.update_layout(template='plotly_dark').write_image('tq_temp_filt_profile.svg')
```

![](./tq_temp_filt_profile.svg)


spatial filter ...

```python
sf = do.DOGSpatialFilter.load(do.DOGSpatialFilter.get_saved_filters()[0])
fig = plot.spat_filt_fit(sf)
# fig.update_layout(template='plotly_dark').write_image('dog_sf_filt_fit.svg')
```

![](./dog_sf_filt_fit.svg)



### Interpolation of combined Spatial and Temporal Filter

```python
fig = plot.joint_sf_tf_amp(tqtf, sf, n_increments=50)
# fig.update_layout(template='plotly_dark').write_image('joint_sf_tf_interp.svg')
```

![](./joint_sf_tf_interp.svg)


### The Orientation Selectivity of a Spatial Filter


Basic polar plot of orientation preference

```python
ori_sf = ff.mk_ori_biased_spatfilt_params_from_spat_filt(sf, circ_var=0.8)
freq = SpatFrequency(4)
fig = plot.orientation_plot(ori_sf, freq)
# fig.update_layout(template='plotly_dark').write_image('ori_polar.svg')
```

![](./ori_polar.svg)


Spatial frequency responses to vertical and horizantally oriented gratings

```python
ori_sf = ff.mk_ori_biased_spatfilt_params_from_spat_filt(sf, circ_var=0.8)
fig = plot.dog_sf_ft_hv(ori_sf, use_log_freqs=True)
# fig.update_layout(template='plotly_dark').write_image('ori_hv_spat_freq.svg')
```

![](ori_hv_spat_freq.svg)



Heatmap of response to orientation x SF
helpful for quickly viewing the overall orientation selectivity of the RF

```python
ori_sf = ff.mk_ori_biased_spatfilt_params_from_spat_filt(sf, circ_var=0.8)
fig = plot.ori_spat_freq_heatmap(ori_sf, n_orientations=16 )
# fig.update_layout(template='plotly_dark').write_image('ori_spat_freq_heatmap.svg')
```

![](./ori_spat_freq_heatmap.svg)


Orientation preference plots (polar) for a matrix of circular variance and SF values

```python
fig = plot.orientation_circ_var_subplots(sf)
fig.update_layout(template='plotly_dark').write_image('ori_spat_freq_subplots.svg')
```

![](ori_spat_freq_subplots.svg)
