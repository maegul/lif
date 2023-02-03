# convenience imports for the general `lgn` api
from .cells import mk_lgn_layer

from ..utils import data_objects as do
from ..utils.units.units import (ArcLength, Time)


# > demonstration params
demo_stparams = do.SpaceTimeParams(
    spat_ext=ArcLength(5), spat_res=ArcLength(1, 'mnt'), temp_ext=Time(1),
    temp_res=Time(1, 'ms'))

demo_lgnparams = do.LGNParams(
    n_cells=20,
    orientation = do.LGNOrientationParams(ArcLength(30), 0.5),
    circ_var = do.LGNCircVarParams('naito_lg_highsf', 'naito'),
    spread = do.LGNLocationParams(2, 'jin_etal_on'),
    filters = do.LGNFilterParams(spat_filters='all', temp_filters='all'),
    F1_amps = do.LGNF1AmpDistParams()
    )
