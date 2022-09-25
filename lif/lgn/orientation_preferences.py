# > Imports
from typing import List, cast

import numpy as np

import plotly.express as px

from ..utils import data_objects as do
from ..utils.units.units import (
    scalar,
    ArcLength
    )
from ..receptive_field.filters import cv_von_mises as cvvm


# > Orientations

# +
def _mk_angle_probs(vm_params: do.VonMisesParams, n_angle_increments: int = 1000):

    angles = ArcLength(
        np.linspace(0, 180, n_angle_increments, endpoint=False))
    probs = cvvm.von_mises(
        angles,
        k=vm_params.k,
        phi=vm_params.phi)
    probs /= probs.sum()

    return angles, probs

def mk_orientations(
        n: int,
        angles: ArcLength[np.ndarray], probs: np.ndarray
        ) -> List[ArcLength[scalar]]:

    random_angles = np.random.choice(angles.deg, p=probs, size=n)
    orientations = [
        ArcLength(a, 'deg')
        for a in random_angles
    ]

    return orientations
# -

# >>> Demo
# +
# mk_orientations(
#     20,
#     *_mk_angle_probs(do.VonMisesParams(
#         phi=ArcLength(10, 'deg'),
#         k=cvvm.cv_k(0.5)  # convert circ_var to kappa value
#         ))
#     )
# -


# > Circular Variance Distribution

# Get Distributions fron Naito, Shou and anywhere else a definition of circ_var
# ... has been taken from

# >> Naito Circ Var Values

# BIG DECISION HERE about which distribution to use!!
# obviously there needs to be some sort of match between the distribution and the definition
# of circ variance used (thus using the Naito data).
# But ... to use the large stimulus or optimimal stimulus data?
# ... Guess I could just use both again.
# ... Using large stimulus data is consistent with the large full-field stimuli many others
# ... have used in their work.  The suppressive effect of the large stimulus (which is well
# ... demonstrated in the Naito paper) should be reasonably presumed to be operating in my model.
# ... in fact ... maybe I should be modelling it??  OR, simply adjusting circular variance
# ... to follow its effect on the magnitude of orientation biases should be sufficient.

# >>> Large + High SF

naito_lg_highsf_raw = {
    # bins are 0.1 circ_var wide
    "hist_mp": [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75],
    "count":
    [11.330471657795853,36.16595133510011,14.306153877245706,19.34192185265844,
        3.204583430949195,2.288986967462225,1.2589440894807837,1.2589440894807837]
}

