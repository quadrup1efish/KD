import pynbody
import numpy as np
from pynbody import units
from pynbody.array import SimArray

@pynbody.derived_array
def jp(sim):
    j = np.cross(sim['pos'], sim['vel'])
    jp= SimArray(np.sqrt(j[:, 0]**2 + j[:, 1]**2), units.kpc*units.km/units.s)
    return jp

@pynbody.derived_array
def e(sim):
    return sim['phi']+sim['ke']

@pynbody.derived_array
def age(sim):
    return sim['tform']

    



