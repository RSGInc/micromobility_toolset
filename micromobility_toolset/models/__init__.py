"""micromobility_toolset.models

Models are imported as ActivitySim injected steps.

This module defines the following constants:
    - STEPS: dictionary of step names, function objects.
    These have been defined by the @step decorator in the
    imported modules.
    - NAMES: step names

The models.run() function takes a step name or list of
names and executes them in order.

See ambag_bike_model.py for a usage example.

"""

from activitysim.core import inject as _inject

from .skim_network import skim_network
from .initial_demand import initial_demand
from .incremental_demand import incremental_demand
from .benefits import benefits
from .assign_demand import assign_demand

STEPS = _inject._DECORATED_STEPS
NAMES = list(STEPS.keys())

def run(names):
    if not isinstance(names, list):
        names = [names]

    if not all(name in STEPS for name in names):
        raise KeyError(f'Invalid step list {names}')

    for name in names:
        step_func = STEPS.get(name)
        step_func()
