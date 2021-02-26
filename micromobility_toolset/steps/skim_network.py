import logging
import numpy as np

from ..model import step

@step()
def skim_network(*scenarios):
    """Skim the network for bike cost matrix.

    This standalone step will generate skims for the provided scenarios.
    """

    for scenario in scenarios:

        scenario.logger.info('getting skims...')

        skims = scenario.skims
        scenario.logger.info(f'zone count: {skims.length}')

        for core in skims.core_names:

            skim_matrix = skims.get_core(core)

            scenario.logger.info(f'{core} min cost: {np.amin(skim_matrix)}')
            scenario.logger.info(f'{core} max cost: {np.amax(skim_matrix)}')
            scenario.logger.info(f'{core} median: {np.median(skim_matrix)}')
            scenario.logger.info(f'{core} mean: {np.mean(skim_matrix)}')
            scenario.logger.info(f'{core} zero count: {np.count_nonzero(skim_matrix==0)}')
            scenario.logger.info(f'{core} non zero count: {np.count_nonzero(skim_matrix)}')
