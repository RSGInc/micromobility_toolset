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

        scenario.logger.info('bike skim stats')

        skim_matrix = scenario.bike_skim

        scenario.logger.info(f'zone count: {skim_matrix.shape[0]}')
        scenario.logger.info(f'min dist: {np.amin(skim_matrix)}')
        scenario.logger.info(f'max dist: {np.amax(skim_matrix)}')
        scenario.logger.info(f'median: {np.median(skim_matrix)}')
        scenario.logger.info(f'mean: {np.mean(skim_matrix)}')
        scenario.logger.info(f'zero count: {np.count_nonzero(skim_matrix==0)}')
        scenario.logger.info(f'non zero count: {np.count_nonzero(skim_matrix)}')

