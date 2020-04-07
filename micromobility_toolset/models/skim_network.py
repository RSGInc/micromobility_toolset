import numpy as np

from activitysim.core.inject import step

from ..utils.io import load_skim


@step()
def skim_network():

    bike_skim = load_skim('bike')
    walk_skim = load_skim('walk')

    print('')
    print('bike skim stats')
    print_skim_stats(bike_skim)

    print('')
    print('walk skim stats')
    print_skim_stats(walk_skim)


def print_skim_stats(skim_matrix):

    print(f'taz count: {skim_matrix.shape[0]}')
    print(f'min dist: {np.amin(skim_matrix)}')
    print(f'max dist: {np.amax(skim_matrix)}')
    print(f'median: {np.median(skim_matrix)}')
    print(f'mean: {np.mean(skim_matrix)}')
    print(f'zero count: {np.count_nonzero(skim_matrix==0)}')
    print(f'non zero count: {np.count_nonzero(skim_matrix)}')
