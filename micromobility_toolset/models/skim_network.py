import numpy as np

from activitysim.core.inject import step

from ..utils.io import load_skim


@step()
def skim_network():
    """Skim the base network for bike and walk distance matrices.

    This standalone step will generate 'base' skims for subsequent
    model steps and will write skim files to the base directory.
    If base skims already exist they will not be overwritten but
    will be loaded and described by `print_skim_stats()`.

    Network and skim configurations should be described in
    'network.yaml' and 'skims.yaml' respectively.

    """

    bike_skim = load_skim('bike', base=True)
    walk_skim = load_skim('walk', base=True)

    print("\nbike skim stats")
    print_skim_stats(bike_skim)

    print("\nwalk skim stats")
    print_skim_stats(walk_skim)


def print_skim_stats(skim_matrix):

    print(f'taz count: {skim_matrix.shape[0]}')
    print(f'min dist: {np.amin(skim_matrix)}')
    print(f'max dist: {np.amax(skim_matrix)}')
    print(f'median: {np.median(skim_matrix)}')
    print(f'mean: {np.mean(skim_matrix)}')
    print(f'zero count: {np.count_nonzero(skim_matrix==0)}')
    print(f'non zero count: {np.count_nonzero(skim_matrix)}')
