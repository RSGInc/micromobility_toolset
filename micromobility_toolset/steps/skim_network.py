import numpy as np

from ..model import step

@step()
def skim_network(*scenarios):
    """Skim the network for bike and walk distance matrices.

    This standalone step will generate skims for the provided scenarios.
    
    Network and skim configurations should be described in
    'network.yaml' and 'skims.yaml' respectively.

    """

    for scenario in scenarios:

        print(f"\ngetting {scenario.name} skims...")

        print(f"\n{scenario.name} bike skim stats")
        print_skim_stats(scenario.bike_skim)

        # print(f"\n{scenario.name} walk skim stats")
        # print_skim_stats(scenario.walk_skim)


def print_skim_stats(skim_matrix):

    print(f'zone count: {skim_matrix.shape[0]}')
    print(f'min dist: {np.amin(skim_matrix)}')
    print(f'max dist: {np.amax(skim_matrix)}')
    print(f'median: {np.median(skim_matrix)}')
    print(f'mean: {np.mean(skim_matrix)}')
    print(f'zero count: {np.count_nonzero(skim_matrix==0)}')
    print(f'non zero count: {np.count_nonzero(skim_matrix)}')
