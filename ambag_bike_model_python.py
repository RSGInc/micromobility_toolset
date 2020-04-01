import argparse

from activitysim.core import pipeline
from activitysim.core import inject
from activitysim.core.config import setting

from bike_demand.models import (
    initial_demand,
    incremental_demand,
    benefits,
    assign_demand)


def run():

    models = [
        'initial_demand',
        'incremental_demand',
        'benefits',
        'assign_demand'
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', dest='type', action='store', choices=models)
    args = parser.parse_args()

    inject.add_injectable('configs_dir', 'ambag_example/configs')
    inject.add_injectable('data_dir', 'ambag_example/data')
    inject.add_injectable('output_dir', 'ambag_example/output')

    inject.add_step('initial_demand', initial_demand)
    inject.add_step('incremental_demand', incremental_demand)
    inject.add_step('benefits', benefits)
    inject.add_step('assign_demand', assign_demand)

    if args.type:
        pipeline.run(models=[args.type])
    else:
        pipeline.run(models=setting('models'))

    pipeline.close_pipeline()


if __name__ == '__main__':
    run()
