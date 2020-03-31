import argparse
import time

from activitysim.core import pipeline
from activitysim.core import inject
from activitysim.core.config import setting

from bike_demand import incremental_demand
from bike_demand import benefits
from bike_demand import assign_demand


def run():

    t1 = time.time()

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

    if args.type:
        pipeline.run(models=[args.type])
    else:
        pipeline.run(models=setting('models'))

    pipeline.close_pipeline()

    print('runtime: ', time.time() - t1)


if __name__ == '__main__':
    run()
