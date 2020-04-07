import argparse

from activitysim.core import pipeline
from activitysim.core import inject
from activitysim.core.config import setting

from micromobility_toolset import models


def run():

    # step methods imported by the models module
    types = [type for type in dir(models) if '__' not in type]

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', dest='type', action='store', choices=types)
    args = parser.parse_args()

    inject.add_injectable('configs_dir', 'ambag_example/configs')
    inject.add_injectable('data_dir', 'ambag_example/base')
    inject.add_injectable('output_dir', 'ambag_example/build')

    if args.type:
        pipeline.run(models=[args.type])
    else:
        pipeline.run(models=setting('models'))

    pipeline.close_pipeline()


if __name__ == '__main__':
    run()
