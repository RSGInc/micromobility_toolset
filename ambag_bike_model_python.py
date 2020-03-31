import argparse
import time

from activitysim.core import pipeline
from activitysim.core import inject

from bike_demand import incremental_demand
from bike_demand import benefits
from bike_demand import assign_demand

usage = "ambag_bike_model_python.py {--type [incremental_demand/benefits/assign_demand]} --base [base_database] --build [build_database] [--base_disk]}"
if __name__ == '__main__':

    t1 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--type',dest='type',action='store')
    args = parser.parse_args()

    inject.add_injectable('configs_dir', 'ambag_example/configs')
    inject.add_injectable('data_dir', 'ambag_example/data')
    inject.add_injectable('output_dir', 'ambag_example/output')

    model = args.type
    if model == "incremental_demand":
        pipeline.run(models=['incremental_demand'])
        pipeline.close_pipeline()
    elif model == "benefits":
        benefits.benefits_main()
    elif model == "assign_demand":
        assign_demand.assign_demand_main()
    else:
        print(usage)

    print('runtime: ', time.time() - t1)
