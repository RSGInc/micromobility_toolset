import argparse
import time

from bike_demand import incremental_demand
from bike_demand import benefits
from bike_demand import assign_demand

usage = "ambag_bike_model_python.py {--type [incremental_demand/benefits/assign_demand]} --base [base_database] --build [build_database] [--base_disk]}"
if __name__ == '__main__':

	t1 = time.time()

	parser = argparse.ArgumentParser()
	parser.add_argument('--type',dest='type',action='store')
	parser.add_argument('--base')
	parser.add_argument('--build')
	parser.add_argument('--base_disk',help='read base skims from disk to speed up incremental demand',action='store_true')
	args = parser.parse_args()

	model = args.type
	if model == "incremental_demand":
		incremental_demand.incremental_demand_main()
	elif model == "benefits":
		benefits.benefits_main()
	elif model == "assign_demand":
		assign_demand.assign_demand_main()
	else:
		print(usage)

	print('runtime: ', time.time() - t1)
