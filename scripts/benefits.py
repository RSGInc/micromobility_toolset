import choice_set, network, config, output, csv, sqlite3, numpy, time, argparse
from input import *

def benefits_main():
	
	# read configuration data
	resources = config.Config()
	
	# parse command line options to get base and build database file locations
	parser = argparse.ArgumentParser(description='Perform benefits calculation for bike mode shift model')
	parser.add_argument('--type') #ignore here
	parser.add_argument('--base',dest='base',action='store')
	parser.add_argument('--build',dest='build',action='store')
	parser.add_argument('--base_disk',help='read base skims from disk to speed up incremental demand',action='store_true') #happens to be irrelevant for benefits
	args = parser.parse_args()
	resources.application_config.base_sqlite_file = args.base
	resources.application_config.build_sqlite_file = args.build
	
	# get number of zones to dimension matrices
	nzones = resources.application_config.num_zones
	
	# read auto times and distances
	auto_skim = read_matrix_from_sqlite(resources,'auto_skim',resources.application_config.base_sqlite_file)
	
	# initialize empty matrices
	delta_trips = numpy.zeros((nzones,nzones,len(resources.mode_choice_config.modes)))
	user_ben = numpy.zeros((nzones,nzones))
	
	# ignore numpy divide by zero errors
	numpy.seterr(divide='ignore',invalid='ignore')
	
	print ''
	print('calculating vmt, emissions, and user benefits...')
	
	# loop over market segments
	for idx in range(len(resources.mode_choice_config.trip_tables)):
		
		# read in trip tables
		base_trips = read_matrix_from_sqlite(resources,resources.mode_choice_config.trip_tables[idx],resources.application_config.base_sqlite_file)
		build_trips = read_matrix_from_sqlite(resources,resources.mode_choice_config.trip_tables[idx],resources.application_config.build_sqlite_file)
		
		# calculate difference in trips
		delta_trips = delta_trips + build_trips - base_trips
		
		# aalculate logsums
		base_logsum = numpy.log(1.0 + numpy.nan_to_num(base_trips[:,:,6]/(numpy.sum(base_trips,2)-base_trips[:,:,6])))
		build_logsum = numpy.log(1.0 + numpy.nan_to_num(build_trips[:,:,6]/(numpy.sum(build_trips,2)-build_trips[:,:,6])))
		
		# calculate user benefits
		user_ben = user_ben - numpy.sum(base_trips,2) * (build_logsum - base_logsum)/resources.mode_choice_config.ivt_coef[idx]
	
	# calculate difference in vmt and vehicle minutes of travel
	delta_minutes = auto_skim[:,:,0] * (delta_trips[:,:,0] + delta_trips[:,:,1] /2.0 + delta_trips[:,:,2] /resources.application_config.sr3_avg_occ)
	delta_miles = auto_skim[:,:,1] * (delta_trips[:,:,0] + delta_trips[:,:,1] /2.0 + delta_trips[:,:,2] /resources.application_config.sr3_avg_occ)
	
	print ''	
	print 'User benefits (min.): ', int(numpy.sum(user_ben))
	print 'Change in bike trips: ', int(numpy.sum(delta_trips[:,:,6]))
	print 'Change in VMT: ', int(numpy.sum(delta_miles))
	
	# calculate difference in pollutants
	delta_pollutants = numpy.zeros((nzones,nzones,len(resources.application_config.pollutants)))
	for idx in range(len(resources.application_config.pollutants)):
		delta_pollutants[:,:,idx] = delta_miles * resources.application_config.grams_per_mile + delta_minutes * resources.application_config.grams_per_minute
		print 'Change in g. ' + resources.application_config.pollutants[idx] + ': ', int(numpy.sum(delta_pollutants[:,:,idx]))

	print ''
	print('writing to disk...')
	output.write_matrix_to_sqlite(user_ben,resources.application_config.build_sqlite_file,'user_ben',['minutes'])
	output.write_matrix_to_sqlite(delta_trips,resources.application_config.build_sqlite_file,'chg_trips',resources.mode_choice_config.modes)
	output.write_matrix_to_sqlite(delta_miles,resources.application_config.build_sqlite_file,'chg_vmt',['value'])
	output.write_matrix_to_sqlite(delta_pollutants,resources.application_config.build_sqlite_file,'chg_emissions',resources.application_config.pollutants)
	
if __name__ == '__main__':
	benefits_main()
