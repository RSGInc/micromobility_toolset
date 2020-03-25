from math import log, sqrt, exp
import copy, random

def get_coef_bounding_box_single_od_pair(net,source,target,previous_means,outer_box,tolerance):
	
	new_box={}
	
	for var in outer_box:
		
		if outer_box[var][1] <= outer_box[var][0]:
			new_box[var] = outer_box[var]
		else:
			myfun =  lambda cur_coef: net.path_trace(
				net.single_source_dijkstra(
					source,
					dict(previous_means,**{var:cur_coef}),
					target=target
				)[1][target],
				var
			)
			
			coef_min_low = coef_min_high = log(outer_box[var][0])
			coef_max_low = coef_max_high = log(outer_box[var][1])
			val_min_low = val_min_high = myfun(exp(coef_min_low))
			val_max_low = val_max_high = myfun(exp(coef_max_low))
			
			if val_min_low == val_max_low:
				
				new_box[var]=[None,None]
			
			else:
			
				while True:
				
					coef_mid_low = (coef_min_low+coef_max_low)/2
					coef_mid_high = (coef_min_high+coef_max_high)/2
					val_mid_low =  myfun(exp(coef_mid_low))
					val_mid_high =  myfun(exp(coef_mid_high))
					
					if val_mid_low == val_min_low:
						coef_min_low = coef_mid_low
					else:
						coef_max_low = coef_mid_low
						val_max_low = val_mid_low
					
					if val_mid_high == val_max_high:
						coef_max_high = coef_mid_high
					else:
						coef_min_high = coef_mid_high
						val_min_high = val_mid_high
				
					if (coef_max_low - coef_min_low) < tolerance:
						break
			
				new_box[var]= (exp(coef_mid_low),exp(coef_mid_high))
				
	return new_box
	
def get_updated_coef_bounding_box(net,paths,previous_box,outer_box,tolerance):
	
	new_box = {}
	temp_box = {}
	previous_means = {}
	counts = {}
	
	for var in previous_box:
		previous_means[var] = sqrt( previous_box[var][0] * previous_box[var][1] )
		new_box[var] = [0,0]
		counts[var] = 0
		
	for path in paths:
		temp_box = get_coef_bounding_box_single_od_pair(net,path[0],path[-1],previous_means,outer_box,tolerance)
		for var in temp_box:
			if temp_box[var][0] is not None:
				new_box[var][0] = new_box[var][0] + log(temp_box[var][0])
				new_box[var][1] = new_box[var][1] + log(temp_box[var][1])
				counts[var] = counts[var] + 1
	
	for var in temp_box:
		new_box[var][0] = exp( new_box[var][0] / counts[var] )
		new_box[var][1] = exp( new_box[var][1] / counts[var] )		
				
	return new_box
	
def get_coef_bounding_box_noniterative(net,paths,config):
	
	outer_box = config.bounding_box_outer_box
	tolerance = config.bounding_box_tolerance
	ref = config.bounding_box_ref_var
	median_vars = config.bounding_box_median_compare
	
	new_box = {}
	temp_box = {}
	previous_means = {ref:1.0}
	counts = {}

	for var in outer_box:
		new_box[var] = [0,0]
		counts[var] = 0

	median_box = {}
	for var in median_vars:
		median_box[var] = outer_box[var]
		del outer_box[var]
	
	print( 'Calibrating coefficient bounding box' )
	print( 'Initial boundary:' )
	print( outer_box )
		
	for path in paths:
		temp_box = get_coef_bounding_box_single_od_pair(net,path[0],path[-1],previous_means,outer_box,tolerance)
		for var in temp_box:
			if temp_box[var][0] is not None:
				new_box[var][0] = new_box[var][0] + log(temp_box[var][0])
				new_box[var][1] = new_box[var][1] + log(temp_box[var][1])
				counts[var] = counts[var] + 1
	
	for var in temp_box:
		new_box[var][0] = exp( new_box[var][0] / counts[var] )
		new_box[var][1] = exp( new_box[var][1] / counts[var] )
		previous_means[var] = sqrt( new_box[var][0] * new_box[var][1] )
	
	
	print( 'Median boundary:' )
	for path in paths:
		temp_box = get_coef_bounding_box_single_od_pair(net,path[0],path[-1],previous_means,median_box,tolerance)
		for var in temp_box:
			if temp_box[var][0] is not None:
				new_box[var][0] = new_box[var][0] + log(temp_box[var][0])
				new_box[var][1] = new_box[var][1] + log(temp_box[var][1])
				counts[var] = counts[var] + 1
				
	for var in temp_box:
		new_box[var][0] = exp( new_box[var][0] / counts[var] )
		new_box[var][1] = exp( new_box[var][1] / counts[var] )
				
	return new_box
	
def get_coef_bounding_box(net,paths,config):
	
	outer_box = config.bounding_box_outer_box
	tolerance = config.bounding_box_tolerance
	maxiters = config.bounding_box_maxiters
	
	previous_box = copy.copy(outer_box)
	iter = 0
	
	print( 'Calibrating coefficient bounding box' )
	print( 'Initial boundary:' )
	print( previous_box )
	
	while iter < maxiters:
		print(('Iteration: ' + str(iter) ))
		terminate_flag = True
		new_box = get_updated_coef_bounding_box(net,paths,previous_box,outer_box,tolerance)
		print( new_box )
		for var in new_box:
			if max(
				abs(
					log(new_box[var][0]) - log(previous_box[var][0])
				),
				abs(
					log(new_box[var][0]) - log(previous_box[var][0])
				)
			)  > tolerance:
				terminate_flag = False
				break
		if terminate_flag:
			print('Terminating with convergence')
			return new_box
		else:
			for var in new_box:
				previous_box[var][0] = new_box[var][0] # sqrt( new_box[var][0] * previous_box[var][0] )
				previous_box[var][1] = new_box[var][1] # sqrt( new_box[var][1] * previous_box[var][1] )
				
		iter = iter + 1

	print('Terminating without convergence')

def generate_choice_set(net,chosen,bounding_box,randomization_scale,num_draws):
	
	source=chosen[0]
	target=chosen[-1]
	
	variable_coefficients={}
	choice_set=[chosen]
	for i in range(num_draws):
		
		#sample random coefficients from bounding box
		for var in bounding_box:
			variable_coefficients[var] = exp(
				random.uniform(
					log(bounding_box[var][0]),
					log(bounding_box[var][1])
				)
			)
			
		#perform generalized cost shortest path search
		choice_set.append( net.single_source_dijkstra(source,variable_coefficients,target,randomization_scale)[1][target] )
		
	return choice_set
	