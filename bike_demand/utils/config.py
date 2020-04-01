class Config():

    def __init__(self):
        """initialize runtime configuration data, void"""

        # initialize children
        self.network_config = NetworkConfig()
        self.choice_set_config = ChoiceSetConfig()
        self.output_config = OutputConfig()
        self.mode_choice_config = ModeChoiceConfig()
        self.application_config = ApplicationConfig()

        #location of cycle tracks map-matched link traversals for route choice estimation
        self.trip_data_filename = 'data/cycletracks.csv'


class NetworkConfig():

    def __init__(self):
        """initialize network configuration data, void"""

        self.link_table = 'link'  # link table name in sqlite database
        self.link_file = 'link.csv'  # link file for csv input

        # link a, and b column names
        self.from_name = 'from_node'
        self.to_name = 'to_node'

        # column names in sqlite database for needed attributes
        self.link_attributes_by_direction = {
            'distance':    ('length','length'),
            'bike_class':    ('bike_class','bike_class'),
            'lanes':        ('ab_ln','ba_ln'),
            'riseft':        ('ab_gain','ba_gain'),
            'link_type':    ('link_type','link_type'),
            'fhwa_fc':        ('fhwa_fc','fhwa_fc'),
            'shuttle_ivt':    ('ab_ivt','ba_ivt'),
            'shuttle_wait':    ('wait','wait')
        }

        self.node_table = 'node'  # node table name in sqlite database
        self.link_file = 'node.csv'  # node file for csv input

        # node attributes
        self.node_name = 'node_id'
        self.node_attributes = {
            'xcoord':'xcoord',
            'ycoord':'ycoord'
        }
        self.node_x_name = 'xcoord'
        self.node_y_name = 'ycoord'

        # name to give boolean centroid connector column in link data
        self.centroid_connector_name = 'centroid'

        # link is a centroid connector if net.get_edge_attribute(edge,centroid_connector_test[0] ) == centroid_connector_test[1]
        self.centroid_connector_test = ('link_type','CENTROID CONNECT')

class ChoiceSetConfig():
    # irrelevant for model application

    def __init__(self):
        """initialize choice set configuration data, void"""

        self.bounding_box_outer_box = {
        'distance':                [1.0, 1.0],
            'dne1':                [0.0001, 100.0],
            'dne2':                [0.0001, 100.0],
            'dne3':                [0.0001, 100.0],
            'dw':                  [0.001, 10000.0],
            'riseft':              [0.0001, 10000.0],
            'turn':                [0.0001, 10000.0],
            'path_onoff':          [0.0001, 10000.0],
            'bike_exclude':        [999.0, 999.0],
            'thru_centroid':       [999.0, 999.0],
        }

        self.bounding_box_tolerance = 0.01
        self.bounding_box_ref_var = 'd0'
        self.bounding_box_median_compare = ['turn']

        self.num_draws = 50
        self.randomization_scale = 0.7

class OutputConfig():
    # irrelevant for model application

    def __init__(self):
        """initialize output configuration data for choice set generation, void"""

        self.choice_set_pathname = 'data/choice_set_paths.csv'
        self.choice_set_linkname = 'data/choice_set_links.csv'
        self.estimation_file = 'data/estimation.csv'

        self.path_size_overlap_var = 'distance'

        self.variables = [
            'd0',
            'd1',
            'd2',
            'd3',
            'dw',
            'riseft',
            'r_turn',
            'l_turn',
            'u_turn',
            'dloc',
            'dcol',
            'dart',
            'dne3loc',
            'dne2art',
            'path_onoff'
        ]

class ModeChoiceConfig():

    def __init__(self):
        """initialize mode choice configuration data, void"""

        # trip tables by market segment
        self.trip_tables = ['hbw1trip','hbw2trip','hbw3trip','hbw4trip',
                    'hscl1trip','hscl2trip','hscl3trip','hscl4trip',
                    'hunv1trip','hunv2trip','hunv3trip','hunv4trip',
                    'nwk1trip','nwk2trip','nwk3trip','nwk4trip',
                    'nhbtrip']

        # coefficients by market segment
        self.ivt_coef = [-0.017,-0.017,-0.017,-0.017,
                         -0.005,-0.005,-0.005,-0.005,
                         -0.061,-0.061,-0.061,-0.061,
                         -0.005,-0.005,-0.005,-0.005,
                         -0.013]

        self.walk_skim_coef = [-0.712,-0.712,-0.712,-0.712,
                               -0.712,-0.712,-0.712,-0.712,
                               -0.712,-0.712,-0.712,-0.712,
                               -0.712,-0.712,-0.712,-0.712,
                               -0.712]

        self.bike_skim_coef = [-0.182,-0.182,-0.182,-0.182,
                               -0.182,-0.182,-0.182,-0.182,
                               -0.182,-0.182,-0.182,-0.182,
                               -0.182,-0.182,-0.182,-0.182,
                               -0.182]

        self.bike_dist_coef_santa_clara = [-0.154,-0.154,-0.154,-0.154,
                                           -0.154,-0.154,-0.154,-0.154,
                                           -0.154,-0.154,-0.154,-0.154,
                                           -0.154,-0.154,-0.154,-0.154,
                                           -0.154]

        # generalized cost coefficients for bike skimming
        self.route_varcoef_bike = {
            'd0':             0.858, # distance on ordinary streets, miles ( bike_class == 0 and lanes > 0 )
            'd1':             0.387, # distance on bike paths ( bike_class == 1 )
            'd2':             0.544, # distance on bike lanes ( bike_class == 2 )
            'd3':             0.773, # distance on bike routes ( bike_class == 3 )
            'dw':             3.449, # distance wrong way ( bike_class == 0 and lanes == 0 )
            'riseft':             0.005, # elevation gain in feet
            'turn':            0.094, # number of turns
            'dne2art':            1.908, # additional penalty for distance on an arterial without bike lane ( fhwa_fc in [1,2,6,11,12,14,77] ) * ( bike_class != 2 )
            'bike_exclude':     999.0, # bikes not allowed ( link_type in ['FREEWAY'] )
            'thru_centroid':    999.0, # centroid to centroid traversal
            'shuttle_ivt':        0.143, # shuttle in-vehicle minutes
            'shuttle_wait':        0.216 # shuttle wait minutes
        }

        # generalized cost coefficients for walk skimming
        self.route_varcoef_walk = {
            'distance':         1.0,
            'bike_exclude':     999.0,
            'thru_centroid':    999.0
        }

        # modes in order of columns in sqlite database trip tables
        self.modes = ['da','s2','s3','wt','dt','wk','bk']

        # maximum cost after which skimming is halted
        self.max_cost_bike = 15.0
        self.max_cost_walk = 10.0

        # ASCs and calibration adjustments
        self.walk_asc = [-0.99,-0.12,-0.38,0.31,
                         0.14,-1.62,-0.05,0.70,
                         -1.036,-1.036,-1.036,-1.036,
                         1.15,0.17,0.18,0.14,
                         0.28]

        for i in range(len(self.walk_asc)):
            self.walk_asc[i] = self.walk_asc[i] + 0.30

        self.bike_asc = [-1.63,-0.76,-1.0,-0.33,
                         -1.18,-2.94,-1.36,-0.62,
                         -1.04,-1.04,-1.04,-1.04,
                         -0.45,-0.46,-0.30,-0.31,
                         -1.43]

        bike_adjust = [-1.75,-1.75,-1.75,-1.75,
                       -1.75,-1.75,-1.75,-1.75,
                       -1.75,-1.75,-1.75,-1.75,
                       -1.75,-1.75,-1.75,-1.75,
                       -1.75]

        for i in range(len(self.bike_asc)):
            self.bike_asc[i] = self.bike_asc[i] + bike_adjust[i]

        # more coefficients
        self.walk_intrazonal = [-0.613,-0.613,-0.613,-0.613,
                                -0.613,-0.613,-0.613,-0.613,
                                -0.613,-0.613,-0.613,-0.613,
                                -0.613,-0.613,-0.613,-0.613,
                                -0.613]

        self.bike_intrazonal = [-1.720,-1.720,-1.720,-1.720,
                                -1.720,-1.720,-1.720,-1.720,
                                -1.720,-1.720,-1.720,-1.720,
                                -1.720,-1.720,-1.720,-1.720,
                                -1.720]

        self.walk_santa_clara = [0.378,0.378,0.378,0.378,
                                 0.378,0.378,0.378,0.378,
                                 0.378,0.378,0.378,0.378,
                                 0.378,0.378,0.378,0.378,
                                 0.378]

        self.walk_santa_clara_intrazonal = [-1.985,-1.985,-1.985,-1.985,
                                            -1.985,-1.985,-1.985,-1.985,
                                            -1.985,-1.985,-1.985,-1.985,
                                            -1.985,-1.985,-1.985,-1.985,
                                            -1.985]

        self.bike_santa_clara = [-0.982,-0.982,-0.982,-0.982,
                                 -0.982,-0.982,-0.982,-0.982,
                                 -0.982,-0.982,-0.982,-0.982,
                                 -0.982,-0.982,-0.982,-0.982,
                                 -0.982]

        self.bike_santa_clara_intrazonal = [0.775,0.775,0.775,0.775,
                                            0.775,0.775,0.775,0.775,
                                            0.775,0.775,0.775,0.775,
                                            0.775,0.775,0.775,0.775,
                                            0.775]

        # county code for Santa Clara in taz table
        self.santa_clara_county_code = 85

        # reduction of motorized utility for UCSC
        self.motorized_ucsc_attr = [-0.0,-0.0,-0.0,-0.0,
                                    -0.0,-0.0,-0.0,-0.0,
                                    -0.0,-0.0,-0.0,-0.0,
                                    -0.0,-0.0,-0.0,-0.0,
                                    -0.0]

        # taz of UCSC
        self.ucsc_taz = 34

class ApplicationConfig():

    def __init__(self):
        """initialize application configuration data, void"""

        # locations of base and build sqlite databases
        self.base_sqlite_file = 'ambag_example/data/example.db'
        self.build_sqlite_file = 'ambag_example/data/example.db'

        # directories for csv inputs
        self.base_data_dir = 'ambag_example/data/'
        self.build_data_dir = 'ambag_example/data/'

        # taz table name and column names
        self.taz_table_name = 'taz'
        self.taz_taz_column = 'taz'
        self.taz_node_column = 'node_id'
        self.taz_county_column = 'co'

        # number of zones
        self.num_zones = 25
        self.max_zone = 1133

        # should tracing be performed in incremental logit, and for which zones and segment
        self.trace = False
        self.trace_ptaz = 1455
        self.trace_ataz = 1493
        self.trace_segment = 'nwk1trip'

        # parameters for emissions estimation
        self.pollutants = ['CO2']
        self.grams_per_mile = [311.0]
        self.grams_per_minute = [79.0]
        self.sr3_avg_occ = 3.25

        # should skims be read from sqlite database or recalculated?
        self.read_base_skims_from_disk = False
        self.read_build_skims_from_disk = False
