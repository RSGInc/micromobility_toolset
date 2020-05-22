"""Run the AMBAG bike model example

This script runs the models in the micromobility_toolset on
a 25-zone example data set.

It instantiates a driver class called 'Scenario' that keeps track of the 'config' and
'data' directories. These directories contain all the necessary instructions and
data to run the models.

  - 'configs' contains settings files for each of the core elements of the
  model. These are .yaml files that provide constants and configuration
  options for the network, skims, landuse data, and trip data.

  - 'data' contains the initial data before any models are run. It
  contains initial trip tables and land use (TAZ) data in either
  CSV or SQLITE format.

This script can be either run with no arguments, which will run the model
list found in settings.yaml:

    python ambag_bike_model.py

Or with the --name argument to run a named model:

    python ambag_bike_model.py --name incremental_demand

The models will generally attempt to load trip/skim data found in the 'build'
and 'base' directories to perform their calculations. If the required data,
cannot be found, the model will create the necessary files with the data it
is provided.
"""


import argparse

from micromobility_toolset import model
from micromobility_toolset.network import preprocessor


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', dest='name', action='store', choices=model.list_steps())
    args = parser.parse_args()

    ambag_base = model.Scenario(
        name='base',
        config='ambag_example/configs',
        data='ambag_example/base')

    ambag_build = model.Scenario(
        name='build',
        config='ambag_example/configs',
        data='ambag_example/build')

    # only use zones affected by network/landuse differences
    model.filter_impact_area(ambag_base, ambag_build)

    if args.name:
        model.run(args.name, ambag_base, ambag_build)

    else:
        # run all steps
        # model.run('initial_demand', ambag_base, ambag_build)
        model.run('incremental_demand', ambag_base, ambag_build)
        model.run('benefits', ambag_base, ambag_build)
        model.run('assign_demand', ambag_base, ambag_build)


@preprocessor()
def preprocess_network(net):
    """add network attributes that are combinations of existing attributes"""

    # add new link attribute columns
    net.add_link_attribute('d0') # distance on ordinary streets, miles
    net.add_link_attribute('d1') # distance on bike paths
    net.add_link_attribute('d2') # distance on bike lanes
    net.add_link_attribute('d3') # distance on bike routes
    net.add_link_attribute('dne1') # distance not on bike paths
    net.add_link_attribute('dne2') # distance not on bike lanes
    net.add_link_attribute('dne3') # distance not on bike routes
    net.add_link_attribute('dw') # distance wrong way
    net.add_link_attribute('auto_permit') # autos permitted
    net.add_link_attribute('bike_exclude') # bikes excluded
    net.add_link_attribute('dloc') # distance on local streets
    net.add_link_attribute('dcol') # distance on collectors
    net.add_link_attribute('dart') # distance on arterials
    net.add_link_attribute('dne3loc') # distance on locals with no bike route
    net.add_link_attribute('dne2art') # distance on arterials with no bike lane

    # loop over links and calculate derived values
    for a in net.adjacency:
        for b in net.adjacency[a]:
            distance = net.get_link_attribute_value((a,b),'distance')
            bike_class = net.get_link_attribute_value((a,b),'bike_class')
            lanes = net.get_link_attribute_value((a,b),'lanes')
            link_type = net.get_link_attribute_value((a,b),'link_type')
            fhwa_fc = net.get_link_attribute_value((a,b),'fhwa_fc')
            net.set_link_attribute_value( (a,b), 'd0', distance * ( bike_class == 0 and lanes > 0 ) )
            net.set_link_attribute_value( (a,b), 'd1', distance * ( bike_class == 1 ) )
            net.set_link_attribute_value( (a,b), 'd2', distance * ( bike_class == 2 ) )
            net.set_link_attribute_value( (a,b), 'd3', distance * ( bike_class == 3 ) )
            net.set_link_attribute_value( (a,b), 'dne1', distance * ( bike_class != 1 ) )
            net.set_link_attribute_value( (a,b), 'dne2', distance * ( bike_class != 2 ) )
            net.set_link_attribute_value( (a,b), 'dne3', distance * ( bike_class != 3 ) )
            net.set_link_attribute_value( (a,b), 'dw', distance * ( bike_class == 0 and lanes == 0 ) )
            net.set_link_attribute_value( (a,b), 'bike_exclude', 1 * ( link_type in ['FREEWAY'] ) )
            net.set_link_attribute_value( (a,b), 'auto_permit', 1 * ( link_type not in ['BIKE','PATH'] ) )
            net.set_link_attribute_value( (a,b), 'dloc', distance * ( fhwa_fc in [19,9] ) )
            net.set_link_attribute_value( (a,b), 'dcol', distance * ( fhwa_fc in [7,8,16,17] ) )
            net.set_link_attribute_value( (a,b), 'dart', distance * ( fhwa_fc in [1,2,6,11,12,14,77] ) )
            net.set_link_attribute_value( (a,b), 'dne3loc', distance * ( fhwa_fc in [19,9] ) * ( bike_class != 3 ) )
            net.set_link_attribute_value( (a,b), 'dne2art', distance * ( fhwa_fc in [1,2,6,11,12,14,77] ) * ( bike_class != 2 ) )

    # add new dual (link-to-link) attribute columns
    net.add_dual_attribute('thru_centroid') # from centroid connector to centroid connector
    net.add_dual_attribute('l_turn') # left turn
    net.add_dual_attribute('u_turn') # u turn
    net.add_dual_attribute('r_turn') # right turn
    net.add_dual_attribute('turn') # turn
    net.add_dual_attribute('thru_intersec') # through a highway intersection
    net.add_dual_attribute('thru_junction') # through a junction

    # loop over pairs of links and set attribute values
    for link1 in net.dual:
        for link2 in net.dual[link1]:

            traversal_type = net.traversal_type(link1,link2,'auto_permit')

            net.set_dual_attribute_value(link1,link2,'thru_centroid', 1 * (traversal_type == 0) )
            net.set_dual_attribute_value(link1,link2,'u_turn', 1 * (traversal_type == 3 ) )
            net.set_dual_attribute_value(link1,link2,'l_turn', 1 * (traversal_type in [5,7,10,13]) )
            net.set_dual_attribute_value(link1,link2,'r_turn', 1 * (traversal_type in [4,6,9,11]) )
            net.set_dual_attribute_value(link1,link2,'turn', 1 * (traversal_type in [3,4,5,6,7,9,10,11,13]) )
            net.set_dual_attribute_value(link1,link2,'thru_intersec', 1 * (traversal_type in [8,12]) )
            net.set_dual_attribute_value(link1,link2,'thru_junction', 1 * (traversal_type == 14) )


if __name__ == '__main__':
    main()
