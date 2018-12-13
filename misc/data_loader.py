import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
from MoviaBusDataset import MoviaBusDataset

soi = MoviaBusDataset.section_of_interest

def load_network(delete_nodes=[], path = '../data/road_network.geojson'):
    road_network = gpd.read_file(path).drop('id', axis = 1)
    forward = road_network.copy().drop(['Oneway', 'MaxSpeedBackward'], axis = 1).rename({'MaxSpeedForward': 'MaxSpeed'}, axis = 1)
    forward['Heading'] = 'Forward'
    backward = road_network[lambda x: x['Oneway'] == 0].copy().drop(['Oneway', 'MaxSpeedForward'], axis = 1).rename({'Source': 'Target', 'Target': 'Source', 'MaxSpeedBackward': 'MaxSpeed'}, axis = 1)
    backward['geometry'] = backward['geometry'].apply(lambda x: LineString(reversed(x.coords)))
    backward['Heading'] = 'Backward'
    road_network = forward.append(
        backward,
        sort = True
    )
    
    road_network['LinkRef'] = road_network.apply(lambda r: '{WayId}:{Source}:{Target}'.format(**r), axis = 1)
    road_network.set_index('LinkRef', inplace = True)
    road_network = road_network[road_network.index.isin(soi)]
    for id_ in delete_nodes:
        del_node_target = road_network.loc[id_]['Target']
        del_node_source = road_network.loc[id_]['Source']
        road_network.loc[road_network['Source']==del_node_target,'Source'] = del_node_source
        road_network.loc[road_network['Target']==del_node_source,'Target'] = del_node_target
        road_network = road_network[road_network.index != id_]
        #print("targets target {}".format(road_network.loc[id_]['Target']['Target']))
        #print("targets source {}".format(road_network.loc[id_]['Target']['Source']))
        #print("b' source {}".format(road_network.loc[id_]['Source']))
        #print("sources {}".format(road_network.loc[id_]['Source']))
    return road_network
#    matched = pd.read_csv('../data/20181001/vehicle-position-matched-online.csv')
#    matched['Time'] = pd.to_datetime(matched['Time'])
#    grp = matched.groupby(['VehicleRef', 'JourneyRef'])
#    data_filter = matched['JourneyRef'].str.extract('(?P<OperatingDayDate>\d{8})L(?P<LineNumber>\d{4})')['LineNumber'].astype(float).isin([1, 4, 8])
#    data_filter &= matched['StopPointRef'].isnull()
#    matched_filter = matched[data_filter].copy()
#    road_network_reduced = road_network.copy()
#    road_network_reduced['Count'] = matched_filter.groupby('LinkRef')['Time'].count()
#    road_network_reduced['Speed_Mean'] = matched_filter.groupby('LinkRef')['Speed'].mean()
#    road_network_reduced = road_network_reduced[road_network_reduced['Count'] > 0]

def adjacency_matrix(road_network):
    nodes = road_network[road_network.index.isin(soi)].index.values
    adjacency_matrix_ = pd.DataFrame(index = nodes, columns = nodes)
    for s in nodes:
        adjacency_matrix_.loc[s,:] = (road_network.loc[s]['Target'] == road_network.loc[nodes]['Source']).astype(int)
    return adjacency_matrix_.values