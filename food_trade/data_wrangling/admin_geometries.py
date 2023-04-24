# Usage: python -m food_trade.data_wrangling.admin_geometries

import os
import pandas as pd
import geopandas as gpd


def count_vertices(row):
    geom = row['geometry']
    multi = geom.type.startswith("Multi")
    if multi:
        n = 0
        # iterate over all parts of multigeometry
        for part in geom:
            n += len(part.exterior.coords)
    else: # if single geometry like point, linestring or polygon
        n = len(geom.exterior.coords)
    return n

if __name__ == '__main__':
    
    if not os.path.exists('../../data/admin_polygons'):
        os.mkdir('../../data/admin_polygons')
              
    admin = gpd.read_file('../../data/admin_polygons.gpkg')
    
    # simplify geometries to make GEE calculations possible
    
    # admin['num_vertices'] = admin.apply(lambda row: count_vertices(row), axis=1)
    # admin.loc[admin['num_vertices']>1000000, 'geometry'] = admin[admin['num_vertices']>1000000]['geometry'].simplify(0.01).buffer(0)
    # admin = admin.drop('num_vertices', axis=1)
    
    admin['geometry'] = admin['geometry'].simplify(0.01).buffer(0)
    
    #convert to shp to upload to GEE
    admin.to_file('../../data/admin_polygons/admin_polygons.shp', driver='ESRI Shapefile')