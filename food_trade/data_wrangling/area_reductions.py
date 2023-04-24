# Usage: python -m food_trade.data_wrangling.area_reductions

import ee
    
ee.Initialize()

def reduce_region(raster, feature): 
    val = raster.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=feature.geometry(),
        scale=raster.projection().nominalScale()
    )
    return ee.Feature(feature.geometry(), val).copyProperties(feature)

def export_features(features, crop, export_to='drive'):
    
    cols = ['ID', 'iso3', 'country', 'admin', 'admin_name', 'latitude', 'longitude']
    
    if export_to=='drive':
        task = ee.batch.Export.table.toDrive(
            collection=features.select(cols+['b1'], cols+[crop+'_sum']),
            description=crop,
            fileNamePrefix=crop+'_areas'
        )
    else:
        task = ee.batch.Export.table.toAsset(
            collection=features.select(cols+['b1'], cols+[crop+'_sum']),
            description=crop,
            assetId=f'users/shruti_jain90/food/{crop}_areas'
        )
    task.start()
    return task

if __name__ == '__main__':
    
    banana_gaez = ee.Image("users/shruti_jain90/food/GAEZAct2015_HarvArea_Banana_Total")
    maize_gaez = ee.Image("users/shruti_jain90/food/GAEZAct2015_HarvArea_Maize_Total")
    soybean_gaez = ee.Image("users/shruti_jain90/food/GAEZAct2015_HarvArea_Soybean_Total")
    coffee_a_spam = ee.Image("users/shruti_jain90/food/spam2010V2r0_global_H_ACOF_A")
    coffee_r_spam = ee.Image("users/shruti_jain90/food/spam2010V2r0_global_H_RCOF_A")
    cattle = ee.Image("users/shruti_jain90/food/5_Ct_2010_Da")
    admin = ee.FeatureCollection("users/shruti_jain90/admin_polygons_simplified")
    pop = ee.Image("users/shruti_jain90/ppp_2020_1km_Aggregated")
    
    # gaez units : 1000 ha
    # spam units : ha
    # cattle units: number
    # population units: number

    # maize
    maize_gaez = maize_gaez.multiply(1000)
    maize_red = admin.map(lambda feature: reduce_region(maize_gaez, feature))
    export_features(maize_red, 'maize')

    # banana
    banana_gaez = banana_gaez.multiply(1000)
    banana_red = admin.map(lambda feature: reduce_region(banana_gaez, feature))
    export_features(banana_red, 'banana')

    # soybean
    soybean_gaez = soybean_gaez.multiply(1000)
    soybean_red = admin.map(lambda feature: reduce_region(soybean_gaez, feature))
    export_features(soybean_red, 'soybean')

    # cattle
    cattle_red = admin.map(lambda feature: reduce_region(cattle, feature))
    export_features(cattle_red, 'cattle')

    # coffee
    coffee_spam = coffee_a_spam.add(coffee_r_spam)
    coffee_red = admin.map(lambda feature: reduce_region(coffee_spam, feature))
    export_features(coffee_red, 'coffee')

    # population
    pop_red = admin.map(lambda feature: reduce_region(pop, feature))
    export_features(pop_red, 'pop')