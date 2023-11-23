# Usage: python -m food_trade.data_wrangling.area_reductions

import ee
    
ee.Initialize()

def reduce_region(raster, feature, reducer=ee.Reducer.sum()): 
    val = raster.reduceRegion(
        reducer=reducer,
        geometry=feature.geometry(),
        scale=raster.projection().nominalScale(),
        maxPixels=1e9
    )
    return ee.Feature(feature.geometry(), val).copyProperties(feature)

def export_features(features, item, export_to='drive'):
    
    cols = ['ID', 'iso3', 'country', 'admin', 'admin_name', 'latitude', 'longitude']
    
    if export_to=='drive':
        task = ee.batch.Export.table.toDrive(
            collection=features.select(cols+['b1'], cols+[item]),
            description=item,
            fileNamePrefix=item
        )
    else:
        task = ee.batch.Export.table.toAsset(
            collection=features.select(cols+['b1'], cols+[item]),
            description=item,
            assetId=f'users/shruti_jain90/food/admin_reductions/{item}'
        )
    task.start()
    return task

if __name__ == '__main__':
    
    admin = ee.FeatureCollection("users/shruti_jain90/admin_polygons_simplified")
    
    #gaez areas
    banana_area_gaez = ee.Image("users/shruti_jain90/food/gaez_2015/area/GAEZAct2015_HarvArea_Banana_Total")
    soybean_area_gaez = ee.Image("users/shruti_jain90/food/gaez_2015/area/GAEZAct2015_HarvArea_Soybean_Total")
    barley_area_gaez = ee.Image("users/shruti_jain90/food/gaez_2015/area/GAEZAct2015_HarvArea_Barley_Total")
    maize_area_gaez = ee.Image("users/shruti_jain90/food/gaez_2015/area/GAEZAct2015_HarvArea_Maize_Total")
    millet_area_gaez = ee.Image("users/shruti_jain90/food/gaez_2015/area/GAEZAct2015_HarvArea_Millet_Total")
    rice_area_gaez = ee.Image("users/shruti_jain90/food/gaez_2015/area/GAEZAct2015_HarvArea_Rice_Total")
    sorghum_area_gaez = ee.Image("users/shruti_jain90/food/gaez_2015/area/GAEZAct2015_HarvArea_Sorghum_Total")
    wheat_area_gaez = ee.Image("users/shruti_jain90/food/gaez_2015/area/GAEZAct2015_HarvArea_Wheat_Total")
    other_cereals_area_gaez = ee.Image("users/shruti_jain90/food/gaez_2015/area/GAEZAct2015_HarvArea_Othercereals_Total")
    
    #gaez production
    barley_production_gaez = ee.Image("users/shruti_jain90/food/gaez_2015/production/GAEZAct2015_Production_Barley_Total")
    maize_production_gaez = ee.Image("users/shruti_jain90/food/gaez_2015/production/GAEZAct2015_Production_Maize_Total")
    millet_production_gaez = ee.Image("users/shruti_jain90/food/gaez_2015/production/GAEZAct2015_Production_Millet_Total")
    rice_production_gaez = ee.Image("users/shruti_jain90/food/gaez_2015/production/GAEZAct2015_Production_Rice_Total")
    sorghum_production_gaez = ee.Image("users/shruti_jain90/food/gaez_2015/production/GAEZAct2015_Production_Sorghum_Total")
    wheat_production_gaez = ee.Image("users/shruti_jain90/food/gaez_2015/production/GAEZAct2015_Production_Wheat_Total")
    other_cereals_production_gaez = ee.Image("users/shruti_jain90/food/gaez_2015/production/GAEZAct2015_Production_Othercereals_Total")
    
    # spam areas
    coffee_a_area_spam = ee.Image("users/shruti_jain90/food/mapspam_2010/spam2010V2r0_global_H_ACOF_A")
    coffee_r_area_spam = ee.Image("users/shruti_jain90/food/mapspam_2010/spam2010V2r0_global_H_RCOF_A")
    
    # livestock
    buffaloes = ee.Image("users/shruti_jain90/food/livestock_2010/5_Bf_2010_Da")
    cattle = ee.Image("users/shruti_jain90/food/livestock_2010/5_Ct_2010_Da")
    chickens = ee.Image("users/shruti_jain90/food/livestock_2010/5_Ch_2010_Da")
    ducks = ee.Image("users/shruti_jain90/food/livestock_2010/5_Dk_2010_Da")
    goats = ee.Image("users/shruti_jain90/food/livestock_2010/5_Gt_2010_Da")
    horses = ee.Image("users/shruti_jain90/food/livestock_2010/5_Ho_2010_Da")
    pigs = ee.Image("users/shruti_jain90/food/livestock_2010/5_Pg_2010_Da")
    sheep = ee.Image("users/shruti_jain90/food/livestock_2010/5_Sh_2010_Da")
    
    # pop indices
    pop = ee.Image("users/shruti_jain90/ppp_2020_1km_Aggregated")

    # ghsl variables
    ghsl_v = ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_V").filter(ee.Filter.eq('system:index','2020')).first()
    ghsl_s = ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S").filter(ee.Filter.eq('system:index','2020')).first()
    
    # gaez area units : 1000 ha
    # gaez production units: 1000 T
    # spam area units : ha
    # livestock units: number
    # population units: number
    # ghsl volum: meter cube
    # ghsl area: meter square
    
    # ## areas
    
    # # banana
    # banana_area_gaez = banana_area_gaez.multiply(1000)
    # banana_area_red = admin.map(lambda feature: reduce_region(banana_area_gaez, feature))
    # export_features(banana_area_red, 'banana_areas')

    # # soybean
    # soybean_area_gaez = soybean_area_gaez.multiply(1000)
    # soybean_area_red = admin.map(lambda feature: reduce_region(soybean_area_gaez, feature))
    # export_features(soybean_area_red, 'soybean_areas')
    
    # # coffee
    # coffee_area_spam = coffee_a_area_spam.add(coffee_r_area_spam)
    # coffee_area_red = admin.map(lambda feature: reduce_region(coffee_area_spam, feature))
    # export_features(coffee_area_red, 'coffee_areas')
    
    # # barley
    # barley_area_gaez = barley_area_gaez.multiply(1000)
    # barley_area_red = admin.map(lambda feature: reduce_region(barley_area_gaez, feature))
    # export_features(barley_area_red, 'barley_areas')
    
    # # maize
    # maize_area_gaez = maize_area_gaez.multiply(1000)
    # maize_area_red = admin.map(lambda feature: reduce_region(maize_area_gaez, feature))
    # export_features(maize_area_red, 'maize_areas')
    
    # # millet
    # millet_area_gaez = millet_area_gaez.multiply(1000)
    # millet_area_red = admin.map(lambda feature: reduce_region(millet_area_gaez, feature))
    # export_features(millet_area_red, 'millet_areas')
    
    # # rice
    # rice_area_gaez = rice_area_gaez.multiply(1000)
    # rice_area_red = admin.map(lambda feature: reduce_region(rice_area_gaez, feature))
    # export_features(rice_area_red, 'rice_areas')
    
    # # sorghum
    # sorghum_area_gaez = sorghum_area_gaez.multiply(1000)
    # sorghum_area_red = admin.map(lambda feature: reduce_region(sorghum_area_gaez, feature))
    # export_features(sorghum_area_red, 'sorghum_areas')
    
    # # wheat
    # wheat_area_gaez = wheat_area_gaez.multiply(1000)
    # wheat_area_red = admin.map(lambda feature: reduce_region(wheat_area_gaez, feature))
    # export_features(wheat_area_red, 'wheat_areas')
    
    # # other cereals
    # other_cereals_area_gaez = other_cereals_area_gaez.multiply(1000)
    # other_cereals_area_red = admin.map(lambda feature: reduce_region(other_cereals_area_gaez, feature))
    # export_features(other_cereals_area_red, 'other_cereals_areas')
    
    # ## production
    
    # # barley
    # barley_production_gaez = barley_production_gaez.multiply(1000)
    # barley_production_red = admin.map(lambda feature: reduce_region(barley_production_gaez, feature))
    # export_features(barley_production_red, 'barley_production')
    
    # # maize
    # maize_production_gaez = maize_production_gaez.multiply(1000)
    # maize_production_red = admin.map(lambda feature: reduce_region(maize_production_gaez, feature))
    # export_features(maize_production_red, 'maize_production')
    
    # # millet
    # millet_production_gaez = millet_production_gaez.multiply(1000)
    # millet_production_red = admin.map(lambda feature: reduce_region(millet_production_gaez, feature))
    # export_features(millet_production_red, 'millet_production')
    
    # # rice
    # rice_production_gaez = rice_production_gaez.multiply(1000)
    # rice_production_red = admin.map(lambda feature: reduce_region(rice_production_gaez, feature))
    # export_features(rice_production_red, 'rice_production')
    
    # # sorghum
    # sorghum_production_gaez = sorghum_production_gaez.multiply(1000)
    # sorghum_production_red = admin.map(lambda feature: reduce_region(sorghum_production_gaez, feature))
    # export_features(sorghum_production_red, 'sorghum_production')
    
    # # wheat
    # wheat_production_gaez = wheat_production_gaez.multiply(1000)
    # wheat_production_red = admin.map(lambda feature: reduce_region(wheat_production_gaez, feature))
    # export_features(wheat_production_red, 'wheat_production')
    
    # # other cereals
    # other_cereals_production_gaez = other_cereals_production_gaez.multiply(1000)
    # other_cereals_production_red = admin.map(lambda feature: reduce_region(other_cereals_production_gaez, feature))
    # export_features(other_cereals_production_red, 'other_cereals_production')

    # ## counts
    # # buffaloes
    # buffaloes_red = admin.map(lambda feature: reduce_region(buffaloes, feature))
    # export_features(buffaloes_red, 'buffaloes_totals')
    
    # # cattle
    # cattle_red = admin.map(lambda feature: reduce_region(cattle, feature))
    # export_features(cattle_red, 'cattle_totals')
    
    # # chickens
    # chickens_red = admin.map(lambda feature: reduce_region(chickens, feature))
    # export_features(chickens_red, 'chickens_totals')
    
    # # ducks
    # ducks_red = admin.map(lambda feature: reduce_region(ducks, feature))
    # export_features(ducks_red, 'ducks_totals')
    
    # # goats
    # goats_red = admin.map(lambda feature: reduce_region(goats, feature))
    # export_features(goats_red, 'goats_totals')
    
    # # horses
    # horses_red = admin.map(lambda feature: reduce_region(horses, feature))
    # export_features(horses_red, 'horses_totals')
    
    # # pigs
    # pigs_red = admin.map(lambda feature: reduce_region(pigs, feature))
    # export_features(pigs_red, 'pigs_totals')
    
    # # sheep
    # sheep_red = admin.map(lambda feature: reduce_region(sheep, feature))
    # export_features(sheep_red, 'sheep_totals')

    # # population
    # pop_red = admin.map(lambda feature: reduce_region(pop, feature))
    # export_features(pop_red, 'pop_totals')
    
    ## ghsl
    # volume_total
    ghsl_v = ghsl_v.reproject(crs='EPSG:4326', scale=ghsl_v.projection().nominalScale())
    ghsl_v_red = admin.map(lambda feature: reduce_region(
        ee.Image(ghsl_v.select('built_volume_total').updateMask(ghsl_v.select('built_volume_total').gt(0))).select(['built_volume_total'], ['b1']), 
        feature
    ))
    export_features(ghsl_v_red, 'built_volume_total')

    # volume_nres
    ghsl_v_red = admin.map(lambda feature: reduce_region(
        ee.Image(ghsl_v.select('built_volume_nres').updateMask(ghsl_v.select('built_volume_nres').gt(0))).select(['built_volume_nres'], ['b1']), 
        feature
    ))
    export_features(ghsl_v_red, 'built_volume_nres')

    # surface area
    ghsl_s = ghsl_s.reproject(crs='EPSG:4326', scale=ghsl_s.projection().nominalScale())
    ghsl_s_red = admin.map(lambda feature: reduce_region(
        ee.Image(ghsl_s.updateMask(ghsl_s.gt(0))).select(['built_surface'], ['b1']), 
        feature
    ))
    export_features(ghsl_s_red, 'built_surface')

    # # total area
    # area = admin.map(lambda feature: feature.set('b1', feature.area()))
    # export_features(area, 'area_total')




    