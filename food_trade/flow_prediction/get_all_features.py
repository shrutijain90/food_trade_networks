# Usage: python -m food_trade.flow_prediction.get_all_features

import geopandas as gpd
import pandas as pd
import numpy as np

# trade data
def get_trade_data():

    # re-exported trade matrix (203 x 203)
    # trade_2017 = pd.read_csv("../../data/FAO_re_export/supply_matrix_cereals_all_2017.csv")
    trade_2017 = pd.read_csv("../../data/FAO_re_export/supply_matrix_Maize (corn)_2017.csv")
    # trade_2017 = pd.read_csv("../../data/FAO_bal_trade_mat/trade_matrix_cereals_all_2017.csv")
    # trade_2017 = pd.read_csv("../../data/FAO_bal_trade_mat/trade_matrix_Maize (corn)_2017.csv")
    
    # trade_2018 = pd.read_csv("../../data/FAO_re_export/supply_matrix_cereals_all_2018.csv")
    trade_2018 = pd.read_csv("../../data/FAO_re_export/supply_matrix_Maize (corn)_2018.csv")
    # trade_2018 = pd.read_csv("../../data/FAO_bal_trade_mat/trade_matrix_cereals_all_2018.csv")
    # trade_2018 = pd.read_csv("../../data/FAO_bal_trade_mat/trade_matrix_Maize (corn)_2018.csv")
    
    # trade_2019 = pd.read_csv("../../data/FAO_re_export/supply_matrix_cereals_all_2019.csv")
    trade_2019 = pd.read_csv("../../data/FAO_re_export/supply_matrix_Maize (corn)_2019.csv")
    # trade_2019 = pd.read_csv("../../data/FAO_bal_trade_mat/trade_matrix_cereals_all_2019.csv")
    # trade_2019 = pd.read_csv("../../data/FAO_bal_trade_mat/trade_matrix_Maize (corn)_2019.csv")
    
    # trade_2020 = pd.read_csv("../../data/FAO_re_export/supply_matrix_cereals_all_2020.csv")
    trade_2020 = pd.read_csv("../../data/FAO_re_export/supply_matrix_Maize (corn)_2020.csv")
    # trade_2020 = pd.read_csv("../../data/FAO_bal_trade_mat/trade_matrix_cereals_all_2020.csv")
    # trade_2020 = pd.read_csv("../../data/FAO_bal_trade_mat/trade_matrix_Maize (corn)_2020.csv")
    
    # trade_2021 = pd.read_csv("../../data/FAO_re_export/supply_matrix_cereals_all_2021.csv")
    trade_2021 = pd.read_csv("../../data/FAO_re_export/supply_matrix_Maize (corn)_2021.csv")
    # trade_2021 = pd.read_csv("../../data/FAO_bal_trade_mat/trade_matrix_cereals_all_2021.csv")
    # trade_2021 = pd.read_csv("../../data/FAO_bal_trade_mat/trade_matrix_Maize (corn)_2021.csv")
    
    trade = pd.concat([trade_2017, trade_2018, trade_2019, trade_2020, trade_2021]).groupby('iso3').mean().reset_index()
    
    # production, consumption, trade [203 FAO countries]
    # trade['production'] = trade.drop('iso3', axis=1).sum(axis=1)
    # prod = trade[['iso3', 'production']]
    # trade = trade.drop('production', axis=1)
    # cons = pd.DataFrame(trade.drop('iso3', axis=1).sum()).reset_index().rename(columns={'index': 'iso3', 0: 'consumption'})
    
    # melt trade data 
    trade = trade.rename(columns={'iso3': 'from_iso3'})
    trade = trade.melt(id_vars=['from_iso3'], value_vars=trade['from_iso3'].values.tolist()).rename(columns={'variable': 'to_iso3', 'value': 'trade'})
    trade = trade[trade['from_iso3']!=trade['to_iso3']]
    
    return trade

# admin reductions
def get_admin_reductions():
    # available for all boundaries, i.e. 240 countries, 3787 admins [only country in FAO data not in boundaries data is TKL but it doesn't have much trade]
    
    def _clean_cols(filepath):
        df_admin = pd.read_csv(filepath).drop(['system:index', '.geo', 'admin', 'admin_name', 'country', 'latitude', 'longitude'], axis=1)
        col = [c for c in df_admin.columns.tolist() if c not in ['iso3', 'ID']][0]
        # agg at country level 
        df_country = df_admin[['iso3', col]].groupby('iso3').sum().reset_index()
        return df_admin, df_country
    
    # crop areas
    barley_a_admin, barley_a_country = _clean_cols('../../data/admin_reductions/barley_areas.csv')
    maize_a_admin, maize_a_country = _clean_cols('../../data/admin_reductions/maize_areas.csv')
    millet_a_admin, millet_a_country = _clean_cols('../../data/admin_reductions/millet_areas.csv')
    rice_a_admin, rice_a_country = _clean_cols('../../data/admin_reductions/rice_areas.csv')
    sorghum_a_admin, sorghum_a_country = _clean_cols('../../data/admin_reductions/sorghum_areas.csv')
    wheat_a_admin, wheat_a_country = _clean_cols('../../data/admin_reductions/wheat_areas.csv')
    other_a_admin, other_a_country = _clean_cols('../../data/admin_reductions/other_cereals_areas.csv')
    
    # crop productions
    barley_p_admin, barley_p_country = _clean_cols('../../data/admin_reductions/barley_production.csv')
    maize_p_admin, maize_p_country = _clean_cols('../../data/admin_reductions/maize_production.csv')
    millet_p_admin, millet_p_country = _clean_cols('../../data/admin_reductions/millet_production.csv')
    rice_p_admin, rice_p_country = _clean_cols('../../data/admin_reductions/rice_production.csv')
    sorghum_p_admin, sorghum_p_country = _clean_cols('../../data/admin_reductions/sorghum_production.csv')
    wheat_p_admin, wheat_p_country = _clean_cols('../../data/admin_reductions/wheat_production.csv')
    other_p_admin, other_p_country = _clean_cols('../../data/admin_reductions/other_cereals_production.csv')
    
    # livestock
    buffaloes_admin, buffaloes_country = _clean_cols('../../data/admin_reductions/buffaloes_totals.csv')
    cattle_admin, cattle_country = _clean_cols('../../data/admin_reductions/cattle_totals.csv')
    chickens_admin, chickens_country = _clean_cols('../../data/admin_reductions/chickens_totals.csv')
    ducks_admin, ducks_country = _clean_cols('../../data/admin_reductions/ducks_totals.csv')
    goats_admin, goats_country = _clean_cols('../../data/admin_reductions/goats_totals.csv')
    horses_admin, horses_country = _clean_cols('../../data/admin_reductions/horses_totals.csv')
    pigs_admin, pigs_country = _clean_cols('../../data/admin_reductions/pigs_totals.csv')
    sheep_admin, sheep_country = _clean_cols('../../data/admin_reductions/sheep_totals.csv')
    
    # population
    pop_admin, pop_country = _clean_cols('../../data/admin_reductions/pop_totals.csv')

    # ghsl
    area_admin, area_country = _clean_cols('../../data/admin_reductions/area_total.csv')
    built_surface_admin, built_surface_country = _clean_cols('../../data/admin_reductions/built_surface.csv')
    built_volume_total_admin, built_volume_total_country = _clean_cols('../../data/admin_reductions/built_volume_total.csv')
    built_volume_nres_admin, built_volume_nres_country = _clean_cols('../../data/admin_reductions/built_volume_nres.csv')
    
    # combine all
    reductions_admin = barley_a_admin.merge(maize_a_admin, how='left').merge(millet_a_admin, how='left').merge(rice_a_admin, how='left').merge(
        sorghum_a_admin, how='left').merge(wheat_a_admin, how='left').merge(other_a_admin, how='left').merge(
        barley_p_admin, how='left').merge(maize_p_admin, how='left').merge(millet_p_admin, how='left').merge(rice_p_admin, how='left').merge(
        sorghum_p_admin, how='left').merge(wheat_p_admin, how='left').merge(other_p_admin, how='left').merge(
        buffaloes_admin, how='left').merge(cattle_admin, how='left').merge(chickens_admin, how='left').merge(ducks_admin, how='left').merge(
        goats_admin, how='left').merge(horses_admin, how='left').merge(pigs_admin, how='left').merge(sheep_admin, how='left').merge(
        pop_admin, how='left').merge(area_admin, how='left').merge(built_surface_admin, how='left').merge(built_volume_total_admin, how='left').merge(
        built_volume_nres_admin, how='left')

    reductions_country = barley_a_country.merge(maize_a_country, how='left').merge(millet_a_country, how='left').merge(rice_a_country, how='left').merge(
        sorghum_a_country, how='left').merge(wheat_a_country, how='left').merge(other_a_country, how='left').merge(
        barley_p_country, how='left').merge(maize_p_country, how='left').merge(millet_p_country, how='left').merge(rice_p_country, how='left').merge(
        sorghum_p_country, how='left').merge(wheat_p_country, how='left').merge(other_p_country, how='left').merge(
        buffaloes_country, how='left').merge(cattle_country, how='left').merge(chickens_country, how='left').merge(ducks_country, how='left').merge(
        goats_country, how='left').merge(horses_country, how='left').merge(pigs_country, how='left').merge(sheep_country, how='left').merge(
        pop_country, how='left').merge(area_country, how='left').merge(built_surface_country, how='left').merge(built_volume_total_country, how='left').merge(
        built_volume_nres_country, how='left')

    return  reductions_admin, reductions_country

# transport data and import tariffs
def get_transport_data(reductions_admin):
    # 216 countries (both from and to), 3619 from admins, 3613 to admins [overall 216 countries, 3619 admins]
    transport_admin = pd.read_parquet('../../data/transport_data/global_lowest_transport_cost.parquet')
    transport_admin = transport_admin[['from_id', 'to_id', 'from_iso3', 'to_iso3', 'freight_USD_t', 'transport_USD_t', 'time_h', 'distance_km',
                           'border_USD_t', 'mode', 'trade_USD_t', 'customs_cost']]

    # getting country level transport data through aggregations weighted by production in exporting region and population+livestock in importing region
    transport_admin = transport_admin.merge(reductions_admin[['ID', 'barley_production', 'maize_production', 'millet_production', 'rice_production',
           'sorghum_production', 'wheat_production', 'other_cereals_production']], left_on='from_id', right_on='ID', how='left').drop('ID', axis=1)
    transport_admin = transport_admin.merge(reductions_admin[['ID', 'buffaloes_totals', 'cattle_totals', 'chickens_totals', 'ducks_totals', 'goats_totals',
           'horses_totals', 'pigs_totals', 'sheep_totals', 'pop_totals']], left_on='to_id', right_on='ID', how='left').drop('ID', axis=1)
    
    transport_admin['pop_livestock'] = transport_admin['pop_totals'] + transport_admin['buffaloes_totals'] + transport_admin['cattle_totals'] \
        + transport_admin['chickens_totals'] + transport_admin['ducks_totals'] + transport_admin['goats_totals'] + transport_admin['horses_totals'] \
        + transport_admin['pigs_totals'] + transport_admin['sheep_totals']
    transport_admin['production'] = transport_admin['barley_production'] + transport_admin['maize_production'] + transport_admin['millet_production'] \
        + transport_admin['rice_production'] + transport_admin['sorghum_production'] + transport_admin['wheat_production'] \
        + transport_admin['other_cereals_production'] 
    transport_admin['weight'] = transport_admin['pop_livestock'] * transport_admin['production']
    
    transport_admin = transport_admin.drop(['production', 'barley_production', 'maize_production', 'millet_production', 
                                'rice_production', 'sorghum_production', 'wheat_production', 'other_cereals_production', 'buffaloes_totals', 
                                'cattle_totals', 'chickens_totals', 'ducks_totals', 'goats_totals',
                                'horses_totals', 'pigs_totals', 'sheep_totals', 'pop_totals', 'pop_livestock'], axis=1)
    
    transport_admin.loc[transport_admin['weight']==0, 'weight'] = transport_admin[transport_admin['weight']>0]['weight'].min()
    transport_admin.loc[transport_admin['customs_cost'].isna(), 'customs_cost'] = 0

    wm = lambda x: np.average(x, weights=transport_admin.loc[x.index, "weight"])
    transport_country = transport_admin.groupby(['from_iso3', 'to_iso3']).agg({'freight_USD_t': wm,
                                                                         'transport_USD_t': wm,
                                                                         'time_h': wm,
                                                                         'distance_km': wm,
                                                                         'border_USD_t': wm,
                                                                         'trade_USD_t': wm,
                                                                         'customs_cost': wm,
                                                                         'mode': pd.Series.mode})

    
    transport_country = transport_country.reset_index()
    transport_country = transport_country[transport_country['from_iso3']!=transport_country['to_iso3']].reset_index(drop=True)
    
    transport_admin = transport_admin.drop('weight', axis=1)

    # import tariffs data
    mm = pd.read_csv('../../data/Import_tariffs/mm2019.csv', sep=';')
    mm = mm[(mm['hs6_rev2017']>=100000) & (mm['hs6_rev2017']<110000)].reset_index(drop=True)
    mm = mm.groupby(['importer', 'exporter']).mean()[['Pref_Applied_AVE']].reset_index()
    mm_country = pd.read_csv('../../data/Import_tariffs/Countries.csv')
    mm = mm.merge(mm_country, right_on='code', left_on='importer').drop('code', axis=1).rename(columns={'iso3': 'to_iso3'})
    mm = mm.merge(mm_country, right_on='code', left_on='exporter').drop('code', axis=1).rename(columns={'iso3': 'from_iso3'})
    mm = mm.drop(['importer', 'exporter'], axis=1)

    # combining transport and tariffs data (replacing nans with zeros)
    transport_admin = transport_admin.merge(mm, how='left')
    transport_admin.loc[transport_admin['Pref_Applied_AVE'].isna(), 'Pref_Applied_AVE'] = 0
    transport_admin['Pref_Applied_AVE'] = transport_admin['Pref_Applied_AVE'] * 0.8 * (transport_admin['transport_USD_t'] + transport_admin['border_USD_t'])
    
    transport_country = transport_country.merge(mm, how='left')
    transport_country.loc[transport_country['Pref_Applied_AVE'].isna(), 'Pref_Applied_AVE'] = 0
    transport_country['Pref_Applied_AVE'] = transport_country['Pref_Applied_AVE'] * 0.8 * (transport_country['transport_USD_t'] + transport_country['border_USD_t'])
    
    return transport_admin, transport_country

# merge data to get master dataframe
def merge_data(log_x=True, log_y=True, trade_thresh=10, get_trade_only=False):
    # trade data, replacing values less than 10 with zero
    trade = get_trade_data()
    trade.loc[trade['trade']<trade_thresh, 'trade'] = 0
    trade['trade_exists'] = 0
    trade.loc[trade['trade']>0, 'trade_exists'] = 1
    # log trade for where it exists (for volume prediction)
    if log_y:
        trade.loc[trade['trade_exists']==1, 'trade'] = trade[trade['trade_exists']==1].apply(lambda row: np.log(row['trade']), axis=1)
    print('trade data done')

    if get_trade_only:
        return trade
    
    # x variables (admin reductions and transport)
    reductions_admin, reductions_country = get_admin_reductions()
    print('admin reductions done')
    transport_admin, transport_country = get_transport_data(reductions_admin)
    print('transport data done')

    # log x variables for both transport and admin reductions
    if log_x:
        for col in ['freight_USD_t', 'transport_USD_t', 'time_h', 'distance_km', 'border_USD_t', 'customs_cost', 'Pref_Applied_AVE']:
            print(f'log {col}')
            transport_admin[col] = transport_admin.apply(lambda row: np.log(row[col]+1), axis=1)
            transport_country[col] = transport_country.apply(lambda row: np.log(row[col]+1), axis=1)

        for col in ['barley_areas', 'maize_areas', 'millet_areas', 'rice_areas', 'sorghum_areas', 'wheat_areas', 'other_cereals_areas', 
                    'barley_production', 'maize_production', 'millet_production', 'rice_production', 'sorghum_production', 'wheat_production', 'other_cereals_production', 
                    'buffaloes_totals', 'cattle_totals', 'chickens_totals', 'ducks_totals', 'goats_totals', 'horses_totals', 'pigs_totals', 'sheep_totals', 'pop_totals',
                    'area_total', 'built_surface', 'built_volume_total', 'built_volume_nres']:
            print(f'log {col}')
            reductions_admin[col] = reductions_admin.apply(lambda row: np.log(row[col]+1), axis=1)
            reductions_country[col] = reductions_country.apply(lambda row: np.log(row[col]+1), axis=1)

    # categorical variable - mode
    transport_country.loc[~transport_country['mode'].isin(['maritime', 'rail', 'road']), 'mode'] = transport_country[
        ~transport_country['mode'].isin(['maritime', 'rail', 'road'])]['mode'].str[0]
    transport_country.loc[transport_country['mode']=='maritime', 'mode'] = 0
    transport_country.loc[transport_country['mode']=='rail', 'mode'] = 1
    transport_country.loc[transport_country['mode']=='road', 'mode'] = 2
    transport_country = transport_country.drop('trade_USD_t', axis=1)
    
    transport_admin.loc[transport_admin['mode']=='maritime', 'mode'] = 0
    transport_admin.loc[transport_admin['mode']=='rail', 'mode'] = 1
    transport_admin.loc[transport_admin['mode']=='road', 'mode'] = 2
    transport_admin = transport_admin.drop('trade_USD_t', axis=1)

    # getting country level data by merging everything (trade, transport, admin reductions)
    df_country = transport_country.merge(trade)
    df_country = df_country.merge(reductions_country, left_on='from_iso3', right_on='iso3').drop(['iso3'], axis=1).rename(columns={
        'barley_areas': 'from_barley_area', 
        'maize_areas': 'from_maize_area', 
        'millet_areas': 'from_millet_area',
        'rice_areas': 'from_rice_area', 
        'sorghum_areas': 'from_sorghum_area', 
        'wheat_areas': 'from_wheat_area', 
        'other_cereals_areas': 'from_other_cereals_area',
        'barley_production': 'from_barley_production', 
        'maize_production': 'from_maize_production', 
        'millet_production': 'from_millet_production',
        'rice_production': 'from_rice_production', 
        'sorghum_production': 'from_sorghum_production', 
        'wheat_production': 'from_wheat_production', 
        'other_cereals_production': 'from_other_cereals_production',
        'buffaloes_totals': 'from_buffaloes', 
        'cattle_totals': 'from_cattle', 
        'chickens_totals': 'from_chickens',
        'ducks_totals': 'from_ducks', 
        'goats_totals': 'from_goats', 
        'horses_totals': 'from_horses', 
        'pigs_totals': 'from_pigs',
        'sheep_totals': 'from_sheep', 
        'pop_totals': 'from_pop',
        'area_total': 'from_area', 
        'built_surface': 'from_built_surface', 
        'built_volume_total': 'from_built_volume_total', 
        'built_volume_nres': 'from_built_volume_nres'})
    
    df_country = df_country.merge(reductions_country, left_on='to_iso3', right_on='iso3').drop(['iso3'], axis=1).rename(columns={
        'barley_areas': 'to_barley_area', 
        'maize_areas': 'to_maize_area', 
        'millet_areas': 'to_millet_area',
        'rice_areas': 'to_rice_area', 
        'sorghum_areas': 'to_sorghum_area', 
        'wheat_areas': 'to_wheat_area', 
        'other_cereals_areas': 'to_other_cereals_area',
        'barley_production': 'to_barley_production', 
        'maize_production': 'to_maize_production', 
        'millet_production': 'to_millet_production',
        'rice_production': 'to_rice_production', 
        'sorghum_production': 'to_sorghum_production', 
        'wheat_production': 'to_wheat_production', 
        'other_cereals_production': 'to_other_cereals_production', 
        'buffaloes_totals': 'to_buffaloes', 
        'cattle_totals': 'to_cattle', 
        'chickens_totals': 'to_chickens',
        'ducks_totals': 'to_ducks', 
        'goats_totals': 'to_goats', 
        'horses_totals': 'to_horses', 
        'pigs_totals': 'to_pigs',
        'sheep_totals': 'to_sheep', 
        'pop_totals': 'to_pop',
        'area_total': 'to_area', 
        'built_surface': 'to_built_surface', 
        'built_volume_total': 'to_built_volume_total', 
        'built_volume_nres': 'to_built_volume_nres'})
    # remove these 5 area codes ('GLP', 'GUF', 'MTQ', 'REU', 'TUV') from data
    # [wouldn't make a difference, since they all have 0 production and consumption]
    df_country = df_country[(~df_country['from_iso3'].isin(['GLP', 'GUF', 'MTQ', 'REU', 'TUV'])) 
    & (~df_country['to_iso3'].isin(['GLP', 'GUF', 'MTQ', 'REU', 'TUV']))].reset_index(drop=True)
    print('df_country done')

    # list of countries in data (will keep only these in the admin level data)
    iso3_codes = list(set(df_country['from_iso3'].values.tolist() + df_country['to_iso3'].values.tolist()))

    # getting admin level data by merging everything (transport and admin reductions)
    df_admin = transport_admin.merge(reductions_admin, left_on='from_id', right_on='ID').drop(['ID', 'iso3'], axis=1).rename(columns={
        'barley_areas': 'from_barley_area', 
        'maize_areas': 'from_maize_area', 
        'millet_areas': 'from_millet_area',
        'rice_areas': 'from_rice_area', 
        'sorghum_areas': 'from_sorghum_area', 
        'wheat_areas': 'from_wheat_area', 
        'other_cereals_areas': 'from_other_cereals_area',
        'barley_production': 'from_barley_production', 
        'maize_production': 'from_maize_production', 
        'millet_production': 'from_millet_production',
        'rice_production': 'from_rice_production', 
        'sorghum_production': 'from_sorghum_production', 
        'wheat_production': 'from_wheat_production', 
        'other_cereals_production': 'from_other_cereals_production',
        'buffaloes_totals': 'from_buffaloes', 
        'cattle_totals': 'from_cattle', 
        'chickens_totals': 'from_chickens',
        'ducks_totals': 'from_ducks', 
        'goats_totals': 'from_goats', 
        'horses_totals': 'from_horses', 
        'pigs_totals': 'from_pigs',
        'sheep_totals': 'from_sheep', 
        'pop_totals': 'from_pop',
        'area_total': 'from_area', 
        'built_surface': 'from_built_surface', 
        'built_volume_total': 'from_built_volume_total', 
        'built_volume_nres': 'from_built_volume_nres'})
    
    df_admin = df_admin.merge(reductions_admin, left_on='to_id', right_on='ID').drop(['ID', 'iso3'], axis=1).rename(columns={
        'barley_areas': 'to_barley_area', 
        'maize_areas': 'to_maize_area', 
        'millet_areas': 'to_millet_area',
        'rice_areas': 'to_rice_area', 
        'sorghum_areas': 'to_sorghum_area', 
        'wheat_areas': 'to_wheat_area', 
        'other_cereals_areas': 'to_other_cereals_area',
        'barley_production': 'to_barley_production', 
        'maize_production': 'to_maize_production', 
        'millet_production': 'to_millet_production',
        'rice_production': 'to_rice_production', 
        'sorghum_production': 'to_sorghum_production', 
        'wheat_production': 'to_wheat_production', 
        'other_cereals_production': 'to_other_cereals_production', 
        'buffaloes_totals': 'to_buffaloes', 
        'cattle_totals': 'to_cattle', 
        'chickens_totals': 'to_chickens',
        'ducks_totals': 'to_ducks', 
        'goats_totals': 'to_goats', 
        'horses_totals': 'to_horses', 
        'pigs_totals': 'to_pigs',
        'sheep_totals': 'to_sheep', 
        'pop_totals': 'to_pop',
        'area_total': 'to_area', 
        'built_surface': 'to_built_surface', 
        'built_volume_total': 'to_built_volume_total', 
        'built_volume_nres': 'to_built_volume_nres'})

    df_admin = df_admin[(df_admin['from_iso3'].isin(iso3_codes)) & (df_admin['to_iso3'].isin(iso3_codes))].reset_index(drop=True)
    print('df_admin done')
    
    return df_country, df_admin

if __name__ == '__main__':
    
    # # thresh 0.001
    # df_country, df_admin = merge_data(log_x=False, log_y=False, trade_thresh=0.001)
    # df_country.to_parquet('../../data/combined_features/using_supply_mat/df_country_log_false_thresh_pt001.parquet.gzip', index=False, compression='gzip')
    # df_admin.to_parquet('../../data/combined_features/df_admin_log_false.parquet.gzip', index=False, compression='gzip')
    
    # df_country, df_admin = merge_data(log_x=True, log_y=True, trade_thresh=0.001)
    # df_country.to_parquet('../../data/combined_features/using_supply_mat/df_country_log_true_thresh_pt001.parquet.gzip', index=False, compression='gzip')
    # df_admin.to_parquet('../../data/combined_features/df_admin_log_true.parquet.gzip', index=False, compression='gzip')
    
    # # thresh 1
    # df_country_trade = merge_data(log_x=False, log_y=False, trade_thresh=1, get_trade_only=True)
    # df_country_feat = pd.read_parquet('../../data/combined_features/using_supply_mat/df_country_log_false_thresh_pt001.parquet.gzip').drop(
    #     ['trade_exists', 'trade'], axis=1)
    # df_country = df_country_feat.merge(df_country_trade)
    # df_country.to_parquet('../../data/combined_features/using_supply_mat/df_country_log_false_thresh_1.parquet.gzip', index=False, compression='gzip')
    
    # df_country_trade = merge_data(log_x=True, log_y=True, trade_thresh=1, get_trade_only=True)
    # df_country_feat = pd.read_parquet('../../data/combined_features/using_supply_mat/df_country_log_true_thresh_pt001.parquet.gzip').drop(
    #     ['trade_exists', 'trade'], axis=1)
    # df_country = df_country_feat.merge(df_country_trade)
    # df_country.to_parquet('../../data/combined_features/using_supply_mat/df_country_log_true_thresh_1.parquet.gzip', index=False, compression='gzip')

    # # thresh 10
    # df_country_trade = merge_data(log_x=False, log_y=False, trade_thresh=10, get_trade_only=True)
    # df_country_feat = pd.read_parquet('../../data/combined_features/using_supply_mat/df_country_log_false_thresh_pt001.parquet.gzip').drop(
    #     ['trade_exists', 'trade'], axis=1)
    # df_country = df_country_feat.merge(df_country_trade)
    # df_country.to_parquet('../../data/combined_features/using_supply_mat/df_country_log_false_thresh_10.parquet.gzip', index=False, compression='gzip')
    
    # df_country_trade = merge_data(log_x=True, log_y=True, trade_thresh=10, get_trade_only=True)
    # df_country_feat = pd.read_parquet('../../data/combined_features/using_supply_mat/df_country_log_true_thresh_pt001.parquet.gzip').drop(
    #     ['trade_exists', 'trade'], axis=1)
    # df_country = df_country_feat.merge(df_country_trade)
    # df_country.to_parquet('../../data/combined_features/using_supply_mat/df_country_log_true_thresh_10.parquet.gzip', index=False, compression='gzip')
    

    for thresh in [0.001, 1, 10]:
        if thresh==0.001:
            strng = 'pt001'
        else:
            strng = str(thresh)
        for lg in [False, True]:
            df_country_trade = merge_data(log_x=lg, log_y=lg, trade_thresh=thresh, get_trade_only=True)
            df_country_feat = pd.read_parquet(
                f'../../data/combined_features/using_supply_mat/df_country_log_{str(lg).lower()}_thresh_pt001.parquet.gzip').drop(
                ['trade_exists', 'trade'], axis=1)
            df_country = df_country_feat.merge(df_country_trade)
            fname = f'../../data/combined_features/maize_using_supply_mat/df_country_log_{str(lg).lower()}_thresh_{strng}.parquet.gzip'
            df_country.to_parquet(fname, index=False, compression='gzip')
            