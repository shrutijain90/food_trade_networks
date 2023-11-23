# Usage: python -m food_trade.demand_supply.balance_trade

import geopandas as gpd
import pandas as pd
import numpy as np

data_dir_prefix = '../../data/'

def get_area_codes():
    
    # preparing area codes (203 in total, although maybe only 198 have data for the later years)
    area_codes = pd.read_csv(f'{data_dir_prefix}countries_codes_and_coordinates.csv')
    area_codes['Alpha-3 code'] = area_codes.apply(lambda row: row['Alpha-3 code'].split('"')[1], axis=1)
    area_codes['Numeric code'] = area_codes.apply(lambda row: int(row['Numeric code'].split('"')[1]), axis=1)
    area_codes = area_codes[['Alpha-3 code', 'Numeric code']].rename(
        columns={'Alpha-3 code': 'iso3', 'Numeric code': 'Area Code (M49)'})
    
    admin = gpd.read_file(f'{data_dir_prefix}admin_polygons.gpkg') # not working due to inconsistencies in environment, hence using the areas file in next line
    area_codes = area_codes.merge(admin[['iso3','country']].drop_duplicates(), how='outer', indicator=True)
    area_codes.loc[area_codes['iso3']=='TKL', '_merge'] = 'both' # because Tokelau is missing from admin areas but is present in FAO data
    area_codes = area_codes[area_codes['_merge']=='both'].reset_index(drop=True)
    area_codes = area_codes.drop('_merge', axis=1)
    area_codes.loc[area_codes['iso3']=='TKL', 'country'] = 'Tokelau'
    
    prod_area_codes = pd.read_csv(f'{data_dir_prefix}FAOSTAT_A-S_E/Production_Crops_Livestock_E_All_Data_(Normalized)/Production_Crops_Livestock_E_AreaCodes.csv',
                                  encoding='latin1')
    FAO_area_codes = prod_area_codes.rename(columns={'M49 Code' : 'Area Code (M49)'})
    FAO_area_codes['Area Code (M49)'] = FAO_area_codes.apply(lambda row: int(row['Area Code (M49)'][1:]), axis=1)
    FAO_area_codes = FAO_area_codes.merge(area_codes)
    FAO_area_codes = FAO_area_codes[['Area Code (M49)', 'country', 'iso3']]
    FAO_area_codes = FAO_area_codes.sort_values(by='iso3')
    
    return FAO_area_codes

def get_prod_matrix(item, year, FAO_area_codes):
    
    # preparing production matrix
    prod = pd.read_csv(f'{data_dir_prefix}FAOSTAT_A-S_E/Production_Crops_Livestock_E_All_Data_(Normalized)/Production_Crops_Livestock_E_All_Data_(Normalized).csv',
                       encoding='latin1')
    prod = prod[['Area Code (M49)', 'Area', 'Item', 'Element', 'Year', 'Unit', 'Value']]
    prod = prod[(prod['Year']==year) & (prod['Item']==item) & (prod['Element']=='Production')].reset_index(drop=True)
    prod = prod.rename(columns={'Value': 'Production'})
    prod = prod.drop('Element', axis=1)
    
    prod['Area Code (M49)'] = prod.apply(lambda row: int(row['Area Code (M49)'][1:]), axis=1)
    prod = prod.merge(FAO_area_codes, how='right')
    prod = prod[['iso3', 'Area Code (M49)', 'country', 'Production']].sort_values(by='iso3')
    prod = prod.fillna(0)
    P = prod[['Production']].to_numpy()
    
    return P

def get_trade_matrix(item, year, FAO_area_codes):
        
    # preparing trade matrix
    mat = pd.read_csv(f'{data_dir_prefix}FAOSTAT_T-Z_E/Trade_DetailedTradeMatrix_E_All_Data_(Normalized)/Trade_DetailedTradeMatrix_E_All_Data_(Normalized).csv',
                      encoding='latin1')
    mat = mat[(mat['Year']==year) & (mat['Item']==item) & (mat['Unit']=='tonnes')][['Reporter Country Code (M49)', 'Partner Country Code (M49)',
                                                                                    'Reporter Countries', 'Partner Countries', 'Item', 'Element', 
                                                                                    'Year', 'Unit', 'Value']].reset_index(drop=True)
    
    # reliability index from "Reconciling bilateral trade data for use in GTAP"
    # accuracy level for each import 
    imports = mat[mat['Element']=='Import Quantity']
    imports = imports.rename(columns={'Reporter Countries': 'Country A', 'Partner Countries': 'Country B',
                                      'Reporter Country Code (M49)': 'Country A Code', 'Partner Country Code (M49)': 'Country B Code'})
    exports = mat[mat['Element']=='Export Quantity']
    exports = exports.rename(columns={'Reporter Countries': 'Country B', 'Partner Countries': 'Country A',
                                      'Reporter Country Code (M49)': 'Country B Code', 'Partner Country Code (M49)': 'Country A Code'})

    df = pd.concat([imports, exports], axis=0, ignore_index=True)
    df = pd.pivot(df, index=['Country A', 'Country B', 'Country A Code', 'Country B Code', 'Item', 'Year', 'Unit'], 
                  columns = 'Element',values = 'Value').reset_index()
    df = df.fillna(0)
    df = df.rename(columns={'Import Quantity': 'Import rep A', 'Export Quantity': 'Export rep B',})
    df = df[(df['Import rep A']!=0) | (df['Export rep B']!=0)].reset_index(drop=True)
    df['AL'] = 2*(df['Import rep A'] - df['Export rep B']).abs()/(df['Import rep A'] + df['Export rep B'])
    row_cond = (df['AL']==2) & (df['Import rep A']<10) & (df['Export rep B']<10)
    df.loc[row_cond, 'AL'] = -1
    
    def _calc_rel_index(g, col, ind_col):
        d = g.copy()
        d = d[d['AL']!=-1]
        # d['WAL'] = d[col] / d[col].sum() * d['AL']
        # d = d[d['WAL']!=d['WAL'].max()]
        RI = d[d['AL']<=0.2][col].sum() / d[col].sum()
        g[ind_col] = RI 
        return g
    
    df = df.groupby('Country A').apply(lambda g: _calc_rel_index(g, 'Import rep A', 'RIM')).reset_index(drop=True)
    df = df.groupby('Country B').apply(lambda g: _calc_rel_index(g, 'Export rep B', 'RIX')).reset_index(drop=True)
    df = df.fillna(0)
    
    def _select_qty(row):
        if row['RIM'] >= row['RIX']:
            return row['Import rep A']
        else:
            return row['Export rep B']
        
    df['From B to A'] = df.apply(lambda row: _select_qty(row), axis=1)
    df['Country A Code'] = df.apply(lambda row: int(row['Country A Code'][1:]), axis=1)
    df['Country B Code'] = df.apply(lambda row: int(row['Country B Code'][1:]), axis=1)
    
    trade_mat = df[['Country A', 'Country B', 'Country A Code', 'Country B Code', 'Item', 'Year',
                    'Unit', 'From B to A']]
    
    trade_mat = trade_mat.merge(FAO_area_codes, left_on='Country A Code', right_on='Area Code (M49)', how='right')
    trade_mat = trade_mat.drop(['Country A', 'Country A Code'], axis=1)
    trade_mat = trade_mat.rename(columns={'Area Code (M49)': 'Country A M49', 'country': 'Country A',
                                          'iso3': 'Country A iso3'})
    trade_mat = trade_mat.sort_values(by='Country A iso3')
    
    def _add_all_countries(m):
        m = m.merge(FAO_area_codes, left_on='Country B Code', right_on='Area Code (M49)', how='right')
        m = m.drop(['Country B', 'Country B Code', 'Country A iso3', 'Country A M49', 'Country A'], axis=1)
        m = m.rename(columns={'Area Code (M49)': 'Country B M49', 'country': 'Country B',
                                              'iso3': 'Country B iso3'})
        m = m.sort_values(by='Country B iso3')
        return m

    trade_mat = trade_mat.groupby(['Country A iso3', 'Country A M49', 'Country A']).apply(lambda g: _add_all_countries(g)).reset_index()

    trade_mat = trade_mat.drop('level_3', axis=1)
    trade_mat['Item'] = item
    trade_mat['Year'] = year
    trade_mat['Unit'] = 'tonnes'
    trade_mat = trade_mat.fillna(0)
    trade_mat = pd.pivot(trade_mat, index=['Country B iso3', 'Country B M49', 'Country B',
                                           'Item', 'Year', 'Unit'], columns = 'Country A iso3',values = 'From B to A').reset_index()
    E = trade_mat.drop(['Country B iso3', 'Country B M49', 'Country B', 'Item', 'Year', 'Unit'], axis=1).to_numpy()
    return E


def re_export_algo(P, E):
    # Implements the trade matrix re-export algorithm as given in Croft et al., 2018 (https://www.sciencedirect.com/science/article/pii/S0959652618326180#appsec2)
    
    # Number of iterations
    N = 10000

    # Number of countries
    num_ctry = len(P)

    # Pre-calculate diagonal Production matrix
    Pd = np.diagflat(P)

    # Pre-allocate Domestic Supply matrix
    D = np.zeros((num_ctry, num_ctry))

    for n in range(1,N+1):
        # STEP 1: allocate production
        # Allocate production to domestic supply
        D += Pd / N

        # STEP 2: perform trade
        # Calculate proportions of domestic supply required for each component of export iteration
        temp1 = E / N / np.tile(np.sum(D, axis=0), (num_ctry, 1)).T

        # Sum to check if greater than 1 (if domestic supply is less than desired export total)
        temp2 = np.tile(np.nansum(temp1, axis=1)[:, np.newaxis], (1, num_ctry))

        # Constrain export greater than domestic supply to be equal to domestic supply
        mask = np.tile(np.nansum(temp1, axis=1) > 1, (num_ctry, 1)).T # or np.tile(np.nansum(temp1, axis=1)[:, np.newaxis]>1, (1, num_ctry))
        temp1[mask] = temp1[mask] / temp2[mask]
        
        # Proportional change in domestic supply
        e_n = np.ones((num_ctry, 1)) - np.nansum(temp1, axis=1)[:, np.newaxis]

        # Apply to domestic supply of domestic production (non-traded component)
        e_n = np.diagflat(e_n) + temp1

        # Take care of 0/0 cases
        e_n[np.isnan(e_n)] = 0

        # Take care of x/0 cases
        e_n[np.isinf(e_n)] = 0

        # Rescale domestic supply to redistribute according to trade
        D = D.dot(e_n)
        
    return D

if __name__ == '__main__':
    
    # Enter crops and years
    items = ['Wheat', 'Maize (corn)', 'Rye', 'Barley', 'Oats', 'Sorghum', 'Rice', 'Buckwheat', 'Millet', 'Quinoa', 'Cereals n.e.c.'] 
    years = [2017, 2018, 2019, 2020, 2021]
    
    FAO_area_codes = get_area_codes()
    
    # create empty matrix
    cereals_P = np.zeros((len(FAO_area_codes),1))
    cereals_E = np.zeros((len(FAO_area_codes), len(FAO_area_codes)))
    cereals_D = np.zeros((len(FAO_area_codes), len(FAO_area_codes)))
    
    for year in years:
        print(year)
        for item in items:
            print(item)
            P = get_prod_matrix(item, year, FAO_area_codes)
            E = get_trade_matrix(item, year, FAO_area_codes)
            D = re_export_algo(P, E)

            df_P = pd.DataFrame(P, columns=['prod'])
            df_P['iso3'] = pd.Series(FAO_area_codes['iso3'].values.tolist(), index=df_P.index)
            first_column = df_P.pop('iso3')
            df_P.insert(0, 'iso3', first_column)
            df_P.to_csv(f'{data_dir_prefix}FAO_prod_mat/prod_matrix_{item}_{year}.csv', index=False)
            
            df_E = pd.DataFrame(E, columns = FAO_area_codes['iso3'].values.tolist())
            df_E['iso3'] = pd.Series(FAO_area_codes['iso3'].values.tolist(), index=df_E.index)
            first_column = df_E.pop('iso3')
            df_E.insert(0, 'iso3', first_column)
            df_E.to_csv(f'{data_dir_prefix}FAO_bal_trade_mat/trade_matrix_{item}_{year}.csv', index=False)
            
            df_D = pd.DataFrame(D, columns = FAO_area_codes['iso3'].values.tolist())
            df_D['iso3'] = pd.Series(FAO_area_codes['iso3'].values.tolist(), index=df_D.index)
            first_column = df_D.pop('iso3')
            df_D.insert(0, 'iso3', first_column)
            df_D.to_csv(f'{data_dir_prefix}FAO_re_export/supply_matrix_{item}_{year}.csv', index=False)

            # add to existing matrix for cereals
            cereals_P = np.add(cereals_P, P)
            cereals_E = np.add(cereals_E, E)
            cereals_D = np.add(cereals_D, D)

        df_cereals_P = pd.DataFrame(cereals_P, columns=['prod'])
        df_cereals_P['iso3'] = pd.Series(FAO_area_codes['iso3'].values.tolist(), index=df_cereals_P.index)
        first_column = df_cereals_P.pop('iso3')
        df_cereals_P.insert(0, 'iso3', first_column)
        df_cereals_P.to_csv(f'{data_dir_prefix}FAO_prod_mat/prod_matrix_cereals_all_{year}.csv', index=False)
        
        df_cereals_E = pd.DataFrame(cereals_E, columns = FAO_area_codes['iso3'].values.tolist())
        df_cereals_E['iso3'] = pd.Series(FAO_area_codes['iso3'].values.tolist(), index=df_cereals_E.index)
        first_column = df_cereals_E.pop('iso3')
        df_cereals_E.insert(0, 'iso3', first_column)
        df_cereals_E.to_csv(f'{data_dir_prefix}FAO_bal_trade_mat/trade_matrix_cereals_all_{year}.csv', index=False)
        
        df_cereals_D = pd.DataFrame(cereals_D, columns = FAO_area_codes['iso3'].values.tolist())
        df_cereals_D['iso3'] = pd.Series(FAO_area_codes['iso3'].values.tolist(), index=df_cereals_D.index)
        first_column = df_cereals_D.pop('iso3')
        df_cereals_D.insert(0, 'iso3', first_column)
        df_cereals_D.to_csv(f'{data_dir_prefix}FAO_re_export/supply_matrix_cereals_all_{year}.csv', index=False)


