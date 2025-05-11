# Usage: python -m food_trade.flow_prediction.run_harmonizing

import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from food_trade.flow_prediction.helper_functions import *

def probs_edit(g, dom_q=0.8, int_q=0.8, threshold=0.5, crop='cereals_all'):
    if g[f'supply_{crop}_exists_pred'].sum()==0:
        return g
    if g['from_iso3'].values[0]==g['to_iso3'].values[0]:
        if g[f'supply_{crop}_exists_pred'].quantile(dom_q)>threshold:
            return g
        g.loc[g[f'supply_{crop}_exists_pred']>=g[f'supply_{crop}_exists_pred'].quantile(dom_q), f'supply_{crop}_exists_pred'] = threshold+0.1
    else:
        if g[f'supply_{crop}_exists_pred'].quantile(int_q)>threshold:
            return g
        g.loc[g[f'supply_{crop}_exists_pred']>=g[f'supply_{crop}_exists_pred'].quantile(int_q), f'supply_{crop}_exists_pred'] = threshold+0.1
    return g


def _rescale(g, crop='cereals_all'):
    if g['from_iso3'].values[0]==g['to_iso3'].values[0]:
        return g
    if g[f'supply_{crop}_pred'].sum()==0:
        return g
    if g[f'supply_{crop}'].values[0] < g[f'supply_{crop}_pred'].sum(): 
        # if target trade is lower, then it doesn't matter that some regions have reached their outflow capacity (even in the constrained situation)
        g['scaling_factor'] = g[f'supply_{crop}'].values[0] / g[f'supply_{crop}_pred'].sum()
    else:
        # if target trade is higher, and some regions have reached their outflow capacity, we need to exclude those in the constrained situation
        target_excl = g[f'supply_{crop}'].values[0] - g[g['exclude']==1][f'supply_{crop}_pred'].sum()
        current_excl = g[g['exclude']==0][f'supply_{crop}_pred'].sum()
        if current_excl==0:
            g.loc[g['exclude']==0, f'supply_{crop}_pred'] = g[g['exclude']==0]['remaining']
            current_excl = g[g['exclude']==0][f'supply_{crop}_pred'].sum()
        g['scaling_factor'] = target_excl / current_excl
        g.loc[g['exclude']==1, 'scaling_factor'] = 1
    g[f'supply_{crop}_pred'] = g[f'supply_{crop}_pred'] * g['scaling_factor']
    return g

def int_rescale(F, df_country, df_ids, beta=0.95, constrained=False, crop='cereals_all'):
    # convert to dataframe
    df_F = pd.DataFrame(F, columns = df_ids['id'].values.tolist())
    df_F['from_id'] = pd.Series(df_ids['id'].values.tolist(), index=df_F.index)
    first_column = df_F.pop('from_id')
    df_F.insert(0, 'from_id', first_column)
    df_F = df_F.melt(id_vars=['from_id'], value_vars=df_ids['id'].values.tolist()).rename(columns={'variable': 'to_id', 'value': f'supply_{crop}_pred'})
    df_F = df_F.merge(df_ids, left_on='from_id', right_on='id').rename(columns={'iso3': 'from_iso3'}).drop('id', axis=1)
    df_F = df_F.merge(df_ids, left_on='to_id', right_on='id').rename(columns={'iso3': 'to_iso3'}).drop('id', axis=1)
    df_F = df_F.merge(df_country[['from_iso3', 'to_iso3', f'supply_{crop}']], how='left')
    
    outflow_status = df_F.groupby('from_id')[[f'supply_{crop}_pred']].sum().reset_index().merge(master_df_admin[['ID', f'{crop}_prod']], left_on='from_id', right_on='ID').drop('ID', axis=1)
    outflow_status['exclude'] = 0
    outflow_status.loc[outflow_status[f'supply_{crop}_pred'] >= beta * beta * outflow_status[f'{crop}_prod'], 'exclude'] = 1
    outflow_status['remaining'] = outflow_status[f'{crop}_prod'] - outflow_status[f'supply_{crop}_pred']
    df_F = df_F.merge(outflow_status[['from_id', 'remaining', 'exclude']])
    if not constrained:
        df_F['exclude'] = 0
    # rescale
    df_F = df_F.groupby(['from_iso3', 'to_iso3']).apply(lambda g: _rescale(g, crop)).reset_index(drop=True)

    # convert to matrix
    F = pd.pivot(df_F, index=['from_id'], columns='to_id', values = f'supply_{crop}_pred').reset_index().drop('from_id', axis=1).to_numpy()

    return F

def harmonize_algorithm(df_mat, master_df_admin, df_country, df_ids, crop, 
                        threshold=0.5, tol=0.05, max_iter=50, alpha=0.1, beta=0.95, dom_q=0.8, int_q=0.8):

    df_mat = df_mat.groupby(['from_iso3', 'to_iso3']).apply(lambda g: probs_edit(g, dom_q, int_q, threshold, crop)).reset_index(drop=True)
    df_mat = df_mat.sort_values(by=['from_id', 'to_id'])
    df_mat_probs = pd.pivot(df_mat, index=['from_id'], columns = 'to_id',values = f'supply_{crop}_exists_pred').reset_index().drop('from_id', axis=1).to_numpy()
    df_mat_trade = pd.pivot(df_mat, index=['from_id'], columns = 'to_id',values = f'supply_{crop}_pred').reset_index().drop('from_id', axis=1).to_numpy()

    F = df_mat_trade
    prob_matrix = df_mat_probs
    F = np.where(prob_matrix >= threshold, F, 0)

    n_regions = master_df_admin['ID'].nunique()
    region_to_country = master_df_admin['iso3'].values
    production = master_df_admin[f'{crop}_prod'].values
    consumption = master_df_admin[f'{crop}_cons'].values

    inflow = F.sum(axis=0)       # shape (n_regions,)
    outflow = F.sum(axis=1)      # shape (n_regions,)
    net = production + inflow - outflow  # net resources at each region
    diff = consumption - net     # positive diff indicates deficit, negative indicates surplus
    tol_threshold = tol * consumption
    
    print(f"Before balancing: Total = {F.sum()}, Sum deficits = {diff[diff > tol_threshold].sum():.2f}, " +
          f"Sum surpluses = {diff[diff < -tol_threshold].sum():.2f}")

    for iteration in range(max_iter):

        # Save F before updating, to later compute max change.
        F_old = F.copy()
        
        # Adjust international flows to match national targets.
        F = int_rescale(F, df_country, df_ids, beta, constrained=False, crop=crop)
    
        max_change = 0
        
        # Adjust flows to meet the consumption constraints.
        inflow = F.sum(axis=0)       # shape (n_regions,)
        outflow = F.sum(axis=1)      # shape (n_regions,)
        net = production + inflow - outflow  # net resources at each region
        diff = consumption - net     # positive diff indicates deficit, negative indicates surplus
        tol_threshold = tol * consumption
    
        # Diagnostic prints:
        print(f"Iteration {iteration}: Sum deficits = {diff[diff > tol_threshold].sum():.2f}, " +
              f"Sum surpluses = {diff[diff < -tol_threshold].sum():.2f}")
    
        # Initialize scaling factors as 1 for every region.
        scaling_cols = np.ones(n_regions)
        scaling_rows = np.ones(n_regions)

        # can change this if the flows seem to be reducing/increasing too much or if nothing changes 
        add_fac = (1 + iteration/(max_iter-1)) * alpha # goes from alpha to 2*alpha 
        sub_fac = (0.5 + iteration/(max_iter-1)) * alpha # goes from 0.5*alpha to 1.5*alpha 
        
        # Loop over each region and apply the appropriate correction.
        for i in range(n_regions):
            if diff[i] > tol_threshold[i]:  # Region i in deficit.
                if inflow[i] > 0:
                    scaling_cols[i] = (inflow[i] + add_fac * diff[i]) / inflow[i]
                if outflow[i] > 0:
                    scaling_rows[i] = (outflow[i] - sub_fac * diff[i]) / outflow[i]
            elif diff[i] < -tol_threshold[i]:  # Region i in surplus.
                if inflow[i] > 0:
                    scaling_cols[i] = (inflow[i] + sub_fac * diff[i]) / inflow[i]
                # Only adjust outflows if region i also has un-exported production.
                if (production[i] - outflow[i]) > 0 and outflow[i] > 0:
                    scaling_rows[i] = (outflow[i] - add_fac * diff[i]) / outflow[i]
        
        # Clip scaling factors to ensure they are not negative.
        scaling_cols = np.maximum(scaling_cols, 0)
        scaling_rows = np.maximum(scaling_rows, 0)
        
        # Update F.
        # Multiply each column i (i.e. inflows to region i) by scaling_cols[i]:
        F = F * scaling_cols[None, :]
        # Multiply each row i (i.e. outflows from region i) by scaling_rows[i]:
        F = F * scaling_rows[:, None]
    
        print("Flow sum after consumption adjustment =", F.sum())
    
        # Enforce constraint: total outflows must not exceed beta*production, 
        # also if production allows, at least 50% consumption must be met by local production (so constraint must take that into account too) 
        # need to justify the second (can do using national scale data, and/or india data?)
        for i in range(n_regions):
            current_outflow = F[i, :].sum()
            constrained_outflow = min(beta * production[i], max(0, production[i]-0.5*consumption[i])) 
            if current_outflow > constrained_outflow: 
                factor = constrained_outflow / current_outflow
                F[i, :] *= factor
        
        print("Flow sum after enforcing production constraint =", F.sum())
        
        # Track maximum change from this consumption adjustment.
        max_change = np.max(np.abs(F - F_old))
        if max_change < 1:
            print("Converged at iteration", iteration)
            break

    F = int_rescale(F, df_country, df_ids, beta, constrained=True, crop=crop)

    inflow = F.sum(axis=0)       # shape (n_regions,)
    outflow = F.sum(axis=1)      # shape (n_regions,)
    net = production + inflow - outflow  # net resources at each region
    diff = consumption - net     # positive diff indicates deficit, negative indicates surplus
    tol_threshold = tol * consumption
    
    # Diagnostic prints:
    print(f"After first loop: Total = {F.sum()}, Sum deficits = {diff[diff > tol_threshold].sum():.2f}, " +
          f"Sum surpluses = {diff[diff < -tol_threshold].sum():.2f}")

    excess_outflow = 0
    for i in range(n_regions):
        current_outflow = F[i, :].sum()
        if current_outflow > production[i]: 
            excess_outflow += current_outflow - production[i]
    print(f'Excess outflow = {excess_outflow}')

    for iteration in range(2*max_iter):

        # Save F to later compute max change.
        F_old = F.copy()
    
        # Compute the residual consumption difference.
        inflow = F.sum(axis=0)
        outflow = F.sum(axis=1)
        net = production + inflow - outflow
        diff = consumption - net  # positive indicates additional flow needed.
    
        # Diagnostic prints:
        print(f"Iteration {iteration}: Sum deficits = {diff[diff > tol_threshold].sum():.2f}, " +
              f"Sum surpluses = {diff[diff < -tol_threshold].sum():.2f}")
    
        countries = np.unique(region_to_country)
        for c in countries:
            # Get indices for regions in country c.
            idx = np.where(region_to_country == c)[0]
            if len(idx) == 0:
                continue
        
            # Compute aggregates for regions in country c.
            # Calculate inflow for each region in idx:
            inflow_c = F[:, idx].sum(axis=0)
            outflow_c = F[idx, :].sum(axis=1)
            net_c = production[idx] + inflow_c - outflow_c
            diff_c = consumption[idx] - net_c  # Positive: deficit, Negative: surplus
        
            # Identify deficit and surplus regions.
            mask_deficit = diff_c > tol_threshold[idx]  # deficit regions
    
            # Identify surplus regions with two conditions:
            # 1. Their net exceeds their consumption (net > consumption)
            # 2. Their total outflow is less than production (so they are not already exporting all their production)
            mask_surplus = (net_c > consumption[idx]) & (outflow_c < production[idx])
        
            if np.sum(mask_deficit) == 0 or np.sum(mask_surplus) == 0:
                continue
        
            deficit_regions = idx[mask_deficit]
            surplus_regions = idx[mask_surplus]
            
            # Total deficit and total available surplus (absolute values).
            total_deficit = diff_c[mask_deficit].sum()
    
            # For each surplus region, compute its available surplus in two ways:
            #   a) net surplus = net - consumption
            #   b) unused production = production - outflow
            # Use the minimum of these two as the effective surplus.
            available_surplus = np.minimum(-diff_c[mask_surplus], production[idx][mask_surplus] - outflow_c[mask_surplus]) # since diff is negative for surplus
            total_surplus = available_surplus.sum()  
        
            # Determine the reallocation amount R.
            R = alpha * min(total_deficit, total_surplus)
            if R <= 0:
                continue
        
            # Allocate extra flow from surplus to deficit regions.
            # For each deficit region, compute its share of total deficit.
            deficit_shares = diff_c[mask_deficit] / total_deficit  if total_deficit > 0 else np.ones(np.sum(mask_deficit))
            # For each surplus region, compute its share of total surplus.
            surplus_shares = available_surplus / total_surplus  if total_surplus > 0 else np.ones(np.sum(mask_surplus))
            
            # For each deficit region i and each surplus region j, allocate extra flow.
            for di, i in enumerate(deficit_regions):
                for sj, j in enumerate(surplus_regions):
                    # The extra flow contributed from donor j to recipient i:
                    delta = R * deficit_shares[di] * surplus_shares[sj]
                    F[j, i] += delta
       
        print(F.sum())
        
        # Track maximum change from this consumption adjustment.
        max_change = np.max(np.abs(F - F_old))
        if max_change < 1:
            print("Converged at iteration", iteration)
            break
    
    return F

if __name__ == '__main__':

    master_df_country = country_totals()
    df_country, df_admin = get_bilateral_data()
    master_df_admin = pred_consumption(master_df_country, df_country, df_admin)
    
    df_country = df_country.sort_values(by=['from_iso3', 'to_iso3']).reset_index(drop=True)
    master_df_admin = master_df_admin.sort_values(by=['iso3', 'ID']).reset_index(drop=True)

    # all ids
    df_ids = df_admin[['from_id', 'from_iso3']].drop_duplicates().sort_values(by='from_id').reset_index(drop=True)
    df_ids = df_ids.rename(columns={'from_id': 'id', 'from_iso3': 'iso3'})
    crop = 'cereals_all'

    # # harmonize and save
    # suffix = 'all'
    # df_mat = pd.read_parquet(f'../../data/machine_learning/admin_predictions/df_mat_{crop}_{suffix}.parquet.gzip')
    # F = harmonize_algorithm(df_mat, master_df_admin, df_country, df_ids, crop)
    # df_F = pd.DataFrame(F, columns = df_ids['id'].values.tolist())
    # df_F['from_id'] = pd.Series(df_ids['id'].values.tolist(), index=df_F.index)
    # first_column = df_F.pop('from_id')
    # df_F.insert(0, 'from_id', first_column)
    # df_F = df_F.melt(id_vars=['from_id'], value_vars=df_ids['id'].values.tolist()).rename(columns={'variable': 'to_id', 'value': f'supply_{crop}_bal'})
    # df_F = df_F.merge(df_ids, left_on='from_id', right_on='id').rename(columns={'iso3': 'from_iso3'}).drop('id', axis=1)
    # df_F = df_F.merge(df_ids, left_on='to_id', right_on='id').rename(columns={'iso3': 'to_iso3'}).drop('id', axis=1)
    # df_F.to_parquet(f'../../data/machine_learning/admin_predictions_harmonized/df_mat_bal_{crop}_{suffix}.parquet.gzip', index=False, compression='gzip')
    
    # run once more on fao based predictions
    suffix = 'fao'
    df_mat = pd.read_parquet(f'../../data/machine_learning/admin_predictions/df_mat_{crop}_{suffix}.parquet.gzip')
    F = harmonize_algorithm(df_mat, master_df_admin, df_country, df_ids, crop)
    df_F = pd.DataFrame(F, columns = df_ids['id'].values.tolist())
    df_F['from_id'] = pd.Series(df_ids['id'].values.tolist(), index=df_F.index)
    first_column = df_F.pop('from_id')
    df_F.insert(0, 'from_id', first_column)
    df_F = df_F.melt(id_vars=['from_id'], value_vars=df_ids['id'].values.tolist()).rename(columns={'variable': 'to_id', 'value': f'supply_{crop}_bal'})
    df_F = df_F.merge(df_ids, left_on='from_id', right_on='id').rename(columns={'iso3': 'from_iso3'}).drop('id', axis=1)
    df_F = df_F.merge(df_ids, left_on='to_id', right_on='id').rename(columns={'iso3': 'to_iso3'}).drop('id', axis=1)
    df_F.to_parquet(f'../../data/machine_learning/admin_predictions_harmonized/df_mat_bal_{crop}_{suffix}.parquet.gzip', index=False, compression='gzip')
    
    # sensitivity analysis
    suffix = 'all'
    df_mat = pd.read_parquet(f'../../data/machine_learning/admin_predictions/df_mat_{crop}_{suffix}.parquet.gzip')
    param_grid = {'threshold': [0.4, 0.5, 0.6],
                  'tol' : [0.05],
                  'max_iter' : [40, 50],
                  'alpha' : [0.1, 0.2],
                  'beta' : [0.9, 0.95],
                  'dom_q' : [0.75, 0.8],
                  'int_q' : [0.75, 0.8]}

    index_labels = []
    i=0
    for params in ParameterGrid(param_grid):
        if i>=0:
            F = harmonize_algorithm(df_mat, master_df_admin, df_country, df_ids, crop, **params)   
            df_F = pd.DataFrame(F, columns = df_ids['id'].values.tolist())
            df_F['from_id'] = pd.Series(df_ids['id'].values.tolist(), index=df_F.index)
            first_column = df_F.pop('from_id')
            df_F.insert(0, 'from_id', first_column)
            df_F = df_F.melt(id_vars=['from_id'], value_vars=df_ids['id'].values.tolist()).rename(columns={'variable': 'to_id', 'value': f'supply_{crop}_bal'})
            df_F = df_F.merge(df_ids, left_on='from_id', right_on='id').rename(columns={'iso3': 'from_iso3'}).drop('id', axis=1)
            df_F = df_F.merge(df_ids, left_on='to_id', right_on='id').rename(columns={'iso3': 'to_iso3'}).drop('id', axis=1)
            df_F.to_parquet(f'../../data/machine_learning/admin_predictions_harmonized/sensitivity_analysis/df_mat_bal_{crop}_{suffix}_{i}.parquet.gzip', 
                            index=False, compression='gzip')        
        index_labels.append({**params})
        i += 1

    param_df = pd.DataFrame(index_labels)
    param_df.to_csv('../../data/machine_learning/admin_predictions_harmonized/sensitivity_analysis/param_grid.csv', index=False)

    


    