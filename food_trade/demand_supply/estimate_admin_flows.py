# Usage: python -m food_trade.demand_supply.estimate_admin_flows

import numpy as np
import pandas as pd
import time

# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(tc, prod_cons):
    """Index regions and map transport cost pairs to integer indices."""
    region_ids = prod_cons['ID'].values
    id_to_idx = {rid: i for i, rid in enumerate(region_ids)}
    region_to_country = prod_cons['iso3'].values

    tc_indexed = tc.copy()
    tc_indexed['from_idx'] = tc_indexed['from_id'].map(id_to_idx)
    tc_indexed['to_idx'] = tc_indexed['to_id'].map(id_to_idx)
    tc_indexed = tc_indexed.dropna(subset=['from_idx', 'to_idx'])
    tc_indexed['from_idx'] = tc_indexed['from_idx'].astype(int)
    tc_indexed['to_idx'] = tc_indexed['to_idx'].astype(int)

    return region_ids, region_to_country, tc_indexed

def prepare_crop_data(prod_cons, region_to_country, region_ids, crop):
    """
    Prepare production, consumption, surplus, and deficit for one crop.
    """
    production = prod_cons[f'{crop}_prod'].values.copy()
    consumption = prod_cons[f'{crop}_cons'].values.copy()

    # Surplus and deficit per region
    balance = production - consumption
    surplus = np.maximum(balance, 0)
    deficit = np.maximum(-balance, 0)
    
    # Strictly balance global surplus and deficit for the algorithm to converge
    total_surplus = surplus.sum()
    total_deficit = deficit.sum()
    target_total = max(total_surplus, total_deficit)
    
    if target_total > 0:
        surplus = surplus * (target_total / (total_surplus + 1e-12))
        deficit = deficit * (target_total / (total_deficit + 1e-12))

    print(f"  Surplus regions: {(surplus > 0).sum()}")
    print(f"  Deficit regions: {(deficit > 0).sum()}")
    print(f"  Balanced global tradable mass: {target_total:,.0f} tonnes")

    return production, consumption, surplus, deficit


# =============================================================================
# UNIFIED TRIPLY-CONSTRAINED GRAVITY MODEL
# =============================================================================

def solve_unified_gravity(surplus, deficit, tc_indexed, fao, crop, 
                          region_to_country, n_regions, beta=1.5, max_iter=200, tol=1.0):
    """
    Simultaneously solves international and domestic flows using a 
    Triply-Constrained Spatial Interaction Model (Sinkhorn-Knopp).
    
    Beta=1.5 is standard in gravity trade literature for cost elasticity.
    """
    print(f"\n  Building Unified Spatial matrix (N={n_regions})...")
    t0 = time.time()
    
    # 1. Map countries to indices for block constraints
    countries = np.unique(region_to_country)
    n_countries = len(countries)
    iso3_to_c_idx = {c: i for i, c in enumerate(countries)}
    region_c_idx = np.array([iso3_to_c_idx[c] for c in region_to_country])
    
    # Indicator matrix U mapping regions to countries (Shape: N x K)
    U = np.zeros((n_regions, n_countries))
    U[np.arange(n_regions), region_c_idx] = 1.0

    # 2. Build dense cost matrix (Shape: N x N)
    print("  Constructing cost matrices...")
    C_mat = np.full((n_regions, n_regions), fill_value=np.inf)
    C_mat[tc_indexed['from_idx'], tc_indexed['to_idx']] = tc_indexed['transport_USD_t']
    
    # Self-trade distance penalty (small cost to allow local balancing)
    min_cost = max(tc_indexed['transport_USD_t'].min(), 1.0)
    np.fill_diagonal(C_mat, min_cost)
    
    # Penalize completely missing links so they don't block the algorithm
    max_valid_cost = tc_indexed['transport_USD_t'].max()
    C_mat[np.isinf(C_mat)] = max_valid_cost * 2.0

    # Initialize Prior matrix with power-law distance decay
    Prior = C_mat ** -beta
    
    # 3. Build Country-to-Country Target Matrix (Shape: K x K)
    print("  Calibrating FAO international targets...")
    fao_col = f'supply_{crop}'
    fao_nonzero = fao[fao[fao_col] > 0]
    
    initial_M = np.zeros((n_countries, n_countries))
    for _, row in fao_nonzero.iterrows():
        if row['from_iso3'] in iso3_to_c_idx and row['to_iso3'] in iso3_to_c_idx:
            i, j = iso3_to_c_idx[row['from_iso3']], iso3_to_c_idx[row['to_iso3']]
            initial_M[i, j] = row[fao_col]

    # Calculate required domestic trade targets (Surplus left in country)
    S_country = U.T @ surplus
    D_country = U.T @ deficit
    
    for i in range(n_countries):
        exports = initial_M[i, :].sum()
        initial_M[i, i] = max(0.1, S_country[i] - exports)

    # Pre-balance the FAO matrix so macro targets strictly match regional mass
    M_mat = initial_M.copy()
    for _ in range(50):
        r = M_mat.sum(axis=1) + 1e-12
        M_mat *= (S_country / r)[:, np.newaxis]
        c = M_mat.sum(axis=0) + 1e-12
        M_mat *= (D_country / c)[np.newaxis, :]

    # 4. Iterative Proportional Fitting (Triply Constrained Sinkhorn)
    print("  Solving multi-constrained Sinkhorn optimization...")
    X = Prior.copy()
    
    for iteration in range(max_iter):
        # Constraint 1: Match Regional Deficit (Columns)
        c = X.sum(axis=0) + 1e-12
        X *= (deficit / c)[np.newaxis, :]
        
        # Constraint 2: Match Regional Surplus (Rows)
        r = X.sum(axis=1) + 1e-12
        X *= (surplus / r)[:, np.newaxis]
        
        # Constraint 3: Match FAO Country-to-Country targets (Blocks)
        # V is the current sum of flows between every country pair
        V = U.T @ X @ U  
        scale = M_mat / np.maximum(V, 1e-12)
        X *= (U @ scale @ U.T)
        
        # Check convergence periodically
        if iteration % 20 == 0 or iteration == max_iter - 1:
            err_r = np.max(np.abs(X.sum(axis=1) - surplus))
            err_c = np.max(np.abs(X.sum(axis=0) - deficit))
            if err_r < tol and err_c < tol:
                print(f"  Converged at iteration {iteration} (Max error: {max(err_r, err_c):.2f})")
                break
    else:
        print(f"  Reached max iterations with final error: {max(err_r, err_c):.2f}")

    elapsed = time.time() - t0
    print(f"  Gravity Model solved in {elapsed:.1f}s")

    # 5. Extract results into DataFrame
    # Filter out computationally tiny flows to keep data clean
    active_idx = np.where(X > 0.1)
    
    from_iso3 = region_to_country[active_idx[0]]
    to_iso3 = region_to_country[active_idx[1]]
    is_domestic = (from_iso3 == to_iso3)
    
    df_flows = pd.DataFrame({
        'from_idx': active_idx[0],
        'to_idx': active_idx[1],
        'from_iso3': from_iso3,
        'to_iso3': to_iso3,
        'flow': X[active_idx],
        'is_domestic': is_domestic,
        'is_self': False # We handle self-consumed separately below
    })
    
    return df_flows


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_crop(prod_cons, fao, tc_indexed, region_to_country, region_ids, n_regions, crop, beta):
    """Run full pipeline for one crop."""

    print(f"\n{'='*70}")
    print(f"CROP: {crop}")
    print(f"{'='*70}")

    # Step 1: Prep masses
    production, consumption, surplus, deficit = prepare_crop_data(
        prod_cons, region_to_country, region_ids, crop
    )

    # Step 2: Unified Gravity routing
    df_flows = solve_unified_gravity(
        surplus, deficit, tc_indexed, fao, crop, 
        region_to_country, n_regions, beta=beta
    )

    # Step 3: Compute self-consumption (production consumed locally before trade)
    self_consumed = np.minimum(production, consumption)
    self_rows = []
    for i in range(n_regions):
        if self_consumed[i] > 0.01:
            self_rows.append({
                'from_idx': i,
                'to_idx': i,
                'from_iso3': region_to_country[i],
                'to_iso3': region_to_country[i],
                'flow': self_consumed[i],
                'is_domestic': True,
                'is_self': True
            })
    
    df_self = pd.DataFrame(self_rows) if self_rows else pd.DataFrame(
        columns=['from_idx', 'to_idx', 'from_iso3', 'to_iso3', 'flow', 'is_domestic', 'is_self']
    )

    # Combine routed trade + self-consumption
    df_all = pd.concat([df_flows, df_self], ignore_index=True)

    # Summary
    intl_total = df_all.loc[df_all['is_domestic'] == False, 'flow'].sum()
    dom_trade = df_all.loc[(df_all['is_domestic'] == True) & (df_all['is_self'] == False), 'flow'].sum()
    self_total = df_all.loc[df_all['is_self'] == True, 'flow'].sum()

    print(f"\n  --- {crop} SUMMARY ---")
    print(f"  Total links: {len(df_all):,}")
    print(f"  Total flow: {df_all['flow'].sum():,.0f}")
    print(f"  International: {intl_total:,.0f}")
    print(f"  Domestic trade: {dom_trade:,.0f}")
    print(f"  Self-consumption: {self_total:,.0f}")

    return df_all


def run_all(tc, prod_cons, fao, beta):
    """Run pipeline for all crops and produce comparisons."""

    region_ids, region_to_country, tc_indexed = prepare_data(tc, prod_cons)
    n_regions = len(region_ids)

    individual_crops = ['wheat', 'rice', 'maize', 'other_cereals']
    all_crops = individual_crops + ['cereals_all']

    results = {}

    for crop in all_crops:
        df = run_crop(prod_cons, fao, tc_indexed, region_to_country,
                      region_ids, n_regions, crop, beta)
        results[crop] = df

    # Combine crop-specific flows
    crop_dfs = [results.get(c) for c in individual_crops if results.get(c) is not None]

    if crop_dfs:
        df_combined = pd.concat(crop_dfs, ignore_index=True)
        df_combined = df_combined.groupby(
            ['from_idx', 'to_idx', 'from_iso3', 'to_iso3', 'is_domestic', 'is_self']
        )['flow'].sum().reset_index()
        results['combined_crops'] = df_combined

        print(f"\n{'='*70}")
        print(f"COMBINED CROP-SPECIFIC FLOWS")
        print(f"{'='*70}")
        print(f"  Active links: {len(df_combined):,}")
        print(f"  Total flow: {df_combined['flow'].sum():,.0f}")

    return results


def add_region_ids(results, region_ids):
    """Replace integer indices with actual region IDs."""
    for key, df in results.items():
        if df is not None and 'from_idx' in df.columns:
            df['from_id'] = df['from_idx'].map(lambda x: region_ids[x])
            df['to_id'] = df['to_idx'].map(lambda x: region_ids[x])
    return results


if __name__ == '__main__':

    tc = pd.read_parquet('../../data/combined_data/transport.parquet.gzip')
    prod_cons = pd.read_csv('../../data/combined_data/admin_data.csv')
    fao = pd.read_csv('../../data/combined_data/fao_trade.csv')

    # Sort for consistent indexing
    prod_cons = prod_cons.sort_values('ID').reset_index(drop=True)

    # Filter out self-supply from bilateral data
    fao = fao[fao['from_iso3'] != fao['to_iso3']]

    for beta in [1.2, 1.5, 1.8]:
        print(f'Running with beta = {beta}')
        # Run the unified gravity model pipeline
        results = run_all(tc, prod_cons, fao, beta=beta)
    
        # Add admin IDs back
        region_ids = prod_cons['ID'].values
        results = add_region_ids(results, region_ids)
    
        for key, df in results.items():
            if df is not None:
                df.to_parquet(
                    f'../../data/outputs/flows_{key}_{beta}.parquet.gzip',
                    index=False, compression='gzip'
                )
                print(f"Saved: flows_{key}_{beta}.parquet.gzip ({len(df):,} rows)")