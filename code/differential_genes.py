### Making a function to be easier
import scanpy as sc
import pandas as pd
import os

def run_sc_de(adata, group_col, subset_celltypes, target_group, reference_group, method='wilcoxon', savedir=None, prefix='DE'):
    """
    Perform differential expression analysis for single-cell data comparing a target group vs reference group.
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing your single-cell data.
    group_col : str
        Column in adata.obs indicating cell types or groups.
    target_group : str
        Name of the group of interest (will be tested vs reference_group).
    reference_group : str
        Name of the reference group (other cells to compare against).
    method : str, default 'wilcoxon'
        Statistical test for rank_genes_groups ('wilcoxon', 't-test', etc.).
    savedir : str, optional
        Directory to save the DE table CSV. If None, will not save.
    prefix : str, default 'DE'
        Prefix for output CSV filename.
    Returns
    -------
    de_results : pd.DataFrame
        Differential expression results with logfoldchange, p-values, adjusted p-values, and percent expression.
    """
    # Subset relevant cells
    adata_subset = adata[adata.obs[group_col].isin(subset_celltypes)].copy()
    # Create a new 'comparison' column for rank_genes_groups
    adata_subset.obs['comparison'] = adata_subset.obs[group_col].apply(
        lambda x: target_group if x == target_group else reference_group
    )
    # Normalize and log-transform
    sc.pp.normalize_total(adata_subset, target_sum=1e4)
    sc.pp.log1p(adata_subset)
    # Run DE analysis
    sc.tl.rank_genes_groups(
        adata_subset,
        groupby='comparison',
        reference=reference_group,
        method=method,
        pts=True
    )
    # Extract results
    genes = adata_subset.uns['rank_genes_groups']['names'][target_group]
    logfc = adata_subset.uns['rank_genes_groups']['logfoldchanges'][target_group]
    pvals = adata_subset.uns['rank_genes_groups']['pvals'][target_group]
    pvals_adj = adata_subset.uns['rank_genes_groups']['pvals_adj'][target_group]
    # Percent expression
    pts_group = adata_subset.uns['rank_genes_groups']['pts']
    pct_expr_in_group = pts_group.loc[genes.tolist(),:]
    # Build DE table
    de_results = pd.DataFrame({
        'gene': genes,
        'logfoldchange': logfc,
        'pvals': pvals,
        'pvals_adj': pvals_adj,
        f'{reference_group}_pct': pct_expr_in_group[reference_group],
        f'{target_group}_pct': pct_expr_in_group[target_group]
    })
    # Save if requested
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        out_file = os.path.join(savedir, f"{prefix}_{target_group}_vs_{reference_group}.csv")
        de_results.to_csv(out_file, sep=",", index=False)
        print(f"Saved DE table to {out_file}")
    return de_results


# savedir = "/mnt/data/projects/all_cancer/VB_foundation_model/colon_HNSCC_HCC/Table/DE_wilcoxon"
# de_HCC = run_sc_de(
#     adata=HCC,
#     group_col='subtypes',
#     subset_celltypes = ['mregDC','cDC1','cDC2'],
#     target_group='mregDC',
#     reference_group='cDC1_cDC2',
#     method='wilcoxon',
#     savedir=savedir,
#     prefix='HCC'
# )

