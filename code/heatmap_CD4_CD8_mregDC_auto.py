import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

def plot_immune_subtype_heatmaps(
    query,
    cancer_name,
    save_dir,
    layers = "counts",
    group_key = "predicted_subtypes",
    gene_dir="/mnt/data/resources/genes",
    n_cells_per_group_cd4=50,
    n_cells_per_group_cd8=50,
    n_cells_per_group_dc=12,
    CD4_vmax = 0.2,
    CD8_vmax = 0.1,
    DC_vmax = 0.3,
    Boolean_val = False
):
    """
    Generates standardized heatmaps for CD4, CD8, and mregDC predicted subtypes using equal cell sampling.
    Parameters:
        query: AnnData object with predicted_subtypes in `.obs`.
        cancer_name: Label to use in saved file names.
        save_dir: Directory to save heatmaps.
        gene_dir: Directory containing marker gene lists.
        n_cells_per_group_cd4: Number of cells per CD4 subtype.
        n_cells_per_group_cd8: Number of cells per CD8 subtype.
        n_cells_per_group_dc: Number of cells per DC subtype.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.chdir(save_dir)
    sc.settings.figdir = save_dir  # sets where save=True plots will be written

    def make_heatmap(celltypes, gene_file, n_cells_per_group, filename_prefix, extra_genes=None, override_genes=None, vmax=0.2, figsize=(5, 9)):
        print(f"\n[•] Processing: {filename_prefix} heatmap...")
        import re
        # Join all celltypes into a regex pattern, e.g., 'Luminal|Basal|Her2'
        pattern = '|'.join(map(re.escape, celltypes))
        # Do partial matching using .str.contains()
        subset = query[query.obs[group_key].str.contains(pattern, case=False, na=False)].copy()

        # subset = query[query.obs[group_key].isin(celltypes)].copy()

        if subset.n_obs == 0:
            print(f"⚠️ No cells found for {filename_prefix} subtypes. Skipping.")
            return

        # Sample equal number of cells per group
        # group_key = group_key
        inds = []
        for group in subset.obs[group_key].unique():
            group_inds = np.where(subset.obs[group_key] == group)[0]
            sampled = np.random.choice(group_inds, n_cells_per_group, replace=Boolean_val)
            inds.extend(sampled)
        adata_equal = subset[inds].copy()

        # Load marker genes
        if override_genes:
            celltype_marker = override_genes
        else:
            celltype_marker = pd.read_csv(os.path.join(gene_dir, gene_file), header=None)[0].tolist()

        present_genes = list(adata_equal.var_names)
        gene_markers = [g for g in celltype_marker if g in present_genes]

        if extra_genes:
            gene_markers += [g for g in extra_genes if g in present_genes]

        if not gene_markers:
            print(f"⚠️ No marker genes found in {filename_prefix} data. Skipping heatmap.")
            return

        # Plot heatmap
        sc.pl.heatmap(
            adata_equal,
            gene_markers,
            groupby=group_key,
            layer = layers,
            standard_scale="var",
            cmap="Reds",
            vmin=0,
            vmax=vmax,
            dendrogram=False,
            show_gene_labels=True,
            # save=f"_{filename_prefix}_{cancer_name}_gene_markers_equal.pdf",
            swap_axes=True,
            figsize=figsize
        )
        
        plt.suptitle(cancer_name+"_"+filename_prefix, fontsize=15)
        plt.savefig(f"{filename_prefix}_{cancer_name}_{layers}_gene_markers_equal.pdf", bbox_inches='tight')  # Manually save
        plt.close()



    # --- CD4 ---
    CD4_types = ["1-Tfh-like", "2-Th1 GZMK", "3-Th1/Th17", "4-Effector-Memory", "5-Memory"]
    make_heatmap(
        CD4_types,
        gene_file="CD4_Nat_Med_genes.txt",
        n_cells_per_group=n_cells_per_group_cd4,
        filename_prefix="CD4",
        vmax = CD4_vmax,
        extra_genes=["CD3D", "CD3E", "CD4", "CD8A", "CD8B"],
        figsize=(5,8.5)
    )

    # --- CD8 ---
    CD8_types = [
        "1-Dysfunctional-terminal", "2-Dysfunctional-progenitor", "3-Dysfunctional-Proliferating",
        "4-Dysfunctional-effector", "5-Effector", "6-Memory", "7-Cytotoxic"
    ]
    make_heatmap(
        CD8_types,
        gene_file="CD8_Nat_Med_genes.txt",
        n_cells_per_group=n_cells_per_group_cd8,
        filename_prefix="CD8",
        vmax = CD8_vmax,
        extra_genes=["CD3D", "CD3E", "CD4", "CD8A", "CD8B"],
        figsize=(5,10)
    )

    # --- mregDC ---
    DC_types = ["cDC1", "cDC2", "mregDC"]
    mregDC_genes = [
        "CLEC9A","XCR1","CADM1","IRF8","CLEC10A","FCER1A","CD1C","GPR183","CCL17",
        "LAMP3","IL4I1","IDO1","CCR7","FSCN1",
        "RELB","CCL22","CXCL9","CXCL10","CCL19","CD40","CD80","CD86",
        "CD274","PDCD1LG2","PVR","FAS","IL15","IL12B"
    ]
    make_heatmap(
        DC_types,
        gene_file=None,
        n_cells_per_group=n_cells_per_group_dc,
        filename_prefix="DC",
        override_genes=mregDC_genes,
        vmax=DC_vmax,
        figsize=(3, 5.5)
    )

    print("\n✅ All heatmaps generated and saved to:", save_dir)

## To run this
# import importlib
# import heatmap_CD4_CD8_mregDC
# importlib.reload(heatmap_CD4_CD8_mregDC)

# # import heatmap_CD4_CD8_mregDC
# heatmap_CD4_CD8_mregDC.plot_immune_subtype_heatmaps(
#     query = NSCLC_adata,
#     cancer_name = "NSCLC",
#     save_dir = "/mnt/data/projects/atlas_integration/analysis/NSCLC/analysis/label_transfer/",
#     gene_dir="/mnt/data/resources/genes",
#     n_cells_per_group_cd4=1000,
#     n_cells_per_group_cd8=379,
#     n_cells_per_group_dc=200,
#     CD4_vmax = 0.1,
#     CD8_vmax = 0.1,
#     DC_vmax = 0.3
# )
