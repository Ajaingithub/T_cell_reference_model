import os
import gc
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity

def run_scanvi_label_transfer(
    query,
    ref,
    cancer_name,
    save_dir,
    exclude_genes_path="/mnt/data/resources/genes/exclude_genes",
    min_cells=100,
    max_epochs=100,
    label_key="subtypes"
):
    """
    SCANVI label transfer from reference to query dataset.
    Parameters:
    - query_path: str, path to query h5ad file
    - ref_path: str, path to reference h5ad file
    - cancer_name: str, label for saving models and outputs
    - save_dir: str, output directory for models and results
    - exclude_genes_path: str, file path to list of genes to exclude
    - min_cells: int, minimum number of cells per sample to keep
    - max_epochs: int, number of training epochs for query adaptation
    - label_key: str, the column name for labels in reference (default: "subtypes")
    """
    os.makedirs(save_dir, exist_ok=True)
    os.chdir(save_dir)
    # Set plotting and precision options
    sc.settings.set_figure_params(dpi=200, frameon=False)
    torch.set_printoptions(precision=3, sci_mode=False)
    torch.set_float32_matmul_precision('medium')
    scvi.settings.data_loading_num_workers = 8
    intersected_genes = ref.var_names.intersection(query.var_names).tolist()
    excluded_genes = pd.read_csv(exclude_genes_path, header=None)[0].tolist()
    filtered_genes = [g for g in intersected_genes if g not in excluded_genes]
    ref = ref[:, filtered_genes].copy()
    ref.write(os.path.join(save_dir, f"HCC_{cancer_name}_reference.h5ad"))
    gc.collect()
    print(f"Reference filtered to {len(filtered_genes)} genes")
    print("[3] Training SCVI on reference...")
    sca.models.SCVI.setup_anndata(ref, batch_key='samples')
    vae = sca.models.SCVI(ref, n_layers=2, encode_covariates=True, deeply_inject_covariates=False,
                          use_layer_norm="both", use_batch_norm="none")
    vae.train()
    ref_model_path = f'HCC_ref_model_remove_batch_CRC_query_{cancer_name}_genes'
    vae.save(ref_model_path, overwrite=True)
    print("[4] Training SCANVI on reference...")
    scanvi_model = sca.models.SCANVI.from_scvi_model(vae, unlabeled_category="Unknown", labels_key=label_key)
    scanvi_model.train(max_epochs=20, batch_size=1024, precision="16-mixed")
    scanvi_model_path = f'{ref_model_path}_scanvi'
    scanvi_model.save(scanvi_model_path, overwrite=True)
    print("[5] Preparing query for label transfer...")
    query = query[:, filtered_genes].copy()
    gc.collect()
    model = sca.models.SCANVI.load_query_data(
        query,
        scanvi_model_path,
        freeze_dropout=True
    )
    model._unlabeled_indices = np.arange(query.n_obs)
    model._labeled_indices = []
    print(f"Labelled Indices: {len(model._labeled_indices)}, Unlabelled Indices: {len(model._unlabeled_indices)}")
    print("[6] Training model on query data...")
    model.train(
        max_epochs=max_epochs,
        plan_kwargs=dict(weight_decay=0.0),
        check_val_every_n_epoch=10
    )
    print("[7] Saving and writing predictions...")
    query_latent = sc.AnnData(model.get_latent_representation())
    query_latent.obs['samples'] = query.obs['samples'].tolist()
    query_latent.obs['predictions'] = model.predict()
    query_latent.write(os.path.join(save_dir, f"{cancer_name}_latent_{label_key}_predicted.h5ad"))
    query.obs[f'predicted_{label_key}'] = query_latent.obs['predictions'].tolist()
    query.obs[f'predicted_{label_key}'] = pd.Categorical(query.obs[f'predicted_{label_key}'])
    query.write(os.path.join(save_dir, f"{cancer_name}_adata_{label_key}_predicted.h5ad"))
    model.save(os.path.join(save_dir, f"{cancer_name}_HCC_surgery_model_{label_key}"))
    print(f"[✓] Finished. Predictions saved to: {save_dir}")
    return(query)

# query = sc.read_h5ad(query_path)
# query.obs['samples'] = query.obs['sample_name']

# # Filter query based on number of cells per sample
# sample_counts = query.obs['samples'].value_counts()
# valid_samples = sample_counts[sample_counts > min_cells].index
# query = query[query.obs['samples'].isin(valid_samples), :].copy()

# # Set counts layer as X
# if 'counts' in query.layers:
#     query.X = query.layers['counts'].copy()
# else:
#     raise ValueError("Query data missing 'counts' layer.")
# print(f"Filtered query to {query.n_obs} cells across {len(valid_samples)} samples")

# print("[2] Loading and preparing reference...")
# ref = sc.read_h5ad(ref_path)

# To run this
# First load the function from here
# import sys
# sys.path.append('/mnt/data/resources/code/')
# import label_transfer

# ## loading the query and reference dataset
# NSCLC_adata = sc.read_h5ad("/mnt/data/projects/atlas_integration/analysis/NSCLC/analysis/saveh5ad/NSCLC_combined.h5ad")
# ref_adata=sc.read_h5ad("/mnt/data2/projects/HNSCC/analysis/saveh5ad/HCC_adata_CRC_HSNCC_genes.h5ad")
# NSCLC_adata = label_transfer.run_scanvi_label_transfer(
#     query = NSCLC_adata,
#     ref = ref_adata,
#     cancer_name = "NSCLC",
#     save_dir = "/mnt/data/projects/atlas_integration/analysis/NSCLC/analysis/label_transfer/",
#     exclude_genes_path="/mnt/data2/resources/exclude_genes",
#     min_cells=100,
#     max_epochs=100,
#     label_key="subtypes"
# )
