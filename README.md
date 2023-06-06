ScRAT: Single-cell RNA (scRNA-seq) ATtention Network for clinical phenotype prediction
==========
Early Phenotype Prediction using scRNA-seq Data from Limited Number of Samples and Minimal Dependency of Cell-type Annotations

# Table of contents
1. [Setup](#setup)
2. [Input Data Format](#input-data-format)
3. [Test Run](#test-run)
4. [Contact](#contact)

## Setup (python>=3.7)

```
git clone https://github.com/yuzhenmao/ScRAT
cd ScRAT
python -m venv scrat
source scrat/bin/activate
pip install -r requirements.txt
```

## Input Data Format
ScRAT requires the following information.
1) scRNA-seq sample Matrices (row major)
2) Metadata for cells: patient id, phenotype labels, cell population

* ### For dataset mentioned in our paper:
Please download datasets from: https://figshare.com/projects/ScRAT_Early_Phenotype_Prediction_From_Single-cell_RNA-seq_Data_using_Attention-Based_Neural_Networks/151659, and put the datasets under `ScRAT/data`. The folder should have the following content:
```
ScRAT/data/
├── SC4
│   ├── cell_type_large.pkl
│   ├── cell_type.pkl
│   ├── covid_pca.npy
│   ├── patient_id.pkl
│   ├── severity_label.pkl
│   └── stage_label.pkl
├── Haniffa
│   ├── cell_type_large.pkl
│   ├── cell_type.pkl
│   ├── Haniffa_X_pca.npy
│   ├── patient_id.pkl
│   └── labels.pkl
├── COMBAT
│   ├── cell_type_large.pkl
│   ├── cell_type.pkl
│   ├── COMBAT_X_pca.npy
│   ├── patient_id.pkl
│   └── labels.pkl

```
(You can also get the raw data from the original papers listed in our paper.)

* ###  For customized dataset:
Please pack the dataset in the h5ad format and set the value of args.dataset to the path of the dataset. Also, please set args.task to 'custom'.

Furthermore, in the dataloader.py file, please modify the following lines: 
  1) line 178 for label dictionary to map string to integer (default: {})
  2) line 185 for patient id (default: data.obs['patient_id'])
  3) line 187 for label, which is clinical phenotype for prediction (default: data.obs['Outcome']) 
  4) line 189 for cell type, which assist for mixup (default: data.obs['cell_type']) 
  
## Test Run
### Demo
```
bash run.sh
```

You can modify the hyper-parameters in `run.sh`.
- task: `stage`, `severity`, `haniffa`, `combat`, `custom`

### Output Template
`out.txt`

## Contact
Please contact us via yuzhenm@sfu.ca if you have any question when running the code.

