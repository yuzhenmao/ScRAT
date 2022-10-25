ScRAT: Single-cell RNA (scRNA-seq) ATtention Network for clinical phenotype prediction
==========
Early Phenotype Prediction using scRNA-seq Data from Limited Number of Samples and Minimal Dependency of Cell-type Annotations

# Table of contents
1. [Setup](#setup)
2. [Inout Data Format](#input-data-format)
3. [Test Run](#test-run)
4. [Contact](#contact)

## Setup

```
git clone https://github.com/yuzhenmao/ScRAT
cd ScRAT
python3 -m venv scrat
source scrat/bin/activate
pip install -r requirements.txt
```

## Input Data Format
ScRAT requires the following information.
* scRNA-seq sample Matrices (row major)
* Metadata for cells: patient id, phenotype labels, cell population

Please download datasets from: https://figshare.com/projects/ScRAT_Early_Phenotype_Prediction_From_Single-cell_RNA-seq_Data_using_Attention-Based_Neural_Networks/151659. And put the datasets under `ScRAT/data`. The folder should have the following content:
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


You can also get the raw data from the original papers listed in our paper.


## Test Run
### Demo
```
bash run.sh
```

You can modify the hyper-parameters in `run.sh`.
- task: `stage`, `severity`, `haniffa`, `combat`

### Output Template
`out.txt`

## Contact
Please contact us via yuzhenm@sfu.ca if you have any problem when running the code.

