ScRAT: Single-cell RNA (scRNA-seq) ATtention Network for clinical phenotype prediction
==========
Early Phenotype Prediction using scRNA-seq Data from Limited Number of Samples and Minimal Dependency of Cell-type Annotations
tallation](#)
# Table of contents
1. [Installation](#installation)
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
* scRNA-seq sample Matrices
* Metadata for cells

## Test Run
Please download datasets from: https://figshare.com/projects/ScRAT_Early_Phenotype_Prediction_From_Single-cell_RNA-seq_Data_using_Attention-Based_Neural_Networks/151659
You can also get the raw data from the original papers listed in our paper.
```
mkdir data
cd data
wget https://figshare.com/projects/ScRAT_Early_Phenotype_Prediction_From_Single-cell_RNA-seq_Data_using_Attention-Based_Neural_Networks/151659
```

### Sample Input Datasets

### Demo
```
bash run.sh
```

## Contact
Please contact us via yuzhenm@sfu.ca if you have any problem when running the code.

