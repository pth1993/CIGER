CIGER - Chemical-induced Gene Expression Ranking


# CIGER - Chemical-induced Gene Expression Ranking
-----------------------------------------------------------------
Code by **Thai-Hoang Pham** at Ohio State University.

## 1. Introduction
**CIGER** is a Python implementation of the neural network-based model that predicts the rankings of genes in the whole 
chemical-induced gene expression profiles given molecular structures.

The experimental results show that CIGER significantly outperforms existing methods in both ranking and classification 
metrics for gene expression prediction task. Furthermore, a new drug screening pipeline based on CIGER is pro-posed to 
select potential treatments for pancreatic cancer from the DrugBank database, thereby showing the effectiveness of 
**CIGER** for phenotypic compound screening of precision drug discovery in practice.

## 2. CIGER architecture

![alt text](docs/fig1.png "CIGER")

Figure 1: Overall architecture of **CIGER**

## 3. Pipeline

![alt text](docs/fig2.png "Pipeline")

Figure 2: Drug screening pipeline using **CIGER**. This model is trained with the LINCS L1000 dataset to learn the 
relation between gene expression profiles and molecular structures (i.e., SMILES). Then molecular structures retrieved 
from the DrugBank database are put into **CIGER** to generate the corresponding gene expression profiles. Finally, 
these profiles are compared with disease profiles calculated from treated and untreated samples to find the most 
potential treatments for that disease.

## 4. Installation

**CIGER** depends on Numpy, SciPy, PyTorch (CUDA toolkit if use GPU), scikit-learn, and RDKit.
You must have them installed before using **CIGER**.

The simple way to install them is using conda:

```sh
	$ conda install numpy scipy scikit-learn rdkit pytorch
```
## 5. Usage

### 5.1. Data

The datasets used to train **CIGER** are located at folder ``CIGER/data/``

### 5.2. Training CIGER

The training script for **CIGER** is located at folder ``CIGER/``

```sh
    $ cd CIGER
    $ bash train.sh
```

## 6. Contact

**Thai-Hoang Pham** < pham.375@osu.edu >

Department of Computer Science and Engineering, Ohio State University, USA
