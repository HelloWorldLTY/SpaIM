# SpaIM

## Single-cell Spatial Transcriptomics Imputation via Style Transfer  [[paper]](https://www.biorxiv.org/content/10.1101/2025.01.24.634756v1.full.pdf)

We introduce SpaIM, a novel style transfer learning model that leverages scRNA-seq data to accurately impute unmeasured gene expressions in spatial transcriptomics (ST) data. SpaIM separates scRNA-seq and ST data into data-agnostic contents and data-specific styles, capturing commonalities and unique differences, respectively. By integrating scRNA-seq and ST strengths, SpaIM addresses data sparsity and limited gene coverage, outperforming existing methods across 53 diverse ST datasets. It also enhances downstream analyses like ligand-receptor interaction detection, spatial domain characterization, and differentially expressed gene identification.
![workflow](./data/Fig1.png)

# Getting Started

## Environment

To get started with SpaIM, please follow the steps below to set up your environment:

```commandline
git clone https://github.com/QSong-github/SpaIM
cd SpaIM
conda env create -f environment.yaml
conda activate SpaIM
```

## Datasets

All datasets used in this study are publicly available. 

- Data sources and details are provided in [`Supplemental_Table_1`](./data/Supplementary_Table_1.xlsx). After downloading the data, follow the processing flow in [get_adata_cluster.py](./data/get_adata_cluster.py) to analyse it for clustering.

- All processed datasets and example 'Dataset 1' can be downloaded at [Zenodo](https://zenodo.org/records/14741028) and [Synapse](https://www.synapse.org/Synapse:syn64421787/files/).

The datasets should be organized in the following structure:
```
|-- dataset
    |-- Dataset1
    |-- Dataset2
    |-- ......
    |-- Dataset52
    |-- Dataset53
```

## SpaIM Training and Testing

Train all 53 datasets with a single command:
```
chmod +x ./*
./run_SpaIM.sh
```

The trained models and metric results will be saved in the following directories:
```
./SpaIM_results/Dataset1/
```

## SpaIM Inference

Run the following command to perform inference:
```
cd test
python SpaIM_imputation.py
```
The inference results will will be saved in './SpaIM_results/Dataset1/impute_sc_result_%d.pkl'.

# Reference
If you find this project is useful for your research, please cite:
```
@article{li2025spaim,
  title={SpaIM: Single-cell Spatial Transcriptomics Imputation via Style Transfer},
  author={Li, Bo and Tang, Ziyang and Budhkar, Aishwarya and Liu, Xiang and Zhang, Tonglin and Yang, Baijian and Su, Jing and Song, Qianqian},
  journal={bioRxiv},
  pages={2025--01},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Acknowledgments

Our code is based on the [neural-style](https://github.com/jcjohnson/neural-style). Special thanks to the authors and contributors for their invaluable work.

