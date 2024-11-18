# SpaIM : Single-cell Spatial Transcriptomics Imputation via Style Transfer

Spatial transcriptomics (ST) technologies have revolutionized our understanding of cellular ecosystems. However, these technologies face challenges such as sparse gene signals and limited gene detection capacities, which hinder their ability to fully capture comprehensive spatial gene expression profiles. To address these limitations, we propose leveraging single-cell RNA sequencing (scRNA-seq), which provides comprehensive gene expression data but lacks spatial context, to enrich ST profiles. Herein, we introduce SpaIM, an innovative style transfer learning model that utilizes scRNA-seq information to predict unmeasured gene expressions in ST data, thereby improving gene coverage and expressions. SpaIM segregates scRNA-seq and ST data into data-agnostic contents and data-specific styles, with the contents capture the commonalities between the two data types, while the styles highlight their unique differences. By integrating the strengths of scRNA-seq and ST, SpaIM overcomes data sparsity and limited gene coverage issues, making significant advancements over 12 existing methods. This improvement is demonstrated across 53 diverse ST datasets, spanning sequencing- and imaging-based spatial technologies in various tissue types. Additionally, SpaIM enhances downstream analyses, including the detection of ligand-receptor interactions, spatial domain characterization, and identification of differentially expressed genes. Released as open-source software, SpaIM increases accessibility for spatial transcriptomics analysis. In summary, SpaIM represents a pioneering approach to enrich spatial transcriptomics using scRNA-seq data, enabling precise gene expression imputation and advancing the field of spatial transcriptomics research.

# Model Architecture

![workflow](./Fig.1.png)
To accurately impute unmeasured gene expressions in spatial transcriptomics (ST) data, we introduce SpaIM, a novel style transfer learning model leveraging scRNA-seq (SC) data. As illustrated in Fig.1a, SpaIM is a multilayer Recursive Style Transfer (ReST) model with layer-wise content- and style-based feature extraction and fusion. Specifically, SpaIM comprises an ST autoencoder (Fig.1b) and an ST generator (Fig.1c). For a single gene, we consider the gene expression pattern across the K single cell clusters as its content, and the unique gene expression pattern across all cells in ST data, which differs from SC data, as its style. The style represents the intrinsic differences in gene expressions between the ST and the SC platforms. The style-transfer learning of SpaIM involves two simultaneous tasks: the ST autoencoder uses the SC data as the reference to disentangle the ST gene expression patterns into content and style, and the ST generator extracts the content from the SC data and transfers the learned style from ST autoencoder to infer ST gene expressions. The ST autoencoder and the ST generator share the same decoder and are co-trained using a joint loss function based on the common genes between ST and SC data. This allows the ST generator to capture the gene expression patterns in the ST data as well as the relation between the ST and SC data. After training, the ST generator is used as a stand-alone model to infer the expression patterns of unmeasured genes in the ST data, using only the SC data as input. In this way, the well trained SpaIM model enables accurate predictions of unmeasured gene expressions, through leveraging the comprehensive gene expression profiles of scRNA-seq and the optimized ST generator.

# Getting Started

## Environment

The required environment has been packaged in the [`requirements.txt`](./requirements.txt) file. Please run the following command to install.

```commandline
git clone https://github.com/QSong-github/SpaIM
cd SpaIM
pip install -r requirements.txt
```

## Datasets

All datasets used in this study are publicly available. 

- Data sources and details are provided in [`Supplemental_Table_1`](./Supplemental_Table_1.xlsx). 

- The NanoString CosMx SMI datasets are available from [https://nanostring.com/products/cosmx-spatial-molecular-imager/nsclc-ffpe-dataset/](https://nanostring.com/products/cosmx-spatial-molecular-imager/nsclc-ffpe-dataset/). 

The datasets structure should be as follows:
```
|-- dataset
    |-- benchmark_datasets
        |-- Dataset1
        |-- Dataset2
        |-- ......
        |-- Dataset44
        |-- Dataset45
    |-- nano5-1
    |-- nano5-2
    |-- nano5-3
    |-- nano6
    |-- nano9-1
    |-- nano9-2
    |-- nano12
    |-- nano13
```

## SpaIM Training and Testing

```
# Train both the benchmark and nano datasets.
./run_SpaIM.sh
```

The trained models and metric results are available in the defined folders:
```
./checkpoint_SpaIM  # for benchmark datasets
./results/checkpoint_SpaIM   # for nano datasets
```



