# SpaIM : Single-cell Spatial Transcriptomics Imputation via Style Transfer

Spatial transcriptomics (ST) technologies have revolutionized our understanding of cellular ecosystems. However, these technologies face challenges such as sparse gene signals and limited gene detection capacities, which hinder their ability to fully capture comprehensive spatial gene expression profiles. To address these limitations, we propose leveraging single-cell RNA sequencing (scRNA-seq), which provides comprehensive gene expression data but lacks spatial context, to enrich ST profiles. Herein, we introduce SpaIM, an innovative style transfer learning model that utilizes scRNA-seq information to predict unmeasured gene expressions in ST data, thereby improving gene coverage and expressions. SpaIM segregates scRNA-seq and ST data into data-agnostic contents and data-specific styles, with the contents capture the commonalities between the two data types, while the styles highlight their unique differences. By integrating the strengths of scRNA-seq and ST, SpaIM overcomes data sparsity and limited gene coverage issues, making significant advancements over existing methods. This is demonstrated across 53 diverse ST datasets, encompassing sequencing-based and imaging-based technologies in various tissue types. Furthermore, SpaIM is released as open-source software, enhancing its accessibility and applicability for spatial transcriptomics data analysis. In summary, SpaIM represents a novel approach to enrich spatial transcriptomics with scRNA-seq data, enabling precise gene expression imputation and pushing the boundaries of spatial transcriptomics research.

# Model Architecture
![workflow](./Fig.1.png)


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
    |-- nano5-6
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

The trained model and metric results are available in the defined folders:
./checkpoint_SpaIM for the benchmark datasets;
./results/checkpoint_SpaIM for the nano datasets.



