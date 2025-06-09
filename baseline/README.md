
# Compared Methods  

Following the *SpatialBenchmarking* [[paper](https://github.com/QuKunLab/SpatialBenchmarking)], we can obtain the [[test_pipeline.py](./test_baseline.py)] for comparison. After running it, you will get the results of Tangram, SpaGE, gimVI, novoSpaRc, SpaOTsc, and stPlus. 

### Additional Comparison Methods  
For the remaining four comparison methods, execute the following scripts:  
- **SpatialScope**: `python ./test_spscope.py` [[source]](./test_spscope.py)  
- **SPRITE**: `python ./test_sprite.py` [[source]](./test_sprite.py)  
- **TISSUE**: `python ./test_TISSUE.py` [[source]](./test_TISSUE.py)  
- **stDiff**: `python ./test_stDiff.py` [[source]](./test_stDiff.py)  

Each script will generate results specific to the respective method. Refer to the individual script files for additional parameters or dependencies.


## Data Sources and Code Availability  

### 1. Methods from the *SpatialBenchmarking* Paper  
The code and data for the following methods can be found in the paper:  
**《Benchmarking spatial and single-cell transcriptomics integration methods for transcript distribution prediction and cell type deconvolution》**  
(https://github.com/QuKunLab/SpatialBenchmarking).

### 2. Recent Methods from Official Repositories  
For newer methods, access their official GitHub repositories for code and data, and follow the `README` instructions to reproduce results:  
- **stDiff**: https://github.com/fdu-wangfeilab/stDiff
- **TISSUE**: https://github.com/sunericd/TISSUE  
- **SPRITE**: https://github.com/sunericd/SPRITE  
