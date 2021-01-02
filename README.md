# Simulations for multi-view mixture modeling paper


The code in this repository reproduces the results of ["Learning Sparsity and Block Diagonal Structure in Multi-View
Mixture Models"](https://arxiv.org/pdf/2012.15313.pdf).


To run all the simulations do the following (note you have to write in a couple hard coded paths for where simulation output data should be saved!)

```
git clone https://github.com/idc9/mvmm_sim@SOMETAG
pip install .

# before running the simulations you should should change the paths in the following files
# mvmm_sim/simulation/Paths.py
# data_analysis_scripts/tcga_single_view_analysis.sh
# data_analysis_scripts/tcga_mvmm_analysis.sh
# data_analysis_scripts/mouse_et_run_analysis.sh

# run both the synthetic data simulations as well as the real data analysis
sh run_all_simulations.sh
```

For questions or feedback please reach out to Iain!