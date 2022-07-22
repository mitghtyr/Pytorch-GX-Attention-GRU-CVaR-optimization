# Pytorch-GX-Attention-GRU-CVaR-optimization
code for implement of the dynamic allocation with GX-GRU network

## Two step interative algorithm for GX-GRU network (Pytorch)

* implement with Pytorch platform
* code is available in the folder: `GRU_network`
    * exit the Naive-GRU with: `nohup python -u GRU_network/naive_dynamic_GRU_one_ahead_month_attention.py>data/gru_save/naive_print_one_ahead_month_attn.log >&1 &`
    * exit the GX-GRU with: `nohup python -u GRU_network/gx_dynamic_GRU_one_ahead_month_attn.py>data/gru_save/gx_print_one_ahead_month_attn.log >&1 &`
* Output in the folder: `data/gru_save`

## CVaR optimization with generative sampling (MATLAB)

* implement with MATLAB platform 

* solve CVaR optimization with `optimization` toolbox provided by MathWorks <https://www.mathworks.com/products/optimization.html>

* code is available in the folder: `CVaR_optimization`
    * perform CVaR optimization for Naive-GRU with: `nohup ./matlab -nosplash -nodisplay -nodesktop < CVaR_optimization/mat_dynamic_optim_month.m > data/cvar_ouput/result_NN_month_time_att.log 2>&1 &`
    * perform CVaR optimization for GX-GRU with: `nohup ./matlab -nosplash -nodisplay -nodesktop < CVaR_optimization/mat_dynamic_optim_month_gx.m > data/cvar_ouput/result_gx_month_time_att.log 2>&1 &`
    * the training and CVaR optimization for DCC model is finished with
        * DCC-DM: `nohup ./matlab -nosplash -nodisplay -nodesktop < CVaR_optimization/Dcc_fitting_month.m > data/cvar_ouput/result_dcc_dm.log 2>&1 &`
        * DCC-MM: `nohup ./matlab -nosplash -nodisplay -nodesktop < CVaR_optimization/DCC_fitting_month_to_month.m > data/cvar_ouput/result_dcc_mm.log 2>&1 &`
        * Note that the fitting of the DCC model is complished with the MFE toolbox <https://www.kevinsheppard.com/code/matlab/mfe-toolbox/>
* output is in the `data/cvar_output` folder
    * final evaluation results for Naive-GRU: `result_NN_month_time_att.log`
    * final evaluation results for GX-GRU: `result_gx_month_time_att.log`
    * final evaluation results for DCC-DM: `result_dcc_dm.log`
    * final evaluation results for DCC-MM: `result_dcc_mm.log`

## Visualization and evaluation (jupyter-notebook)

* we provide the visualization code in a jupyter-notebook
* the notebook is at `visualization`
* those saved figures are located at `data/visualization_output`
