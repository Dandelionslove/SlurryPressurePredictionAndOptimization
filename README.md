# Prediction and Optimization of Slurry Pressure in the Shield Tunneling by Ensemble Learning Algorithms and Greedy Search Strategy

Running environment:

```shell
python==3.7.0
Flask==1.1.1
Flask-APScheduler==1.12.1
joblib==0.13.2
numpy==1.16.5
pandas==0.25.1
requests==2.22.0
scikit-learn==0.22.2.post1
scikit-opt==0.6.1
```

The data collected by interfaces and aranged by rings are saved in ***data_ring_extracted_by_day/***.

The pre-processed data are in ***preprocessed_data/t60_s5/***. And the script ***data_preparation.py*** is for data engineering for model training and test. 

To adjust the hyper-parameters and evaluate the models, run the following scripts:

```
# Decision tree, Random Forest, GBDT, XGBoost, LightGBM

python script ml_models.py

# Back-propagation neural network

python script bpnn_model.py

# Multi-linear regression and Stacking model.

python script ml_model_linear_stack.py
```

The hyper-parameter optimization results will be saved in the dir ***gscv_results/t60_s5_k6/***. The evaluation results along with the fitting figures for each model on the test set will be saved in the dir ***test_results/t60_s5_k6/***. The optimal model will be saved in the dir ***models/t60_s5_k6/***.

The optimization process can be executed by running the script ***optimization_process.py***, and the pre-requisite for running this script is that the optimal model is abtained in the dir ***models/t60_s5_k6/***.

