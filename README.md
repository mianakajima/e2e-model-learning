# e2e-model-learning
Reproduce paper "Task-based End-to-end Model Learning in Stochastic Optimization" by Donti et al

I write about my findings [here](https://mianakajima.github.io/projects/task-based-ML/)! 

Repository organization: 
- `data`: contains data from reproduced paper and cleaned version in `processed_data` folder
- `figures`: generated figures during training
- `notebooks`: Jupyter notebooks for exploration 
- `results`: result outputs from model training
  - metrics.csv has training and test RMSE, accuracy and task loss
  - hourly_results.csv has test RMSE, accuracy and task loss averaged per hour
  - pt files are the saved pytorch models
  - csv files ending with "predictions" are the loads predicted by the model 
- `scripts`: contains all scripts for training models
  - main.py: This is the main file that needs to be run. 
  - model_setup.py: Has hyperparameters of model
  - models.py: Pytorch models defined for neural network and optimization problem.
  - tests.py: Scratch file for testing code
  - train_eval.py: Helper functions for training and testing models. 
  
  
Use `conda env create -f environment.yml` to create a conda environment with given yml file.
