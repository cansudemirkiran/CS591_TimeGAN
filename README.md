# CS591_TimeGAN
CS591 Deep Learning Course Project

Author: Cansu Demirkiran
cansu@bu.edu

Reimplementation of Time Series Generative Adversarial Network by Yoon et al. 
Source: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

**How to run:**

There are two ways of running the code. 

**1. Run in your local or remote server**

`Clone the repo to your local directory`

`python -m venv timegan_venv`

`source timegan_venv/bin/activate`

`pip install -r requirements.txt`

`jupyter notebook timegan_notebook`

**2. Run in Google Colab**

Clone the repo and move repo to drive

-------------------------------------------

**Explanation for files**

timegan_notebook.ipynb -> Main Colab notebook file to call necessary functions and run the model. (Please refer to information above for running)

tgan.py -> Training file. It takes real data and paramteters as input and outputs the synthetic data and stored loss values

data_load.py ->Include functions for loading datasets, generating sine wave dataset

test.py -> Sequence predictors. Has two functions, one for the model trained with real and one for the model trained with synthetic data. Outputs mean absolute error values (MAE). 

metrics -> t-SNE visualization code. Taken from original implementation of TimeGAN

plots -> Loss plots for different datasets
