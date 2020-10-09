# Model Test

Build a comprehensive benchmark of DeepCTR applied on an extensive list of freely available CTR datasets.

## Disclaimer 
This is work in progress. API will change significantly.

## Architecture and main concepts: 
---
There are 3 main concepts in the ModelTest: the datasets, the paradigm and the evaluation. 

### Datasets
A dataset handle and abstract low level access to the data. the dataset will takes data stored locally, in the format in which they have been downloaded, and will convert them into a DataFrame. 

### Paradigm
A paradigm defines how the raw data will be converted to what ready to be processed by a algorithm. This is a function of the paradigm used, i.e. different preprocessing is necessary for DeepFM vs DIN paradigms.

### Evaluations
An evaluation defines how we go from data to a generalization statistic (AUC score, f-score, accuracy, etc) 
