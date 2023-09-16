# Explainable proportion estimation for DNA analysis

This project includes a python system to classify a dataset of DNA reads (either ``.bam`` or ``.fasta`` files) and output the estimated proportion of different species in the dataset.
It also provides explanations to the output that make it more interpretale and resolve ambiguities in the output and provide further insights.
The 2 notions of explanations the system provides are Counterfactuals, and attribution (using estimation of Shapley values for each DNA sequence and label).

## Getting Started

These instructions will give you a copy of the project up and running on your local machine.

### Prerequisites

Requirements for running the GUI and the system on your **linux machine** (or use [WSL](https://docs.microsoft.com/en-us/windows/wsl/install) on a windows machine): 
- [python](https://www.python.org/)
- [Jupyter notebooks](https://jupyter.org/)
- [biopython](https://biopython.org/)
- [SHAP](https://shap.readthedocs.io/en/latest/index.html)
- [pysam](https://pysam.readthedocs.io/en/latest/api.html)
- [joblib](https://joblib.readthedocs.io/en/latest/)
- [bioconda](https://bioconda.github.io/)
- [termcolor](https://anaconda.org/conda-forge/termcolor)

### Work with the python code

The important methods are:
- Constructor (gets dataset, references) and runs the preProcessing (ComputeDataProbability) of the proportion estimation maximum likelihood algorithm
- ``estimate_proportions`` - estimating the proportions of the dataset (after running the constructor)
- ``estimate_shapley_value_for_read`` - returning for each data point in the dataset an explanation vector of the estimation of the scaled Shapley values (how the data point is influencing the output of the algorithm)
-  ``generateCounterFactualMinimalSetToRemoveAndChangeMax`` - calculating a counter factual - meaning what is the minimal subset that removing it from the dataset would cause the dominant label to change.
-  ``getRankingOfA_d_s_Values`` - to get the estimation of the order of the Shapley values using the A_d_s values 


### Run the project

1. Clone this repo ``git clone https://github.com/Amitbergman/ExplainableProportionEstimationForDNAAnalysis.git``
2. Run ``Jupyter lab`` in root directory.
3. Run the ``Demo`` notebook for a step by step demonstration.
4. Run the ``GUI`` notebook for the GUI that interacts with the system.
5. Run the ``UseCaseRealData`` notebook for a full execution example on real datasets. 
6. Choose a dataset to analyze from the datasets under the data directory.

## Authors
  - **Amit Bergman**
