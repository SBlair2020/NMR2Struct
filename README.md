# Accurate and efficient structure elucidation from routine one-dimensional NMR spectra using multitask machine learning

Official implementation of:

**Accurate and efficient structure elucidation from routine one-dimensional NMR spectra using multitask machine learning**

Frank Hu, Michael S. Chen, Grant M. Rotskoff, Matthew W. Kanan, Thomas E. Markland

**Abstract:** Rapid determination of molecular structures can greatly accelerate workflows across many chemical disciplines. However, elucidating structure using only one-dimensional (1D) NMR spectra, the most readily accessible data, remains an extremely challenging problem because of the combinatorial explosion of the number of possible molecules as the number of constituent atoms is increased. Here, we introduce a multitask machine learning framework that predicts the molecular structure (formula and connectivity) of an unknown compound solely based on its 1D <sup>1</sup>H and/or <sup>13</sup>C NMR spectra. First, we show how a transformer architecture can be constructed to efficiently solve the task, traditionally performed by chemists, of assembling large numbers of molecular fragments into molecular structures. Integrating this capability with a convolutional neural network, we build an end-to-end model for predicting structure from spectra that is fast and accurate. We demonstrate the effectiveness of this framework on molecules with up to 19 heavy (non-hydrogen) atoms, a size for which there are trillions of possible structures. Without relying on any prior chemical knowledge such as the molecular formula, we show that our approach predicts the exact molecule 69.6% of the time within the first 15 predictions, reducing the search space by up to 11 orders of magnitude.

Paper available at: https://pubs.acs.org/doi/10.1021/acscentsci.4c01132

Data available on Zenodo at: https://zenodo.org/records/13892026

# Installation
We recommend having Anaconda installed as your package manager. To install NMR2Struct, do the following:
```
git clone https://github.com/MarklandGroup/NMR2Struct.git
cd NMR2Struct
conda env create -f environment.yml
conda activate NMR_env
pip install -e .
```

Make sure to run ``conda update conda`` before installing to ensure your anaconda distribution is up to date. 

