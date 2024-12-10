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

# The configuration file

All of the functionality in NMR2Struct is controlled via a YAMl configuration file system. A YAML file consists of sections with fields that can have different user-specified values. While most fields in a YAML file is populated with strings, fields can also be set to booleans (```True``` and ```False```), integers, floats, lists of values, and None types with ```null```. The YAML files in NMR2Struct have the following sections:
- **global_args**: Controls general arguments such as the seed, number of GPUs, where to save everything, and tensor datatypes
- **data**: Determines how input and target data are processed for training and inference, and where to find data files
- **model**: Model architecture arguments that are used for instantiating the model
- **training**: Arguments for the optimization protocol, such as the data split, number of epochs, optimizer, etc.
- **analysis**: Arguments for computing performance metrics on the model 

For the ```data``` section specifically, the ```spectra_file```, ```label_file```, and ```smiles_file``` fields contain lists of files and there should be the same number of elements in each list, even if some elements are set to ```null```. There is an element-wise correspondence of the elements in each list, so the first file spectrum, label, and smiles file together forms one dataset, and so on. This is designed so that the training and losses can be computed separately for each dataset and is useful for more advanced early-stopping experiments. The files should have the following formats:
- **spectra**: These should be hdf5 files that contain one dataset called 'spectra' where each row is a one-dimensional vector representing the concatenated <sup>1</sup>H NMR and <sup>13</sup>C NMR spectrum.
- **label**: These should beb hdf5 files that contain one dataset called 'substructure_labels' where each row is a one-dimensional binary vector representing the substructures detected in each molecule.
- **smiles**: This should be a numpy file that contains a single array with all SMILES strings encoded into binary format. Be sure to do ```x.decode('utf-8')``` when reading strings from this format.


# Training 
To train a substructure-to-structure transformer, follow these steps:
1. Modify the training configuration file under ```example_configs/training_substruct_to_struct.yaml``` to point to the correct substructure file under the ```label_file``` field and the correct smiles file under the ```smiles_file``` field in the ```data``` section. The ```spectra_file``` field can be set to a list of ```null``` values because spectra are not used for training the substructure-to-structure model.
2. Modify the config file to point to the correct splitting file under the ```splits``` field in the ```training``` section. The splitting file should be a dictionary of three fields which specify the indices that should go into the training, validation, and testing set, respectively. If you do not want to provide a splitting dictionary, set this field to ```null```. 
3. Copy the training config file into a directory where you want to run the training.
4. Run the following command (assuming you named your yaml file to be ```config.yaml```):
```
nmr_train config.yaml
```

A similar procedure is used for training a spectrum-to-structure multitask model, but make sure to use the ```example_configs/training_spectrum_to_struct.yaml``` file as the starting point and set the ```spectra_file``` field to list the correct spectra files.

# Inference (with full dataset)
To infer a substructure-to-structure transformer for SMILES generation, first make sure to set the alphabet correctly with:
```yaml
inference:
  ...
  run_inference_args: 
    ...
    pred_gen_opts:
      ...
      alphabet: checkpoints/alphabet.npy
```
which is the alphabet generated from training. 

> [!NOTE]
> If you want to use the alphabet that was used in the original model from the paper, put the path to the file ```example_configs/alphabet.npy``` instead.

Once the alphabet is set, just run:
```
nmr_infer config.yaml 0 1
```
where ```config.yaml``` is your modified training script from the training section which should contain an inference section tailored for the substructure-to-structure transformer. The 0 and 1 refer to the local rank and total number of GPU processes and is used to control multi-GPU inference. For most cases, inference on one GPU is sufficient, so we use a local rank of 0 and 1 GPU process. Make sure to set the ```splits``` field correctly under the ```inference``` section. The same procedure holds for inferring SMILES from a spectrum-to-structure multitask model. By default, the yaml files are configured to have the models generate SMILES strings. 

To infer substructures from the multitask model, modify the inference section as follows, being sure to fill in the splits file which denotes which parts of the dataset belong to the test set:
```yaml
inference:
  model_selection: lowest
  splits: 
    - PATH TO SPLITS FILE OR NULL
  train_size: 0.8 
  val_size: 0.1 
  test_size: 0.1 
  dloader_args:
    shuffle: false
    batch_size: 1024
  sets_to_run: ['test']
  run_inference_args: 
    pred_gen_fn: 'infer_multitask_model_substructures'
    pred_gen_opts:
      track_gradients: true
    write_freq: 100
```
Running inference writes the raw predictions as an hdf5 file to the save directory that you specified under the ```global_args``` section, which should also be where the model checkpoints and tensorboard files are saved.

> [!WARNING]
> This inference workflow is intended for running inference on a model on a dataset that was used for training, i.e. getting predictions over a test set to then compute metrics on. If you want to infer a single example, please refer to the section on inference with specific examples.

# Inference (on a specific example)
To infer a specific example, 

# Analysis
If you want to calculate metrics for substructures, configure the ```analysis``` section of your YAML file as follows:
```yaml
analysis:
  analysis_type: "substructure"
  pattern: "predictions_dataset_0_[0-9]+.h5"
```
This will calculate metrics such as RMSEs, F1-scores, BCE values, etc. 

For SMILES analysis, configure the ```analysis``` section as follows:
```yaml
analysis:
  analysis_type: "SMILES"
  pattern: "predictions_dataset_0_[0-9]+.h5"
  f_addn_args: 
    substructures: PATH TO SUBSTRUCTURE FILE
```
where the substructures is set to point to the ```example_configs/substructures_957.pkl``` file which contains the SMART strings for the 957 substructures used. 

> [!WARNING]
> Analysis is only useful when you are training a model on a dataset with known targets and you want to compare its predictions against the target values. If you are just doing a one-off inference using the model, analysis will not work. 

# Tensorboard visualization
NMR2Struct uses Tensorboard to visualize the learning curves when training the models, and it is installed with the environment. To use tensorboard, do the following:
```
cd /path/to/training/directory
tensorboard --logdir=./
```
This should automatically do port-forwarding to your browser where you will see the Tensorboard dashboard.


