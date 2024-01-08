# NMR to Chemical Formula 

## Usage info
The following entry points are implemented:
- `nmr_train`: Calls the main() method for training and requires a YAML config file input
- `nmr_infer`: Calls the main() method for inference and requires a YAML config file input
- `available_models` : Prints out a list of currently implemented models
- `input_formats` : Prints out a list of implemented input formats for data
- `target_formats` : Prints out a list of implemented target formats for data
- `network_components` : Prints out a summary of the available source and target embeddings and the source and target forward functions used by the Transformer
