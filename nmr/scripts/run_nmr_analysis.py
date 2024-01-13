import argparse
import yaml
import h5py
import os
from nmr.analysis.analysis_runner import (
    process_substructure_predictions,
    process_SMILES_predictions,
    run_process_parallel
)
from nmr.analysis.util import (
    intake_data,
)
from nmr.analysis.postprocessing import (
    collate_predictions,
    postprocess_save_SMILES_results,
    postprocess_save_substructure_results
)

def get_args() -> dict:
    parser = argparse.ArgumentParser(description='Run NMR analysis')
    parser.add_argument('config_file', type = str, help = 'The path to the YAML configuration file')
    args = parser.parse_args()
    listdoc = yaml.safe_load(open(args.config_file, 'r'))
    return (
        listdoc['global_args'],
        listdoc['analysis']
    )

def main() -> None:
    print("Parsing arguments...")   
    global_args, analysis_args = get_args()
    file_handles = intake_data(analysis_args['savedir'], analysis_args['pattern'])
    all_sets = file_handles[0].keys()

    if analysis_args['analysis_type'] == 'substructure':
        print("Analyzing substructure results")
        result_dict = {}
        for set_name in all_sets:
            selected_handles = [f[set_name] for f in file_handles]
            #Gather predictions together because the metrics are computed over all predictions
            #   and targets together
            collated_targets, collated_predictions = collate_predictions(selected_handles)
            result_dict[set_name] = process_substructure_predictions(collated_predictions,
                                                                     collated_targets)
        postprocess_save_substructure_results(global_args['savedir'],
                                              result_dict)
    
    elif analysis_args['analysis_type'] == 'SMILES':
        print("Analyzing SMILES results")
        with h5py.File(os.path.join(global_args['savedir'], 'processed_predictions.h5'), 'w') as f:
            for set_name in all_sets:
                selected_handles = [f[set_name] for f in file_handles]
                curr_results = run_process_parallel(process_SMILES_predictions,
                                                    analysis_args['f_addn_args'],
                                                    selected_handles,
                                                    os.cpu_count())
                postprocess_save_SMILES_results(f, set_name, curr_results)

if __name__ == "__main__":
    main()