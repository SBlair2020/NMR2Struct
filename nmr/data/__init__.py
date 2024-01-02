from .build_dataset import create_dataset

def get_input_generators():
    all_input_generators = [
        'SubstructureRepresentationOneIndexed',
        'SubstructureRepresentationBinary'
    ]
    print(all_input_generators)

def get_target_generators():
    all_target_generators = [
        'SMILESRepresentationTokenized',
        'SubstructureRepresentationBinary',
        'SubstructureRepresentationOneIndexed',
    ]
    print(all_target_generators)