#!/usr/bin/env python3
# read_paint_preds.py

import os
import h5py
import numpy as np

def read_hdf5_in_chunks(h5_file, dataset_names, chunk_size=1000000):
    """
    Generator to read specified datasets from an HDF5 file in chunks.

    Parameters:
    - h5_file (str): Path to the HDF5 file.
    - dataset_names (list of str): Names of datasets to read.
    - chunk_size (int): Number of samples to read per chunk.

    Yields:
    - dict: A dictionary where keys are dataset names and values are NumPy arrays.
    """
    with h5py.File(h5_file, 'r') as f:
        # Verify that all requested datasets exist
        for name in dataset_names:
            if name not in f:
                raise KeyError(f"Dataset '{name}' not found in '{h5_file}'.")

        # Get the total number of samples from the first dataset
        total_samples = f[dataset_names[0]].shape[0]
        print(f"Total number of samples: {total_samples}")

        # Calculate the number of chunks
        num_chunks = (total_samples + chunk_size - 1) // chunk_size
        print(f"Reading data in {num_chunks} chunks of up to {chunk_size} samples each.")

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_samples)
            data = {}
            for name in dataset_names:
                data[name] = f[name][start_idx:end_idx]
            yield data
            print(f"Yielded chunk {i + 1}/{num_chunks}: samples {start_idx} to {end_idx}")

def main():
    # Path to the HDF5 file
    target_skypatch_id = 164  # Adjust as needed
    h5_file = f"./paint_models/paint_preds_skypatch_{target_skypatch_id}.h5"
    print('Paint h5 file read', h5_file)

    # Datasets to read
    datasets = ['SED', 'core_tag']

    # Define chunk size (number of samples per chunk)
    chunk_size = 1000000  # Adjust based on available memory

    # Check if the file exists
    if not os.path.isfile(h5_file):
        raise FileNotFoundError(f"The file '{h5_file}' does not exist.")

    # Example usage: Iterate over chunks and process them
    for chunk_idx, data_chunk in enumerate(read_hdf5_in_chunks(h5_file, datasets, chunk_size)):
        sed = data_chunk['SED']          # Shape: (chunk_size, num_features)
        core_tag = data_chunk['core_tag']  # Shape: (chunk_size,)

        # Example processing: Print shapes and some statistics
        print(f"\nProcessing Chunk {chunk_idx + 1}:")
        print(f"SED shape: {sed.shape}")
        print(f"core_tag shape: {core_tag.shape}")
        
        '''
        # Example: Compute mean and standard deviation of SED for the chunk
        sed_mean = np.mean(sed, axis=0)
        sed_std = np.std(sed, axis=0)
        print(f"SED Mean (first 5 features): {sed_mean[:5]}")
        print(f"SED Std Dev (first 5 features): {sed_std[:5]}")

        # Example: Count unique core_tags in the chunk
        unique_tags, counts = np.unique(core_tag, return_counts=True)
        print(f"Unique core_tags in this chunk: {unique_tags}")
        print(f"Counts per core_tag: {counts}")
        '''

        # TODO: Add your custom processing logic here
        # For example, saving processed data, aggregating results, etc.

    print("\nFinished reading all chunks.")

if __name__ == '__main__':
    main()
