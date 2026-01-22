import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # Disable HDF5 file locking
os.environ['NUMEXPR_MAX_THREADS'] = '1'        # Limit numexpr to 1 thread
os.environ['OMP_NUM_THREADS'] = '1'            # Limit OpenMP to 1 thread
os.environ['MKL_NUM_THREADS'] = '1'            # Limit MKL to 1 thread
os.environ["CUDA_VISIBLE_DEVICES"] = ""        # Force CPU usage

import numpy as np
import h5py
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import load_model
import concurrent.futures
import multiprocessing as mp
import yaml_read
from argparse import ArgumentParser
import time


def parse_arguments():
    parser = ArgumentParser(description="Run processing for a specific sky patch.")
    parser.add_argument('--skypatchID', type=int, help='Specify the skypatchID for this run.')
    return parser.parse_args()

def load_config(file_path, skypatch_lightcone_id=None):
    config = yaml_read.yaml_config(file_path)
    if skypatch_lightcone_id is not None:
        config['hacc_simulation']['skypatchID'] = skypatch_lightcone_id
    return config


def load_model_and_scalers(model_path, scaler_input_path, scaler_output_path):
    model = load_model(model_path)
    with open(scaler_input_path, 'rb') as f:
        scaler_input = pickle.load(f)
    with open(scaler_output_path, 'rb') as f:
        scaler_output = pickle.load(f)
    return model, scaler_input, scaler_output

def load_data_chunk(h5_file, start_idx, end_idx):
    # Read a chunk of data from HDF5 file
    combined_redshift = []
    combined_sfh = []
    combined_core_tag = []
    total_samples = 0
    print(f'Loading HDF5 chunks from {start_idx} to {end_idx}')

    with h5py.File(h5_file, 'r') as f:
        for core_key in f.keys():
            redshift = f[core_key]['redshift'][...]
            sfh = f[core_key]['sfh'][...]
            core_tag = f[core_key]['core_tag'][...]

            num_samples_in_core = redshift.shape[0]
            if total_samples + num_samples_in_core <= start_idx:
                # Skip this core key as it's before the desired range
                total_samples += num_samples_in_core
                continue
            elif total_samples >= end_idx:
                # We've reached the end of the desired range
                break
            else:
                # Determine the indices to include from this core key
                start = max(0, start_idx - total_samples)
                end = min(num_samples_in_core, end_idx - total_samples)
                combined_redshift.append(redshift[start:end])
                combined_sfh.append(sfh[start:end])
                combined_core_tag.append(core_tag[start:end])
                total_samples += num_samples_in_core

    redshift_chunk = np.concatenate(combined_redshift, axis=0)
    sfh_chunk = np.concatenate(combined_sfh, axis=0)
    core_tag_chunk = np.concatenate(combined_core_tag, axis=0)
    print(f'Loaded HDF5 chunks from {start_idx} to {end_idx}')
    return redshift_chunk, sfh_chunk, core_tag_chunk

def process_chunk(args):
    chunk_idx, h5_file, start_idx, end_idx, model_path, scaler_input_path, scaler_output_path = args
    # Load model and scalers inside the process
    model, scaler_input, scaler_output = load_model_and_scalers(model_path, scaler_input_path, scaler_output_path)
    redshift_chunk, sfh_chunk, core_tag_chunk = load_data_chunk(h5_file, start_idx, end_idx)
    # Preprocess data
    sps_inputs = np.concatenate((redshift_chunk[:, np.newaxis], sfh_chunk.reshape(-1, sfh_chunk.shape[-1])), axis=1)
    sps_inputs_scaled = scaler_input.transform(sps_inputs)
    # Make predictions
    mags_scaled = model.predict(sps_inputs_scaled)
    # Unscale outputs
    mags = scaler_output.inverse_transform(mags_scaled)
    print(f'Processed chunk {chunk_idx} from {start_idx} to {end_idx} with mags shape: {mags.shape}')
    return chunk_idx, start_idx, end_idx, mags, core_tag_chunk

def main():
    
    args = parse_arguments()
    config = load_config('config_LJ.yml', args.skypatchID)
    target_skypatch_id = config['hacc_simulation']['skypatchID']

    # Set multiprocessing start method to 'spawn'
    mp.set_start_method('spawn', force=True)
    
    start_time = time.time()

    # Set parameters
    model_path = './trained_painting_NNs/spec_mlp_eline_damp_central_noncentral_z10.h5'  # Adjust as needed
    scaler_input_path = './trained_painting_NNs/eline_input_scale_damp_central_noncentral_z10.pkl'
    scaler_output_path = './trained_painting_NNs/eline_output_scale_damp_central_noncentral_z10.pkl'
    
    h5_file = f"./mocks/finished_mocks/supermock_lightcone_skypatch_{target_skypatch_id}.h5"  # Path to your HDF5 file
    
    output_file = f"./mocks/paint_models/paint_preds_skypatch_{target_skypatch_id}.h5"  # Output file
    print('H5 catalog: ', h5_file)
    # Determine total number of samples
    num_samples = 0
    with h5py.File(h5_file, 'r') as f:
        for core_key in f.keys():
            num_samples += f[core_key]['redshift'].shape[0]

    # Define chunk size
    chunk_size = 1000000  # Adjust as needed
    print(f"Chunk size: {chunk_size}.")
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    print(f"Total number of chunks: {num_chunks}.")

    # Create a list of arguments for each chunk
    args_list = []
    for i in range(num_chunks):
        print(f"Preparing chunk indices: {i} ...")
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_samples)
        args = (i, h5_file, start_idx, end_idx, model_path, scaler_input_path, scaler_output_path)
        args_list.append(args)

    # Load model to get output dimension
    model, scaler_input, scaler_output = load_model_and_scalers(model_path, scaler_input_path, scaler_output_path)
    output_dim = model.output_shape[1]
    print(f"Output dimension: {output_dim}")

    # Create the output HDF5 file and datasets
    with h5py.File(output_file, 'w') as f_out:
        # Create datasets with the total size known
        dset_sed = f_out.create_dataset('SED', shape=(num_samples, output_dim), dtype='float32', compression="gzip")
        dset_core_tag = f_out.create_dataset('core_tag', shape=(num_samples,), dtype='int64', compression="gzip")

        # Calculate max_workers
        max_workers = min(num_chunks, os.cpu_count(), 16)  # Limit to 16 workers or adjust as needed

        # Process chunks in parallel
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context('spawn')) as executor:

            futures = [executor.submit(process_chunk, args) for args in args_list]

            for future in concurrent.futures.as_completed(futures):
                chunk_idx, start_idx, end_idx, mags, core_tag_chunk = future.result()
                print(f"Writing chunk {chunk_idx} data to HDF5 file at indices {start_idx}:{end_idx}")

                # Write data to datasets
                dset_sed[start_idx:end_idx, :] = mags
                dset_core_tag[start_idx:end_idx] = core_tag_chunk

        print(f"Predictions and core_tags saved to {output_file}")
        elapsed_time = (time.time() - start_time) / 3600
        print(f"Processing completed in {elapsed_time:.2f} hours.")

if __name__ == '__main__':
    main()
