import os
os.environ['NUMEXPR_MAX_THREADS'] = '128'  # Set this to a suitable value based on your system
print("Environment:", os.environ)
print("CPU count:", os.cpu_count())
import sys
sys.path.append('/lcrc/project/cosmo_ai/nramachandra/Projects/SPHEREx/MAH/HACCnPaint/Cores/PaintCores_lcx/')
import numexpr
import time
import glob
import numpy as np
import pandas as pd
import h5py
from multiprocessing import Pool
from numba import config
from argparse import ArgumentParser

import simulation_models
import cosmology_utils
import cores_analysis
import bpl_analysis
import haccytrees.mergertrees
from haccytrees.coretrees.coretree_reader import corematrix_reader
import yaml_read

config.THREADING_LAYER = 'workqueue'

def configure_environment():
    os.environ["CUDA_32VISIBLE_DEVICES"] = ""
    numexpr.set_num_threads(128)

def parse_arguments():
    parser = ArgumentParser(description="Run processing for a specific sky patch.")
    parser.add_argument('--skypatchID', type=int, help='Specify the skypatchID for this run.')
    return parser.parse_args()

def load_config(file_path, skypatch_lightcone_id=None):
    config = yaml_read.yaml_config(file_path)
    if skypatch_lightcone_id is not None:
        config['hacc_simulation']['skypatchID'] = skypatch_lightcone_id
    return config

def initialize_output_file(output_file):
    with h5py.File(output_file, "w") as f:
        f.attrs["description"] = "Processed core subvolume matches"

def find_all_subfiles_lightcone(target_skypatch_lightcone_id, dir_in):
    return glob.glob(os.path.join(dir_in, f"lc_cores-*.{target_skypatch_lightcone_id}.hdf5"))

def extract_data_from_file(file_path):
    with h5py.File(file_path, 'r') as hdf:
        data = {key: hdf[key][:] for key in hdf.keys()}
    return pd.DataFrame(data)

# ###########################################


def process_core_file(unique_core_idx, single_skypatch_lightcone_all_data, dirIn_core, SIMULATION, alphaq_analysis_age, V_alphaq, config):
    print(f"Processing core subvolume: {unique_core_idx}")

    # Step 1: Define utility functions
    def initialize_cosmology(config):
        smdpl_cosmo = simulation_models.get_cosmo(config['reference_simulation']['name'])
        redshift = np.array([0.5, 0.25, 0.0])
        cosmic_time = smdpl_cosmo.age(redshift)
        t0 = cosmic_time[-1]
        print(f"Age of the universe: {t0} Gyrs")
        return smdpl_cosmo, cosmic_time

    def get_bpl_times_scale(config):
        return bpl_analysis.bpl_times_scale(config['reference_simulation']['sim_dir'])

    def prepare_time_array(scale_bpl, time_bpl, target_scale):
        time_idx = np.where(scale_bpl == target_scale)[0][0]
        return time_bpl[:time_idx + 1]

    def load_and_process_data(sim_dir, scale, galaxy_type, smdpl_cosmo, time_arr, V_bpl):
        data = bpl_analysis.load_bpl_data(sim_dir, f'{scale:.6f}', galaxy_type)
        sorted_mpeak = bpl_analysis.peak_mass_smdpl(data)
        infall_time = bpl_analysis.infall_time_smdpl(smdpl_cosmo, data)
        rank_sorted_mpeak = bpl_analysis.mass_rank_smdpl(data, V_bpl)
        t50 = bpl_analysis.mass_time_smdpl(time_arr, data, 0.5)
        t25 = bpl_analysis.mass_time_smdpl(time_arr, data, 0.25)
        # bpl_sfh = data["sfr_history_all_prog"]
        bpl_sfh = np.pad(data["sfr_history_all_prog"], ((0, 0), (0, 117 - data["sfr_history_all_prog"].shape[1])), mode='constant')
        bpl_sm = data["sm"]
        return {
            "mass_times": np.array([t50, t25, sorted_mpeak, infall_time]).T,
            "sfh": bpl_sfh,
            "sm": bpl_sm
        }

    def kdtree_match(bpl_summary, lc_summary, bpl_sfh, bpl_sm):
        from scipy.spatial import KDTree
        bpl_tree = KDTree(bpl_summary)
        distance, bpl_index = bpl_tree.query(lc_summary, k=1)
        bpl_match_sfh = bpl_sfh[bpl_index]
        bpl_match_sm = bpl_sm[bpl_index]
        return bpl_match_sfh, bpl_match_sm

    # Step 2: Initialize cosmology and BPL data
    smdpl_cosmo, cosmic_time = initialize_cosmology(config)
    time_bpl, scale_bpl = get_bpl_times_scale(config)

    time_arr_a1 = prepare_time_array(scale_bpl, time_bpl, 1.0)
    time_arr_a06 = prepare_time_array(scale_bpl, time_bpl, 0.664300)

    centrals_data = load_and_process_data(config['reference_simulation']['sim_dir'], 1.0, "Centrals", smdpl_cosmo, time_arr_a1, V_alphaq)
    noncentrals_a1_data = load_and_process_data(config['reference_simulation']['sim_dir'], 1.0, "Non-centrals", smdpl_cosmo, time_arr_a1, V_alphaq)
    noncentrals_a06_data = load_and_process_data(config['reference_simulation']['sim_dir'], 0.664300, "Non-centrals", smdpl_cosmo, time_arr_a06, V_alphaq)

    # Step 3: Load core forest data
    core_forest_file_name = f"{dirIn_core}m000p.coreforest.{unique_core_idx}.hdf5"
    forest_matrices = corematrix_reader(core_forest_file_name, SIMULATION)

    file_idx_selection_criteria = single_skypatch_lightcone_all_data["file_idx"] == unique_core_idx
    row_indx_selected_by_file_idx = single_skypatch_lightcone_all_data[file_idx_selection_criteria]["row_idx"] 

    ####### CRITICAL #######
    ## From Patricia: Michael may have changed this to row_idx_global -- current row_idx is wrong. 
    ## wrong histories in many core files
    ## Not sure if it's moved to lcrc yet -- it may be on eagle. 


    col_indx_selected_by_file_idx = single_skypatch_lightcone_all_data[file_idx_selection_criteria]['snapnum']

    match_data = {key: forest_matrices[key][row_indx_selected_by_file_idx, col_indx_selected_by_file_idx] for key in 
                  ['vx', 'vy', 'vz', 'fof_halo_tag', 'central', 'merged', 'core_state']}
    match_data.update({
        'core_tag': single_skypatch_lightcone_all_data[file_idx_selection_criteria]["core_tag"],
        'x': single_skypatch_lightcone_all_data[file_idx_selection_criteria]["x"],
        'y': single_skypatch_lightcone_all_data[file_idx_selection_criteria]["y"],
        'z': single_skypatch_lightcone_all_data[file_idx_selection_criteria]["z"]
    })
    
    # NOTE: 
    # core_tag_1 = forest_matrices['core_tag'][row_indx_selected_by_file_idx, col_indx_selected_by_file_idx]
    # core_tag_2 = single_skypatch_all_data[file_idx_selection_criteria]["core_tag"]
    # core_tag_1 and core_tag_2 SHOULD match
    ### CHECK THIS AGAIN
    
    # Step 6: Add spatial and velocity data
    match_xyz = np.array([match_data['x'], match_data['y'], match_data['z']]).T
    match_vel_xyz = np.array([match_data["vx"], match_data["vy"], match_data["vz"]]).T

    # match_redshift = cosmology_utils.redshift_from_xyz(match_xyz)
    match_redshift = np.array([1/single_skypatch_lightcone_all_data[file_idx_selection_criteria]["scale_factor"] ][0]) - 1
    
    z_pec, z_tot, _, _, _, _, _ = cosmology_utils.pecZ(match_xyz[:, 0], match_xyz[:, 1], match_xyz[:, 2],
                                                       match_vel_xyz[:, 0], match_vel_xyz[:, 1], match_vel_xyz[:, 2],
                                                       match_redshift, obs=np.zeros(3))
    ra_match, dec_match = cosmology_utils.ra_dec(match_data['x'], match_data['y'], match_data['z'])

    match_data.update({
        'ra': ra_match,
        'dec': dec_match,
        'redshift': z_tot
    })

    # Step 4: Compute host properties and analysis metrics
    mask = forest_matrices['top_host_row'] > 1
    _full_idx = (forest_matrices['top_host_row'][mask], forest_matrices['snapnum'][mask])
    parent_fof_mass = np.empty_like(forest_matrices['infall_tree_node_mass'])
    parent_fof_mass[:] = -1
    parent_fof_mass[mask] = forest_matrices['infall_tree_node_mass'][_full_idx]
    ## changed from infall_fof_halo_mass to  infall_tree_node_mass
    match_fof_halo_mass_row = parent_fof_mass[row_indx_selected_by_file_idx, :]

    match_state_row = forest_matrices['core_state'][row_indx_selected_by_file_idx, :]
    time_infall = cores_analysis.infall_time_alphaq(match_state_row, alphaq_analysis_age)
    t50_a1 = cores_analysis.mass_time_alphaq(match_fof_halo_mass_row, alphaq_analysis_age, 0.5)
    t25_a1 = cores_analysis.mass_time_alphaq(match_fof_halo_mass_row, alphaq_analysis_age, 0.25)
    peak_mass = cores_analysis.peak_mass_HACC(match_fof_halo_mass_row)
    rank_peak_mass = cores_analysis.mass_rank_alphaq(match_fof_halo_mass_row, V_alphaq)

    match_data.update({
        'time_infall': time_infall,
        't50_a1': t50_a1,
        't25_a1': t25_a1,
        'peak_mass': peak_mass,
        'rank_peak_mass': rank_peak_mass
    })

    # Step 5: Perform HACC-BPL Cross-matching
    central_cond = (match_data['central'] == 1)
    satellite_cond = (match_data['central'] == 0) & (match_data['merged'] == 0)
    merged_cond = (match_data['central'] == 0) & (match_data['merged'] == 1)

    match_mass_time = np.array([match_data['central'], match_data['merged'], t50_a1, t25_a1, peak_mass, time_infall]).T

    central_indices = np.where(central_cond)[0]
    satellite_indices = np.where(satellite_cond)[0]
    merged_indices = np.where(merged_cond)[0]

    match_mass_time_central = match_mass_time[central_indices, 2:]
    match_mass_time_satellite = match_mass_time[satellite_indices, 2:]

    a_merged_infall = 1 / (1 + match_data['redshift'][merged_cond])
    merged_high_a_indices = merged_indices[a_merged_infall > 0.664300]
    merged_low_a_indices = merged_indices[a_merged_infall <= 0.664300]

    match_mass_time_merged_high_a = match_mass_time[merged_high_a_indices, 2:]
    match_mass_time_merged_low_a = match_mass_time[merged_low_a_indices, 2:]

    match_sfh_central, match_sm_central = kdtree_match(centrals_data["mass_times"], match_mass_time_central, centrals_data["sfh"], centrals_data["sm"])
    match_sfh_satellite, match_sm_satellite = kdtree_match(noncentrals_a1_data["mass_times"], match_mass_time_satellite, noncentrals_a1_data["sfh"], noncentrals_a1_data["sm"])
    match_sfh_merged_high_a, match_sm_merged_high_a = kdtree_match(noncentrals_a1_data["mass_times"], match_mass_time_merged_high_a, noncentrals_a1_data["sfh"], noncentrals_a1_data["sm"])
    match_sfh_merged_low_a, match_sm_merged_low_a = kdtree_match(noncentrals_a06_data["mass_times"], match_mass_time_merged_low_a, noncentrals_a06_data["sfh"], noncentrals_a06_data["sm"])

    num_galaxies = len(match_mass_time)
    sfh_full = np.zeros((num_galaxies, 117))
    stellar_mass_full = np.zeros(num_galaxies)

    sfh_full[central_indices] = match_sfh_central
    stellar_mass_full[central_indices] = match_sm_central
    sfh_full[satellite_indices] = match_sfh_satellite
    stellar_mass_full[satellite_indices] = match_sm_satellite
    sfh_full[merged_high_a_indices] = match_sfh_merged_high_a
    stellar_mass_full[merged_high_a_indices] = match_sm_merged_high_a
    sfh_full[merged_low_a_indices] = match_sfh_merged_low_a
    stellar_mass_full[merged_low_a_indices] = match_sm_merged_low_a

    match_data.update({
        'sfh': sfh_full,
        'stellar_mass': stellar_mass_full
    })
    
    print('Processed core: ', unique_core_idx, ' with Number of galaxy matches: ', num_galaxies)

    return match_data

##################################

# def process_batch(args):
#     batch_indices, single_skypatch_lightcone_all_data, dir_in_core, simulation, alphaq_analysis_age, v_alphaq, config, output_file = args
#     results = []
#     for unique_core_idx in batch_indices:
#         match_data = process_core_file(unique_core_idx, single_skypatch_lightcone_all_data, dir_in_core, simulation, alphaq_analysis_age, v_alphaq, config)
#         results.append((unique_core_idx, match_data))
#     with h5py.File(output_file, "a") as f:
#         for unique_core_idx, match_data in results:
#             grp = f.create_group(f"core_{unique_core_idx}")
#             for key, value in match_data.items():
#                 grp.create_dataset(key, data=value)
#     return len(batch_indices)

def process_batch(args):
    batch_indices, single_skypatch_lightcone_all_data, dir_in_core, simulation, alphaq_analysis_age, v_alphaq, config, output_file = args
    results = []
    for unique_core_idx in batch_indices:
        match_data = process_core_file(unique_core_idx, single_skypatch_lightcone_all_data, dir_in_core, simulation, alphaq_analysis_age, v_alphaq, config)
        results.append((unique_core_idx, match_data))
    with h5py.File(output_file, "a") as f:
        for unique_core_idx, match_data in results:
            group_name = f"core_{unique_core_idx}"
            if group_name in f:
                print(f"Warning: Group {group_name} already exists. Skipping to avoid overwriting.")
                continue
            grp = f.create_group(group_name)
            for key, value in match_data.items():
                grp.create_dataset(key, data=value, compression="gzip")
    return len(batch_indices)

def validate_hdf5_file(output_file):
    total_matches = 0
    with h5py.File(output_file, "r") as f:
        for core_key in f.keys():
            total_matches += f[core_key]['x'].shape[0]
    print(f"Total number of galaxies in the HDF5 file: {total_matches}")

def load_matches_from_h5(output_file):
    combined_data = {}
    with h5py.File(output_file, "r") as f:
        for core_key in f.keys():
            for key in f[core_key].keys():
                if key not in combined_data:
                    combined_data[key] = []
                combined_data[key].append(f[core_key][key][...])
    # Concatenate lists of arrays into single arrays for each key
    for key in combined_data:
        combined_data[key] = np.concatenate(combined_data[key], axis=0)
    return combined_data

def main():
    configure_environment()
    args = parse_arguments()
    config = load_config('config_LJ.yml', args.skypatchID)
    skypatch_lightcone_id = config['hacc_simulation']['skypatchID']
    dir_in_lc = '/lcrc/project/cosmo_ai/mbuehlmann/LastJourney/core-lc/output/'
    dir_in_core = '/lcrc/group/cosmodata/simulations/LastJourney/coretrees/forest/'

    print(f"Sky patch ID: {skypatch_lightcone_id}")
    single_lightcone_skypatch_all_subfiles = find_all_subfiles_lightcone(skypatch_lightcone_id, dir_in_lc)
    single_skypatch_lightcone_all_data = pd.concat([extract_data_from_file(file) for file in single_lightcone_skypatch_all_subfiles], ignore_index=True)
    print(f"Found {len(single_lightcone_skypatch_all_subfiles)} files for sky patch {skypatch_lightcone_id}.")

    simulation = haccytrees.Simulation.simulations[config['hacc_simulation']['name']]
    alphaq_analysis_age = simulation_models.get_analysis_steps(config['hacc_simulation']['name'])
    v_alphaq = config['hacc_simulation']['side_length']**3

    # file_idx_iters = np.unique(single_skypatch_lightcone_all_data["file_idx"])
    
    file_idx_iters = np.unique(single_skypatch_lightcone_all_data["file_idx"], 
                               return_counts=True)[0][np.argsort(-np.unique(single_skypatch_lightcone_all_data["file_idx"], 
                                                                            return_counts=True)[1])]  ## DO NOT REMOVE THIS
    
    # file_idx_iters = file_idx_iters[:2]   ################################ FOR PROTOTYPE TESTING ONLY -- KEEP IT
    # file_idx_iters = [25, 1]
    
    output_file = f"../PaintCores_lcx/mocks/supermock_lightcone_skypatch_{skypatch_lightcone_id}.h5"
    initialize_output_file(output_file)

    batch_size = 1024
    batches = [file_idx_iters[i:i + batch_size] for i in range(0, len(file_idx_iters), batch_size)]

    start_time = time.time()
    print(f"Processing {len(file_idx_iters)} core subvolumes in batches of {batch_size}.")

    with Pool(processes=48) as pool:
        args_list = [(batch, single_skypatch_lightcone_all_data, dir_in_core, simulation, alphaq_analysis_age, v_alphaq, config, output_file) for batch in batches]
        for processed_count in pool.imap(process_batch, args_list):
            print(f"Processed {processed_count} cores in this batch.")

    elapsed_time = (time.time() - start_time) / 3600
    print(f"Processing completed in {elapsed_time:.2f} hours.")

    print(f"Loading results from {output_file}...")
    validate_hdf5_file(output_file)
    matches_for_lc = load_matches_from_h5(output_file)
    print(f"Loaded data for {len(matches_for_lc)} matches.")
    print(f"First match keys: {matches_for_lc.keys()}")
    print(f"Number of matches: {matches_for_lc['x'].shape}")
    print(f"Sky patch {skypatch_lightcone_id} processing completed.")

if __name__ == "__main__":
    main()