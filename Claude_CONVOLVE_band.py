import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import h5py
import concurrent.futures
import multiprocessing as mp
import time

import yaml_read
# import simulation_models
from argparse import ArgumentParser

from scipy.integrate import simps
from scipy.interpolate import interp1d as interp1d
import glob
import matplotlib.pylab as plt
import pickle
import os
import numpy as np
import h5py
import time
from mpi4py import MPI

#############################

def parse_arguments():
    parser = ArgumentParser(description="Run processing for a specific sky patch.")
    parser.add_argument('--skypatchID', type=int, help='Specify the skypatchID for this run.')
    return parser.parse_args()

def load_config(file_path, skypatch_lightcone_id=None):
    config = yaml_read.yaml_config(file_path)
    if skypatch_lightcone_id is not None:
        config['hacc_simulation']['skypatchID'] = skypatch_lightcone_id
    return config

#############################

def load_matches_from_h5_supermock(output_file):
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

def load_matches_from_h5_sed(output_file):
    combined_data = {}
    with h5py.File(output_file, "r") as f:
        for key in f.keys():
            combined_data[key] = f[key][...]
    return combined_data

#############################
# Utility for SED chunking
#############################

def get_total_sed_samples(sed_file):
    with h5py.File(sed_file, "r") as f:
        return f['SED'].shape[0]  # total # of galaxies in the SED file

def load_sed_chunk(sed_file, start_idx, end_idx):
    with h5py.File(sed_file, "r") as f:
        sed_chunk = f['SED'][start_idx:end_idx, :]
    return sed_chunk

#############################
# Your wave, filter, and bandpass code
#############################

def load_wave_unred():
    dirIn1 = '/lcrc/project/cosmo_ai/nramachandra/Projects/SPHEREx/MAH/TrainingData/Data/Damp_red_centrals_100k_z10/'
    dirIn2 = '/lcrc/project/cosmo_ai/nramachandra/Projects/SPHEREx/MAH/TrainingData/Data/Damp_red_noncentrals_100k_z10/'
    nranks = 16
    rnd_seed = 42
    wave_cen = np.concatenate([np.load(dirIn1 + 'wave_spec' + str(rnd_seed) + '_rank' + str(rank) + '.npy')
                               for rank in range(nranks)], axis=0)
    reds_cen = np.concatenate([np.load(dirIn1 + 'redshift' + str(rnd_seed) + '_rank' + str(rank) + '.npy')
                               for rank in range(nranks)], axis=0)
    wave_unred_cen = wave_cen/(1+reds_cen)[:, np.newaxis]
    nranks = 16
    rnd_seed = 14
    wave_sat = np.concatenate([np.load(dirIn2 + 'wave_spec' + str(rnd_seed) + '_rank' + str(rank) + '.npy')
                               for rank in range(nranks)], axis=0)
    reds_sat = np.concatenate([np.load(dirIn2 + 'redshift' + str(rnd_seed) + '_rank' + str(rank) + '.npy')
                               for rank in range(nranks)], axis=0)
    wave_unred_sat = wave_sat/(1+reds_sat)[:, np.newaxis]
    wave_unred = wave_unred_cen[0]
    return wave_unred

# ------------------------------------------------
# The rest of your photometric_bandpass code, etc.
# (unchanged from your script)
# ------------------------------------------------

#SPHEREx bands
def load_indiv_filter(filtfile, 
                      norm=True):
    
    bandpass_name = filtfile.split('.')[0].split('/')[-1]
    
    x = np.loadtxt(filtfile)
    nonz = (x[:,1] != 0.)
    bandpass_wav = x[nonz,0]*1e-4
    bandpass_val = x[nonz,1]

    if norm:
        bandpass_val /= np.sum(bandpass_val)

    cenwav = np.dot(bandpass_wav, bandpass_val)
    # cenwav = np.dot(x[nonz,0], x[nonz,1])

    return bandpass_wav, bandpass_val, cenwav, bandpass_name


def load_sphx_filters(filtdir='data/spherex_filts/', 
                      to_um=True):

    ''' 
    Loads files, returns list of central wavelengths and list of wavelengths/filter responses. 
    Converts wavelengths to microns unless otherwise specified.
    '''

    bandpass_wavs, bandpass_vals, central_wavelengths, bandpass_names = [], [], [], []
    bband_idxs = np.arange(1, 7)
    
    for bandidx in bband_idxs:
        filtfiles = glob.glob(filtdir+'SPHEREx_band'+str(bandidx)+'*.dat')
        for filtfile in filtfiles:

            bandpass_wav, bandpass_val, cenwav, bandpass_name = load_indiv_filter(filtfile)
            bandpass_names.append(bandpass_name)

            bandpass_wavs.append(bandpass_wav)
            bandpass_vals.append(bandpass_val)
            central_wavelengths.append(cenwav)

    return np.array(central_wavelengths), np.array(bandpass_wavs), np.array(bandpass_vals), np.array(bandpass_names)



## SDSS bands

def load_indiv_filter_sdss(filtfile, 
                           norm=True):
    
    bandpass_name = filtfile.split('.')[0].split('/')[-1]
    
    x = np.loadtxt(filtfile)
    nonz = (x[:,1] != 0.)
    bandpass_wav = x[nonz,0]*1e-4
    bandpass_val = x[nonz,1]

    if norm:
        bandpass_val /= np.sum(bandpass_val)

    cenwav = np.dot(bandpass_wav, bandpass_val)
    # cenwav = np.dot(x[nonz,0], x[nonz,1])

    return bandpass_wav, bandpass_val, cenwav, bandpass_name

def load_sdss_filters(filtdir='data/sdss_filts/', 
                      to_um=True):

    ''' 
    Loads files, returns list of central wavelengths and list of wavelengths/filter responses. 
    Converts wavelengths to microns unless otherwise specified.
    '''

    bandpass_wavs, bandpass_vals, central_wavelengths, bandpass_names = [], [], [], []
    bband_idxs = ['u', 'g', 'r', 'i', 'z']
    
    for bandidx in bband_idxs:
        filtfiles = glob.glob(filtdir+'SLOAN_SDSS.'+str(bandidx)+'*.dat')
        for filtfile in filtfiles:
            

            bandpass_wav, bandpass_val, cenwav, bandpass_name = load_indiv_filter_sdss(filtfile)
            bandpass_names.append(bandpass_name)

            bandpass_wavs.append(bandpass_wav)
            bandpass_vals.append(bandpass_val)
            central_wavelengths.append(cenwav)

    return np.array(central_wavelengths), np.array(bandpass_wavs), np.array(bandpass_vals), np.array(bandpass_names)



def load_survey_pickle(survey, 
                       all_filters_pickle):
    
    spherex_filter_pickle, lsst_filter_pickle, cosmos_filter_pickle, wise_filter_pickle, ls_filter_pickle, mass2_filter_pickle, f784_filter_pickle = all_filters_pickle
    
    
    if (survey=='LSST'):
        FILTER_NAME = lsst_filter_pickle
    elif (survey=='SPHEREx'):
        FILTER_NAME = spherex_filter_pickle
    elif (survey=='COSMOS'):
        FILTER_NAME = cosmos_filter_pickle
    elif (survey=='WISE'):
        FILTER_NAME = wise_filter_pickle
    elif (survey=='LEGACYSURVEY'):
        FILTER_NAME = ls_filter_pickle
    elif (survey=='2MASS'):
        FILTER_NAME = mass2_filter_pickle
    elif (survey=='F784'):
        FILTER_NAME = f784_filter_pickle
         
        
    else: 
        raise NotImplementedError("Filter specifications not included")
        
    with open(FILTER_NAME, 'rb') as f:
     central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = pickle.load(f)
    
    return central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names


def clip_bandpass_vals(bandpass_wavs, 
                       bandpass_vals):
    
    all_clip_bandpass_wav, all_clip_bandpass_vals = [], []

    for b in range(len(bandpass_wavs)):
        nonz_bandpass_val = (bandpass_vals[b] > 0)
        clip_bandpass_wav = bandpass_wavs[b][nonz_bandpass_val]
        clip_bandpass_vals = bandpass_vals[b][nonz_bandpass_val]
        all_clip_bandpass_wav.append(clip_bandpass_wav)
        all_clip_bandpass_vals.append(clip_bandpass_vals)

    return all_clip_bandpass_wav, all_clip_bandpass_vals


def sed_to_mock_phot(central_wavelengths, 
                     sed_um_wave, 
                     sed_mJy_flux, 
                     bandpass_wavs, 
                     bandpass_vals, 
                     interp_kind='linear', 
                     plot=True, 
                     clip_bandpass=True):
    
    # central wavelengths in micron
    if clip_bandpass:
        all_clip_bandpass_wav, all_clip_bandpass_vals = clip_bandpass_vals(bandpass_wavs, bandpass_vals)

    sed_interp = interp1d(sed_um_wave,
                          sed_mJy_flux,
                          kind=interp_kind,
                          bounds_error=False, 
                          fill_value = 0.0)

    band_fluxes = np.zeros_like(central_wavelengths)

    for b, bandpass_wav in enumerate(bandpass_wavs):
        # fluxes in mJy
        if clip_bandpass:
            band_fluxes[b] = np.dot(all_clip_bandpass_vals[b], sed_interp(all_clip_bandpass_wav[b]))
        else:
            band_fluxes[b] = np.dot(bandpass_vals[b], sed_interp(bandpass_wav))

    flux = 1e3*band_fluxes # uJy
    appmag_ext = -2.5*np.log10(flux)+23.9

    if plot:

        wav_um = np.array(central_wavelengths)

        plt.figure(figsize=(12, 4))
        plt.title('sed uJy flux')
        plt.plot(sed_um_wave, 1e3*sed_mJy_flux, color='k', zorder=5, alpha=0.5)
        plt.scatter(wav_um, flux, color='r', label='bandpass-convolved fluxes', s=30)
        # plt.ylim(0, 1.2*np.max(flux))
        plt.xlabel('um', fontsize=16)
        plt.ylabel('uJy', fontsize=16)
        plt.tick_params(labelsize=14)
        plt.legend()
        plt.show()
        
    plt.close("all")

    return flux, appmag_ext, band_fluxes



def load_model_and_scalers(scaler_output_path):
    import pickle

    with open(scaler_output_path, 'rb') as f:
        scaler_output = pickle.load(f)
    
    return scaler_output

def unscale_sed(scaler_output, scaled_sed):
    model_output_log = scaler_output.inverse_transform(scaled_sed)
    sed_unscaled = 10**(model_output_log)
    
    return sed_unscaled
    

def photometric_bandpass(pcolor_all, wave_unnred, redshift_in, all_filter_pickle, wave_lims):

    lsst_mags = np.zeros(shape=(pcolor_all.shape[0], 6))
    spherex_mags = np.zeros(shape=(pcolor_all.shape[0], 102))
    cosmos_mags = np.zeros(shape=(pcolor_all.shape[0], 31))
    wise_mags = np.zeros(shape=(pcolor_all.shape[0], 7))
    ls_mags = np.zeros(shape=(pcolor_all.shape[0], 8))
    mass2_mags = np.zeros(shape=(pcolor_all.shape[0], 3))
    f784_mags = np.zeros(shape=(pcolor_all.shape[0], 2))
    
    
    scaler_output_path = './trained_painting_NNs/eline_output_scale_damp_central_noncentral_z10.pkl'    
    scaler_output = load_model_and_scalers(scaler_output_path)


    
    for galID in range(pcolor_all.shape[0]):


        # redsh = redshift_in[0, galID]
        redsh = redshift_in[galID]

        # wave_red_cuts = wave_unnred[:, np.where( (wave_unnred[0]>wave_lims[0]) & (wave_unnred[0]<wave_lims[1]) )][0, 0, :]*(1 + redsh)

        wave_mask = np.where( (wave_unnred>wave_lims[0]) & (wave_unnred<wave_lims[1]))
        
        wave_red_cuts = wave_unnred[wave_mask]*(1 + redshift_in[galID])
        
        sed_um_wave = wave_red_cuts/1e4
        # sed_mJy_flux = unscale(mag_test, scaler_y)[galID]*1e-3 #mJy
        
        # sed_mJy_flux =  pcolor_all[0, galID, :].T*1e3
        
        ####################################################################
        
        # print('pcolor_all: ', pcolor_all.shape) ## (1000000, 1963)
        sed_scaled = pcolor_all[galID, :].reshape(-1, 1).T  #### (1, 1963)
        # print('sed_scaled: ', sed_scaled.shape) ## (1, 1963)
        
        sed_unscaled = unscale_sed(scaler_output, sed_scaled)[0, wave_mask][0]
    
        sed_mJy_flux = sed_unscaled*1e3
        
        # print('sed_mJy_flux: ', sed_mJy_flux.shape) ## (921,)?
        
        ####################################################################
        


        # central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = painting.load_sphx_filters(filtdir=spherex_filters_dir, to_um=True)
        central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = load_survey_pickle('SPHEREx', all_filter_pickle)

        flux_spherex, appmag_ext_spherex, band_fluxes_spherex = sed_to_mock_phot(central_wavelengths, 
                                                         sed_um_wave, 
                                                         sed_mJy_flux, 
                                                         bandpass_wavs, 
                                                         bandpass_vals, 
                                                         interp_kind='linear', 
                                                         plot=False, 
                                                         clip_bandpass=True)


        # central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = painting.load_lsst_filters(filtdir=sdss_filters_dir, to_um=True)
        central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = load_survey_pickle('LSST', all_filter_pickle)

        flux_lsst, appmag_ext_lsst, band_fluxes_lsst = sed_to_mock_phot(central_wavelengths, 
                                                         sed_um_wave, 
                                                         sed_mJy_flux, 
                                                         bandpass_wavs, 
                                                         bandpass_vals, 
                                                         interp_kind='linear', 
                                                         plot=False, 
                                                         clip_bandpass=True)





        central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = load_survey_pickle('COSMOS', all_filter_pickle)

        flux_cosmos, appmag_ext_cosmos, band_fluxes_cosmos = sed_to_mock_phot(central_wavelengths, 
                                                         sed_um_wave, 
                                                         sed_mJy_flux, 
                                                         bandpass_wavs, 
                                                         bandpass_vals, 
                                                         interp_kind='linear', 
                                                         plot=False, 
                                                         clip_bandpass=True)

        central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = load_survey_pickle('WISE', all_filter_pickle)

        flux_wise, appmag_ext_wise, band_fluxes_wise = sed_to_mock_phot(central_wavelengths, 
                                                         sed_um_wave, 
                                                         sed_mJy_flux, 
                                                         bandpass_wavs, 
                                                         bandpass_vals, 
                                                         interp_kind='linear', 
                                                         plot=False, 
                                                         clip_bandpass=True)

        central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = load_survey_pickle('LEGACYSURVEY', all_filter_pickle)

        flux_ls, appmag_ext_ls, band_fluxes_ls = sed_to_mock_phot(central_wavelengths, 
                                                         sed_um_wave, 
                                                         sed_mJy_flux, 
                                                         bandpass_wavs, 
                                                         bandpass_vals, 
                                                         interp_kind='linear', 
                                                         plot=False, 
                                                         clip_bandpass=True)


        central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = load_survey_pickle('2MASS', all_filter_pickle)

        flux_2mass, appmag_ext_2mass, band_fluxes_2mass = sed_to_mock_phot(central_wavelengths, 
                                                         sed_um_wave, 
                                                         sed_mJy_flux, 
                                                         bandpass_wavs, 
                                                         bandpass_vals, 
                                                         interp_kind='linear', 
                                                         plot=False, 
                                                         clip_bandpass=True)

        central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = load_survey_pickle('F784', all_filter_pickle)

        flux_f784, appmag_ext_f784, band_fluxes_f784 = sed_to_mock_phot(central_wavelengths, 
                                                         sed_um_wave, 
                                                         sed_mJy_flux, 
                                                         bandpass_wavs, 
                                                         bandpass_vals, 
                                                         interp_kind='linear', 
                                                         plot=False, 
                                                         clip_bandpass=True)




        lsst_mags[galID] = appmag_ext_lsst
        spherex_mags[galID] = appmag_ext_spherex
        cosmos_mags[galID] = appmag_ext_cosmos
        wise_mags[galID] = appmag_ext_wise
        ls_mags[galID] = appmag_ext_ls
        mass2_mags[galID] = appmag_ext_2mass
        f784_mags[galID] = appmag_ext_f784
    

    print('SPHEREx mags', spherex_mags.shape)
    return lsst_mags, spherex_mags, cosmos_mags, wise_mags, ls_mags, mass2_mags, f784_mags

'''
#############################
# The chunked process function
#############################

from mpi4py import MPI
import os, multiprocessing, numpy as np, h5py, time

# Assume the following functions are defined:
# parse_arguments(), load_config(), load_matches_from_h5_supermock(),
# get_total_sed_samples(), load_wave_unred(), load_sed_chunk(), photometric_bandpass()

def process_subchunk(args):
    sub_idx, local_start, local_end, local_sed_chunk, local_redshift_chunk, wave_unred, wave_lims, all_filters = args
    print(f"[DEBUG][Worker] Processing subchunk {sub_idx}: indices {local_start}-{local_end-1}")
    sed_sub = local_sed_chunk[local_start:local_end]
    redshift_sub = local_redshift_chunk[local_start:local_end]
    mags_tuple = photometric_bandpass(sed_sub, wave_unred, redshift_sub, all_filters, wave_lims)
    print(f"[DEBUG][Worker] Finished subchunk {sub_idx}")
    return sub_idx, local_start, local_end, mags_tuple

def main():
    comm = MPI.COMM_WORLD
    global_rank = comm.Get_rank()
    global_size = comm.Get_size()
    hostname = MPI.Get_processor_name()
    # Create a communicator grouping processes on the same node.
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
    node_size = node_comm.Get_size()
    print(f"[DEBUG] Global rank {global_rank}/{global_size} on {hostname}, node rank {node_rank}/{node_size}")
    # Only the node leader will do work; others exit.
    if node_rank != 0:
        print(f"[DEBUG] Global rank {global_rank} is not node leader; waiting at barrier and exiting.")
        comm.Barrier()
        exit(0)
    # Create a communicator for all node leaders.
    leaders_comm = comm.Split(color=1, key=global_rank)
    leaders_rank = leaders_comm.Get_rank()
    leaders_size = leaders_comm.Get_size()
    print(f"[DEBUG] Node leader global rank {global_rank} is leader rank {leaders_rank}/{leaders_size}")

    start_time = time.time()
    # Only one (the leader with leaders_rank==0) loads config and full data.
    if leaders_rank == 0:
        args = parse_arguments()
        config = load_config('config_LJ.yml', args.skypatchID)
        target_skypatch_id = config['hacc_simulation']['skypatchID']
        print(f"[DEBUG][Leader {leaders_rank}] Target sky patch: {target_skypatch_id} | Total leaders: {leaders_size}")
        supermock_file = f"../PaintCores_lcx/mocks/finished_mocks/supermock_lightcone_skypatch_{target_skypatch_id}.h5"
        matches_for_lc = load_matches_from_h5_supermock(supermock_file)
        redshift_all = matches_for_lc['redshift']
        num_samples = redshift_all.shape[0]
        print(f"[DEBUG][Leader {leaders_rank}] Loaded supermock with {num_samples} galaxies.")
        sed_file = f"../PaintCores_lcx/mocks/paint_models/finished_paints/paint_preds_skypatch_{target_skypatch_id}.h5"
        total_sed_samples = get_total_sed_samples(sed_file)
        if total_sed_samples != num_samples:
            raise ValueError(f"Mismatch: supermock has {num_samples}, SED file has {total_sed_samples}.")
        photom_file = f"../PaintCores_lcx/mocks/photometry/photometry_skypatch_{target_skypatch_id}.h5"
        wave_unred = load_wave_unred()
        wave_lims = [config['painting_model']['wave_min'], config['painting_model']['wave_max']]
        all_filters = [config['painting_model'][key] for key in ['spherex_filter_pickle',
                                                                 'lsst_filter_pickle',
                                                                 'cosmos_filter_pickle',
                                                                 'wise_filter_pickle',
                                                                 'legacysurvey_filter_pickle',
                                                                 'mass2_filter_pickle',
                                                                 'f784_filter_pickle']]
    else:
        target_skypatch_id = None
        redshift_all = None
        num_samples = None
        sed_file = None
        photom_file = None
        wave_unred = None
        wave_lims = None
        all_filters = None
    # Broadcast needed variables among node leaders.
    target_skypatch_id = leaders_comm.bcast(target_skypatch_id, root=0)
    num_samples = leaders_comm.bcast(num_samples, root=0)
    sed_file = leaders_comm.bcast(sed_file, root=0)
    photom_file = leaders_comm.bcast(photom_file, root=0)
    wave_unred = leaders_comm.bcast(wave_unred, root=0)
    wave_lims = leaders_comm.bcast(wave_lims, root=0)
    all_filters = leaders_comm.bcast(all_filters, root=0)
    # Split the workload among node leaders.
    if leaders_rank == 0:
        chunk_sizes = [num_samples // leaders_size + (1 if i < num_samples % leaders_size else 0) for i in range(leaders_size)]
        displs = [sum(chunk_sizes[:i]) for i in range(leaders_size)]
        print(f"[DEBUG][Leader {leaders_rank}] Chunk sizes: {chunk_sizes}")
    else:
        chunk_sizes = None
        displs = None
    chunk_sizes = leaders_comm.bcast(chunk_sizes, root=0)
    displs = leaders_comm.bcast(displs, root=0)
    my_chunk_size = chunk_sizes[leaders_rank]
    my_start_idx = displs[leaders_rank]
    my_end_idx = my_start_idx + my_chunk_size
    print(f"[DEBUG][Leader {leaders_rank}] Processing galaxies {my_start_idx} to {my_end_idx-1} ({my_chunk_size} galaxies)")
    # Distribute redshift array among node leaders.
    if leaders_rank == 0:
        for i in range(1, leaders_size):
            comm.Send(redshift_all[displs[i]:displs[i]+chunk_sizes[i]], dest=displs[i], tag=77)
        my_redshift_chunk = redshift_all[my_start_idx:my_end_idx]
    else:
        my_redshift_chunk = np.empty(my_chunk_size, dtype=np.float64)
        comm.Recv(my_redshift_chunk, source=0, tag=77)
    print(f"[DEBUG][Leader {leaders_rank}] Received redshift chunk with shape: {my_redshift_chunk.shape}")
    # Each node leader loads its SED chunk.
    print(f"[DEBUG][Leader {leaders_rank}] Loading SED chunk from {sed_file} indices {my_start_idx} to {my_end_idx-1}")
    my_sed_chunk = load_sed_chunk(sed_file, my_start_idx, my_end_idx)
    # Divide local chunk into sub-chunks for multiprocessing.
    num_workers = os.cpu_count()
    subchunk_size = 1000  # adjust as needed
    tasks = []
    sub_idx = 0
    for local_start in range(0, my_chunk_size, subchunk_size):
        local_end = min(local_start + subchunk_size, my_chunk_size)
        tasks.append((sub_idx, local_start, local_end, my_sed_chunk, my_redshift_chunk,
                      wave_unred, wave_lims, all_filters))
        sub_idx += 1
    print(f"[DEBUG][Leader {leaders_rank}] Created {len(tasks)} tasks for multiprocessing using {num_workers} workers.")
    pool = multiprocessing.Pool(processes=num_workers)
    results = pool.map(process_subchunk, tasks)
    pool.close()
    pool.join()
    print(f"[DEBUG][Leader {leaders_rank}] Multiprocessing complete with {len(results)} results.")
    # Sort and combine sub-chunk results.
    results.sort(key=lambda x: x[1])
    dataset_names = ['LSST', 'SPHEREx', 'COSMOS', 'WISE', 'LEGACYSURVEY', '2MASS', 'F784']
    combined = {name: [] for name in dataset_names}
    for sub_idx, local_start, local_end, mags_tuple in results:
        for i, name in enumerate(dataset_names):
            combined[name].append(mags_tuple[i])
    local_results = {}
    for name in dataset_names:
        local_results[name] = np.concatenate(combined[name], axis=0)
        print(f"[DEBUG][Leader {leaders_rank}] Combined {name} shape: {local_results[name].shape}")
    # File writing among node leaders (using a barrier for synchronization).
    leaders_comm.Barrier()
    if leaders_rank == 0:
        print(f"[DEBUG][Leader {leaders_rank}] Creating output HDF5 file: {photom_file}")
        with h5py.File(photom_file, 'w') as fout:
            shapes = {'LSST': (num_samples, 6),
                      'SPHEREx': (num_samples, 102),
                      'COSMOS': (num_samples, 31),
                      'WISE': (num_samples, 7),
                      'LEGACYSURVEY': (num_samples, 8),
                      '2MASS': (num_samples, 3),
                      'F784': (num_samples, 2)}
            for name in dataset_names:
                fout.create_dataset(name, shapes[name], dtype='float32', compression="gzip")
    leaders_comm.Barrier()
    for name in dataset_names:
        with h5py.File(photom_file, 'r+') as fout:
            dset = fout[name]
            dset[my_start_idx:my_end_idx] = local_results[name].astype('float32')
        print(f"[DEBUG][Leader {leaders_rank}] Written {name} data to indices {my_start_idx} to {my_end_idx-1}")
    leaders_comm.Barrier()
    if leaders_rank == 0:
        elapsed = (time.time()-start_time)/3600
        print(f"[DEBUG][Leader {leaders_rank}] All done in {elapsed:.2f} hours.")

if __name__=='__main__':
    main()
    
    
'''


from mpi4py import MPI
import h5py
import numpy as np
import time
import os
import argparse

# These functions are assumed to be defined elsewhere:
# load_config, load_matches_from_h5_supermock, get_total_sed_samples, load_wave_unred,
# load_sed_chunk, photometric_bandpass

#####################################
# The chunk processing function
#####################################
def process_chunk(args):
    # Unpack the arguments: idx, start index, end index, SED file path, redshift slice,
    # wavelength array (unreddened), wavelength limits, and filter pickles.
    (idx, sidx, eidx, sed_file, redshift_all, wave_unred, wave_lims, all_filters) = args
    # Load a slice of SED from disk
    sed_chunk = load_sed_chunk(sed_file, sidx, eidx)
    # redshift_all is assumed to be the slice corresponding to this chunk
    z_chunk = redshift_all  # already sliced before passing
    # Perform the bandpass convolutions (returns a tuple of arrays)
    mags_tuple = photometric_bandpass(sed_chunk, wave_unred, z_chunk, all_filters, wave_lims)
    return mags_tuple

#####################################
# Master process: dynamic scheduling loop
#####################################
def master_loop(tasks, sed_file, supermock_file, wave_unred, wave_lims, all_filters, dsets):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    next_task = 0  # index in tasks list
    num_tasks = len(tasks)
    active_workers = 0

    # Initially send one task to each worker (ranks 1..size-1)
    for worker in range(1, size):
        if next_task < num_tasks:
            # Each task is (i, sidx, eidx)
            comm.send(tasks[next_task], dest=worker, tag=1)
            print(f"[Master] Sent task {tasks[next_task][0]} to worker {worker}")
            next_task += 1
            active_workers += 1
        else:
            comm.send(None, dest=worker, tag=0)
    
    # Optionally, master can process tasks as well when running in a single-rank job.
    # Here we assume rank0 is dedicated to scheduling.
    # Now receive completions and dispatch remaining tasks.
    while active_workers > 0:
        # Receive a completion message from any worker
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=2)
        source = msg.get('worker')
        chunk_id = msg.get('chunk_id')
        print(f"[Master] Received completion of chunk {chunk_id} from worker {source}")
        active_workers -= 1  # one worker finished its task
        # If there are still tasks left, send the next one to this worker.
        if next_task < num_tasks:
            comm.send(tasks[next_task], dest=source, tag=1)
            print(f"[Master] Sent task {tasks[next_task][0]} to worker {source}")
            next_task += 1
            active_workers += 1
        else:
            # No tasks left: tell worker to terminate.
            comm.send(None, dest=source, tag=0)
    print("[Master] All tasks completed.")

#####################################
# Worker process: process tasks as received from master
#####################################
def worker_loop(sed_file, supermock_file, wave_unred, wave_lims, all_filters, dsets):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    while True:
        # Receive a task assignment from master
        status = MPI.Status()
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == 0 or task is None:
            # Termination signal received
            print(f"[Worker {rank}] No more tasks. Exiting loop.")
            break
        # Unpack task info: (chunk_id, sidx, eidx)
        chunk_id, sidx, eidx = task
        print(f"[Worker {rank}] Processing chunk {chunk_id}: indices {sidx}–{eidx}")
        try:
            # Open supermock file and read redshift slice for this chunk
            with h5py.File(supermock_file, 'r') as f_super:
                redshift_slice = f_super['redshift'][sidx:eidx]
            # Prepare arguments and process the chunk
            args_tuple = (chunk_id, sidx, eidx, sed_file, redshift_slice, wave_unred, wave_lims, all_filters)
            mags_tuple = process_chunk(args_tuple)
            # Unpack the resulting photometry arrays
            ml, ms, mc, mw, mlgs, mm2, mf7 = mags_tuple
            # Write results into the correct slices of each dataset
            dsets['LSST'][sidx:eidx]         = ml
            dsets['SPHEREx'][sidx:eidx]      = ms
            dsets['COSMOS'][sidx:eidx]       = mc
            dsets['WISE'][sidx:eidx]         = mw
            dsets['LEGACYSURVEY'][sidx:eidx] = mlgs
            dsets['2MASS'][sidx:eidx]        = mm2
            dsets['F784'][sidx:eidx]         = mf7
            # Inform master of completion; send a simple dict with info.
            comm.send({'worker': rank, 'chunk_id': chunk_id}, dest=0, tag=2)
            print(f"[Worker {rank}] Finished chunk {chunk_id}.")
        except Exception as e:
            # If an error occurs, print and then abort the MPI job.
            print(f"[Worker {rank}] Error processing chunk {chunk_id}: {e}")
            comm.Abort(1)

#####################################
# Main function for MPI-parallel execution
#####################################
def main():
    import resource, os

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # memory in MB
    with open(f"mem_usage_rank{rank}.log", "a") as f:
        f.write(f"{os.getpid()} rank {rank}: {mem_mb} MB\n")
    
    # Parse arguments (e.g., --skypatchID)
    parser = argparse.ArgumentParser()
    parser.add_argument("--skypatchID", type=int, required=True)
    args = parser.parse_args()
    
    # Rank 0 loads configuration and broadcasts it
    if rank == 0:
        config = load_config('config_LJ.yml', args.skypatchID)
    else:
        config = None
    config = comm.bcast(config, root=0)
    target_skypatch_id = config['hacc_simulation']['skypatchID']
    if rank == 0:
        print(f"[Rank 0] Sky patch: {target_skypatch_id}")
    
    # Define file paths based on target_skypatch_id
    supermock_file = f"../PaintCores_lcx/mocks/finished_mocks/supermock_lightcone_skypatch_{target_skypatch_id}.h5"
    sed_file       = f"../PaintCores_lcx/mocks/paint_models/finished_paints/paint_preds_skypatch_{target_skypatch_id}.h5"
    photom_file    = f"../PaintCores_lcx/mocks/photometry/photometry_skypatch_{target_skypatch_id}.h5"
    
    # Rank 0 loads supermock to get number of galaxies and verifies counts;
    # then broadcasts num_samples to all.
    if rank == 0:
        matches_for_lc = load_matches_from_h5_supermock(supermock_file)
        redshift_all = matches_for_lc['redshift']
        num_samples = redshift_all.shape[0]
        print(f"[Rank 0] Loaded supermock with {num_samples} galaxies total.")
    else:
        num_samples = None
    num_samples = comm.bcast(num_samples, root=0)
    
    # Rank 0 verifies SED file sample count matches redshift count.
    if rank == 0:
        total_sed_samples = get_total_sed_samples(sed_file)
        if total_sed_samples != num_samples:
            raise ValueError(f"Mismatch in sample counts: supermock has {num_samples}, SED file has {total_sed_samples}.")
    
    # Load wavelength and filter info (common to all)
    wave_unred = load_wave_unred()
    wave_lims  = [config['painting_model']['wave_min'], config['painting_model']['wave_max']]
    all_filters = [config['painting_model'][key] for key in 
                   ['spherex_filter_pickle','lsst_filter_pickle','cosmos_filter_pickle',
                    'wise_filter_pickle','legacysurvey_filter_pickle','mass2_filter_pickle','f784_filter_pickle']]
    
    # Define chunking parameters (here using a chunk size of 1,000,000 galaxies)
    chunk_size = 50_000 # 1_000_000
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    if rank == 0:
        print(f"[Rank 0] Total galaxies: {num_samples} -> chunk_size={chunk_size} => {num_chunks} chunks")
    
    # Create list of tasks; each task is a tuple: (chunk_id, sidx, eidx)
    tasks = []
    for i in range(num_chunks):
        sidx = i * chunk_size
        eidx = min((i+1)*chunk_size, num_samples)
        tasks.append((i, sidx, eidx))
    
    # Open the output HDF5 file in parallel (all ranks do this)
    out_f = h5py.File(photom_file, 'w', driver='mpio', comm=comm)
    # Create output datasets (using same shapes as before)
    shapes = [6, 102, 31, 7, 8, 3, 2]
    dset_names = ['LSST', 'SPHEREx', 'COSMOS', 'WISE', 'LEGACYSURVEY', '2MASS', 'F784']
    dsets = {}
    for i, name in enumerate(dset_names):
        dsets[name] = out_f.create_dataset(name, (num_samples, shapes[i]), dtype='float32', compression="gzip")
    
    # Synchronize all ranks before processing
    comm.Barrier()
    start_time = time.time()
    
    # If only one MPI rank, process all tasks sequentially.
    if size == 1:
        for (chunk_id, sidx, eidx) in tasks:
            print(f"[Rank 0] Processing chunk {chunk_id}: indices {sidx}–{eidx}")
            with h5py.File(supermock_file, 'r') as f_super:
                redshift_slice = f_super['redshift'][sidx:eidx]
            args_tuple = (chunk_id, sidx, eidx, sed_file, redshift_slice, wave_unred, wave_lims, all_filters)
            mags_tuple = process_chunk(args_tuple)
            ml, ms, mc, mw, mlgs, mm2, mf7 = mags_tuple
            dsets['LSST'][sidx:eidx]         = ml
            dsets['SPHEREx'][sidx:eidx]      = ms
            dsets['COSMOS'][sidx:eidx]       = mc
            dsets['WISE'][sidx:eidx]         = mw
            dsets['LEGACYSURVEY'][sidx:eidx] = mlgs
            dsets['2MASS'][sidx:eidx]        = mm2
            dsets['F784'][sidx:eidx]         = mf7
            print(f"[Rank 0] Finished chunk {chunk_id}")
    else:
        # In multi-rank runs, rank 0 acts as master, and others are workers.
        if rank == 0:
            master_loop(tasks, sed_file, supermock_file, wave_unred, wave_lims, all_filters, dsets)
        else:
            worker_loop(sed_file, supermock_file, wave_unred, wave_lims, all_filters, dsets)
    
    comm.Barrier()  # Wait for all processes to finish
    if rank == 0:
        elapsed_hrs = (time.time() - start_time) / 3600
        print(f"[Rank 0] All done in {elapsed_hrs:.2f} hours.")
    out_f.close()

if __name__ == '__main__':
    main()