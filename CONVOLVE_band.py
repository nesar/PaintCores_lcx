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

#############################
# The chunked process function
#############################

def process_chunk(args):
    (idx, sidx, eidx,
     sed_file, redshift_all,
     wave_unred, wave_lims, all_filters) = args

    # Load a slice of SED from disk
    sed_chunk = load_sed_chunk(sed_file, sidx, eidx)

    # We already have redshift in memory
    z_chunk = redshift_all[sidx:eidx]

    # Then do the bandpass computations
    mags_tuple = photometric_bandpass(sed_chunk, wave_unred, z_chunk, all_filters, wave_lims)
    return idx, sidx, eidx, mags_tuple

#############################
# main()
#############################


def main():
    # Load config
    # config = yaml_read.yaml_config('config_LJ.yml')
    # target_skypatch_id = config['hacc_simulation']['skypatchID']
    # Overwrite or not:
    # target_skypatch_id = 30
    
    args = parse_arguments()
    config = load_config('config_LJ.yml', args.skypatchID)
    target_skypatch_id = config['hacc_simulation']['skypatchID']
    
    print("Sky patch:", target_skypatch_id)

    # supermock has subgroups -> we read them all into memory
    supermock_file = f"../PaintCores_lcx/mocks/finished_mocks/supermock_lightcone_skypatch_{target_skypatch_id}.h5"
    matches_for_lc = load_matches_from_h5_supermock(supermock_file)
    # get the entire redshift array in memory
    redshift_all = matches_for_lc['redshift']
    num_samples = redshift_all.shape[0]
    print("Loaded supermock with {} galaxies total.".format(num_samples))

    # SED file is top-level -> chunk read it
    sed_file = f"../PaintCores_lcx/mocks/paint_models/finished_paints/paint_preds_skypatch_{target_skypatch_id}.h5"
    total_sed_samples = get_total_sed_samples(sed_file)
    if total_sed_samples != num_samples:
        raise ValueError(f"Mismatch in sample counts: supermock has {num_samples}, SED file has {total_sed_samples}.")

    # Output HDF5
    photom_file = f"../PaintCores_lcx/mocks/photometry/photometry_skypatch_{target_skypatch_id}.h5"

    # Wave, filters, etc.
    wave_unred = load_wave_unred()
    wave_lims = [config['painting_model']['wave_min'], config['painting_model']['wave_max']]
    spherex_filter_pickle = config['painting_model']['spherex_filter_pickle']
    lsst_filter_pickle = config['painting_model']['lsst_filter_pickle']
    cosmos_filter_pickle = config['painting_model']['cosmos_filter_pickle']
    wise_filter_pickle = config['painting_model']['wise_filter_pickle']
    ls_filter_pickle = config['painting_model']['legacysurvey_filter_pickle']
    mass2_filter_pickle = config['painting_model']['mass2_filter_pickle']
    f784_filter_pickle = config['painting_model']['f784_filter_pickle']
    all_filters = [spherex_filter_pickle, lsst_filter_pickle, cosmos_filter_pickle,
                   wise_filter_pickle, ls_filter_pickle, mass2_filter_pickle, f784_filter_pickle]

    # Prepare parallel
    mp.set_start_method('spawn', force=True)
    start_time = time.time()

    chunk_size = 1_000_000
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    print(f"Total galaxies: {num_samples} -> chunk_size={chunk_size} => {num_chunks} chunks")

    # 7 sets of shapes: [LSST(6), SPHEREx(102), COSMOS(31), WISE(7), LEGACYSURVEY(8), 2MASS(3), F784(2)]
    shapes = [6, 102, 31, 7, 8, 3, 2]

    with h5py.File(photom_file, 'w') as fout:
        dset_lsst    = fout.create_dataset('LSST',        (num_samples, shapes[0]), dtype='float32', compression="gzip")
        dset_spherex = fout.create_dataset('SPHEREx',     (num_samples, shapes[1]), dtype='float32', compression="gzip")
        dset_cosmos  = fout.create_dataset('COSMOS',      (num_samples, shapes[2]), dtype='float32', compression="gzip")
        dset_wise    = fout.create_dataset('WISE',        (num_samples, shapes[3]), dtype='float32', compression="gzip")
        dset_ls      = fout.create_dataset('LEGACYSURVEY',(num_samples, shapes[4]), dtype='float32', compression="gzip")
        dset_m2      = fout.create_dataset('2MASS',       (num_samples, shapes[5]), dtype='float32', compression="gzip")
        dset_f784    = fout.create_dataset('F784',        (num_samples, shapes[6]), dtype='float32', compression="gzip")

        # Build argument list
        args_list = []
        for i in range(num_chunks):
            sidx = i * chunk_size
            eidx = min((i+1)*chunk_size, num_samples)
            args_list.append((
                i, sidx, eidx, 
                sed_file,
                redshift_all,  # entire array in memory
                wave_unred,
                wave_lims,
                all_filters
            ))

        max_workers = min(num_chunks, os.cpu_count(), 16)

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers,
                                                    mp_context=mp.get_context('spawn')) as executor:
            futures = [executor.submit(process_chunk, a) for a in args_list]
            for future in concurrent.futures.as_completed(futures):
                idx, sidx, eidx, mtuple = future.result()
                ml, ms, mc, mw, mlgs, mm2, mf7 = mtuple

                # Write chunk’s photometry
                dset_lsst[sidx:eidx]    = ml
                dset_spherex[sidx:eidx] = ms
                dset_cosmos[sidx:eidx]  = mc
                dset_wise[sidx:eidx]    = mw
                dset_ls[sidx:eidx]      = mlgs
                dset_m2[sidx:eidx]      = mm2
                dset_f784[sidx:eidx]    = mf7

                print(f"Chunk {idx} done: {sidx}–{eidx}")

    elapsed_hrs = (time.time() - start_time)/3600
    print(f"All done in {elapsed_hrs:.2f} hours.")

if __name__ == '__main__':
    main()
    
    
'''


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parse_arguments()
    config = load_config('config_LJ.yml', args.skypatchID)
    target_skypatch_id = config['hacc_simulation']['skypatchID']

    if rank == 0:
        print(f"Sky patch: {target_skypatch_id}")
        supermock_file = f"../PaintCores_lcx/mocks/finished_mocks/supermock_lightcone_skypatch_{target_skypatch_id}.h5"
        matches_for_lc = load_matches_from_h5_supermock(supermock_file)
        redshift_all = matches_for_lc['redshift']
        num_samples = redshift_all.shape[0]
        print(f"Loaded supermock with {num_samples} galaxies.")

        sed_file = f"../PaintCores_lcx/mocks/paint_models/finished_paints/paint_preds_skypatch_{target_skypatch_id}.h5"
        total_sed_samples = get_total_sed_samples(sed_file)
        assert total_sed_samples == num_samples, "Mismatch in sample counts."

        # wave_unred = np.linspace(3500, 8000, 100)  # Replace with actual wave_unred logic
        # wave_lims = [config['painting_model']['wave_min'], config['painting_model']['wave_max']]
        # all_filters = config['painting_model']['filters']  # Replace with actual filters
        
        wave_unred = load_wave_unred()        
        wave_lims = [config['painting_model']['wave_min'], config['painting_model']['wave_max']]
        spherex_filter_pickle = config['painting_model']['spherex_filter_pickle']
        lsst_filter_pickle = config['painting_model']['lsst_filter_pickle']
        cosmos_filter_pickle = config['painting_model']['cosmos_filter_pickle']
        wise_filter_pickle = config['painting_model']['wise_filter_pickle']
        ls_filter_pickle = config['painting_model']['legacysurvey_filter_pickle']
        mass2_filter_pickle = config['painting_model']['mass2_filter_pickle']
        f784_filter_pickle = config['painting_model']['f784_filter_pickle']
        all_filters = [spherex_filter_pickle, lsst_filter_pickle, cosmos_filter_pickle, 
                       wise_filter_pickle, ls_filter_pickle, mass2_filter_pickle, f784_filter_pickle]
    
    
    else:
        redshift_all = None
        wave_unred = None
        wave_lims = None
        all_filters = None
        num_samples = None
        sed_file = None

    redshift_all = comm.bcast(redshift_all, root=0)
    wave_unred = comm.bcast(wave_unred, root=0)
    wave_lims = comm.bcast(wave_lims, root=0)
    all_filters = comm.bcast(all_filters, root=0)
    num_samples = comm.bcast(num_samples, root=0)
    sed_file = comm.bcast(sed_file, root=0)

    chunk_size = (num_samples + size - 1) // size
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, num_samples)

    sed_chunk = load_sed_chunk(sed_file, start_idx, end_idx)
    redshift_chunk = redshift_all[start_idx:end_idx]
    mags_tuple = photometric_bandpass(sed_chunk, wave_unred, redshift_chunk, all_filters, wave_lims)

    gathered_results = comm.gather((start_idx, mags_tuple), root=0)

    if rank == 0:
        photom_file = f"../PaintCores_lcx/mocks/photometry/photometry_skypatch_{target_skypatch_id}.h5"
        with h5py.File(photom_file, 'w') as fout:
            for start_idx, mags_tuple in gathered_results:
                lsst_mags, spherex_mags, *_ = mags_tuple
                fout.create_dataset(f'LSST_{start_idx}', data=lsst_mags)
                fout.create_dataset(f'SPHEREx_{start_idx}', data=spherex_mags)
        print("All done!")

if __name__ == '__main__':
    main()

'''