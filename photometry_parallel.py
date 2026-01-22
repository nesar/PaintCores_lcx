import numpy as np
import painting
from multiprocessing import Pool

def process_galaxy(args):
    galID, pcolor_all, wave_unnred, redshift_in, all_filter_pickle, wave_lims = args

    redsh = redshift_in[0, galID]
    wave_red_cuts = wave_unnred[:, np.where((wave_unnred[0] > wave_lims[0]) & (wave_unnred[0] < wave_lims[1]))][0, 0, :] * (1 + redsh)
    sed_um_wave = wave_red_cuts / 1e4
    sed_mJy_flux = pcolor_all[0, galID, :].T * 1e3

    filters = ['SPHEREx', 'LSST', 'COSMOS', 'WISE', 'LEGACYSURVEY', '2MASS', 'F784']
    mags = {}

    for filter_name in filters:
        central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = painting.load_survey_pickle(filter_name, all_filter_pickle)
        _, appmag_ext, _ = painting.sed_to_mock_phot(
            central_wavelengths, sed_um_wave, sed_mJy_flux,
            bandpass_wavs, bandpass_vals,
            interp_kind='linear', plot=False, clip_bandpass=True
        )
        mags[filter_name] = appmag_ext

    return mags

def photometric_bandpass_parallel_batched(pcolor_all, wave_unnred, redshift_in, all_filter_pickle, wave_lims, processes=64, batch_size=1000):
    num_galaxies = pcolor_all.shape[1]

    # Prepare arguments for all galaxies
    args_list = [(galID, pcolor_all, wave_unnred, redshift_in, all_filter_pickle, wave_lims) for galID in range(num_galaxies)]

    # Split into batches
    batches = [args_list[i:i + batch_size] for i in range(0, len(args_list), batch_size)]

    # Initialize arrays for storing results
    lsst_mags = np.zeros((num_galaxies, 6))
    spherex_mags = np.zeros((num_galaxies, 102))
    cosmos_mags = np.zeros((num_galaxies, 31))
    wise_mags = np.zeros((num_galaxies, 7))
    ls_mags = np.zeros((num_galaxies, 8))
    mass2_mags = np.zeros((num_galaxies, 3))
    f784_mags = np.zeros((num_galaxies, 2))

    # Process each batch in parallel
    for batch_idx, batch in enumerate(batches):
        print(f"Processing batch {batch_idx + 1}/{len(batches)}...")
        with Pool(processes=processes) as pool:
            results = pool.map(process_galaxy, batch)

        # Aggregate results from the batch
        for i, mags in enumerate(results):
            galID = batch[i][0]  # Extract galaxy ID from arguments
            lsst_mags[galID] = mags['LSST']
            spherex_mags[galID] = mags['SPHEREx']
            cosmos_mags[galID] = mags['COSMOS']
            wise_mags[galID] = mags['WISE']
            ls_mags[galID] = mags['LEGACYSURVEY']
            mass2_mags[galID] = mags['2MASS']
            f784_mags[galID] = mags['F784']

    return lsst_mags, spherex_mags, cosmos_mags, wise_mags, ls_mags, mass2_mags, f784_mags


# HOW TO CALL: 

# all_mags = photometric_bandpass_parallel_batched(
#     pcolor_all_satellite, wave_unnred_satellite, redshift_in_satellite, 
#     all_filter_pickle, wave_lims, processes=64, batch_size=1000
# )
# lsst_mags_satellite, spherex_mags_satellite, cosmos_mags_satellite, wise_mags_satellite, ls_mags_satellite, mass2_mags_satellite, f784_mags_satellite = all_mags