import painting
import numpy as np


def photometric_bandpass(pcolor_all, wave_unnred, redshift_in, all_filter_pickle, wave_lims):

    lsst_mags = np.zeros(shape=(pcolor_all.shape[1], 6))
    spherex_mags = np.zeros(shape=(pcolor_all.shape[1], 102))
    cosmos_mags = np.zeros(shape=(pcolor_all.shape[1], 31))
    wise_mags = np.zeros(shape=(pcolor_all.shape[1], 7))
    ls_mags = np.zeros(shape=(pcolor_all.shape[1], 8))
    mass2_mags = np.zeros(shape=(pcolor_all.shape[1], 3))
    f784_mags = np.zeros(shape=(pcolor_all.shape[1], 2))


    
    for galID in range(pcolor_all.shape[1]):


        redsh = redshift_in[0, galID]
        wave_red_cuts = wave_unnred[:, np.where( (wave_unnred[0]>wave_lims[0]) & (wave_unnred[0]<wave_lims[1]) )][0, 0, :]*(1 + redsh)
        sed_um_wave = wave_red_cuts/1e4
        # sed_mJy_flux = unscale(mag_test, scaler_y)[galID]*1e-3 #mJy
        sed_mJy_flux =  pcolor_all[0, galID, :].T*1e3


        # central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = painting.load_sphx_filters(filtdir=spherex_filters_dir, to_um=True)
        central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = painting.load_survey_pickle('SPHEREx', all_filter_pickle)

        flux_spherex, appmag_ext_spherex, band_fluxes_spherex = painting.sed_to_mock_phot(central_wavelengths, 
                                                         sed_um_wave, 
                                                         sed_mJy_flux, 
                                                         bandpass_wavs, 
                                                         bandpass_vals, 
                                                         interp_kind='linear', 
                                                         plot=False, 
                                                         clip_bandpass=True)


        # central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = painting.load_lsst_filters(filtdir=sdss_filters_dir, to_um=True)
        central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = painting.load_survey_pickle('LSST', all_filter_pickle)

        flux_lsst, appmag_ext_lsst, band_fluxes_lsst = painting.sed_to_mock_phot(central_wavelengths, 
                                                         sed_um_wave, 
                                                         sed_mJy_flux, 
                                                         bandpass_wavs, 
                                                         bandpass_vals, 
                                                         interp_kind='linear', 
                                                         plot=False, 
                                                         clip_bandpass=True)





        central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = painting.load_survey_pickle('COSMOS', all_filter_pickle)

        flux_cosmos, appmag_ext_cosmos, band_fluxes_cosmos = painting.sed_to_mock_phot(central_wavelengths, 
                                                         sed_um_wave, 
                                                         sed_mJy_flux, 
                                                         bandpass_wavs, 
                                                         bandpass_vals, 
                                                         interp_kind='linear', 
                                                         plot=False, 
                                                         clip_bandpass=True)

        central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = painting.load_survey_pickle('WISE', all_filter_pickle)

        flux_wise, appmag_ext_wise, band_fluxes_wise = painting.sed_to_mock_phot(central_wavelengths, 
                                                         sed_um_wave, 
                                                         sed_mJy_flux, 
                                                         bandpass_wavs, 
                                                         bandpass_vals, 
                                                         interp_kind='linear', 
                                                         plot=False, 
                                                         clip_bandpass=True)

        central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = painting.load_survey_pickle('LEGACYSURVEY', all_filter_pickle)

        flux_ls, appmag_ext_ls, band_fluxes_ls = painting.sed_to_mock_phot(central_wavelengths, 
                                                         sed_um_wave, 
                                                         sed_mJy_flux, 
                                                         bandpass_wavs, 
                                                         bandpass_vals, 
                                                         interp_kind='linear', 
                                                         plot=False, 
                                                         clip_bandpass=True)


        central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = painting.load_survey_pickle('2MASS', all_filter_pickle)

        flux_2mass, appmag_ext_2mass, band_fluxes_2mass = painting.sed_to_mock_phot(central_wavelengths, 
                                                         sed_um_wave, 
                                                         sed_mJy_flux, 
                                                         bandpass_wavs, 
                                                         bandpass_vals, 
                                                         interp_kind='linear', 
                                                         plot=False, 
                                                         clip_bandpass=True)

        central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = painting.load_survey_pickle('F784', all_filter_pickle)

        flux_f784, appmag_ext_f784, band_fluxes_f784 = painting.sed_to_mock_phot(central_wavelengths, 
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
    

    return lsst_mags, spherex_mags, cosmos_mags, wise_mags, ls_mags, mass2_mags, f784_mags