import numpy as np
# from emulator_utils.pre_process import load_scaler
# from emulator_utils.split import random_holdout
# from emulator_utils.surrogates import load_mlp, mcdrop_pred
from scipy.integrate import simps
from scipy.interpolate import interp1d as interp1d
import glob
import matplotlib.pylab as plt
import pickle

from tensorflow.keras import Sequential
from keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Activation, Dropout, Flatten, Input
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from pickle import dump, load



def unscale(scaled_data, scaler):
    """
    Takes processed data to the original raw format

    Parameters
    -------
    scaled_data: float
        scaled data
    scaler: func
        scaling function

    Returns
    ----------
    data1d_batch: float
        add explanation


    """
    unscaled_data = scaler.inverse_transform(scaled_data)
    return unscaled_data



def load_mlp(fileout):
    
    print('Model loaded from: '+fileout)

    # load a trained model
    model = tf.keras.models.load_model(fileout)
    return model


def mcdrop_pred(param_in_unscaled, model, scaler_in, scaler_out):
    num_mc_samples = 100
    partial_model = Model(model.layers[0].input, model.output)
    
    input_params_scaled = scaler_in.transform(param_in_unscaled)

    ## Draw MC samples 
    Yt_hat_unscaled = np.array([unscale(partial_model(input_params_scaled, training=True), scaler_out) for _ in range(num_mc_samples)])
    
    y_mean_unscaled = np.mean(Yt_hat_unscaled, axis=0)
    y_std_unscaled = np.std(Yt_hat_unscaled, axis=0)
    
    return Yt_hat_unscaled, y_mean_unscaled, y_std_unscaled


def random_holdout(input_data, output_data, split_fraction):
    """
    Used for train-test splitting of data. The datapoints are randomly selection. 
    TO-DO: fix random seed?

    Parameters
    ----------
    input_data: float
        insert explanation
    output_data: float
        insert explanation
    split_fraction: float
        insert explanation

    Returns
    -------
    train_data: float
        insert explanation
    test_data: float
        insert explanation
    train_target: float
        insert explanation
    test_target: float
        insert explanation

    """

    train_data, test_data, train_target, test_target = train_test_split(input_data, output_data, test_size=split_fraction, random_state=1)
    return train_data, test_data, train_target, test_target


def load_scaler(filepath):
    print('Loading the pre-processing pipeline from: ' + filepath +'.pkl')
    scaler = load(open(filepath+'.pkl', 'rb'))
    return scaler

def load_pretrained_models(model_dirIn, 
                           wave_dirIn, 
                           gal_type):
    
    
    if (gal_type == "Central"):
        
        # model_dirIn = '/lcrc/project/cosmo_ai/nramachandra/Projects/Ulab_ldrd/emulator_utils/test/model/'
        scaler = load_scaler(model_dirIn + 'eline_input_scale_damp_central_z10')
        scaler_y = load_scaler(model_dirIn + 'eline_output_scale_damp_central_z10')
        mlp = load_mlp(model_dirIn + 'spec_mlp_eline_damp_central_z10')

        # wave_dirIn = '/lcrc/project/cosmo_ai/nramachandra/Projects/SPHEREx/MAH/TrainingData/Data/Damp_red/'
        # wave = np.load(wave_dirIn + 'wave_spec.npy')
        # reds = np.load(wave_dirIn + 'redshift.npy')
        
        rnd_seed = 42
        nranks = 16
        wave = np.concatenate([np.load(wave_dirIn + 'Damp_red_centrals_100k_z10/' + 'wave_spec' + str(rnd_seed) + '_rank' + str(rank) + '.npy') for rank in range(nranks)], axis=0)
        reds = np.concatenate([np.load(wave_dirIn + 'Damp_red_centrals_100k_z10/' + 'redshift' + str(rnd_seed) + '_rank' + str(rank) + '.npy') for rank in range(nranks)], axis=0)

    
    elif (gal_type == "Satellite"):
    
        # model_dirIn = '/lcrc/project/cosmo_ai/nramachandra/Projects/Ulab_ldrd/emulator_utils/test/model/'
        scaler = load_scaler(model_dirIn + 'eline_input_scale_damp_noncentral_z10')
        scaler_y = load_scaler(model_dirIn + 'eline_output_scale_damp_noncentral_z10')
        mlp = load_mlp(model_dirIn + 'spec_mlp_eline_damp_noncentral_z10')

        # wave_dirIn = '/lcrc/project/cosmo_ai/nramachandra/Projects/SPHEREx/MAH/TrainingData/Data/Damp_red/'
        # wave = np.load(wave_dirIn + 'wave_spec.npy')
        # reds = np.load(wave_dirIn + 'redshift.npy')
        
        rnd_seed = 14
        nranks = 16
        wave = np.concatenate([np.load(wave_dirIn + 'Damp_red_noncentrals_100k_z10/' + 'wave_spec' + str(rnd_seed) + '_rank' + str(rank) + '.npy') for rank in range(nranks)], axis=0)
        reds = np.concatenate([np.load(wave_dirIn + 'Damp_red_noncentrals_100k_z10/' +'redshift' + str(rnd_seed) + '_rank' + str(rank) + '.npy') for rank in range(nranks)], axis=0)
    
    return mlp, scaler, scaler_y, wave, reds


def paint_sed(mlp_central, 
              scaler, 
              scaler_y, 
              redshift, 
              sfh, 
              wave, 
              reds, 
              plot_every):
    
    redshift_in = redshift[::plot_every][np.newaxis, :]
    sfh_in = sfh[::plot_every, 0, :].T
    sps_inputs = np.concatenate((redshift_in, sfh_in), axis=0).T

    pcolor_all, _, _ = mcdrop_pred(sps_inputs, mlp_central, scaler, scaler_y)
    
    
    ######################

    wave_unnred = np.zeros_like(wave)

    for sedID in range(wave_unnred.shape[0]):
        wave_unnred[sedID] = wave[sedID, :]/(1 + np.array(reds[sedID]))

    return pcolor_all, wave_unnred, redshift_in, sfh_in


def lum_solar(SED, 
              redshift_in, 
              wave, 
              cosmo):
    
    sed_Jy = SED
    sed_ergscm2A = sed_Jy*2.998e+18*1.0e-23/np.asarray(wave)/np.asarray(wave) #erg/s/cm2/A
    intSED = simps(sed_ergscm2A, wave) #erg/s/cm2
    dd1 = cosmo.luminosityDistance(redshift_in) #Mpc?

    dd1cm = dd1*3.086e+24 #cm
    fac = 4*np.pi*(dd1cm**2) #cm2
    lum = fac*intSED #erg/s
    lum_sol = lum/(3.826*1e33)
    
    return lum_sol

def lum(SED, 
        redshift_in, 
        wave, 
        cosmo):
    intSED = simps(SED, wave)
    dd1 = cosmo.luminosityDistance(redshift_in)
    fac = 4*np.pi*(dd1**2)
    lum = fac*intSED
    return lum  


def calc_luminosity(pcolor_all, 
                    wave_unnred, 
                    redshift_in, 
                    cosmo, 
                    wave_min, 
                    wave_max):
    
    match_lum = np.zeros_like(redshift_in[0])
    
    wave_mask = np.where( (wave_unnred[0]>wave_min) & (wave_unnred[0]<wave_max))

    for galID in range(redshift_in[0].shape[0]):
        if (galID % 50000 == 0): 
            print(galID)
        each_wave_red = wave_unnred[:, wave_mask][0, 0, :]*(1 + redshift_in[0][galID])
        match_lum[galID] = lum_solar(pcolor_all[0, galID, :], redshift_in[0][galID], each_wave_red, cosmo)
        # match_lum[galID] = lum(pcolor_all[0, galID, :], redshift_in[0][galID], each_wave_red)
    
    return match_lum



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
