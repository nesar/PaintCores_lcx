import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from mpi4py import MPI
import numpy as np
import h5py
import time
import glob
import pickle
from argparse import ArgumentParser
import yaml_read
import simulation_models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
import multiprocessing as mp

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--skypatchID', type=int, help='Specify skypatchID.')
    return parser.parse_args()

def load_config(file_path, skypatch_lightcone_id=None):
    config = yaml_read.yaml_config(file_path)
    if skypatch_lightcone_id is not None: config['hacc_simulation']['skypatchID'] = skypatch_lightcone_id
    return config

def load_matches_from_h5_supermock(output_file):
    combined_data = {}
    with h5py.File(output_file, "r") as f:
        for core_key in f.keys():
            for key in f[core_key].keys():
                if key not in combined_data: combined_data[key] = []
                combined_data[key].append(f[core_key][key][...])
    for key in combined_data:
        combined_data[key] = np.concatenate(combined_data[key], axis=0)
    return combined_data

def get_total_sed_samples(sed_file):
    with h5py.File(sed_file, "r") as f:
        return f['SED'].shape[0]

def load_sed_chunk(sed_file, start_idx, end_idx):
    with h5py.File(sed_file, "r") as f:
        return f['SED'][start_idx:end_idx, :]

def load_wave_unred():
    dirIn1 = '/lcrc/project/cosmo_ai/nramachandra/Projects/SPHEREx/MAH/TrainingData/Data/Damp_red_centrals_100k_z10/'
    dirIn2 = '/lcrc/project/cosmo_ai/nramachandra/Projects/SPHEREx/MAH/TrainingData/Data/Damp_red_noncentrals_100k_z10/'
    nranks = 16; rnd_seed = 42
    wave_cen = np.concatenate([np.load(dirIn1 + 'wave_spec'+str(rnd_seed)+'_rank'+str(rank)+'.npy') for rank in range(nranks)],0)
    reds_cen = np.concatenate([np.load(dirIn1 + 'redshift'+str(rnd_seed)+'_rank'+str(rank)+'.npy') for rank in range(nranks)],0)
    wave_unred_cen = wave_cen/(1+reds_cen)[:,None]
    nranks = 16; rnd_seed = 14
    wave_sat = np.concatenate([np.load(dirIn2 + 'wave_spec'+str(rnd_seed)+'_rank'+str(rank)+'.npy') for rank in range(nranks)],0)
    reds_sat = np.concatenate([np.load(dirIn2 + 'redshift'+str(rnd_seed)+'_rank'+str(rank)+'.npy') for rank in range(nranks)],0)
    wave_unred_sat = wave_sat/(1+reds_sat)[:,None]
    return wave_unred_cen[0]

def load_indiv_filter(filtfile, norm=True):
    bandpass_name = filtfile.split('.')[0].split('/')[-1]
    x = np.loadtxt(filtfile)
    nonz = (x[:,1]!=0.)
    bandpass_wav = x[nonz,0]*1e-4
    bandpass_val = x[nonz,1]
    if norm: bandpass_val /= np.sum(bandpass_val)
    cenwav = np.dot(bandpass_wav, bandpass_val)
    return bandpass_wav, bandpass_val, cenwav, bandpass_name

def load_sphx_filters(filtdir='data/spherex_filts/'):
    bandpass_wavs, bandpass_vals, central_wavelengths, bandpass_names = [],[],[],[]
    for bandidx in range(1,7):
        for filtfile in glob.glob(filtdir+'SPHEREx_band'+str(bandidx)+'*.dat'):
            bandpass_wav,bandpass_val,cenwav,bandpass_name=load_indiv_filter(filtfile)
            bandpass_names.append(bandpass_name); bandpass_wavs.append(bandpass_wav)
            bandpass_vals.append(bandpass_val); central_wavelengths.append(cenwav)
    return np.array(central_wavelengths),np.array(bandpass_wavs),np.array(bandpass_vals),np.array(bandpass_names)

def load_indiv_filter_sdss(filtfile, norm=True):
    bandpass_name = filtfile.split('.')[0].split('/')[-1]
    x = np.loadtxt(filtfile)
    nonz = (x[:,1]!=0.)
    bandpass_wav = x[nonz,0]*1e-4
    bandpass_val = x[nonz,1]
    if norm: bandpass_val/=np.sum(bandpass_val)
    cenwav = np.dot(bandpass_wav, bandpass_val)
    return bandpass_wav, bandpass_val, cenwav, bandpass_name

def load_sdss_filters(filtdir='data/sdss_filts/'):
    bband_idxs=['u','g','r','i','z']
    bandpass_wavs,bandpass_vals,central_wavelengths,bandpass_names=[],[],[],[]
    for bandidx in bband_idxs:
        for filtfile in glob.glob(filtdir+'SLOAN_SDSS.'+str(bandidx)+'*.dat'):
            bandpass_wav,bandpass_val,cenwav,bandpass_name=load_indiv_filter_sdss(filtfile)
            bandpass_names.append(bandpass_name); bandpass_wavs.append(bandpass_wav)
            bandpass_vals.append(bandpass_val); central_wavelengths.append(cenwav)
    return np.array(central_wavelengths),np.array(bandpass_wavs),np.array(bandpass_vals),np.array(bandpass_names)

def load_survey_pickle(survey, all_filters_pickle):
    spherex_filter_pickle,lsst_filter_pickle,cosmos_filter_pickle,wise_filter_pickle,ls_filter_pickle,mass2_filter_pickle,f784_filter_pickle=all_filters_pickle
    if survey=='LSST': pkl=lsst_filter_pickle
    elif survey=='SPHEREx': pkl=spherex_filter_pickle
    elif survey=='COSMOS': pkl=cosmos_filter_pickle
    elif survey=='WISE': pkl=wise_filter_pickle
    elif survey=='LEGACYSURVEY': pkl=ls_filter_pickle
    elif survey=='2MASS': pkl=mass2_filter_pickle
    elif survey=='F784': pkl=f784_filter_pickle
    else: raise NotImplementedError("Filter not included")
    with open(pkl,'rb') as f:
        cw,bw,bv,bn=pickle.load(f)
    return cw,bw,bv,bn

def clip_bandpass_vals(bandpass_wavs, bandpass_vals):
    all_w,all_v=[],[]
    for b in range(len(bandpass_wavs)):
        nz=(bandpass_vals[b]>0)
        all_w.append(bandpass_wavs[b][nz])
        all_v.append(bandpass_vals[b][nz])
    return all_w,all_v

def sed_to_mock_phot(central_wavelengths, sed_um_wave, sed_mJy_flux, bandpass_wavs, bandpass_vals, interp_kind='linear', plot=True, clip_bandpass=True):
    if clip_bandpass: cw,cv=clip_bandpass_vals(bandpass_wavs, bandpass_vals)
    sed_interp=interp1d(sed_um_wave, sed_mJy_flux, kind=interp_kind, bounds_error=False, fill_value=0.)
    band_fluxes=np.zeros_like(central_wavelengths)
    for b in range(len(bandpass_wavs)):
        if clip_bandpass: band_fluxes[b]=np.dot(cv[b], sed_interp(cw[b]))
        else: band_fluxes[b]=np.dot(bandpass_vals[b], sed_interp(bandpass_wavs[b]))
    flux=1e3*band_fluxes
    appmag_ext=-2.5*np.log10(flux)+23.9
    if plot:
        plt.figure(figsize=(12,4))
        plt.plot(sed_um_wave,1e3*sed_mJy_flux,color='k',zorder=5,alpha=0.5)
        plt.scatter(central_wavelengths,flux,color='r',s=30)
        plt.close("all")
    return flux, appmag_ext, band_fluxes

def photometric_bandpass(pcolor_all, wave_unnred, redshift_in, all_filter_pickle, wave_lims):
    n=len(redshift_in)
    lsst_mags=np.zeros((n,6)); spherex_mags=np.zeros((n,102))
    cosmos_mags=np.zeros((n,31)); wise_mags=np.zeros((n,7))
    ls_mags=np.zeros((n,8)); mass2_mags=np.zeros((n,3)); f784_mags=np.zeros((n,2))
    for galID in range(n):
        redsh=redshift_in[galID]
        mask=np.where((wave_unnred>wave_lims[0])&(wave_unnred<wave_lims[1]))
        wave_red=wave_unnred[mask]*(1+redsh); sed_um_wave=wave_red/1e4
        sed_mJy_flux=pcolor_all[galID,mask][0]*1e3
        cw,bw,bv,bn=load_survey_pickle('SPHEREx',all_filter_pickle)
        _,appmag_ext_spherex,_=sed_to_mock_phot(cw,sed_um_wave,sed_mJy_flux,bw,bv,plot=False)
        cw,bw,bv,bn=load_survey_pickle('LSST',all_filter_pickle)
        _,appmag_ext_lsst,_=sed_to_mock_phot(cw,sed_um_wave,sed_mJy_flux,bw,bv,plot=False)
        cw,bw,bv,bn=load_survey_pickle('COSMOS',all_filter_pickle)
        _,appmag_ext_cosmos,_=sed_to_mock_phot(cw,sed_um_wave,sed_mJy_flux,bw,bv,plot=False)
        cw,bw,bv,bn=load_survey_pickle('WISE',all_filter_pickle)
        _,appmag_ext_wise,_=sed_to_mock_phot(cw,sed_um_wave,sed_mJy_flux,bw,bv,plot=False)
        cw,bw,bv,bn=load_survey_pickle('LEGACYSURVEY',all_filter_pickle)
        _,appmag_ext_ls,_=sed_to_mock_phot(cw,sed_um_wave,sed_mJy_flux,bw,bv,plot=False)
        cw,bw,bv,bn=load_survey_pickle('2MASS',all_filter_pickle)
        _,appmag_ext_2mass,_=sed_to_mock_phot(cw,sed_um_wave,sed_mJy_flux,bw,bv,plot=False)
        cw,bw,bv,bn=load_survey_pickle('F784',all_filter_pickle)
        _,appmag_ext_f784,_=sed_to_mock_phot(cw,sed_um_wave,sed_mJy_flux,bw,bv,plot=False)
        lsst_mags[galID]=appmag_ext_lsst; spherex_mags[galID]=appmag_ext_spherex
        cosmos_mags[galID]=appmag_ext_cosmos; wise_mags[galID]=appmag_ext_wise
        ls_mags[galID]=appmag_ext_ls; mass2_mags[galID]=appmag_ext_2mass; f784_mags[galID]=appmag_ext_f784
    return lsst_mags,spherex_mags,cosmos_mags,wise_mags,ls_mags,mass2_mags,f784_mags

def main():
    comm=MPI.COMM_WORLD; rank=comm.Get_rank(); size=comm.Get_size()
    args=parse_arguments()
    if rank==0:
        config=load_config('config_LJ.yml',args.skypatchID)
        target_skypatch_id=config['hacc_simulation']['skypatchID']
        print(f"Sky patch: {target_skypatch_id}")
        supermock_file=f"../PaintCores_lcx/mocks/finished_mocks/supermock_lightcone_skypatch_{target_skypatch_id}.h5"
        matches_for_lc=load_matches_from_h5_supermock(supermock_file)
        redshift_all=matches_for_lc['redshift']
        num_samples=redshift_all.shape[0]
        print(f"Loaded supermock with {num_samples} galaxies.")
        sed_file=f"../PaintCores_lcx/mocks/paint_models/finished_paints/paint_preds_skypatch_{target_skypatch_id}.h5"
        total_sed=get_total_sed_samples(sed_file)
        assert total_sed==num_samples,"Mismatch in sample counts."
        wave_unred=load_wave_unred()
        wave_lims=[config['painting_model']['wave_min'], config['painting_model']['wave_max']]
        spherex= config['painting_model']['spherex_filter_pickle']
        lsst=    config['painting_model']['lsst_filter_pickle']
        cosmos=  config['painting_model']['cosmos_filter_pickle']
        wise=    config['painting_model']['wise_filter_pickle']
        ls=      config['painting_model']['legacysurvey_filter_pickle']
        m2=      config['painting_model']['mass2_filter_pickle']
        f784=    config['painting_model']['f784_filter_pickle']
        all_filters=[spherex,lsst,cosmos,wise,ls,m2,f784]
    else:
        redshift_all=None; wave_unred=None; wave_lims=None; all_filters=None
        num_samples=None; sed_file=None
    redshift_all=comm.bcast(redshift_all,root=0)
    wave_unred=comm.bcast(wave_unred,root=0)
    wave_lims=comm.bcast(wave_lims,root=0)
    all_filters=comm.bcast(all_filters,root=0)
    num_samples=comm.bcast(num_samples,root=0)
    sed_file=comm.bcast(sed_file,root=0)
    chunk_size=(num_samples+size-1)//size
    start_idx=rank*chunk_size
    end_idx=min(start_idx+chunk_size,num_samples)
    sed_chunk=load_sed_chunk(sed_file,start_idx,end_idx)
    redshift_chunk=redshift_all[start_idx:end_idx]
    mags_tuple=photometric_bandpass(sed_chunk, wave_unred, redshift_chunk, all_filters, wave_lims)
    results=comm.gather((start_idx,mags_tuple),root=0)
    if rank==0:
        photom_file=f"../PaintCores_lcx/mocks/photometry/photometry_skypatch_{target_skypatch_id}.h5"
        with h5py.File(photom_file,'w') as fout:
            for (si,mt) in results:
                ml,ms,mc,mw,mls,mm2,mf7=mt
                fout.create_dataset(f'LSST_{si}',data=ml)
                fout.create_dataset(f'SPHEREx_{si}',data=ms)
                fout.create_dataset(f'COSMOS_{si}',data=mc)
                fout.create_dataset(f'WISE_{si}',data=mw)
                fout.create_dataset(f'LEGACYSURVEY_{si}',data=mls)
                fout.create_dataset(f'2MASS_{si}',data=mm2)
                fout.create_dataset(f'F784_{si}',data=mf7)
        print("All done!")

if __name__=='__main__':
    main()