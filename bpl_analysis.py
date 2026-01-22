import numpy as np
from umachine_pyio.load_mock import load_mock_from_binaries

def read_umachine_data(volumes, path, properties):
    data = load_mock_from_binaries(subvolumes=volumes, root_dirname=path, galprops=properties)
    return data



def load_bpl_data(dirIn, a_name, subhalo_type):
    
    volumes = np.arange(5) #anything <24
    properties = list(("halo_id", "mpeak", "mpeak_history_main_prog" , "sfr_history_all_prog", "sfr_history_main_prog", "sm", "upid", "a_first_infall", "a_last_infall"))
    
    # a_name_list = ['0.664300', '1.000000']
    a_name = 'a_'+ a_name
    
    if (subhalo_type=="Centrals"):

        path_bpl = dirIn+a_name
        data = read_umachine_data(volumes, path_bpl, properties)
        data.sort("mpeak")
        data.reverse()

        data = data[np.where(data['mpeak'] > 1e11)]
        data = data[data['upid'] == -1] ## Centrals only
        
    elif (subhalo_type=="Non-centrals"):

        path_bpl = dirIn+a_name
        data = read_umachine_data(volumes, path_bpl, properties)
        data.sort("mpeak")
        data.reverse()

        data = data[np.where(data['mpeak'] > 1e11)]
        data = data[data['upid'] != -1] ## Non-centrals only

    return data

def infall_time_smdpl(cosmo, data):
    sorted_a_infall1_a1 = data["a_first_infall"]
    infall_redshift_a1 = np.array((1/sorted_a_infall1_a1) - 1)
    infall_time_a1 = cosmo.age(infall_redshift_a1)
    
    return infall_time_a1

def peak_mass_smdpl(data):
    sorted_mpeak_a1 = data["mpeak"]
    return sorted_mpeak_a1


def mass_rank_smdpl(data, volume):
    sorted_mpeak_a1 = peak_mass_smdpl(data)
    rank_sorted_mpeak = np.argsort(sorted_mpeak_a1)/volume
    return rank_sorted_mpeak
    

def bpl_times_scale(dirIn):
    time_bpl = np.loadtxt(dirIn + 'smdpl_cosmic_time.txt')
    scale_bpl = np.loadtxt(dirIn + 'smdpl_scale_list.txt')
    return time_bpl, scale_bpl



def _mass_times_indx_smdpl(data, cut):

    half_mpeak = np.log10(cut*data["mpeak"])
    mpeak_history = np.log10( data['mpeak_history_main_prog'] + 1e-32)
    timeX_indx = np.argmin(np.abs(mpeak_history.T - half_mpeak), axis=0)
    
    return timeX_indx

def mass_time_smdpl(time, data, cut):
    tcut_a_idx_bpl = _mass_times_indx_smdpl(data, cut)
    tcut_a_bpl = time[tcut_a_idx_bpl]
    return tcut_a_bpl