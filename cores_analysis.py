import h5py
import numpy as np
from haccytrees.coretrees.coretree_reader import corematrix_reader
import haccytrees.mergertrees


def read_core_matrices(fileIn):

    with h5py.File(fileIn, "r") as f:
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

    #     print(type(f[a_group_key])) 

    #     peak_masses = np.max(f['fof_halo_mass'][()], axis=1)
    #     peak_mass_sort_idx = np.argsort(peak_masses)[::-1]   

    #     core_status = f['core_state'][()][peak_mass_sort_idx]
    #     core_tag = f['core_tag'][()][peak_mass_sort_idx]
    #     core_radius = f['core_radius'][()][peak_mass_sort_idx]
    #     core_x = f['core_x'][()][peak_mass_sort_idx]
    #     core_y = f['core_y'][()][peak_mass_sort_idx]
    #     core_z = f['core_z'][()][peak_mass_sort_idx]

    #     fof_halo_tag = f['fof_halo_tag'][()][peak_mass_sort_idx]
    #     parent_fof = f['parent_fof'][()][peak_mass_sort_idx]
    #     fof_halo_mass = f['fof_halo_mass'][()][peak_mass_sort_idx]
    #     top_host_row = f['top_host_row'][()][peak_mass_sort_idx]

        core_status = f['core_state'][()]
        core_tag = f['core_tag'][()]
        core_radius = f['core_radius'][()]
        core_x = f['core_x'][()]
        core_y = f['core_y'][()]
        core_z = f['core_z'][()]
        core_vx = f['core_vx'][()]
        core_vy = f['core_vy'][()]
        core_vz = f['core_vz'][()]

        fof_halo_tag = f['fof_halo_tag'][()]
        parent_fof = f['parent_fof'][()]
        fof_halo_mass = f['fof_halo_mass'][()]
        top_host_row = f['top_host_row'][()]
      
    print('\n')
    print("Core matrices dimensions: ", core_status.shape)
    print('\n')
    print('Based on final time step')
    fracCentrals = fof_halo_mass[core_status[:, -1] == 0].shape[0]/fof_halo_mass.shape[0]
    print('Fraction of Centrals: ', fracCentrals )

    fracSatellite = fof_halo_mass[core_status[:, -1] == 1].shape[0]/fof_halo_mass.shape[0]
    print('Fraction of Satellites: ', fracSatellite )

    fracMerged = fof_halo_mass[core_status[:, -1] == 2].shape[0]/fof_halo_mass.shape[0]
    print('Fraction of Merged cores: ', fracMerged )
        
    return core_status, core_tag, core_radius, core_x, core_y, core_z, core_vx, core_vy, core_vz, fof_halo_tag, parent_fof, fof_halo_mass, top_host_row


def read_core_matrices_LJ(core_forest_file_name):
    
    simulation = haccytrees.Simulation.simulations["LastJourney"]
    forest_file = h5py.File(core_forest_file_name)
    forest_data = {k: d[:] for k, d in forest_file["data"].items()}
    # corematrix_reader(filename: str, simulation: Union[Simulation, str], include_fields: List[str] = None)
    forest_matrices = corematrix_reader(core_forest_file_name, simulation)
    print('ALL KEYS: ', forest_matrices.keys())
    #####################
    
    ### FOR ANY HOST PROPERTIES, USE THIS SNIPPET. 

    mask = forest_matrices["top_host_row"] > 1
    _full_idx = (forest_matrices["top_host_row"][mask], forest_matrices["snapnum"][mask])
    parent_fof_mass = np.empty_like(forest_matrices["infall_fof_halo_mass"])
    parent_fof_mass[:] = -1
    parent_fof_mass[mask] = forest_matrices["infall_fof_halo_mass"][_full_idx]
    
    # to calculate relative positions, similarly create parent_fof_x, parent_fof_y  and parent_fof_z  
    # using either infall_fof_halo_center_x or the core positions x , should be the same
    # then you can compare parent_fof_x vs x to get the relative offset
    ######################
    
    core_status = forest_matrices['core_state'][()]
    core_tag = forest_matrices['core_tag'][()]
    core_radius = forest_matrices['radius'][()]   ## 'core_radius' before
    core_x = forest_matrices['x'][()] ## 'core_x' before
    core_y = forest_matrices['y'][()]
    core_z = forest_matrices['z'][()]

    core_vx = forest_matrices['vx'][()] ## 'core_x' before
    core_vy = forest_matrices['vy'][()]
    core_vz = forest_matrices['vz'][()]
    
    
    fof_halo_tag = forest_matrices['fof_halo_tag'][()]
    parent_fof = forest_matrices['fof_halo_tag'][()] ## 'parent_fof' before
    # fof_halo_mass = forest_matrices['fof_halo_mass'][()] ## will be different now -- use indices
    top_host_row = forest_matrices['top_host_row'][()]  
    fof_halo_mass = parent_fof_mass ## mass of the host halo 
    
    ######################
    
    valid_cores = np.where(np.max(fof_halo_mass, axis=1) != -1)[0]

    core_status = core_status[valid_cores]
    core_tag = core_tag[valid_cores]
    core_radius = core_radius[valid_cores]
    core_x = core_x[valid_cores]
    core_y = core_y[valid_cores]
    core_z = core_z[valid_cores]
    
    core_vx = core_vx[valid_cores]
    core_vy = core_vy[valid_cores]
    core_vz = core_vz[valid_cores]
    
    
    fof_halo_tag = fof_halo_tag[valid_cores]
    parent_fof = parent_fof[valid_cores]
    top_host_row = top_host_row[valid_cores]
    fof_halo_mass = fof_halo_mass[valid_cores]
    
    
    print('\n')
    print("Core matrices dimensions: ", core_status.shape)
    print('\n')
    print('Based on final time step')
    fracCentrals = fof_halo_mass[core_status[:, -1] == 0].shape[0]/fof_halo_mass.shape[0]
    print('Fraction of Centrals: ', fracCentrals )

    fracSatellite = fof_halo_mass[core_status[:, -1] == 1].shape[0]/fof_halo_mass.shape[0]
    print('Fraction of Satellites: ', fracSatellite )

    fracMerged = fof_halo_mass[core_status[:, -1] == 2].shape[0]/fof_halo_mass.shape[0]
    print('Fraction of Merged cores: ', fracMerged )
    
    return core_status, core_tag, core_radius, core_x, core_y, core_z, core_vx, core_vy, core_vz, fof_halo_tag, parent_fof, fof_halo_mass, top_host_row
    
    






def _first_nonzero(arr, axis, invalid_val=-1):
    # mask = arr!=0
    mask = ( (arr!=0) & (arr!=-1) )   ## Changed for LJ

    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def infall_time_alphaq(core_status, HACC_analysis_age):
    first_nonzero_status = _first_nonzero(core_status, axis=1, invalid_val=-1)
    time_infall = HACC_analysis_age[first_nonzero_status]
    return time_infall


def _mass_times_indx_HACC(fof_halo_mass, cut):
    peak = np.max(fof_halo_mass, axis=1)
    # half_mpeak = np.log10(cut*peak)
    # mpeak_history = np.log10(fof_halo_mass + 1e-32 )
    
    half_mpeak = np.log10(cut*peak + 2) ## Changed for LJ 
    mpeak_history = np.log10(fof_halo_mass + 2 ) ## Changed for LJ 
    timeX_indx = np.argmin(np.abs(mpeak_history.T - half_mpeak), axis=0)
    
    return timeX_indx

def mass_time_alphaq(fof_halo_mass, HACC_analysis_age, cut):
    tcut_idx = _mass_times_indx_HACC(fof_halo_mass, cut)
    tcut_idx =  np.where(tcut_idx < 97, tcut_idx, 96)  ## Changed for LJ TEMPORARILY becasue of LC step mistmatch

    tcut = HACC_analysis_age[tcut_idx]
    
    return tcut



def peak_mass_HACC(fof_halo_mass):
    peak_mass = np.max(fof_halo_mass, axis=1)
    return peak_mass

def mass_rank_alphaq(fof_halo_mass, volume):
    peak_mass = peak_mass_HACC(fof_halo_mass)
    # rank_mass = np.arange(peak_mass.shape[0])/volume
    rank_mass = 1.0*np.argsort(peak_mass)/volume
    return rank_mass