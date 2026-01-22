import numpy as np
import pygio


def match_lc_tree(parent_fofID_intimestep, 
                  lc_ID):
    '''
    return sorted index and matching indices
    '''
    sort_idx = parent_fofID_intimestep.argsort()
        
    ### CHECK THIS -- done to avoid below errors
    # IndexError: index 533584 is out of bounds for axis 0 with size 533584    
    
    lc_ID = lc_ID[lc_ID > parent_fofID_intimestep.min()]
    lc_ID = lc_ID[lc_ID < parent_fofID_intimestep.max()]

    ################
    
    searchsorted_idx = sort_idx[np.searchsorted(parent_fofID_intimestep, lc_ID, sorter = sort_idx)]
    match_indx = np.where(parent_fofID_intimestep[searchsorted_idx] == lc_ID)
    return searchsorted_idx, match_indx


# https://halotools.readthedocs.io/en/latest/_modules/halotools/utils/crossmatch.html#crossmatch
# HALOTOOL implementation differs sligthy. Why? -- only unique/ non-unique issue


def matched_halo_lightcone(core_type, 
                           HACC_analysis_steps, 
                           HACC_step_X_start, 
                           HACC_step_X_end, 
                           dirInLC, 
                           parent_fof, 
                           core_status, 
                           matching_summaries):
    
    ## change from HACC simulation steps to redshifts
    
    if (core_type=="Central"):
        core_status_selected = 0
    if (core_type=="Satellite"):
        core_status_selected = 1
    if (core_type=="Merged"):
        core_status_selected = 2
    
    step_num_arr = HACC_analysis_steps[HACC_step_X_start:HACC_step_X_end]

    match_id = []
    match_xyz = np.empty((0, 3), dtype=float)
    match_vel_xyz = np.empty((0, 3), dtype=float)
    match_redshift = []
    
    
    matched_summaries = np.empty((0, matching_summaries.shape[1]), dtype=float)
    # match_rank_time = np.empty((0, 2), dtype=float)

    total_lc_objects = 0 
    
    for step_num in step_num_arr:
        
        

        time_stepID = np.where(HACC_analysis_steps==step_num)
        # print(step_num, time_stepID[0][0])

        # lc_data = pygio.read_genericio(dirInLC + 'lc_halos_'+str(step_num)+'/lc_halos.' + str(step_num))
        # lc_data_id = np.abs(lc_data['id'])

            
        lc_data = pygio.read_genericio(dirInLC + 'lcHalos'+str(step_num)+'/lc_intrp_halos_matched.' + str(step_num)) # Changed for LJ
        lc_data_id = np.abs(lc_data['fof_halo_tag']) # # Changed for LJ
        
        total_lc_objects += lc_data_id.shape[0]
        
        

        if (core_status_selected==2):
            core_merger_cond = np.where( 
                (core_status[:, HACC_step_X_start] != 2) & 
                (core_status[:, HACC_step_X_end] == 2) )
            
            HACC_parent_fof_new =  parent_fof[core_merger_cond]
        
        else: 
            HACC_parent_fof_new =  parent_fof[core_status[:, -1] == core_status_selected] ## For HACC centrals only - change this to redshift dependent metrics
        
        # print(HACC_parent_fof_new.shape, matching_summaries.shape)

        searchsorted_idx, match_indx = match_lc_tree(HACC_parent_fof_new[:, time_stepID[0][0] + 1], # +1 from Patricia's debugging 
                                                                             lc_data_id)

        # print('z=',  1./cosmology_utils.a(step_num) - 1)
        # print(lc_data['id'][match_indx].shape, matching_summaries[searchsorted_idx][match_indx].shape)

        match_id = np.append(match_id, lc_data_id[match_indx])
        match_xyz = np.append(match_xyz, np.array([lc_data['x'][match_indx], lc_data['y'][match_indx], lc_data['z'][match_indx]]).T, axis=0)
        
        match_vel_xyz = np.append(match_vel_xyz, np.array([lc_data['vx'][match_indx], lc_data['vy'][match_indx], lc_data['vz'][match_indx]]).T, axis=0)
        
        ### match_xyz will be improved in the future -- currently satellites have the same position as the halo centers from the lightcones. We will use the index information in the future to compute the offset (dx dy dz relative distances) from the halo center to compute the match_xyz for satellites. 
        ### top host row will be used. it will have the central info (index?)
        match_redshift = np.append(match_redshift, ( (1./lc_data['a'][match_indx]) - 1))
        
        matched_summaries = np.append(matched_summaries, matching_summaries[searchsorted_idx][match_indx], axis=0)
        # match_rank_time = np.append(match_rank_time, rank_time_0[searchsorted_idx][match_indx], axis=0)
     
    print('Total Lightcone objects: ', total_lc_objects)
    
    return match_id, match_xyz, match_vel_xyz, match_redshift, matched_summaries

