import h5py

def save_hdf5(fileIn, galaxy_id, ra_catalog, dec_catalog, redshift_catalog, redshift_total_catalog, mstar_catalog, mhalo_catalog, halo_id, is_central, SFH_catalog, SFH_tt_catalog, pcolor_catalog, lum_catalog, wave_catalog, position_x, position_y, position_z, velocity_x, velocity_y, velocity_z, sdss_mags, spherex_mags, cosmos_mags, wise_mags, ls_mags, mass2_mags, f784_mags):
    
    
    print(ra_catalog.shape, redshift_catalog.shape, mstar_catalog.shape, mhalo_catalog.shape, SFH_catalog.shape, pcolor_catalog.shape, lum_catalog.shape, galaxy_id.shape)

    with h5py.File(fileIn, 'w') as hf:

            hf.create_dataset('galaxy_id', data=galaxy_id, compression="gzip", compression_opts=9)
            hf.create_dataset('ra_true', data=ra_catalog, compression="gzip", compression_opts=9)
            hf.create_dataset('dec_true', data=dec_catalog, compression="gzip", compression_opts=9)
            hf.create_dataset('redshift_true', data=redshift_total_catalog, compression="gzip", compression_opts=9)
            hf.create_dataset('redshift_total', data=redshift_catalog, compression="gzip", compression_opts=9)
            # hf.create_dataset('sSFR', data=sSFR_catalog, compression="gzip", compression_opts=9)
            hf.create_dataset('halo_id', data=halo_id, compression="gzip", compression_opts=9)
            hf.create_dataset('is_central', data=is_central, compression="gzip", compression_opts=9)
            hf.create_dataset('stellar_mass', data=mstar_catalog, compression="gzip", compression_opts=9)
            hf.create_dataset('halo_mass', data=mhalo_catalog, compression="gzip", compression_opts=9)
            hf.create_dataset('SFH', data=SFH_catalog, compression="gzip", compression_opts=9)
            hf.create_dataset('time_bins_SFH', data=SFH_tt_catalog, compression="gzip", compression_opts=9)
            # hf.create_dataset('EBV', data=ebv_catalog, compression="gzip", compression_opts=9)
            # hf.create_dataset('Metallicity', data=metal_catalog, compression="gzip", compression_opts=9)
            hf.create_dataset('SED', data=pcolor_catalog, compression="gzip", compression_opts=9)
            hf.create_dataset('SED_wavelength', data=wave_catalog, compression="gzip", compression_opts=9)
            hf.create_dataset('luminosity', data=lum_catalog, compression="gzip", compression_opts=9)
            
            hf.create_dataset('position_x', data=position_x, compression="gzip", compression_opts=9)
            hf.create_dataset('position_y', data=position_y, compression="gzip", compression_opts=9)
            hf.create_dataset('position_z', data=position_z, compression="gzip", compression_opts=9)
            
            hf.create_dataset('velocity_x', data=velocity_x, compression="gzip", compression_opts=9)
            hf.create_dataset('velocity_y', data=velocity_y, compression="gzip", compression_opts=9)
            hf.create_dataset('velocity_z', data=velocity_z, compression="gzip", compression_opts=9)
            
            
            # for band in 'ugriz':
            hf.create_dataset('mag_u_sdss', data=sdss_mags[:, 0], compression="gzip", compression_opts=9)
            hf.create_dataset('mag_g_sdss', data=sdss_mags[:, 1], compression="gzip", compression_opts=9)
            hf.create_dataset('mag_r_sdss', data=sdss_mags[:, 2], compression="gzip", compression_opts=9)
            hf.create_dataset('mag_i_sdss', data=sdss_mags[:, 3], compression="gzip", compression_opts=9)
            hf.create_dataset('mag_z_sdss', data=sdss_mags[:, 4], compression="gzip", compression_opts=9)
            hf.create_dataset('mag_Y_sdss', data=sdss_mags[:, 5], compression="gzip", compression_opts=9)
            
            for spherex_band in range(102):
                hf.create_dataset('mag_{}_spherex'.format(str(spherex_band)), 
                                  data=spherex_mags[:, spherex_band], 
                                  compression="gzip", 
                                  compression_opts=9)
                
            for cosmos_band in range(31):
                hf.create_dataset('mag_{}_cosmos'.format(str(cosmos_band)), 
                                  data=cosmos_mags[:, cosmos_band], 
                                  compression="gzip", 
                                  compression_opts=9)
 
            for wise_band in range(7):
                hf.create_dataset('mag_{}_wise'.format(str(wise_band)), 
                                  data=wise_mags[:, wise_band], 
                                  compression="gzip", 
                                  compression_opts=9)
            
            for ls_band in range(8):
                hf.create_dataset('mag_{}_ls'.format(str(ls_band)), 
                                  data=wise_mags[:, wise_band], 
                                  compression="gzip", 
                                  compression_opts=9)
                
            for mass2_band in range(3):
                hf.create_dataset('mag_{}_2mass'.format(str(mass2_band)), 
                                  data=mass2_mags[:, mass2_band], 
                                  compression="gzip", 
                                  compression_opts=9)
                
            for f784_band in range(2):
                hf.create_dataset('mag_{}_f784'.format(str(f784_band)), 
                                  data=f784_mags[:, f784_band], 
                                  compression="gzip", 
                                  compression_opts=9)

    hf.close()
    
    print('Saved: '+ fileIn)
        
    
