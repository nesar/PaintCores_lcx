import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap
from getdist import plots, MCSamples
import corner
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as mcolors


# plt.rcParams.update({
#     "text.usetex": True,
#     "figure.facecolor": "w"
# })


import SetPub
SetPub.set_pub()


def basemap_plot(ra_sky, dec_sky):
    from mpl_toolkits.basemap import Basemap
    # plt.style.use('dark_background')
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    fig = plt.figure(figsize=(10, 5))

    ##########################
    ax = fig.add_subplot(111)

    # Only plotting random 10,000 galaxies
    n_gal = 500
    random_gal_indices = np.random.randint(low=0, high=dec_sky.shape[0], size=n_gal)

    ra_octant = np.array(ra_sky[random_gal_indices])
    dec_octant = np.array(dec_sky[random_gal_indices])

    # Define the orthographic projection centered on the equator and prime meridian
    m = Basemap(projection='moll', lat_0=90, lon_0=180, resolution='c')
    # Convert RA, Dec to x, y coordinates for plotting
    x, y = m(ra_octant, dec_octant)

    # Plot the sky distribution
    m.scatter(x, y, s=1, c='white', alpha=0.25, edgecolors='w', linewidth=1)

    # Draw parallels and meridians
    m.drawparallels(np.arange(-90.,90.,45), color='yellow', textcolor='yellow', linewidth=2)
    m.drawmeridians(np.arange(0.,360.,45), color='yellow', textcolor='yellow', linewidth=2)
    m.drawmapboundary(fill_color='black')
    # m.drawcoastlines(color='black', linewidth=0.5)

    plt.suptitle('Sky Distribution of Galaxies in full sky', fontsize=20)
    plt.show()
    


def plot_cores_2d(core_x, core_y, core_z, core_radius, core_status, begin_xyz, end_xyz):
    plt.clf()
    plt.close('all')


    selectIDs = np.where((core_x > begin_xyz[1]) & 
                         (core_x < end_xyz[2]) & 
                         (core_y > begin_xyz[1])  & 
                         (core_y < end_xyz[1]) & 
                         (core_z > begin_xyz[2])  & 
                         (core_z < end_xyz[2]) )


    color_arr = ['k', 'r', 'b']

    fig, ax = plt.subplots()

    for sel in range(selectIDs[0].shape[0]):
        ax.add_patch(plt.Circle(
            (core_x[selectIDs[0], -1][sel], 
             core_y[selectIDs[0], -1][sel] ), 
            30*core_radius[selectIDs[0], -1][sel], 
            color= color_arr[core_status[selectIDs[0], -1][sel]], 
            alpha=0.8, fill=False))

    ax.set_aspect('equal', adjustable='datalim')
    ax.plot()   #Causes an autoscale update.
    plt.savefig('plots/core2d.png', bbox_inches='tight')

    plt.show()
    
    
    
def plot_accretion_histories(HACC_analysis_age, fof_halo_mass, core_status, core_type):
    plt.clf()
    plt.close('all')
    
    if (core_type=="Central"):
        core_status_selected = 0
    if (core_type=="Satellite"):
        core_status_selected = 1
    if (core_type=="Merged"):
        core_status_selected = 2
     
    plt.figure(14, figsize=(12, 4))
    for random_halo_idx in np.random.randint(low=0, high=90000, size=4): #random haloes
        plt.plot(HACC_analysis_age, np.log10(fof_halo_mass[core_status[:, -1] == core_status_selected][random_halo_idx]), '-')

    plt.xlabel('Time')
    plt.ylabel(r'log($M_{peak}$)')

    plt.title(core_type + ' core -- mass accretion history')
    # plt.legend(ncol=1)
    plt.xlim(0.1, 13.9)
    plt.savefig('plots/mah.png', bbox_inches='tight')
    plt.show()
    
    
def plot_core_status(core_status):
    plt.clf()
    plt.close('all')
    
    fig = plt.figure(12, figsize=(9 , 10))
    ax1 = fig.add_subplot(1,1,1)


    cMap = ListedColormap(['white', 'blue', 'red'])

    im = ax1.imshow(core_status[-400: -350, :], cmap=cMap)

    spacing = 2.0 # This can be your user specified spacing. 
    minorLocator = MultipleLocator(spacing)
    ax1.yaxis.set_minor_locator(minorLocator)
    # ax1.yaxis.set_major_locator(minorLocator)
    ax1.grid(which = 'minor')


    # cbar = fig.colorbar(im, orientation='horizontal')

    cbar = fig.colorbar(im, ticks=[0.5, 1, 1.5], orientation='horizontal')
    cbar.ax.set_xticklabels(['Central', 'Satellite', 'Merged'])  # horizontal colorbar

    cbar.set_label('Core status', rotation=0)

    plt.xlabel('Snapshot timesteps')
    plt.ylabel('core ID')

    plt.title('Cores - Core status')
    plt.savefig('plots/core_status.png', bbox_inches='tight')
    plt.show()
    
    
def plot_summary_histograms(summary, core_status, summary_type):
    plt.clf()
    plt.close('all')
    
    '''
    summary == mass, time, rank_peak_mass
    
    
    '''
    if (summary_type == "Time infall"):
        plt.figure(28, figsize=(9,6))

        plt.hist(summary[core_status[:, -1] == 1], bins=40, histtype='step', label='Satellite', lw=1.5)
        plt.hist(summary[core_status[:, -1] == 2], bins=40, histtype='step', label='Merged', lw=1.5)
        plt.hist(summary[core_status[:, -1] != 0], bins=40, label='all non-centrals', histtype='stepfilled', alpha=0.2)

        plt.title(summary_type + ' - Summary')
        plt.xlabel('Time')
        plt.ylabel('N')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.savefig('plots/hist_'+ summary_type +'.png', bbox_inches='tight')
        plt.show()
 

    elif (summary_type == "Peak mass"):
        
        
        plt.figure(29, figsize=(9,6))
        plt.hist(np.log10(summary[core_status[:, -1] == 0] + 1e-32), bins=100, histtype='step', label='Centrals', lw=1.5);
        plt.hist(np.log10(summary[core_status[:, -1] == 1] + 1e-32), bins=100, histtype='step', ls='dashed', label='Satellites', lw=1.5);
        plt.hist(np.log10(summary[core_status[:, -1] == 2] + 1e-32), bins=100, histtype='step', alpha=0.6, label='Merged cores', lw=1.5);


        plt.title(summary_type + ' - Summary')

        plt.yscale('log')
        plt.xlabel(r'$log(M_{peak})$')
        plt.ylabel(r'$N(log(M_{peak}))$')

        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.savefig('plots/hist_'+ summary_type +'.png', bbox_inches='tight')
        plt.show()
        
        
    elif (summary_type == 'Rank Peak core mass'):
        plt.figure(25, figsize=(9,6))
        plt.hist(summary[core_status[:, -1] == 0], density=True, alpha=0.4, lw=1.5, label='Centrals', bins=100);
        plt.hist(summary[core_status[:, -1] == 1], density=True, alpha=0.4, lw=1.5, label='Satellites', bins=100);
        plt.hist(summary[core_status[:, -1] == 2], density=True, alpha=0.4, lw=1.5, label='Merged', bins=100);

        # plt.hist(rank_sorted_mpeak, bins = 10, density=True);
        plt.title(summary_type + ' - Summary')

        # plt.yscale('log')
        plt.xlabel(r'r/Vol')
        plt.ylabel(r'N(r/Vol)')

        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.savefig('plots/hist_'+ summary_type +'.png', bbox_inches='tight')
        
        plt.show()

    else:
        plt.figure(30, figsize=(9,6))
        plt.hist(summary[core_status[:, -1] == 1], bins=40, histtype='step', lw=1.5, label='Satellites')
        plt.hist(summary[core_status[:, -1] == 2], bins=40, histtype='step', lw=1.5, label='Merged cores')
        plt.hist(summary[core_status[:, -1] == 0], bins=40, histtype='step', lw=1.5, label='Centrals')

        plt.title(summary_type + ' - Summary')
        plt.xlabel('Time')
        plt.ylabel('N')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.savefig('plots/hist_'+ summary_type +'.png', bbox_inches='tight')

        plt.show()
        
        
def plot_triangle_core(time_25, time_50, time_infall, peak_mass, core_status):
    plt.clf()
    plt.close('all')
    
    s0_HACC = [time_25[core_status[:, -1] == 0], time_50[core_status[:, -1] == 0], time_infall[core_status[:, -1] == 0], np.log10(peak_mass[core_status[:, -1] ==0]) ] 
    s1_HACC = [time_25[core_status[:, -1] == 1], time_50[core_status[:, -1] == 1], time_infall[core_status[:, -1] == 1], np.log10(peak_mass[core_status[:, -1] ==1]) ] 
    s2_HACC = [time_25[core_status[:, -1] == 2], time_50[core_status[:, -1] == 2], time_infall[core_status[:, -1] == 2], np.log10(peak_mass[core_status[:, -1] ==2]) ] 

    allLabels = ['t_{25}', 't_{50}', 't_{infall}', 'log(M_{peak})']

    print(allLabels)

    print(10*'=')

    samples0 = MCSamples(samples= s0_HACC, 
                         labels = allLabels, 
                         names= allLabels, 
                         label = 'HACC sim Centrals', 
                         settings={'mult_bias_correction_order':-1,'smooth_scale_2D':2, 'smooth_scale_1D':2})
    
    samples1 = MCSamples(samples= s1_HACC, 
                         labels = allLabels, 
                         names= allLabels, 
                         label = 'HACC sim Unmerged cores', 
                         settings={'mult_bias_correction_order':-1,'smooth_scale_2D':2, 'smooth_scale_1D':2})
    
    
    samples2 = MCSamples(samples= s2_HACC, 
                         labels = allLabels, 
                         names= allLabels, 
                         label = 'HACC sim Merged cores', 
                         settings={'mult_bias_correction_order':-1,'smooth_scale_2D':2, 'smooth_scale_1D':2})



    g = plots.get_subplot_plotter(subplot_size=5)
    g.settings.axes_fontsize=27
    g.settings.axes_labelsize = 27
    g.settings.legend_fontsize = 27
    g.settings.fontsize = 27
    g.settings.alpha_filled_add=0.3
    # g.settings.title_limit_fontsize = 27
    g.settings.solid_contour_palefactor = 0.5
    g.settings.num_plot_contours = 3

    g.triangle_plot([samples0, samples1, samples2], allLabels , filled=[False, False, False, False], contour_colors=['green', 'red', 'black', 'indigo'], contour_lws=2)
    # g.triangle_plot([samples0], allLabels , filled=[False, False, False, False], contour_colors=['green', 'red', 'black', 'indigo'], contour_lws=2)

    g.export('plots/HACC_sim_time_mass.png')

    plt.legend()
    plt.show()
    

def plot_traingle_bpl_hacc_centrals(time_25_hacc, time_50_hacc, peak_mass_hacc, core_status, time_25_bpl, time_50_bpl, peak_mass_bpl):
    plt.clf()
    plt.close('all')


    s0_HACC = [time_25_hacc[core_status[:, -1] == 0], time_50_hacc[core_status[:, -1] == 0], np.log10(peak_mass_hacc[core_status[:, -1] ==0]) ] 

    s0_bpl = [time_25_bpl, time_50_bpl, np.log10(peak_mass_bpl)] 
    # s1_bpl = [t25_a06, t50_a06, np.log10(sorted_mpeak_a06), sorted_a_infall1_a06] 
    # s2_bpl = [t25_a05, t50_a05, np.log10(sorted_mpeak_a05), sorted_a_infall1_a05] 
    allLabels = ['t_{25}', 't_{50}', 'log(M_{peak})']


    # s0 = [np.log10(sorted_mpeak_a1), sorted_a_infall1_a1] 
    # s1 = [np.log10(sorted_mpeak_a06), sorted_a_infall1_a06] 
    # s2 = [np.log10(sorted_mpeak_a05), sorted_a_infall1_a05, sorted_a_infall2_a05] 
    # allLabels = ['log(M_{peak})', 'a-first-infall', 'a-last-infall']


    print(allLabels)

    print(10*'=')

    samples0_bpl = MCSamples(samples= s0_bpl, 
                             labels = allLabels, 
                             names= allLabels, 
                             label = 'BPL a=1', 
                             settings={'mult_bias_correction_order':-1,'smooth_scale_2D':2, 'smooth_scale_1D':2})
    
    samples0_HACC = MCSamples(samples= s0_HACC, 
                                labels = allLabels, 
                                names= allLabels, 
                                label = 'HACC Centrals',
                                settings={'mult_bias_correction_order':-1,'smooth_scale_2D':2, 'smooth_scale_1D':2})

    # samples1_bpl = MCSamples(samples= s1, labels = allLabels, names= allLabels, label = 'a=0.664')
    # samples2_bpl = MCSamples(samples= s2, labels = allLabels, names= allLabels, label = 'a=0.5')


    g = plots.get_subplot_plotter(subplot_size=5)
    g.settings.axes_fontsize=27
    g.settings.axes_labelsize = 27
    g.settings.legend_fontsize = 27
    g.settings.fontsize = 27
    g.settings.alpha_filled_add=0.1
    # g.settings.title_limit_fontsize = 27
    g.settings.solid_contour_palefactor = 0.2
    g.settings.num_plot_contours = 2

    # g.triangle_plot([samples0_bpl, samples1_bpl, samples2_bpl], allLabels , filled=[False, False, False, False, False], contour_colors=['green', 'red', 'black', 'indigo', 'blue'], contour_lws=3)
    g.triangle_plot([samples0_bpl, samples0_HACC], allLabels , filled=[True, False, False, False, False], contour_colors=['green', 'red', 'black', 'indigo', 'blue'], contour_lws=1)

    g.export('plots/bpl_time_mass.png')

    plt.legend()
    plt.show()

    
    
    
def plot_traingle_bpl_hacc_satellites(time_50_hacc, time_infall_hacc, peak_mass_hacc, core_status, time_50_bpl, time_infall_bpl, peak_mass_bpl):
    plt.clf()
    plt.close('all')


    s0_HACC = [time_50_hacc[core_status[:, -1] == 1], time_infall_hacc[core_status[:, -1] == 1], np.log10(peak_mass_hacc[core_status[:, -1] ==1]) ] 

    s0_bpl = [time_50_bpl, time_infall_bpl, np.log10(peak_mass_bpl)] 
    # s1_bpl = [t25_a06, t50_a06, np.log10(sorted_mpeak_a06), sorted_a_infall1_a06] 
    # s2_bpl = [t25_a05, t50_a05, np.log10(sorted_mpeak_a05), sorted_a_infall1_a05] 
    allLabels = ['t_{50}', 't_{infall}', 'log(M_{peak})']


    # s0 = [np.log10(sorted_mpeak_a1), sorted_a_infall1_a1] 
    # s1 = [np.log10(sorted_mpeak_a06), sorted_a_infall1_a06] 
    # s2 = [np.log10(sorted_mpeak_a05), sorted_a_infall1_a05, sorted_a_infall2_a05] 
    # allLabels = ['log(M_{peak})', 'a-first-infall', 'a-last-infall']


    print(allLabels)

    print(10*'=')

    samples0_bpl = MCSamples(samples= s0_bpl, labels = allLabels, names= allLabels, label = 'BPL a=1')
    samples0_HACC = MCSamples(samples= s0_HACC, labels = allLabels, names= allLabels, label = 'HACC sim Centrals')

    # samples1_bpl = MCSamples(samples= s1, labels = allLabels, names= allLabels, label = 'a=0.664')
    # samples2_bpl = MCSamples(samples= s2, labels = allLabels, names= allLabels, label = 'a=0.5')


    g = plots.get_subplot_plotter(subplot_size=5)
    g.settings.axes_fontsize=27
    g.settings.axes_labelsize = 27
    g.settings.legend_fontsize = 27
    g.settings.fontsize = 27
    g.settings.alpha_filled_add=0.1
    # g.settings.title_limit_fontsize = 27
    g.settings.solid_contour_palefactor = 0.2
    g.settings.num_plot_contours = 2

    # g.triangle_plot([samples0_bpl, samples1_bpl, samples2_bpl], allLabels , filled=[False, False, False, False, False], contour_colors=['green', 'red', 'black', 'indigo', 'blue'], contour_lws=3)
    g.triangle_plot([samples0_bpl, samples0_HACC], allLabels , filled=[True, False, False, False, False], contour_colors=['green', 'red', 'black', 'indigo', 'blue'], contour_lws=1)

    g.export('plots/bpl_time_mass_satellites.png')

    plt.legend()
    plt.show()

    
    
def plot_SMHM_comparison(Mpeak, Mstar, plt_title):
    plt.clf()
    plt.close('all')
    
    # SMHM

    behroozi = np.loadtxt('/lcrc/project/cosmo_ai/nramachandra/Projects/SPHEREx/MAH/data/mstar_mhalo/Behroozi2012.txt', delimiter=',')
    moster = np.loadtxt('/lcrc/project/cosmo_ai/nramachandra/Projects/SPHEREx/MAH/data/mstar_mhalo/Moster2013.txt', delimiter=',')

    

    plt.figure(32, figsize=(9,6))
    
    h = 1.0

    corner.hist2d( x=np.log10(Mpeak), 
                   y=np.log10(Mstar), 
                          bins=30, 
                          smooth=1.0,
                          new_fig=False, 
                          labels = 'Synthetic', 
                          color = 'r', 
                          fill_contours=True, 
                          levels=(0.9, 0.95, 0.98,),
                          alpha=(0.1, 0.2, 0.3, 0.4, ),
                          range=[[11, 13], [8, 12]],
                          plot_density=True, 
                          plot_contours=True,
                          data_kwargs = {"ms":1, "alpha":0.6}
                          )

    plt.plot(behroozi[:, 0]*h, behroozi[:, 1]*h, lw =2, ls='dashed', color='g', label='Behroozi 2012')
    plt.plot(moster[:, 0]*h, moster[:, 1]*h, lw=2, ls='dashed', color='b', label='Moster 2013')

    # plt.xscale('log')
    # plt.yscale('log')

    plt.xlim(11, 13)
    plt.ylim(8, 12)
    plt.xlabel(r'$log(M_{peak})$')
    plt.ylabel(r'$log(M_{star})$')
    plt.legend(title='SMHM relation', bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title('SMHM - '+ plt_title)
    plt.show()
    plt.savefig('plots/smhm_'+plt_title+'.png', bbox_inches='tight')
    
    
def plot_GSMF(Mstar, bpl_data, plt_title):
    plt.clf()
    plt.close('all')
    # https://astronomy.stackexchange.com/questions/21138/how-do-i-create-a-galaxy-stellar-mass-function
    
    nbins = 25                             #Number of bins to divide data into
    V     = 256**3                             #Survey volume in Mpc3
    Phi,edg = np.histogram(np.log10(Mstar),bins=nbins) #Unnormalized histogram and bin edges
    dM    = edg[1] - edg[0]                 #Bin size
    Max   = edg[0:-1] + dM/2.               #Mass axis
    Phi   = Phi / V / dM                    #Normalize to volume and bin size

    cond_bpl = bpl_data['sm'][bpl_data['upid'] == -1] > 1e9
    Mstar_bpl  = bpl_data['sm'][bpl_data['upid'] == -1][cond_bpl]
    V_bpl     = 400**3                             #Survey volume in Mpc3
    Phi_bpl ,edg_bpl  = np.histogram(np.log10(Mstar_bpl), bins=nbins) #Unnormalized histogram and bin edges
    dM_bpl     = edg_bpl [1] - edg_bpl [0]                 #Bin size
    Max_bpl    = edg_bpl [0:-1] + dM_bpl /2.               #Mass axis
    Phi_bpl    = Phi_bpl  / V_bpl / dM_bpl                 #Normalize to volume and bin size

    
    # plt.clf()
    plt.figure(33, figsize=(9,6))
    
    plt.title('Galaxy Stellar mass function --' + plt_title, fontsize=14)
    plt.yscale('log')
    plt.xlabel(r'$\log(M_\star\,/\,M_\odot)$', fontsize=14)
    plt.ylabel(r'$\Phi\,/\,\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}$', fontsize=14)
    plt.plot(Max, Phi, ls='-', label='HACC sim lightcone at z cut')
    plt.plot(Max_bpl, Phi_bpl, ls='-', label='BPL snapshot at z=0')

    plt.xlim(9.0, 11.8)

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
    plt.savefig('plots/gsmf' + plt_title + '.png', bbox_inches='tight')
    plt.show()
    
    



def plot_skymap(ra, dec, scatter_size, scatter_color, plt_title):
    plt.clf()
    plt.close('all')
    
    fig = plt.figure(figsize=(12, 12))

    plt.xlabel("RA", color='black', fontsize=25);
    plt.ylabel("Dec", color='black', fontsize=25);

    m = Basemap(projection = 'ortho', lon_0 = 50, lat_0 = 0,
                llcrnrx=-3000000, llcrnry=1000000, urcrnrx=2000000, urcrnry=3500000, 
                resolution = 'l')

    
    sizeO = np.log10( scatter_size )
    # # rescaledO = (sizeO - np.min(sizeO))/ (np.max(sizeO) - np.min(sizeO))*(100)
    rescaledO =  (sizeO-np.min(sizeO))**3
    reverseO = np.cbrt(rescaledO) + np.min(sizeO)


    x, y = m(ra, dec)  # transform coordinates

    sc = plt.scatter(x, y, s=rescaledO, 
                     marker='o', 
                     c=np.log10(scatter_color), 
                     alpha=0.3, 
                     lw = 0, 
                     cmap=plt.get_cmap('coolwarm'), 
                     # vmin=14, 
                     # vmax=18, 
                     label=str(reverseO) ) 

    # plt.rcParams["legend.markerscale"] = 0.9
    # plt.legend(*sc.legend_elements(prop="sizes", alpha=1.0, num=4), loc="upper right", title="rescaled log(sSFR)")


    # handles, labels = sc.legend_elements(prop="sizes", alpha=1.0, num=4)     
    # labels_new = ["< 5000", "< 20000", " <50000", "> 50000"]     
    # plt.legend(handles, labels_new, loc="upper right", title="log(Mstar)")

    # cbar_ax = fig.add_axes([1.01, 0.3, 0.02, 0.3])
    
    clb = plt.colorbar(sc,  shrink = 0.4)#, cax=cbar_ax)
    clb.set_label('log(Luminosity)', rotation=270, fontsize=15, labelpad=15)


    m.drawparallels(np.arange(-90.,120.,10.), color='yellow', textcolor='yellow', linewidth=1.5)
    m.drawmeridians(np.arange(0.,420.,10.), color='yellow', textcolor='yellow', linewidth=1.5)
    m.drawmapboundary(fill_color='black')

    plt.title("Skymap -" + plt_title)
    


    plt.savefig('plots/sky_lum_'+ plt_title +'.png',bbox_inches='tight',dpi = 300)
    plt.show()

    
    

def plot_SED(pcolor_all, wave_unnred, redshift_in, wave_min, wave_max, plt_title):
    plt.clf()
    plt.close('all')
    
    np.random.seed(33)
    plt.figure(figsize = (12, 4))

    # galID_arr =  np.arange(0, pcolor_all.shape[1])
    galID_arr =  np.random.randint(0, pcolor_all.shape[1], 5)

    colorparams = galID_arr
    colormap = plt.get_cmap('viridis', 10)
    normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))


    for idx, galID in enumerate(galID_arr):

        color = colormap(normalize(galID))

        wave_mask = np.where( (wave_unnred[0]>wave_min) & (wave_unnred[0]<wave_max))


        each_wave_red = wave_unnred[:, wave_mask][0, 0, :]*(1 + redshift_in[0][galID])

        plt.plot(each_wave_red, pcolor_all[1, galID].T,  color = color, alpha=1, label='z=%.2f'%redshift_in[0, galID]);
        # plt.plot(wave, np.median(pcolor_all[:, galID, :].T, axis=1)*1./(4*np.pi*dd1**2),  color = color, alpha=1, linewidth = 0.8, label='z=%.2f'%redshift_in[0, idx]);
        # plt.plot(wave, np.median(pcolor_all[:, galID, :].T, axis=1),  color = color, alpha=1, linewidth = 0.8, linestyle = '-.', label='z=%.2f'%redshift_in[0, idx]);


    plt.ylabel('Flux per unit wavelength')
    plt.xlabel('Wavelength (Angstrom) ')

    plt.yscale('log')
    plt.legend(ncol=1, title='Redshift')
    plt.xlim(wave_min, wave_max)

    plt.savefig('plots/sed_'+ plt_title +'.png',bbox_inches='tight',dpi = 300)
    plt.show()
    
    