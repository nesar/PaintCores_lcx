from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
from astropy.cosmology import WMAP7 as cosmo
import astropy.constants as const
import simulation_models

def a(step):
    '''
    Since the step number corresponds to the higher redshift of the lightcone interval
    '''
    a = 1/201 + (1-1/201)/500*(step+1)
    return a

def redshift(a):
    z = (1/a) - 1
    return z


def ra_dec(x_Mpc, y_Mpc, z_Mpc):  
    c = SkyCoord(x=x_Mpc, y=y_Mpc, z=z_Mpc, unit='Mpc', representation_type='cartesian')
    c.representation_type = 'spherical'
    return c.ra, c.dec


#### Using peculiar velocities in updating RA/dec


def adjust_position_with_velocity(x_Mpc, y_Mpc, z_Mpc, vx, vy, vz, delta_t):
    # Convert velocity from km/s to Mpc/Myr (common astronomical units)
    # 1 km/s = 1.0227121650537077 * 10^-6 Mpc/Myr
    conversion_factor = 1.0227121650537077 * 10**-6
    
    # Adjust positions based on peculiar velocities and time interval
    x_Mpc += vx * conversion_factor * delta_t
    y_Mpc += vy * conversion_factor * delta_t
    z_Mpc += vz * conversion_factor * delta_t
    
    return x_Mpc, y_Mpc, z_Mpc

###################################################################
###################################################################

def get_lightcone_coords(x_Mpc_com, y_Mpc_com, z_Mpc_com):
    coords = SkyCoord(
        x=x_Mpc_com,
        y=y_Mpc_com,
        z=z_Mpc_com,
        unit="Mpc",
        representation_type="cartesian",
    )
    coords.representation_type = "spherical"
    return coords

def redshift_from_xyz(match_xyz):
    cosmo = simulation_models.get_cosmo("LastJourney")
    h = cosmo.h
    coords = get_lightcone_coords(match_xyz[:, 0]*h, match_xyz[:, 1]*h, match_xyz[:, 2]*h);
    coord_dist = coords.distance

    #  generate distance estimates for values between min and max redshifts
    zmin = np.min(0.0001)
    zmax = np.max(2)
    Nzgrid = 100

    zgrid = np.logspace(np.log10(zmin), np.log10(zmax), Nzgrid)
    # CDgrid = cosmology.comoving_distance(zgrid) * h

    CDgrid = cosmo.comovingDistance(z_min=0.0, z_max=zgrid, transverse=True)*h

    redshift_compute = np.interp(coord_dist.Mpc, CDgrid, zgrid)
    
    return redshift_compute


def pecZ(x, y, z, vx, vy, vz, z_hubb, obs=np.zeros(3)):
    # get relative position (r vector) of and position unit vector toward each object
    r_rel = np.vstack((x, y, z)).T - obs
    r_rel_mag = np.linalg.norm(r_rel, axis=1)
    r_rel_hat = r_rel / r_rel_mag[:, np.newaxis]

    # dot velocity vectors with relative position unit vector to get peculiar velocity
    v = np.vstack((vx, vy, vz)).T
    v_pec = np.einsum('ij,ij->i', v, r_rel_hat)

    # find total and peculiar z_hubb
    c = const.c.value / 1000  # speed of light in km/s
    z_pec = np.sqrt((1 + v_pec / c) / (1 - v_pec / c)) - 1
    z_tot = (1 + z_hubb) * (1 + z_pec) - 1

    # find the distorted distance from applying Hubble's law using the new z_tot z_hubbs
    a = 1 / (1 + z_hubb)
    r_dist = r_rel_mag + v_pec / 100.0 / cosmo.efunc(z_hubb) / a

    return z_pec, z_tot, v_pec, v_pec * a, r_rel_mag, r_rel_mag * a, r_dist



# def pecZ_snapshot(x, vx, z_hubb):
#     c = const.c.value / 1000  # speed of light in km/s
#     z_pec = np.sqrt((1 + vx / c) / (1 - vx / c)) - 1
#     z_tot = (1 + z_hubb) * (1 + z_pec) - 1

#     return z_pec, z_tot


###################################################################
###################################################################



'''
# Co-ords + redshift TO-DO:

1. Correct ra/dec
2. Check the snapshot based discontinuities
3. peculiar-z 
4. Satellite cores dx,dy,dz

'''

'''
# https://github.com/LSSTDESC/lsstdesc-diffsky/blob/d2bb23cca8a963f61b6144b5e27e69a366cbe662/lsstdesc_diffsky/write_mock_to_disk_singlemet.py#L1595

def get_sky_coords(
    dc2, cosmology, redshift_method="halo", Nzgrid=50, source_galaxy_tag="source_galaxy"
):
    #  compute galaxy redshift, ra and dec
    if redshift_method is not None:
        print(
            "\n.....Generating lightcone redshifts using {} method".format(
                redshift_method
            )
        )
        r = np.sqrt(dc2["x"] * dc2["x"] + dc2["y"] * dc2["y"] + dc2["z"] * dc2["z"])
        mask = r > 5000.0
        if np.sum(mask) > 0:
            print("WARNING: Found {} co-moving distances > 5000".format(np.sum(mask)))

        dc2["redshift"] = dc2["target_halo_redshift"]  # copy halo redshifts to galaxies
        H0 = cosmology.H0.value
        if redshift_method == "galaxy":
            #  generate distance estimates for values between min and max redshifts
            zmin = np.min(dc2["redshift"])
            zmax = np.max(dc2["redshift"])
            zgrid = np.logspace(np.log10(zmin), np.log10(zmax), Nzgrid)
            CDgrid = cosmology.comoving_distance(zgrid) * H0 / 100.0
            #  use interpolation to get redshifts for satellites only
            sat_mask = dc2[source_galaxy_tag + "upid"] != -1
            dc2["redshift"][sat_mask] = np.interp(r[sat_mask], CDgrid, zgrid)

        dc2["dec"] = 90.0 - np.arccos(dc2["z"] / r) * 180.0 / np.pi  # co-latitude
        dc2["ra"] = np.arctan2(dc2["y"], dc2["x"]) * 180.0 / np.pi
        dc2["ra"][(dc2["ra"] < 0)] += 360.0  # force value 0->360

        print(
            ".......min/max z for shell: {:.3f}/{:.3f}".format(
                np.min(dc2["redshift"]), np.max(dc2["redshift"])
            )
        )
    return dc2



# https://github.com/LSSTDESC/lsstdesc-diffsky/blob/d2bb23cca8a963f61b6144b5e27e69a366cbe662/lsstdesc_diffsky/pecZ.py

dot_vmap = jjit(vmap(jnp.dot, in_axes=(0, 0)))


def pecZ(x, y, z, vx, vy, vz, z_hubb, obs=np.zeros(3)):
    """
    This function calculates peculiar z_hubbs for n-body simulation objects, given
    their comoving position and velocity, and returns some other useful products
    of the calculation.
    Joe Hollowed COSMO-HEP 2017
    Parameters
    ----------
    x: x-position for each object, in form [x1, x2,... xn]
    :param y: y-position for each object, in form of param x
    :param z: z-position for each object, in form of param x
    :param vx: x-velocity for each object, in form [vx1, vx2,... vxn]
    :param vy: y-velocity for each object, in form of param vx
    :param vz: z-velocity for each object, in form of param vx
    :param z_hubb: cosmological redshifts for each object, in form [z1, z2,... zn]
    :param obs: The coordinates of the observer, in form [x, y, z]
    Returns
    -------
             - the peculiar z_hubb in each object in form of param redshift
             - the total observed z_hubb (cosmological+peculiar)
             - the peculiar velocity of each object, in the form of param vx,
               where negative velocities are toward the observer, in comoving km/s
             - the line-of-sight velocity, in proper km/s (peculiar velocity * a)
             - distance from observer to object in comoving Mpc
             - distance from observer to object in kpc proper (comoving dist * a)
             - distorted distance from observer to object in Mpc proper
    """

    # get relative position (r vector) of and position unit vector toward each object
    r_rel = np.array([x, y, z]).T - obs
    r_rel_mag = np.linalg.norm(r_rel, axis=1)
    r_rel_hat = np.divide(r_rel, np.array([r_rel_mag]).T)

    # dot velocity vectors with relative position unit vector to get peculiar velocity
    v = np.array([vx, vy, vz]).T
    v_pec = dot_vmap(v, r_rel_hat)

    # find total and peculiar z_hubb (full relativistic expression)
    c = const.c.value / 1000
    z_pec = np.sqrt((1 + v_pec / c) / (1 - v_pec / c)) - 1
    z_tot = (1 + z_hubb) * (1 + z_pec) - 1

    # find the distorted distance from appliying Hubble's law using the new
    # z_tot z_hubbs
    a = 1 / (1 + z_hubb)
    r_dist = r_rel_mag + v_pec / 100.0 / cosmo.efunc(z_hubb) / a
    # pdb.set_trace()

    return z_pec, z_tot, v_pec, v_pec * a, r_rel_mag, r_rel_mag * a, r_dist




def pecZ_snapshot(x, vx, z_hubb):
    c = const.c.value / 1000
    z_pec = np.sqrt((1 + vx / c) / (1 - vx / c)) - 1
    z_tot = (1 + z_hubb) * (1 + z_pec) - 1

    return z_pec, z_tot
    
    
'''