from colossus.cosmology import cosmology
import haccytrees.mergertrees
import numpy as np


def get_cosmo(simulation):


    if (simulation=="AlphaQ"):
        sim = haccytrees.Simulation.simulations[simulation]
        cosmo = cosmology.setCosmology(simulation, 
                                       dict(flat=True, 
                                            H0=sim.cosmo.h*100, 
                                            Om0=sim.cosmo.Omega_m, 
                                            Ob0=sim.cosmo.Omega_b, 
                                            sigma8=sim.cosmo.s8, 
                                            ns=sim.cosmo.ns))
        return cosmo

    elif (simulation=="LastJourney"):
        sim = haccytrees.Simulation.simulations[simulation]
        cosmo = cosmology.setCosmology(simulation, 
                                       dict(flat=True, 
                                            H0=sim.cosmo.h*100, 
                                            Om0=sim.cosmo.Omega_m, 
                                            Ob0=sim.cosmo.Omega_b, 
                                            sigma8=sim.cosmo.s8, 
                                            ns=sim.cosmo.ns))
        return cosmo


    elif (simulation=="SMDPL"):
        cosmo = cosmology.setCosmology(simulation, 
                                       dict(flat=True, 
                                            H0=0.6777*100, 
                                            Om0=0.307115, 
                                            Ob0=0.048206, 
                                            sigma8=0.8228, 
                                            ns=0.96))
        return cosmo

    else:
        return NotImplemented


def get_analysis_steps(simulation):
    if (simulation=="AlphaQ"):
        hacc_analysis_steps = np.array([44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 59, 60,
        62, 63, 65, 67, 68, 70, 72, 74, 76, 77, 79, 81, 84, 86, 88,
        90, 92, 95, 97, 100, 102, 105, 107, 110, 113, 116, 119, 121,
        124, 127, 131, 134, 137, 141, 144, 148, 151, 155, 159, 163, 167,
        171, 176, 180, 184, 189, 194, 198, 203, 208, 213, 219, 224, 230,
        235, 241, 247, 253, 259, 266, 272, 279, 286, 293, 300, 307, 315,
        323, 331, 338, 347, 355, 365, 373, 382, 392, 401, 411, 421, 432,
        442, 453, 464, 475, 487, 499])
        return hacc_analysis_steps

    elif (simulation=="LastJourney"):
        hacc_analysis_steps = np.sort(np.array([42, 43,  44,  45,  46,  48,  49,  50,  52,  53,  54,  56,  57,  59,
                60,  62,  63,  65,  67,  68,  70,  72,  74,  76,  77,  79,  81,
                84,  86,  88,  90,  92,  95,  97, 100, 102, 105, 107, 110, 113,
               116, 119, 121, 124, 127, 131, 134, 137, 141, 144, 148, 151, 155,
               159, 163, 167, 171, 176, 180, 184, 189, 194, 198, 203, 208, 213,
               219, 224, 230, 235, 241, 247, 253, 259, 266, 272, 279, 286, 293,
               300, 307, 315, 323, 331, 338, 347, 355, 365, 373, 382, 392, 401,
               411, 421, 432, 442, 453, 464, 475, 487, 499])).astype('int')
        return hacc_analysis_steps
    else:
        return NotImplemented
