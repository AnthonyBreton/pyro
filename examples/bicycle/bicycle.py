# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:01:07 2018

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import vehicle
###############################################################################

def main():
    # Vehicule dynamical system
    sys = vehicle.KinematicBicyleModel()

    # Set default wheel steering angle and velocity
    sys.ubar = np.array([0.01,15])

    # Plot open-loop behavior
    sim = sys.compute_trajectory( np.array([0,0,0]) , 10 )
    sys.plot_trajectory(sim)

    # Animate the simulation
    sys.animate_simulation(sim)

if __name__ == "__main__":
    main()