#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:19:16 2020

@author: alex
------------------------------------


Fichier d'amorce pour les livrables de la problématique GRO640'


"""

from typing import List
import numpy as np

from pyro.control.robotcontrollers import EndEffectorPD
from pyro.control.robotcontrollers import EndEffectorKinematicController


###################
# Part 1
###################

def dh2T( r , d , theta, alpha ):
    """

    Parameters
    ----------
    r     : float 1x1
    d     : float 1x1
    theta : float 1x1
    alpha : float 1x1
    
    4 paramètres de DH

    Returns
    -------
    T     : float 4x4 (numpy array)
            Matrice de transformation

    """
    
    T = np.array([
        [np.cos(theta),     -np.sin(theta) * np.cos(alpha),     np.sin(theta) * np.sin(alpha),      r * np.cos(theta)],
        [np.sin(theta),     np.cos(theta) * np.cos(alpha),      -np.cos(theta) * np.sin(alpha),     r * np.sin(theta)],
        [0,                 np.sin(alpha),                      np.cos(alpha),                      d],
        [0,                 0,                                  0,                                  1]
        ])
    
    return T



def dhs2T( r: List, d: List, theta: List, alpha: List):
    """

    Parameters
    ----------
    r     : float nx1
    d     : float nx1
    theta : float nx1
    alpha : float nx1
    
    Colonnes de paramètre de DH

    Returns
    -------
    WTT     : float 4x4 (numpy array)
              Matrice de transformation totale de l'outil

    """
    
    WTT = np.identity((4))

    for index in range(len(r)):
        WTT = np.dot(WTT, dh2T(r[index] , d[index], theta[index], alpha[index]))
    
    return WTT


def f(q: List):
    """
    

    Parameters
    ----------
    q : float 6x1
        Joint space coordinates

    Returns
    -------
    r : float 3x1 
        Effector (x,y,z) position

    """
    r_dh =  [0.0,               0.033,      0.155,          0.135,  0.0,            0.0,        0.0]
    d =     [0.071,             0.076,      0.0,            0.0,    0.0,            0.217,      q[5]]
    theta = [np.pi/2 - q[0],    0.0,        np.pi/2 - q[1], -q[2],   np.pi/2 - q[3], -q[4],       0.0]
    alpha = [0.0,               np.pi/2,    0.0,            0.0,    np.pi/2,        np.pi/2,    0.0]
    
    WTT = dhs2T(r_dh, d, theta, alpha)

    r = np.array([WTT[:-1,3]])
    
    return r


###################
# Part 2
###################
    
class CustomPositionController( EndEffectorKinematicController ) :
    
    ############################
    def __init__(self, manipulator ):
        """ """
        
        EndEffectorKinematicController.__init__( self, manipulator, 1)
        
        ###################################################
        # Vos paramètres de loi de commande ici !!
        ###################################################
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback law: u = c(y,r,t)
        
        INPUTS
        y = q   : sensor signal vector  = joint angular positions      dof x 1
        r = r_d : reference signal vector  = desired effector position   e x 1
        t       : time                                                   1 x 1
        
        OUPUTS
        u = dq  : control inputs vector =  joint velocities             dof x 1
        
        """
        
        # Feedback from sensors
        q = y
        
        # Jacobian computation
        J = self.J( q )
        
        # Ref
        r_desired   = r
        r_actual    = self.fwd_kin( q )
        
        # Error
        error  = r_desired - r_actual

        # Gain
        lambda_error = 1.5
        lambda_speed = 0.1

        # Compute the desired effector velocity 
        dr_r = lambda_error * error
        # From effector speed to joint speed
        dq = np.linalg.inv(J.T @ J + (lambda_speed**2) * np.identity(3)) @ J.T @ dr_r

        return dq
    
    
###################
# Part 3
###################
        

        
class CustomDrillingController( EndEffectorPD ) :
    """ 

    """
    
    ############################
    def __init__(self, robot_model ):
        """ """
        
        EndEffectorPD.__init__( self , robot_model )
        
        # Label
        self.name = 'Custom Drilling Controller'
        
        ###################################################
        # Vos paramètres de loi de commande ici !!
        ###################################################
        self.robot_model = robot_model
        self.is_over_drill_point = False
        self.is_done_drilling = False
        # Target effector force
        self.rbar = np.array([0,0,-200]) 
        
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
        
        # Ref
        f_e = r
        
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )

        # Jacobian computation
        J = self.J( q )

        ## Ref
        r_d = np.array([0.25, 0.25, 0.4])
        r_actual = self.fwd_kin( q )

        K = 100
        B = 50

        is_approaching = not np.isclose(r_d, r_actual, atol=0.001).all() and self.is_over_drill_point == False
        is_drilling = not np.isclose(0.2, r_actual[2], atol=0.001) and self.is_done_drilling == False
        is_out_of_hole = np.isclose(0.4, r_actual[2], atol=0.01)

        if is_approaching:
            f_e = np.array([0,0,0]) # Control in postion by impedance to approach point no force
        else:
            self.is_over_drill_point = True
            if is_drilling:
                pass # Control in postion by impedance and force with the -200N to drill down
            else:
                self.is_done_drilling = True
                if is_out_of_hole:
                    f_e = np.array([0,0,0]) # stop motion
                else:
                    f_e = np.array([0,0,45]) # Control in postion by impedance and force to drill back up

        tau = J.T @ ( K * ( r_d - r_actual ) + B * ( - J @ dq ) + f_e ) + self.robot_model.g(q)

        return tau
        
    
###################
# Part 4
###################
        
    
def goal2r( r_0 , r_f , t_f ):
    """
    
    Parameters
    ----------
    r_0 : numpy array float 3 x 1
        effector initial position
    r_f : numpy array float 3 x 1
        effector final position
    t_f : float
        time 

    Returns
    -------
    r   : numpy array float 3 x l
    dr  : numpy array float 3 x l
    ddr : numpy array float 3 x l

    """
    # Time discretization
    l = 1000 # nb of time steps
    
    # Number of DoF for the effector only
    m = 3
    
    r = np.zeros((m,l))
    dr = np.zeros((m,l))
    ddr = np.zeros((m,l))
    
    #################################
    # Votre code ici !!!
    ##################################
    
    
    return r, dr, ddr


def r2q( r, dr, ddr , manipulator ):
    """

    Parameters
    ----------
    r   : numpy array float 3 x l
    dr  : numpy array float 3 x l
    ddr : numpy array float 3 x l
    
    manipulator : pyro object 

    Returns
    -------
    q   : numpy array float 3 x l
    dq  : numpy array float 3 x l
    ddq : numpy array float 3 x l

    """
    # Time discretization
    l = r.shape[1]
    
    # Number of DoF
    n = 3
    
    # Output dimensions
    q = np.zeros((n,l))
    dq = np.zeros((n,l))
    ddq = np.zeros((n,l))
    
    #################################
    # Votre code ici !!!
    ##################################
    
    
    return q, dq, ddq



def q2torque( q, dq, ddq , manipulator ):
    """

    Parameters
    ----------
    q   : numpy array float 3 x l
    dq  : numpy array float 3 x l
    ddq : numpy array float 3 x l
    
    manipulator : pyro object 

    Returns
    -------
    tau   : numpy array float 3 x l

    """
    # Time discretization
    l = q.shape[1]
    
    # Number of DoF
    n = 3
    
    # Output dimensions
    tau = np.zeros((n,l))
    
    #################################
    # Votre code ici !!!
    ##################################
    
    
    return tau