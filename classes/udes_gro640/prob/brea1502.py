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
        J_pseudo = J.T @ np.linalg.inv( J @ J.T )
        
        # Ref
        r_desired   = r


        # Option avec le temps
        #if t < 0.5:
        #    r_desired = r + np.array([1.0 , 0.0])
        # Option par la norme de la pseudo inverse
        #if np.linalg.norm(J_pseudo) < 1.35: # chiffre magic no me likey
        #    r_desired = r + np.array([1.0 , 0.0])
        
        r_actual    = self.fwd_kin( q )
        
        # Error
        error  = r_desired - r_actual

        # Gain
        lambda_gain = 1.5

        # Compute the desired effector velocity 
        dr_r = lambda_gain * error # Place holder
        # From effector speed to joint speed
        dq = J_pseudo @ dr_r

        if abs(dq[0]) > 5.0 or abs(dq[1]) > 5.0 or abs(dq[2]) > 5.0:
            return np.array([2.0, 2.0, 2.0])

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
        
        # Target effector force
        self.rbar = np.array([0,0,0]) 
        
        
    
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
        
        ##################################
        # Votre loi de commande ici !!!
        ##################################
        
        tau = np.zeros(self.m)  # place-holder de bonne dimension
        
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