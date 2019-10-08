# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:43:21 2019

@author: alxgr
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.control import controller
from pyro.dynamic import manipulator
###############################################################################


###############################################################################
# Mother Class
###############################################################################

class RobotController( controller.StaticController ) :
    """     """
    ############################
    def __init__(self, dof = 1):
        """ """

        # Dimensions
        self.k = dof 
        self.m = dof
        self.p = dof * 2  # y = x
        
        controller.StaticController.__init__(self, self.k, self.m, self.p)
        
        # Label
        self.name = 'Robot Controller'
        
        # DoF
        self.dof = dof
        
    #############################
    def x2q( self, x ):
        """ from state vector (x) to angle and speeds (q,dq) """
        
        q  = x[ 0        :     self.dof   ]
        dq = x[ self.dof : 2 * self.dof   ]
        
        return [ q , dq ]
        


###############################################################################
# Joint PID
###############################################################################
        
class JointPID( RobotController ) :
    """ 
    Linear controller for mechanical system with full state feedback (y=x)
    Independent PID for each DOF
    ---------------------------------------
    r  : reference signal_proc vector  dof x 1
    y  : sensor signal_proc vector     n   x 1
    u  : control inputs vector    dof x 1
    t  : time                     1   x 1
    ---------------------------------------
    u = c( y , r , t ) = (r - q) * kp + ( dq ) * kd + int(r - q) * ki

    """
    
    ############################
    def __init__(self, dof = 1, kp = 1, ki = 0, kd = 0):
        """ """
        
        RobotController.__init__( self , dof )
        
        # Label
        self.name = 'Joint PID Controller'
        
        # Gains
        self.kp = np.ones( dof  ) * kp
        self.kd = np.ones( dof  ) * kd
        self.ki = np.ones( dof  ) * ki
        
        # TODO Update this is not a static controller !!!!
        # Integral Memory
        self.dt    = 0.001
        self.e_int = np.zeros( dof )
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal_proc vector     p x 1
        r  : reference signal_proc vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
        
        u = np.zeros(self.m) 
        
        # Ref
        q_d = r
        
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )
        
        # Error
        e  = q_d - q
        de =     - dq
        ie = self.e_int + e * self.dt
        
        # PIDs
        u = e * self.kp + de * self.kd + ie * self.ki
        
        # Memory
        self.e_int =  ie
        
        return u
    


###############################################################################
# End-Effector PID
###############################################################################
        
class EndEffectorPID( RobotController ) :
    """ 
    PID in effector coordinates, using the Jacobian of the system
    ---------------------------------------
    r  : reference signal_proc vector  e   x 1
    y  : sensor signal_proc vector     n   x 1
    u  : control inputs vector    dof x 1
    t  : time                     1   x 1
    ---------------------------------------
    u = c( y , r , t ) = (r - q) * kp + ( dq ) * kd + int(r - q) * ki

    """
    
    ############################
    def __init__(self, manipulator, kp = 1, ki = 0, kd = 0):
        """ """
        
        # Using from model
        self.fwd_kin = manipulator.forward_kinematic_effector
        self.J       = manipulator.J
        self.e       = manipulator.e # nb of effector dof
        
        RobotController.__init__( self , manipulator.dof )
        
        # Label
        self.name = 'End-Effector PID Controller'
        
        # Gains
        self.kp = np.ones( self.dof  ) * kp
        self.kd = np.ones( self.dof  ) * kd
        self.ki = np.ones( self.dof  ) * ki
        
        # TODO Update this is not a static controller !!!!
        # Integral Memory
        self.dt    = 0.001
        self.e_int = np.zeros( self.dof )
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal_proc vector     p x 1
        r  : reference signal_proc vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
        
        u = np.zeros(self.m) 
        
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )
        
        # Jacobian computation
        J = self.J( q )
        
        # Ref
        r_desired   = r
        r_actual    = self.fwd_kin( q )
        dr          = np.dot( J , dq )
        
        # Error
        e  = r_desired - r_actual
        de =           - dr
        ie = self.e_int + e * self.dt
        
        # Effector space PID
        f = e * self.kp + de * self.kd + ie * self.ki
        
        # From effector force to joint torques
        u = np.dot( J.T , f )
        
        # Memory
        self.e_int =  ie
        
        return u
    
    
###############################################################################
# Kinematic Controller
###############################################################################
        
class EndEffectorKinematicController( RobotController ) :
    """ 
    Kinematic effector coordinates controller using the Jacobian of the system
    ---------------------------------------
    r  : reference signal_proc vector  e   x 1
    y  : sensor signal_proc vector     dof x 1
    u  : control inputs vector    dof x 1
    t  : time                     1   x 1
    ---------------------------------------
    u = c( y , r , t ) = J(q)^T *  [ (r - r_robot(q)) * k ]

    """
    
    ############################
    def __init__(self, manipulator, k = 0):
        """ """
        
        # Using from model
        self.fwd_kin = manipulator.forward_kinematic_effector
        self.J       = manipulator.J
        self.e       = manipulator.e # nb of effector dof
        
        # Dimensions
        self.dof = manipulator.dof
        self.k   = self.e 
        self.m   = self.dof
        self.p   = self.dof
        
        controller.StaticController.__init__(self, self.k, self.m, self.p)

        # Label
        self.name = 'End Effector Kinematic Controller'
        
        # Gains
        self.gains = np.ones( self.e  ) * k
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal_proc vector     p x 1
        r  : reference signal_proc vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
        
        #u = np.zeros(self.m) 
        
        # Feedback from sensors
        q = y
        
        # Jacobian computation
        J = self.J( q )
        
        # Ref
        r_desired   = r
        r_actual    = self.fwd_kin( q )
        
        # Error
        e  = r_desired - r_actual
        
        # Effector space PID
        dr_r = e * self.gains
        
        # From effector force to joint torques
        if self.dof == self.e:
            dq_r = np.dot( np.linalg.inv( J ) , dr_r )
            
        elif self.dof > self.e:
            J_pinv = np.linalg.pinv( J )
            dq_r   = np.dot( J_pinv , dr_r )
            
        else:
            #TODO
            pass
        
        return dq_r
    
    
    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":
    pass