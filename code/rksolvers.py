# This folder contains all functions related to the runga kutta solver

import numpy as np

def rk4(x,t,tau,derivsRK,**kwargs):
  """
  Runge-Kutta integrator (4th order). Calling format derivsRK(x,t,**kwargs).
  Inputs:
      x               current value of dependent variable
      t               independent variable (usually time)
      tau             step size (usually timestep)
      derivsRK        right hand side of the ODE; derivsRK is the
                        name of the function which returns dx/dt
      **kwargs        arguments for the derivRK function
  Outputs:
      xout            new value of x after a step of size tau
  """
  half_tau = 0.5*tau
  F1 = derivsRK(x,t,**kwargs)
  t_half = t + half_tau
  xtemp = x + half_tau*F1
  F2 = derivsRK(xtemp,t_half,**kwargs)
  xtemp = x + half_tau*F2
  F3 = derivsRK(xtemp,t_half,**kwargs)
  t_full = t + tau
  xtemp = x + tau*F3
  F4 = derivsRK(xtemp,t_full,**kwargs)
  xout = x + tau/6.*(F1 + F4 + 2.*(F2+F3))
  return xout

def rka(x,t,tau,err,derivsRK,**kwargs):
    """
    Adaptive Runge-Kutta routine
    Inputs:
        x          Current value of the dependent variable
        t          Independent variable (usually time)
        tau        Step size (usually time step)
        err        Desired fractional local truncation error
        derivsRK   Right hand side of the ODE; derivsRK is the
                        name of the function which returns dx/dt
                        Calling format derivsRK(x,t).
        **kwargs    arguments for the derivRK function
    Outputs:
        xSmall     New value of the dependent variable
        t          New value of the independent variable
        tau        Suggested step size for next call to rka
    """
    # Set initial variables
    tSave = t;  xSave = x       # Save initial values
    safe1 = .9;  safe2 = 4.     # Safety factors
    eps = np.spacing(1)         # Smallest value
    # Loop over maximum number of attempts to satisfy error bound
    maxTry = 100
    for iTry in range(1,maxTry):
      # Take the two small time steps
      half_tau = 0.5 * tau
      xTemp = rk4(xSave,tSave,half_tau,derivsRK,**kwargs)
      t = tSave + half_tau
      xSmall = rk4(xTemp,t,half_tau,derivsRK,**kwargs)
      # Take the single big time step
      t = tSave + tau
      xBig = rk4(xSave,tSave,tau,derivsRK,**kwargs)
      # Compute the estimated truncation error
      scale = err * (np.abs(xSmall) + np.abs(xBig))/2.
      xDiff = xSmall - xBig
      errorRatio = np.max([np.abs(xDiff)/(scale + eps)])
      # print safe1,tau,errorRatio
      # Estimate news tau value (including safety factors)
      tau_old = tau
      tau = safe1*tau_old*errorRatio**(-0.20)
      tau = np.max([tau,tau_old/safe2])
      tau = np.min([tau,safe2*tau_old])
      # If error is acceptable, return computed values
      if errorRatio < 1 :
        xSmall = (16.*xSmall - xBig)/15. # correction
        return xSmall, t, tau
    # Issue error message if error bound never satisfied
    print ('ERROR: Adaptive Runge-Kutta routine failed')
    return

def planet_derivs(s,t,**kwargs):
    """
    Returns right-hand side of Kepler ODE; used by Runge-Kutta routines
    Inputs
        s      State vector [r(1) r(2) v(1) v(2)]
        t      Time (not used)
    Output
        deriv  Derivatives [dr(1)/dt dr(2)/dt dv(1)/dt dv(2)/dt]
    """
    sun_mass = kwargs['sun_mass']
    r = np.array([s[1], s[2]]); v = np.array([s[3] ,s[4]])
    accel = -4*np.pi**2*sun_mass*r/np.linalg.norm(r)**3 # accel from sun
    return np.array([s[0], v[0], v[1], accel[0], accel[1]])

def mission_derivs(s,t,**kwargs):
    """
    Returns right-hand side of Kepler ODE; used by Runge-Kutta routines
    Inputs
        s      State vector [[m r(1) r(2) v(1) v(2)]...]
        t      Time (not used)
    Output
        deriv  Derivatives [[dr(1)/dt dr(2)/dt dv(1)/dt dv(2)/dt]...]
    """
    sun_mass = kwargs['sun_mass']; new_s = []
    for i, si in enumerate(s):
        r = np.array([si[1], si[2]]); v = np.array([si[3], si[4]])
        accel = -4*np.pi**2*sun_mass*r/np.linalg.norm(r)**3 # accel from the sun
        for j, sj in enumerate(s): # accel from the other bodies
            rdiff = r - np.array([sj[1], sj[2]])
            if np.linalg.norm(rdiff) != 0.0:
                accel += -4*np.pi**2*sj[0]*rdiff/np.linalg.norm(rdiff)**3
        new_s.append([si[0], v[0], v[1], accel[0], accel[1]])
    return np.array(new_s)
