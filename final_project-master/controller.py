"""
author: Peter Huang
email: hbd730@gmail.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
import model.params as params
from math import sin, cos
import matplotlib.pyplot as plt
from model.quadcopter import Quadcopter
from sklearn.metrics import mean_squared_error

# PID Controller for quacopter
# Return [F, M] F is total force thrust, M is 3x1 moment matrix

# Constants for translational motion
k_p_x = 3
k_i_x = 8
k_d_x = 8

k_p_y = 3
k_i_y = 8
k_d_y = 8

k_p_z = 1000
k_i_z = 150
k_d_z = 200


# Constants for rotational motion [roll(phi), pitch(theta), and yaw(psi) angles] 
k_p_phi = 40
k_i_phi = 40
k_d_phi = 3

k_p_theta = 40
k_i_theta = 40
k_d_theta = 3

k_p_psi = 20
k_i_psi = 20
k_d_psi = 5

# ---------This part added for this project----------
# Integral error initalization for each motion
integral_error_x = 0.0
integral_error_y = 0.0
integral_error_z = 0.0
integral_error_phi = 0.0
integral_error_theta = 0.0
integral_error_psi = 0.0

# Simulation time step
dt = 1.0 / 200.0

# Performance tracking variables 
p_data = {
  'time': [],
  'thrust': [],
  'moment_x': [],
  'moment_y': [],
  'moment_z': []
}

# This variable keep tracks the controller for how many times it has been called
c_count = 0

# Vertical test mode flag
VERTICAL_ONLY_MODE = False

# Activation and p only mode flag setting for ziegler nichols method
zn_tuning_active = False 
zn_p_only_mode = False  

# Some important variable to tune the vertical PID constants with using Ziegler-Nichols method 
_last_z_error = 0.0      
_zero_crossings_count = 0
_last_zero_crossing_time = 0.0
_oscillation_periods_temp = []
_ultimate_period_detected = 0.0
# ----------------------------------------------

# A run function for PID constroller
def run(quad, des_state):
    
    # ---------This part added for this project----------
    # If just vertical motion used it enters here
    if VERTICAL_ONLY_MODE:
        return run_vertical_only(quad, des_state)
    
    else:
    
        # global variables for both translational and rotational motion integral error
        global integral_error_x, integral_error_y, integral_error_z
        global integral_error_phi, integral_error_theta, integral_error_psi

        # global variables for performance tracking and counting the controller
        global p_data, c_count
        # --------------------------------------------------
        
        # These are extracts current quadcopter's position, velocity, attitude and angular velocity
        x, y, z = quad.position()
        x_dot, y_dot, z_dot = quad.velocity()
        phi, theta, psi = quad.attitude()
        p, q, r = quad.omega()

        # These are extracts the desired quadcopter's position, velocity, acceleration, yaw angle and yaw dot
        des_x, des_y, des_z = des_state.pos
        des_x_dot, des_y_dot, des_z_dot = des_state.vel
        des_x_ddot, des_y_ddot, des_z_ddot = des_state.acc
        des_psi = des_state.yaw
        des_psi_dot = des_state.yawdot

        # ---------This part added for this project----------
        # These are differences in between desired and actual positions of quadcopter
        error_x = des_x - x
        error_y = des_y - y
        error_z = des_z - z

        # Maximum integral boundry
        max_integral = 10.0


        # Continuously added integral error term with maximum integral boundry for each translational motion term and 
        # after it is clipped to avoid excessive accumulation 
        integral_error_x += error_x * dt
        integral_error_x = np.clip(integral_error_x, -max_integral, max_integral)

        integral_error_y += error_y * dt
        integral_error_y = np.clip(integral_error_y, -max_integral, max_integral)

        integral_error_z += error_z * dt
        integral_error_z = np.clip(integral_error_z, -max_integral, max_integral)
        # ------------------------------------------------
        
        # Commanded accelerations
        commanded_r_ddot_x = des_x_ddot + k_d_x * (des_x_dot - x_dot) + k_i_x * integral_error_x + k_p_x * (des_x - x)
        commanded_r_ddot_y = des_y_ddot + k_d_y * (des_y_dot - y_dot) + k_i_y * integral_error_y + k_p_y * (des_y - y)
        commanded_r_ddot_z = des_z_ddot + k_d_z * (des_z_dot - z_dot) + k_i_z * integral_error_z + k_p_z * (des_z - z)

        # In here it converts the vertical acceleration into desired thrust
        F = params.mass * (params.g + commanded_r_ddot_z)
        
        # Moment                      # Desired upward acceleration from PID
        p_des = 0
        q_des = 0
        r_des = des_psi_dot
        
        # These equations are converts the desired x/y accelerations to desired roll and pitch attitudes
        des_phi = 1 / params.g * (commanded_r_ddot_x * sin(des_psi) - commanded_r_ddot_y * cos(des_psi))
        des_theta = 1 / params.g * (commanded_r_ddot_x * cos(des_psi) + commanded_r_ddot_y * sin(des_psi))
    
        
        # ---------This part added for this project----------
        # These are calculates the angular error in between desired and current attitude
        error_phi = des_phi - phi
        error_theta = des_theta - theta
        error_psi = des_psi - psi

        # This is the normalization part for the yaw error in the range of [-pi, pi]
        if error_psi > np.pi:
            error_psi -= 2 * np.pi
        elif error_psi < -np.pi:
            error_psi += 2 * np.pi

        # Continuously added integral error term with maximum integral boundry for each rotational motion term and 
        # after it is clipped to avoid excessive accumulation 
    
        integral_error_phi += error_phi * dt
        integral_error_phi = np.clip(integral_error_phi, -max_integral, max_integral)

        integral_error_theta += error_theta * dt
        integral_error_theta = np.clip(integral_error_theta, -max_integral, max_integral)
    
        integral_error_psi += error_psi * dt
        integral_error_psi = np.clip(integral_error_psi, -max_integral, max_integral)
        # -------------------------------------------------
        
        
        # Calculating torque vector with using PID control on angular error
        M = np.array([[k_p_phi * (des_phi - phi) + k_i_phi * integral_error_phi + k_d_phi * (p_des - p),
                    k_p_theta * (des_theta - theta) + k_i_theta * integral_error_theta + k_d_theta * (q_des - q),
                    k_p_psi * (des_psi - psi) + k_i_psi * integral_error_psi + k_d_psi * (r_des - r)]]).T

        
        # ---------This part added for this project----------
        # Storing the performance data
        c_count += 1
        c_time = c_count * dt # Basic time calculation

        # Storing each variable in the relevevant key of p_data dictionary
        p_data['time'].append(c_time)
        p_data['thrust'].append(F)
        p_data['moment_x'].append(M[0, 0])
        p_data['moment_y'].append(M[1, 0])
        p_data['moment_z'].append(M[2, 0])

        
        # Record of every time, actual and desired vertical distance
        track_height(c_time, z, des_z)
        # ---------------------------------------------------
        
        return F, M
    

# ---------This part added for this project----------
# This is a run function for running the quadcopter only in vertical motion  
def run_vertical_only(quad, des_state):
    
    # Global initialization of some important variables for both just vertical movment and also for ziegler nichols tuning    
    global integral_error_z, p_data, c_count
    global zn_tuning_active, zn_p_only_mode
    global _last_z_error, _zero_crossings_count, _last_zero_crossing_time, _oscillation_periods_temp, _ultimate_period_detected

    
    # Extracting the only vertical states of quadcopter like for actual position and velocity
    z = quad.position()[2]
    z_dot = quad.velocity()[2]
    des_z = des_state.pos[2]
    des_z_dot = des_state.vel[2]
    des_z_ddot = des_state.acc[2]

    # Integral error calculation for just vertical movment quadcopter
    error_z = des_z - z

    # If ziegler nichols P-only mode, disable the I and D terms
    if zn_p_only_mode:
        integral_error_z = 0.0 
        commanded_z_ddot = (des_z_ddot + 
                            k_p_z * error_z +
                            k_i_z * integral_error_z +
                            k_d_z * 0)
    
    # Else calculate normal vertical PID with anti-windup. 
    else:    
        # An integral error calculation before adding the anti-windup to the program
        '''integral_error_z += error_z * dt
        integral_error_z = np.clip(integral_error_z, -10.0, 10.0)'''


        # A vertical PID calculation with just added desired acceleration variable
        commanded_unclipped_z_ddot = (des_z_ddot +
                            k_p_z * error_z +
                            k_i_z * integral_error_z +
                            k_d_z * (des_z_dot - z_dot))
    
        # Thrust calculation (without unclipped)
        F_unclipped = params.mass * (params.g + commanded_unclipped_z_ddot)
        
    # Finding if the thrust is saturated and decide if integral should be stopped
        if (F_unclipped > params.maxF and error_z > 0) or \
            (F_unclipped < params.minF and error_z < 0):
                # This denies accumulating the integral error when output is saturated
                pass
        else:
            # Accumulating the integral error normally if not saturated
            integral_error_z += error_z * dt
            
        # To make it safe always clip the integral error
        integral_error_z = np.clip(integral_error_z, -10.0, 10.0)

        # Calculate the actual commanded accleration that is going to be used
        commanded_z_ddot = (des_z_ddot +
                            k_p_z * error_z +
                            k_i_z * integral_error_z +
                            k_d_z * (des_z_dot - z_dot))

    # Final thrust calculation
    F = params.mass * (params.g + commanded_z_ddot)

    # Clipping the final thrust output to its physical limits
    F_clipped = np.clip(F, params.minF, params.maxF)
    F = F_clipped

    
    # Moments is not possible because all 4 rotors are locked identical
    M = np.zeros((3,1))

    
    # This is an oscillation detection for zn method for only if error is very significant
    if zn_tuning_active and zn_p_only_mode and abs(error_z) > 0.05: 
        
        # Getting the current simulation time
        current_sim_time = c_count * dt 

        # Count the zero crossings if error changes the sign
        if (_last_z_error > 0 and error_z < 0) or (_last_z_error < 0 and error_z > 0):
            _zero_crossings_count += 1
            
            # If the last zero crossing time different then zero calculate the current period 
            # and continue the calculation
            if _last_zero_crossing_time != 0.0:
                current_period = current_sim_time - _last_zero_crossing_time
                
                # Avoiding the very small periods from noise and storing the estimated oscillated periods
                if current_period > 0.1: 
                    _oscillation_periods_temp.append(current_period) 
                    
                    # It needs six half cycles for consistant calculation
                    if len(_oscillation_periods_temp) >= 6:

                        # The calculation of average and standard deviation of oscillation periods 
                        avg_period = np.mean(_oscillation_periods_temp[-6:])
                        std_dev_period = np.std(_oscillation_periods_temp[-6:])
                        
                        # Checking the consistency of periods 
                        if std_dev_period < 0.1 * avg_period:
                            _ultimate_period_detected = avg_period # Setting the Tu if its consistent
            
            # Updating the last zero crossing time with the current simulation time 
            _last_zero_crossing_time = current_sim_time 
            
        # The last error updation for every next iteration
        _last_z_error = error_z

    
    # Storing the performance data like original. For example, simulation time and thrust
    c_count += 1
    c_time = c_count * dt
    p_data['time'].append(c_time)
    p_data['thrust'].append(F)
    p_data['moment_x'].append(0.0)
    p_data['moment_y'].append(0.0)
    p_data['moment_z'].append(0.0)
    
    
    # Calling the tracking of height variable
    track_height(c_time, z, des_z)

    return F, M
# ---------------------------------------------------


# ---------This part added for this project----------
# A reset function for zn variables
def reset_zn_tuning_variables():
    global _last_z_error, _zero_crossings_count, _last_zero_crossing_time, _oscillation_periods_temp, _ultimate_period_detected
    _last_z_error = 0.0
    _zero_crossings_count = 0
    _last_zero_crossing_time = 0.0
    _oscillation_periods_temp = []
    _ultimate_period_detected = 0.0
# --------------------------------------------------


# ---------This part added for this project----------
# A function for plotting the thrust and moment performance
def plotting_the_performance():

    # Checking whether time variable different then zero
    if len(p_data['time']) == 0:
        print("No performance data collected!")
        return
    
    # The title of grpahs
    plt.figure(figsize=(8, 6), num="Controller Performance")

    # The graph of Thrust (F) variable
    plt.subplot(2, 1, 1)
    plt.plot(p_data['time'], p_data['thrust'], 'b-', linewidth=2, label='Thrust (F)')
    plt.title('Thrust Force (F)', fontsize=14)
    plt.ylabel('Force (N)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # The graph of Moments (M) variable
    plt.subplot(2, 1, 2)
    plt.plot(p_data['time'], p_data['moment_x'], 'r-', linewidth=2, label='Moment X')
    plt.plot(p_data['time'], p_data['moment_y'], 'g-', linewidth=2, label='Moment Y')
    plt.plot(p_data['time'], p_data['moment_z'], 'b-', linewidth=2, label='Moment Z')
    plt.title('Moment Vector (M)', fontsize=14)
    plt.ylabel('Moment (Nm)')
    plt.xlabel('Time (s)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Make the screen layout more tight
    plt.tight_layout()

    # Show the plot to visible
    plt.show(block=False)
    plt.pause(120)

# -------------------------------------------------


# ---------This part added for this project----------
# Some dictionary variables for height tracking
height_data = {
    'time': [],
    'actual': [],
    'desired': [],
    'error': [],
}

# A function for resetting the height data
def reset_height_data():
    
    global height_data
    height_data = {'time': [], 'actual': [], 'desired': [], 'error': []}

# --------------------------------------------------


# ---------This part added for this project----------
# A simple function to assign time, actual and desired height measures and thrust value
def track_height(time, z_actual, z_desired):
    height_data['time'].append(time)
    height_data['actual'].append(z_actual)
    height_data['desired'].append(z_desired)
    height_data['error'].append(z_desired - z_actual)
    

# A function to analyze height measurement and important metrics for comparison
def analyze_height():

    # If there is no time data recorded justv warn the user
    if len(height_data['time']) == 0:
        print("No height data!")
        return
    
    # Display a frame with size of 10 by 6 cm
    plt.figure(figsize=(10,6))

    # This below is a subplot for height tracking
    plt.subplot(2,1,1)
    plt.plot(height_data['time'], height_data['actual'], 'b-', label='Actual', linewidth=2)
    plt.plot(height_data['time'], height_data['desired'], 'r--', label='Desired', linewidth=2)
    plt.title('Height Tracking')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.grid(True)

    # This below is a subplot for analyzing the height error
    plt.subplot(2,1,2)
    plt.plot(height_data['time'], height_data['error'], 'r-', linewidth=2)
    plt.title('Height Error')
    plt.ylabel('Error (m)')
    plt.grid(True)

    # Some adjustments for all graphs
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(120)

    
    # Some metiric adjustments for easy usage
    times = np.array(height_data['time'])
    actuals = np.array(height_data['actual'])
    desireds = np.array(height_data['desired'])
    
    # Finding the step response
    try:
        step_idx = np.where(desireds > 0)[0][0]
        step_time = times[step_idx]
        target_height = desireds[step_idx]
    except IndexError:
        return {'rise_time': float('inf'), 'settling_time': float('inf')}

    # Isolating the post-step response
    post_step_times = times[step_idx:] - step_time
    post_step_actuals = actuals[step_idx:]
    post_step_desireds = desireds[step_idx:]

    
    # The calculation of rise time with 10% to 90% of the target height
    try:
        rise_10_idx = np.where(post_step_actuals >= 0.1 * target_height)[0][0]
        rise_90_idx = np.where(post_step_actuals >= 0.9 * target_height)[0][0]
        rise_time = post_step_times[rise_90_idx] - post_step_times[rise_10_idx]
    except IndexError:
        rise_time = float('inf')

    
    # The calculation of settling time 
    tolerance = 0.02 * target_height
    unsettled = np.where(np.abs(post_step_actuals - target_height) > tolerance)[0]
    if len(unsettled) > 0:
        last_unsettled_idx = unsettled[-1]
        settling_time = post_step_times[last_unsettled_idx]
    else:
        settling_time = 0.0

    
    # The calculation of overshoot
    peak_actual = np.max(post_step_actuals)
    if target_height > 0:
        overshoot = ((peak_actual - target_height) / target_height) * 100
        if overshoot < 0: 
            overshoot = 0.0
    else: 
        overshoot = float('nan') 


    # The calculation of steady-state error
    if len(post_step_actuals) > 0:
        steady_state_value = post_step_actuals[-1]
        steady_state_error = target_height - steady_state_value
    else:
        steady_state_error = float('nan') 
    
    
    # The calculation of mean squared error
    if len(post_step_actuals) > 0:
        errors = post_step_desireds - post_step_actuals
        mean_squared_error = np.mean(errors**2)
    else:
        mean_squared_error = float('nan')
    

    # Assigning the all metrics to this dictionary 
    metrics = {
        'rise_time': rise_time,
        'settling_time': settling_time,
        'overshoot': overshoot,
        'steady_state_error': steady_state_error,
        'mean_squared_error': mean_squared_error
    }

    return metrics
# --------------------------------------------

# ---------This part added for this project----------
# A function for mse calculation that just works for grid search method
def mse_calculation_for_grid_search():

    # Retrieve the actual and desired meter values 
    actuals = np.array(height_data['actual'])
    desireds = np.array(height_data['desired'])
    
    # Calculate the desired and actual meter values after step response 
    try:
        step_idx = np.where(desireds > 0)[0][0]
        post_step_actuals = actuals[step_idx:]
        post_step_desireds = desireds[step_idx:]
    except IndexError:
        return {'mean_squared_error': float('nan')}
    
    # The calculation of mean squared error
    errors = post_step_desireds - post_step_actuals
    mean_squared_error = np.mean(errors**2)
    
    return mean_squared_error
# ----------------------------------------------------
