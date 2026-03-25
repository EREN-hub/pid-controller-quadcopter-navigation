"""
author: Peter Huang, Antonio Cuni
email: hbd730@gmail.com, anto.cuni@gmail.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

from quadPlot import plot_quad_3d
import controller
import trajGen
import trajGen3D
import sys
import matplotlib.pyplot as plt
from model.quadcopter import Quadcopter
from quadPlot import plot_position_over_time
import numpy as np
from collections import namedtuple

animation_frequency = 50
control_frequency = 200 # Hz for attitude control loop
control_iterations = int(control_frequency / animation_frequency)
dt = 1.0 / control_frequency
time = [0.0]

def attitudeControl(quad, time, waypoints, coeff_x, coeff_y, coeff_z):
    desired_state = trajGen3D.generate_trajectory(time[0], 1.2, waypoints, coeff_x, coeff_y, coeff_z)
    F, M = controller.run(quad, desired_state)
    quad.update(dt, F, M)
    time[0] += dt

# ---------This part added for this project----------
DesiredState = namedtuple('DesiredState', 'pos vel acc yaw yawdot')

# A function for simulating the step response  
def run_step_response_simulation():

    # A quadcopter position initalization 
    quadcopter = Quadcopter(pos=(0,0,0), attitude=(0,0,0))
    
    # Two waypoints position initalization
    waypoints = np.array([[0,0,0], [0,0,4.0]])

    # Time initalization for step response function
    time[0] = 0.0

    # A function calling for reseting the height variables for a run
    controller.reset_height_data()


    # A control loop function for this simulation
    def step_response_control_loop(i):
        
        # A for loop for number of iterations
        for _ in range(control_iterations):
        
            # Make the quadcopter's target height 4.0 if time passes two second 
            target_height = 4.0 if time[0] > 2.0 else 0.0

            # Initilalizing the desired state as just for the vertical position for this simulation
            desired_state = DesiredState(
                pos = np.array([0.0, 0.0, target_height]),
                vel = np.array([0.0, 0.0, 0.0]),
                acc = np.array([0.0, 0.0, 0.0]),
                yaw = 0.0,
                yawdot = 0.0   
            )

            # Getting thrust and moment values for this simulations desired state
            F, M = controller.run(quadcopter, desired_state)

            # Updating the quadcopter with related function using the thrust and moment values 
            # then updating the time for next iteration
            quadcopter.update(dt, F, M)
            time[0] += dt
        
        return quadcopter.world_frame()
    
    
    # Plotting the simulation to analyze the step response
        
    plot_quad_3d(waypoints, step_response_control_loop)
# ---------------------------------------------------

# ---------This part added for this project----------
# A function for simulating the comparison of PID constant tuning 
def run_pid_comparison_simulation():
   
    # Storing the original gains for using them later 
    original_gains = {
        'p': controller.k_p_z,
        'i': controller.k_i_z,
        'd': controller.k_d_z
    }

    # A dictionary for default and tuned set of constants to compare the statistics
    gain_sets = {
        "Original Untuned": {'p': 1000, 'i': 150, 'd': 200},
        "Tuned": {'p': 200, 'i': 25, 'd': 70}
    }

    results = {}

    # A for loop to run the step response simulation for untuned and tuned constants
    for name, gains in gain_sets.items():
        print("\n" + "="*50)
        print(f"Running simulation for: {name.upper()} gains")
        print(f"P={gains['p']}, I={gains['i']}, D={gains['d']}")
        print("="*50)

        # Setting the PID control gains for the current run
        controller.k_p_z = gains['p']
        controller.k_i_z = gains['i']
        controller.k_d_z = gains['d']

        # Setting on the vertical mode for the current run
        controller.VERTICAL_ONLY_MODE = True

        # Run the step response simulation with current assigned gains
        run_step_response_simulation()

        # Analyzing the metrics and storing them in here with related verison of gains
        metrics = controller.analyze_height()
        results[name] = metrics
        
        # Waiting the user for continuing the simulation until it presses the Enter
        if name != list(gain_sets.keys())[-1]:
            input("\nPress Enter to run the simulation with the next set of gains...")

    # Exiting from the vertical mode to continue from printing the all metrics
    controller.VERTICAL_ONLY_MODE = False 

    # Restoring the original gains for future use
    controller.k_p_z = original_gains['p']
    controller.k_i_z = original_gains['i']
    controller.k_d_z = original_gains['d']

    # Printing the final comparison table to understand the differences in between two different gain
    print("\n\n" + "="*60)
    print("           PID GAINS COMPARISON RESULTS")
    print("="*60)
    print(f"{'Metric':<20} | {'Original Untuned':<20} | {'Tuned':<20}")
    print("-"*60)
    
    # The rise time comparison
    rt_orig = f"{results['Original Untuned']['rise_time']:.3f} s"
    rt_tuned = f"{results['Tuned']['rise_time']:.3f} s"
    print(f"{'Rise Time (10-90%)':<20} | {rt_orig:<20} | {rt_tuned:<20}")

    # The settling time comparison
    st_orig = f"{results['Original Untuned']['settling_time']:.3f} s"
    st_tuned = f"{results['Tuned']['settling_time']:.3f} s"
    print(f"{'Settling Time (+-2%)':<20} | {st_orig:<20} | {st_tuned:<20}")
    
    # The overshoot comparison
    o_orig = f"{results['Original Untuned']['overshoot']:.3f} %"
    o_tuned = f"{results['Tuned']['overshoot']:.3f} %"
    print(f"{'Overshoot':<20} | {o_orig:<20} | {o_tuned:<20}")

    # Steady state error comparison
    st_s_e_orig = f"{results['Original Untuned']['steady_state_error']:.3f} m"
    st_s_e_tuned = f"{results['Tuned']['steady_state_error']:.3f} m"
    print(f"{'Steady State Error':<20} | {st_s_e_orig:<20} | {st_s_e_tuned:<20}")

    # Mean Squared error comparison
    mse_orig = f"{results['Original Untuned']['mean_squared_error']:.3f} m^2"
    mse_tuned = f"{results['Tuned']['mean_squared_error']:.3f} m^2"
    print(f"{'Mean Squared Error':<20} | {mse_orig:<20} | {mse_tuned:<20}")

    print("="*60)
    
    # Show the all plots until the simulation user closes them
    plt.show() 
# ---------------------------------------------------

# ---------This part added for this project----------
# Function for grid search tuning
def run_grid_search_tuning():
    print("\n" + "="*60)
    print("Starting Grid Search PID Tuning for Z-axis")
    print("============================================================")

    # Turn on the vertical mode and deactivate zn modes for just to make sure Ziegler-Nichols method closed
    controller.VERTICAL_ONLY_MODE = True
    controller.zn_tuning_active = False
    controller.zn_p_only_mode = False

    # Arranging the PID constant ranges in order to tune
    kp_range = np.arange(200, 400, 25)  
    ki_range = np.arange(20, 60, 5)   
    kd_range = np.arange(30, 80, 10)  

    # Two constants for early iteration termination 
    SETTLE_ERROR_TOLERANCE = 0.07
    SETTLE_TIME_REQUIRED = 0.5
    
    #Initalizing the best MSE and best parameters dictionary for PID
    best_mse = float('inf')
    best_params = {'p': None, 'i': None, 'd': None}
    
    # Calculation of total combination
    total_combinations = len(kp_range) * len(ki_range) * len(kd_range)
    
    # Initializing a count for tracking
    current_combination_count = 0

    # Some print statements before search
    print(f"Testing {total_combinations} combinations...")
    print(f"Kp Range: {kp_range}")
    print(f"Ki Range: {ki_range}")
    print(f"Kd Range: {kd_range}")
    print("-" * 60)

    # Get the original gains from controller
    original_gains = {
        'p': controller.k_p_z,
        'i': controller.k_i_z,
        'd': controller.k_d_z
    }

    # Iteration through all combinations
    for kp in kp_range:
        for ki in ki_range:
            for kd in kd_range:

                # Counting the total combination
                current_combination_count += 1
                
                # Setting the current combination of PID constants for current run
                controller.k_p_z = kp
                controller.k_i_z = ki
                controller.k_d_z = kd

                # Resetting the quadcopter and the global time
                tuning_quadcopter = Quadcopter(pos=(0.0,0.0,0.0), attitude=(0,0,0))
                time[0] = 0.0 

                # Resetting height data and zn tuning variables for to make sure this simulation is not in the zn tuning
                controller.reset_height_data()
                controller.reset_zn_tuning_variables()

                # Initialize duration time and control steps
                run_duration = 10.0 
                num_control_steps = int(run_duration * control_frequency)

                # Initialize settling time for checking
                s_check_start_time = 0.0
                
                # Simulation loop for every combination of PID
                for step_idx in range(num_control_steps):
                    
                    #Taking the current simulation time 
                    current_time_in_sim = time[0]
                    
                    # Target height initialization
                    target_height = 4.0 if current_time_in_sim > 2.0 else 0.0 

                    # Desired state initialization
                    desired_state = DesiredState(
                        pos = np.array([0.0, 0.0, target_height]),
                        vel = np.array([0.0, 0.0, 0.0]),
                        acc = np.array([0.0, 0.0, 0.0]),
                        yaw = 0.0,
                        yawdot = 0.0
                    )
                    
                    # Thrust and moment calculation for updating quadcopter movment and time update 
                    F, M = controller.run(tuning_quadcopter, desired_state)
                    tuning_quadcopter.update(dt, F, M)
                    time[0] += dt

                    # Start to check after the step input
                    if current_time_in_sim > 2.0: 
                        # Calculation of current error
                        current_error = abs(target_height - tuning_quadcopter.position()[2])

                        # Checking for error tolerance
                        if current_error <= SETTLE_ERROR_TOLERANCE:
                            if s_check_start_time == 0.0:
                                
                                # Starting the timer if just entered the tolerance band
                                s_check_start_time = current_time_in_sim
                            
                            # If quadcopter stayed within tolerance for a little time
                            elif (current_time_in_sim - s_check_start_time) >= SETTLE_TIME_REQUIRED:
                                print(f"  Quadcopter movment finished. Moving to analyze part.")
                                break 
                        else:
                            # Resetting the timer if the quadcopter is not settled at tolerance band
                            s_check_start_time = 0.0

                # Mean Squared Error calculation for current combination of PID constants
                c_mse = controller.mse_calculation_for_grid_search()

                print(f"Combination {current_combination_count}/{total_combinations}: Kp={kp}, Ki={ki}, Kd={kd} -> MSE={c_mse:.4f}")

                # Checking if the best mean squared error achieved
                if c_mse < best_mse:

                    # Assign current mse if achieved
                    best_mse = c_mse

                    # Then assign current combination of PID constants
                    best_params = {'p': kp, 'i': ki, 'd': kd}
                    print(f"  --> NEW BEST: MSE={best_mse:.4f} with Kp={kp}, Ki={ki}, Kd={kd}")

    # Restoring the original gains for further use
    controller.k_p_z = original_gains['p']
    controller.k_i_z = original_gains['i']
    controller.k_d_z = original_gains['d']

    # Printing out the best PID parameters and the MSE that achieved with them
    print("\n" + "="*60)
    print("Grid Search Complete!")
    print("============================================================")
    print(f"Optimal PID Parameters:")
    print(f"  Kp_Z: {best_params['p']}")
    print(f"  Ki_Z: {best_params['i']}")
    print(f"  Kd_Z: {best_params['d']}")
    print(f"  Achieved MSE: {best_mse:.4f}")
    print("="*60)
# ---------------------------------------------------

# ---------This part added for this project----------
# A function for tuning the vertical PID constants with using the ziegler-nichols method
def run_ziegler_nichols_tuning():
    print("\n" + "="*50)
    print("Starting Ziegler-Nichols Tuning (Z-axis)")
    print("Increasing Kp until continuous oscillation...")
    print("="*50)

    # Open up the vertical, zn tuning and zn-p only mode for tuning 
    controller.VERTICAL_ONLY_MODE = True
    controller.zn_tuning_active = True
    controller.zn_p_only_mode = True 

    # Saving the original vertical PID constants for restoring them later
    original_k_p_z = controller.k_p_z
    original_k_i_z = controller.k_i_z
    original_k_d_z = controller.k_d_z

    # Set up other vertical constants to zero for zn tuning
    controller.k_i_z = 0.0
    controller.k_d_z = 0.0

    # Set up the flag of Ku, ultimate gain and period
    ku_found = False
    ultimate_gain = 0.0
    ultimate_period = 0.0

    # Initializing the Kp and increase it gradually in order to tune
    controller.k_p_z = 650 
    p_gain_increment = 5.0 

    # Initializing the limit for Kp
    max_p_gain = 2500.0 

    # Initializing the quadcopter position for vertical tuning
    tuning_quadcopter = Quadcopter(pos=(0.0,0.0,0.0), attitude=(0,0,0))
    target_height = 4.0 # Set up the target height for quadcopter

    # A while loop to find ultimate gain (Ku) and ultimate period (Tu)
    while not ku_found and controller.k_p_z < max_p_gain:
        
        # Resetting the quadcopter position and time for each new Kp test
        tuning_quadcopter = Quadcopter(pos=(0.0,0.0,0.0), attitude=(0,0,0))
        time[0] = 0.0

        # Clearing the height data fpr new plots and resetting zn variables for new Kp test
        controller.reset_height_data() 
        controller.reset_zn_tuning_variables() 

        # Runing the simulation for a fixed duration to find the oscillation
        test_duration_per_kp = 15.0 
        num_frames_per_test = int(test_duration_per_kp * animation_frequency)

        # A double for loop for every number of frames per test for every iteration of control
        for i in range(num_frames_per_test):
            for _ in range(control_iterations):
               
                # Defining the desired state for vertical step response
                desired_state = DesiredState(
                    pos = np.array([0.0, 0.0, target_height]),
                    vel = np.array([0.0, 0.0, 0.0]),
                    acc = np.array([0.0, 0.0, 0.0]),
                    yaw = 0.0,
                    yawdot = 0.0
                )
                
                # Finding the thrust and moment values for this tuning  simulation
                F, M = controller.run(tuning_quadcopter, desired_state)

                # Updating the quadcopter's state with each calculated thrust and moment values 
                # and update the time for next iteration 
                tuning_quadcopter.update(dt, F, M)
                time[0] += dt

                # Checking the ultimate period if was detected by the controller
                if controller._ultimate_period_detected != 0.0:
                    ultimate_gain = controller.k_p_z
                    ultimate_period = controller._ultimate_period_detected
                    # If the ultimate gain found set the related flag true and break inner control loop
                    ku_found = True
                    break 
            # If the ultimate gain found it also break from the animation loop
            if ku_found:
                break 
        
        # If there is no oscillation detected for this Kp variable, increase the Kp and try it again
        if not ku_found: 
            controller.k_p_z += p_gain_increment
        # Otherwise plot the oscillation if that is already detected
        else:
            print("Continuous oscillation detected!")
            controller.analyze_height()

    # Resetting the zn tuning flags to zero in any case
    controller.zn_tuning_active = False
    controller.zn_p_only_mode = False

    # If ultimate gain found print some information like ultimate period 
    # and ultimate gain itself
    if ku_found:
        print("\n" + "="*50)
        print("Ziegler-Nichols Tuning Results (Z-axis):")
        print(f"Ultimate Gain (Ku): {ultimate_gain:.3f}")
        print(f"Ultimate Period (Tu): {ultimate_period:.3f} seconds")
        print("="*50)

        # The calculation of zn PID constants with using the related standard PID formulas
        zn_kp = 0.6 * ultimate_gain
        zn_ki = (1.2 * ultimate_gain) / ultimate_period
        zn_kd = 0.075 * ultimate_gain * ultimate_period

        # The printing of calculated PID terms
        print("\nCalculated Ziegler-Nichols PID Gains (for Z-axis):")
        print(f"   Kp_Z (P term): {zn_kp:.3f}")
        print(f"   Ki_Z (I term): {zn_ki:.3f}")
        print(f"   Kd_Z (D term): {zn_kd:.3f}")
        
    
    # An error message if ultimate gain cannot found
    else:
        print("\n" + "="*50)
        print("Ziegler-Nichols Tuning Failed: Could not find continuous oscillation within limits.")
        print("  Suggestions: Adjust initial Kp_Z, p_gain_increment, max_p_gain, or test_duration_per_kp.")
        print("  The system might be too damped, or parameters might be outside reasonable range.")
        print("="*50)

    # Restoring the original PID gains
    controller.k_p_z = original_k_p_z
    controller.k_i_z = original_k_i_z
    controller.k_d_z = original_k_d_z

    # Show the all plots until the simulation user closes them
    plt.show() 

# ---------------------------------------------------


def main():
    pos = (0.5,0,0)
    attitude = (0,0,0)
    quadcopter = Quadcopter(pos, attitude)
    waypoints = trajGen3D.get_helix_waypoints(0, 9)
    
    # Trajectory spline coefficients
    (coeff_x, coeff_y, coeff_z) = trajGen3D.get_MST_coefficients(waypoints)
    
    def control_loop(i):
        for _ in range(control_iterations):
            attitudeControl(quadcopter, time, waypoints, coeff_x, coeff_y, coeff_z)
        return quadcopter.world_frame()

    # Visualazing the quadcopter simulation 
    plot_quad_3d(waypoints, control_loop)
   
    # ---------This part added for this project----------
    # Plotting the summary of thrust and three moment vectors when simulation ends
    controller.plotting_the_performance()
    
    # Plotting the graph of position over time
    plot_position_over_time()
    # --------------------------------------------------

if __name__ == "__main__":
    
    # ---------This part added for this project----------
    # A condition for step response test
    if len(sys.argv) > 1 and sys.argv[1] == 'step_test':
        controller.VERTICAL_ONLY_MODE = True
        run_step_response_simulation() 
        controller.analyze_height()
        controller.VERTICAL_ONLY_MODE = False
    
    # An if condition for comparing the untuned and tuned PID constant performance metrics
    elif len(sys.argv) > 1 and sys.argv[1] == 'pid_compare':
        run_pid_comparison_simulation()
        
    # A condition to tune PID constants with using Ziegler-Nichols method
    elif len(sys.argv) > 1 and sys.argv[1] == 'zn_tune':
        run_ziegler_nichols_tuning()

    # A condition to tune PID constants with using grid search method
    elif len(sys.argv) > 1 and sys.argv[1] == 'grid_search':
        run_grid_search_tuning()
    # ------------------------------------------------

    # Default simulation to execute
    else:
        main()
