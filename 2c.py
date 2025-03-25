import numpy as np
import matplotlib.pyplot as plt

def const_delta(t):
        return 0.1

# Placement loc (0) for rear wheel, (1) for front - Dnoe
def get_new_m_a_b(m, a, b, added_w, placement_loc):
    new_m = m + added_w

    # car could be split into 2 point masses at
    # front & rear tires. So the new CoG is just the weighted
    # avg of the new rear & front weights
    front_mass = m / 2 + added_w * placement_loc
    rear_mass = m / 2 + added_w * 1 - placement_loc

    # Fix the coords system @ b = 0
    total = a + b
    frac = rear_mass / new_m
    new_b = total * frac
    new_a = total - new_b

    return new_m, new_a, new_b

def c1():
    def task_a_model(y, t) -> np.ndarray:
        return solver.bicycle_model(solver.A, solver.B, y, const_delta(t))

    added_w = np.linspace(100, 118, 11)
    cog_values = [get_new_m_a_b(1400, 1.14, 1.33, w, 0) for w in added_w]

    u = 227 * 1000 / 3600
    init_vector = np.array([[0],[0]])
    t_final = 20
    init_t = 0
    step_size = 0.2
    max_iter = int(t_final / step_size)

    histories = []
    labels = []

    for i, (m, a, b) in enumerate(cog_values):
        solver = BicycleSolver(
            m = m,
            a = a,
            b = b,
            C_alpha_f = 25000,
            C_alpha_r = 21000,
            I_z = 2420,
            u = u
        )

        res, hist = solver.solve(solver.rk4, task_a_model, init_vector, init_t, max_iter, step_size)
        stable = solver.check_stability()
        print(f"Stablility of {added_w[i]}kg: {stable}")

        histories.append(hist)
        labels.append(f'{round(added_w[i],2)} kg added')

    solver.visualize_results(histories, labels, init_t, t_final, max_iter, title="Acceleration VS Time with Varied Speed", titleb="Yaw Rate VS Time with Varied Speed")

    #will reach stable between 100 abd 130kg
    #between 115 and 118kg
    ##becomes stable once 117kg of mass is added

def c2():
    #C2 Standard Conditions
    # Convert speed from km/h to m/s
    u_values_km = np.linspace(50, 250, 6)  # km/h
    u_values = [u * 1000 / 3600 for u in u_values_km]  # m/s

    init_vector = np.array([[0],[0]])
    t_final = 20
    init_t = 0
    step_size = 0.2
    max_iter = int(t_final / step_size)

    #iterate through delta
    delta = [0.05, 0.1, 0.15, 0.2, 0.25]


    histories = []
    labels = []

    # Simulate car behavior for different steering angles
    for u in u_values:
        for delta_value in delta:
            solver = BicycleSolver(
                m=1400, 
                a=1.14,  
                b=1.33,  
                C_alpha_f=25000,  
                C_alpha_r=21000,  
                I_z = 2420,
                u=u  
            )

            def task_a_model(y,t) -> np.ndarray:
                return solver.bicycle_model(solver.A, solver.B, y, delta_value)

            res, hist = solver.solve(solver.rk4, task_a_model, init_vector, init_t, max_iter, step_size)
        

            histories.append(hist)
            labels.append(f'δ = {delta_value:.2f} rad, u = {u * 3600 / 1000:.0f} km/hr')


    solver.visualize_results2(histories, labels, init_t, t_final, max_iter, title="Simulation at Different Speeds and Steering Angles")


    # C2 stability check
    for i, history in enumerate(histories):
        y_difference = [history[j][0] - history[j-1][0] for j in range(1, len(history))]
        yaw_difference = [history[j][1] - history[j-1][1] for j in range(1, len(history))]

        if (abs(y_difference[-1]) > abs(y_difference[-5]) or
            abs(yaw_difference[-1]) > abs(yaw_difference[-5])):
        
            u_index = i // len(delta)
            delta_index = i % len(delta)
            print(f'Diverges at {u_values_km[u_index]:.0f} km/h and {delta[delta_index]:.2f} rad')
        else:
            u_index = i // len(delta)
            delta_index = i % len(delta)
            print(f'Converges at u {u_values_km[u_index]:.0f} km/h and {delta[delta_index]:.2f} rad')


    ##C2 - Standard conditions trajectory
    # Convert speed from km/h to m/s
    u_values_km = np.linspace(50, 250, 6)  # km/h
    u_values = [u * 1000 / 3600 for u in u_values_km]  # m/s

    init_vector = np.array([[0], [0], [0], [0]])
    t_final = 20
    init_t = 0
    step_size = 0.2
    max_iter = int(t_final / step_size)

    # Iterate through delta
    delta = [0.05, 0.1, 0.15, 0.2, 0.25]

    # Simulate car behavior for different steering angles and speeds
    for u in u_values:
        for delta_value in delta:
            solver = BicycleSolver(
                m=1400, 
                a=1.14,  
                b=1.33,  
                C_alpha_f=25000,  
                C_alpha_r=21000,  
                I_z=2420,
                u=u  
            )

            def task_a_model(y, t) -> np.ndarray:
                return solver.bicycle_model(solver.A_4, solver.B_4, y, delta_value)

            res, hist = solver.solve(solver.rk4, task_a_model, init_vector, init_t, max_iter, step_size)
            ground_velocity = solver.ground_velocity(hist)
            ground_position = solver.ground_position(ground_velocity, step_size)
            
            # Plot the trajectory for this combination of u and delta
            plt.plot(ground_position[:, 0], ground_position[:, 1], label=f'δ = {delta_value:.2f} rad, u = {u * 3600 / 1000:.0f} km/hr')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Vehicle Trajectory in Standard Conditions")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


    ### Task C2 - Winter Conditions

    # Convert speed from km/h to m/s
    u_values_km = np.linspace(50, 250, 6)  # km/h
    u_values = [u * 1000 / 3600 for u in u_values_km]  # m/s

    init_vector = np.array([[0],[0]])
    t_final = 20
    init_t = 0
    step_size = 0.2
    max_iter = int(t_final / step_size)

    #iterate through delta
    delta = [0.05, 0.1, 0.15, 0.2, 0.25]


    histories = []
    labels = []

    # Simulate car behavior for different steering angles
    for u in u_values:
        for delta_value in delta:
            solver = BicycleSolver(
                m=1400, 
                a=1.14,  
                b=1.33,  
                C_alpha_f=6667,  
                C_alpha_r=5600,  
                I_z = 2420,
                u=u  
            )

            def task_a_model(y,t) -> np.ndarray:
                return solver.bicycle_model(solver.A, solver.B, y, delta_value)

            res, hist = solver.solve(solver.rk4, task_a_model, init_vector, init_t, max_iter, step_size)
        

            histories.append(hist)
            labels.append(f'δ = {delta_value:.2f} rad, u = {u * 3600 / 1000:.0f} km/hr')


    solver.visualize_results2(histories, labels, init_t, t_final, max_iter, title="Simulation at Different Speeds and Steering Angles in Winter Conditions")


    # C2 stability check
    for i, history in enumerate(histories):
        y_difference = [history[j][0] - history[j-1][0] for j in range(1, len(history))]
        yaw_difference = [history[j][1] - history[j-1][1] for j in range(1, len(history))]

        if (abs(y_difference[-1]) > abs(y_difference[-5]) or
            abs(yaw_difference[-1]) > abs(yaw_difference[-5])):
        
            u_index = i // len(delta)
            delta_index = i % len(delta)
            print(f'Diverges at {u_values_km[u_index]:.0f} km/h and {delta[delta_index]:.2f} rad')
        else:
            u_index = i // len(delta)
            delta_index = i % len(delta)
            print(f'Converges at u {u_values_km[u_index]:.0f} km/h and {delta[delta_index]:.2f} rad')


    ##C2 - Winter conditions trajectory
    # Convert speed from km/h to m/s
    u_values_km = np.linspace(50, 250, 6)  # km/h
    u_values = [u * 1000 / 3600 for u in u_values_km]  # m/s

    init_vector = np.array([[0], [0], [0], [0]])
    t_final = 20
    init_t = 0
    step_size = 0.2
    max_iter = int(t_final / step_size)

    # Iterate through delta
    delta = [0.05, 0.1, 0.15, 0.2, 0.25]

    # Simulate car behavior for different steering angles and speeds
    for u in u_values:
        for delta_value in delta:
            solver = BicycleSolver(
                m=1400, 
                a=1.14,  
                b=1.33,  
                C_alpha_f=6667,  
                C_alpha_r=5600,  
                I_z=2420,
                u=u  
            )

            def task_a_model(y, t) -> np.ndarray:
                return solver.bicycle_model(solver.A_4, solver.B_4, y, delta_value)

            res, hist = solver.solve(solver.rk4, task_a_model, init_vector, init_t, max_iter, step_size)
            ground_velocity = solver.ground_velocity(hist)
            ground_position = solver.ground_position(ground_velocity, step_size)
            
            # Plot the trajectory for this combination of u and delta
            plt.plot(ground_position[:, 0], ground_position[:, 1], label=f'δ = {delta_value:.2f} rad, u = {u * 3600 / 1000:.0f} km/hr')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Vehicle Trajectory in Winter Conditions")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # C2 stability check
    for i, history in enumerate(histories):
        y_difference = [history[j][0] - history[j-1][0] for j in range(1, len(history))]
        yaw_difference = [history[j][1] - history[j-1][1] for j in range(1, len(history))]

        if (abs(y_difference[-1]) > abs(y_difference[-5]) or
            abs(yaw_difference[-1]) > abs(yaw_difference[-5])):
        
            u_index = i // len(delta)
            delta_index = i % len(delta)
            print(f'Diverges at {u_values_km[u_index]:.0f} km/h and {delta[delta_index]:.2f} rad')
        else:
            u_index = i // len(delta)
            delta_index = i % len(delta)
            print(f'Converges at u {u_values_km[u_index]:.0f} km/h and {delta[delta_index]:.2f} rad')


def c3():
    ###C3 - A.1 ICY CONDITIONS
    # Piecewise condition to model icy road 
    def summer_tires(delta : float):
        if delta <= 0.06:
            return 20000
        elif delta > 0.2:
            return 0
        else:
            return 100
        
    def winter_tires(delta):
        if delta <= 0.06:
            return 20000
        elif 0.06 < delta <= 0.3:
            return 5000
        else:
            return 0
        
    init_vector = np.array([[0],[0],[0],[0]])
    steering_angles = [0.02, 0.04, 0.6, 0.1, 0.14, 0.18, 0.2, 0.22]

    init_t = 0
    t_final = 1000

    step_size = 0.1
    max_iter = int(t_final / step_size)

    plt.figure(figsize=(8, 6))
    for steering_angle in steering_angles:
        solver = BicycleSolver(
            m = 1400,
            a = 1.14,
            b = 1.33,
            C_alpha_f = summer_tires(steering_angle),
            C_alpha_r = summer_tires(steering_angle),
            I_z = 2420,
            u = 150 * 1000 / 3600
        )

        def task_a_model(y, t) -> np.ndarray:
            return solver.bicycle_model(solver.A_4, solver.B_4, y, steering_angle)

        res, hist = solver.solve(solver.rk4, task_a_model, init_vector, init_t, max_iter, step_size)
        ground_velocity = solver.ground_velocity(hist)
        ground_position = solver.ground_position(ground_velocity,step_size)
        plt.plot(ground_position[:,0], ground_position[:,1], label=f"{steering_angle}rads/sec")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Steering angle in Icy Condition without Winter Tires")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    for steering_angle in steering_angles:
        solver = BicycleSolver(
            m = 1400,
            a = 1.14,
            b = 1.33,
            C_alpha_f = winter_tires(steering_angle),
            C_alpha_r = winter_tires(steering_angle),
            I_z = 2420,
            u = 150 * 1000 / 3600
        )

        def task_a_model(y, t) -> np.ndarray:
            return solver.bicycle_model(solver.A_4, solver.B_4, y, steering_angle)

        res, hist = solver.solve(solver.rk4, task_a_model, init_vector, init_t, max_iter, step_size)
        ground_velocity = solver.ground_velocity(hist)
        ground_position = solver.ground_position(ground_velocity,step_size)
        plt.plot(ground_position[:,0], ground_position[:,1], label=f"{steering_angle}rads/sec")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Steering angle in Icy Condition with Winter Tires")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


    ###C3 - A.2 WINTER TIRES

    init_vector = np.array([[0],[0]])
    t_final = 20
    init_t = 0
    step_size = 0.2
    max_iter = int(t_final / step_size) 

    delta = [0.02, 0.04, 0.6, 0.1, 0.14, 0.18, 0.2, 0.22]

    #piecewise condition to model icy road 
    def winter_tires(delta):
        if delta <= 0.06:
            return 20000
        elif 0.06 < delta <= 0.3:
            return 5000
        else:
            return 0
        
    histories = []
    labels = []

    # Simulate car behavior for different steering angles
    for i, delta_value in enumerate(delta):
        solver = BicycleSolver(
            m=1400, 
            a=1.14,  
            b=1.33,  
            C_alpha_f = winter_tires(delta_value),
            C_alpha_r = winter_tires(delta_value),
            I_z = 2420,
            u = 75 * 1000 / 3600 
        )

        def task_a_model(y,t) -> np.ndarray:
            return solver.bicycle_model(solver.A, solver.B, y, delta_value)

        res, hist = solver.solve(solver.rk4, task_a_model, init_vector, init_t, max_iter, step_size)


        histories.append(hist)
        labels.append(f'δ= {delta_value:.2f} rad')


    solver.visualize_results(histories, labels, init_t, t_final, max_iter, title= "Icy Conditions with Winter Tires")