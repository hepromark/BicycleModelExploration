import numpy as np
import matplotlib.pyplot as plt 

from BicycleSolver import BicycleSolver

init_vector = np.array([[0],[0],[0],[0]])

init_t = 0
t_final = 100
plt.figure(figsize=(8, 6))

step_size = .01
steering_angle = 0.1
max_iter = int(t_final / step_size)

speeds = list(range(1,225,25))
yaw_rate = []

for speed in speeds:
    solver = BicycleSolver(
            m = 1400,
            a = 1.14,
            b = 1.33,
            C_alpha_f = 25000,
            C_alpha_r = 21000,
            I_z = 2420,
            u = speed * 1000 / 3600 # 
        )

    def task_a_model(y, t) -> np.ndarray:
        return solver.bicycle_model(solver.A_4, solver.B_4, y, steering_angle)

    res, hist_1 = solver.solve(solver.rk4, task_a_model, init_vector, init_t, max_iter, step_size)
    ground_velocity = solver.ground_velocity(hist_1)
    ground_position = solver.ground_position(ground_velocity,step_size)
    plt.plot(ground_position[:,0], ground_position[:,1], label=f"{speed}km/h")
    yaw_rate.append(hist_1[max_iter - 1,3])

# Creating approximate linear fit of first two points
slope = (yaw_rate[1] - yaw_rate[0] ) / 25
x_points = np.linspace(1,225,50)
y_points = np.multiply(x_points,slope)
plt.plot(x_points,y_points, "--", label="Linear Approximation", color="black")

plt.xlabel("u (km/h)")
plt.ylabel("Yaw Rate(rads/s)")
plt.title("Yaw Rate VS Car Reference Frame Speed(u)")
plt.grid(True, linestyle='-', alpha=0.6)
plt.legend(loc="upper right")
plt.show()