import numpy as np
import matplotlib.pyplot as plt 

from BicycleSolver import BicycleSolver

init_vector = np.array([[0],[0],[0],[0]])
init_t = 0
t_final = 1000

step_size = .1
max_iter = int(t_final / step_size)

plt.figure(figsize=(8, 6))

for steering_angle in [.1,.2,.3,.4,.5]:
    solver = BicycleSolver(
        m = 1400,
        a = 1.14,
        b = 1.33,
        C_alpha_f = 25000,
        C_alpha_r = 21000,
        I_z = 2420,
        u = 75 * 1000 / 3600
    )

    def task_a_model(y, t) -> np.ndarray:
        return solver.bicycle_model(solver.A_4, solver.B_4, y, steering_angle)

    res, hist = solver.solve(solver.rk4, task_a_model, init_vector, init_t, max_iter, step_size)
    ground_velocity = solver.ground_velocity(hist)
    ground_position = solver.ground_position(ground_velocity,step_size)
    plt.plot(ground_position[:,0], ground_position[:,1], label=f"{steering_angle}rad/s")

plt.xlabel("X(m)")
plt.ylabel("Y(m)")
plt.title("Handling Behaviour of Car with Step Steering Experiment")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc="upper right")
plt.show()