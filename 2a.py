import numpy as np

def a2():
    # Grid Independence Check
    def const_delta(t):
        return 0.1

    highly_exact_step_size = 0.0000001

    init_vector = np.array([[0.0],[0.0]], dtype=np.float64)
    t_final = 1
    init_t = 0

    euler_results = []
    rk4_results = []

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
        return solver.bicycle_model(solver.A, solver.B, y, const_delta(t))

    max_iteration = int(t_final / highly_exact_step_size)
    _, exact_sol = solver.solve(solver.rk4, task_a_model, init_vector, 0, max_iteration, highly_exact_step_size, t_final)

    log_step_size = np.linspace(-1, -3, 3)
    print(log_step_size)
    grid_values = [np.power(10, i) for i in log_step_size]
    print(grid_values)

    for name, iterator in [("RK4", solver.rk4), ("Euler's", solver.eulers_method)]:
        log_error = []
        for step_size in grid_values:
            max_iteration = int(t_final / step_size)
            _, hist = solver.solve(iterator, task_a_model, init_vector, 0, max_iteration, step_size)
            # Compute log error
            target = exact_sol[-1] # we compute error on the last iteration
            error = target - hist[-1]
            log_error.append(np.log10(np.linalg.norm(error)))

        plt.plot(log_step_size, log_error, label=f"{name}")

    plt.xlabel('Log step size')
    plt.ylabel("Log Error")
    plt.title(f"{name} Grid Independence Check")
    plt.legend()
    plt.grid()
    plt.show()