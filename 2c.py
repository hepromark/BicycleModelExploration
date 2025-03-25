import numpy as np

def c():
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