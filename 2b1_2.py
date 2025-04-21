import numpy as np
from BicycleSolver import BicycleSolver


def const_delta(t):
    return 0.1

def b1():
    u_values_km = np.array([20, 50, 75, 100, 200, 220,240,260,280,300])
    u_values = [u * 1000 / 3600 for u in u_values_km]

    init_vector = np.array([[0],[0]], dtype=np.float64)
    t_final = 20
    init_t = 0
    step_size = 0.2
    max_iter = int(t_final / step_size)

    histories = []
    labels = []

    y_accel_hists = []

    for u in u_values:
        y_accel_hist = []
        solver = BicycleSolver(
            m = 1400,
            a = 1.14,
            b = 1.33,
            C_alpha_f = 25000,
            C_alpha_r = 21000,
            I_z = 2420,
            u = u
        )

        def task_a_model(y, t) -> np.ndarray:
            return solver.bicycle_model(solver.A, solver.B, y, const_delta(t))

        res, hist = solver.solve(solver.rk4, task_a_model, init_vector, init_t, max_iter, step_size, y_accel_hist=y_accel_hist)

        histories.append(hist)
        stable = solver.check_stability()
        labels.append(f'{str(round(u *3.6))}km/h | stable: {stable}')
        y_accel_hists.append(y_accel_hist)

    solver.visualize_results(histories, labels, init_t, t_final, max_iter, title="Acceleration VS Time with Varied Speed", titleb="Yaw Rate VS Time with Varied Speed", y_accel_hists=y_accel_hists)

def b2():
    u_values_km = np.linspace(228, 230, 100)
    u_values = [u * 1000 / 3600 for u in u_values_km]

    init_vector = np.array([[0],[0]], dtype=np.float64)
    t_final = 20
    init_t = 0
    step_size = 0.2
    max_iter = int(t_final / step_size)

    histories = []
    labels = []

    y_accel_hists = []

    for u in u_values:
        y_accel_hist = []
        solver = BicycleSolver(
            m = 1400,
            a = 1.14,
            b = 1.33,
            C_alpha_f = 25000,
            C_alpha_r = 21000,
            I_z = 2420,
            u = u
        )

        def task_a_model(y, t) -> np.ndarray:
            return solver.bicycle_model(solver.A, solver.B, y, const_delta(t))

        if solver.check_stability():
            print(f'{u * 3600/1000:.4f}km/h is stable')
        else:
            print(f'{u * 3600/1000:.4f}km/h is unstable')
            break

        
if __name__=="__main__":
    b1()
    b2()