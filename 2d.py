import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.optimize import fsolve
from copy import deepcopy

from BicycleSolver import BicycleSolver


def const_delta(t):
    return 0.1

def explicit_osc():
    modified_C_a = [29000, 33000, 37000, 41000]
    init_vector = np.array([[0],[0]], dtype=np.float64)

    """ Eplicit Oscillations """
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
        return solver.bicycle_model(solver.A, solver.B, y, 0.1)
    t_final = 20
    init_t = 0
    step_size = 0.05
    histories = []
    labels = []
    max_iter = int(t_final / step_size)
    for i in range(len(modified_C_a)):
        print("war")
        solver = BicycleSolver(

            m = 1400,
            a = 1.14,
            b = 1.33,
            C_alpha_f = 25000,
            C_alpha_r = modified_C_a[i],
            I_z = 2420,
            u = 75 * 1000 / 3600
        )

        res, hist = solver.solve(solver.rk4, task_a_model, init_vector, init_t, max_iter, step_size)
        histories.append(hist)
        labels.append(f'c_ar: {modified_C_a[i]}')

    solver.visualize_results(histories, labels, init_t, t_final, max_iter, "Lateral Velocity with Explicit Oscillations", "Yaw Rate with Explicit Oscillations")

def reduced_mass():
    modified_m = [1400, 1300, 1200, 1100, 1000, 900, 800]
    init_vector = np.array([[0],[0]], dtype=np.float64)

    """
    Variable Mass
    """
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
        return solver.bicycle_model(solver.A, solver.B, y, 0.1)
    t_final = 20
    init_t = 0
    step_size = 0.05
    histories = []
    labels = []

    max_iter = int(t_final / step_size)
    for i in range(len(modified_m)):

        solver = BicycleSolver(

            m = modified_m[i],
            a = 1.14,
            b = 1.33,
            C_alpha_f = 25000,
            C_alpha_r = 21000,
            I_z = 2420,
            u = 75 * 1000 / 3600
        )

        res, hist = solver.solve(solver.rk4, task_a_model, init_vector, init_t, max_iter, step_size)
        histories.append(hist)
        labels.append(f'c_ar: {modified_m[i]}')

    solver.visualize_results(histories, labels, init_t, t_final, max_iter, "y Acceleration with Different Vehicle Masses", "Yaw Acceleration with Different Vehicle Masses")

def compute_d(C_alpha_f, car, m, u):
    return -1 * (C_alpha_f + car) / (m * u)

def compute_e(a, C_alpha_f, b, car, m, u):
    return (-1 * (a * C_alpha_f - b * car) / (m * u)) - u

def compute_f(a, C_alpha_f, b, car, k, u):
    return -1 * (a * C_alpha_f - b * car) / (k * u)

def compute_g(a, C_alpha_f, b, car, k, u):
    return -1 * (a**2 * C_alpha_f + b**2 * car) / (k * u)

def find_discriminant_0(var, u_var, k_vars):
    """
    Generalized function to solve for any unknown variable.
    :list var: List containing the unknown variable.
    :string unknown_var: The variable to solve for.
    :dict k_vars: Dictionary of known variables.
    """
    # Update the known variables dictionary with the unknown one
    k_vars[u_var] = var[0]

    d = compute_d(k_vars["C_alpha_f"], k_vars["C_alpha_r"], k_vars["m"], k_vars["u"])
    e = compute_e(k_vars["a"], k_vars["C_alpha_f"], k_vars["b"], k_vars["C_alpha_r"], k_vars["m"], k_vars["u"])
    f = compute_f(k_vars["a"], k_vars["C_alpha_f"], k_vars["b"], k_vars["C_alpha_r"], k_vars["k"], k_vars["u"])
    g = compute_g(k_vars["a"], k_vars["C_alpha_f"], k_vars["b"], k_vars["C_alpha_r"], k_vars["k"], k_vars["u"])

    lhs = (d + g) ** 2
    rhs = 4 * (d * g - e * f)
    
    return lhs - rhs

def find_det_0(var, u_var, k_vars):
    """
    Generalized function to solve for any unknown variable.
    :list var: List containing the unknown variable.
    :string unknown_var: The variable to solve for.
    :dict k_vars: Dictionary of known variables.
    """
    # Update the known variables dictionary with the unknown one
    k_vars[u_var] = var[0]

    d = compute_d(k_vars["C_alpha_f"], k_vars["C_alpha_r"], k_vars["m"], k_vars["u"])
    e = compute_e(k_vars["a"], k_vars["C_alpha_f"], k_vars["b"], k_vars["C_alpha_r"], k_vars["m"], k_vars["u"])
    f = compute_f(k_vars["a"], k_vars["C_alpha_f"], k_vars["b"], k_vars["C_alpha_r"], k_vars["k"], k_vars["u"])
    g = compute_g(k_vars["a"], k_vars["C_alpha_f"], k_vars["b"], k_vars["C_alpha_r"], k_vars["k"], k_vars["u"])

    lhs = (d*g)
    rhs = (e*f)
    
    # solve for discriminant = 0
    return lhs - rhs


def find_unknown(func, unknown_var, known_vars, initial_guess=25000):
    solution = fsolve(func, [initial_guess], args=(unknown_var, known_vars))
    return solution[0]


def under_over_unbound(unknown_variable, known_values):
    resultdisc = find_unknown(find_discriminant_0, unknown_variable, known_values)
    resultdet = find_unknown(find_det_0, unknown_variable, known_values)
    modified_C_a = [resultdet - 1000, (resultdet + resultdisc) / 2, resultdisc + 10000]

    """ 
    Over, under, unbounded
    """

    init_vector = np.array([[0],[0]], dtype=np.float64)
    t_final = 20
    init_t = 0
    step_size = 0.2
    max_iter = int(t_final / step_size)

    histories = []
    labels = []

    for i in range(len(modified_C_a)):
        solver = BicycleSolver(

            m = 1400,
            a = 1.14,
            b = 1.33,
            C_alpha_f = 25000,
            C_alpha_r = modified_C_a[i],
            I_z = 2420,
            u = 75 * 1000 / 3600
        )
        def task_a_model(y, t) -> np.ndarray:
            return solver.bicycle_model(solver.A, solver.B, y, const_delta(t))

        res, hist = solver.solve(solver.rk4, task_a_model, init_vector, init_t, max_iter, step_size)

        histories.append(hist)
        labels.append(f'$c_{{ar}}$: {modified_C_a[i]}')

    solver.visualize_results(histories, labels, init_t, t_final, max_iter, "Lateral Velocity with Variable Damping" , "Yaw Rate with Variable Damping")


# Define the function to plot eigenvalues for varying C_alpha_f
def plot_eigenvalues_vs_caf(BicycleSolver, unknown_variable, known_values, b=False):
    resultdisc = find_unknown(find_discriminant_0, unknown_variable, known_values)
    resultdet = find_unknown(find_det_0, unknown_variable, known_values)
    
    caf_arr = np.linspace(17000, 30000, 50)
    if b:
        caf_arr = np.concatenate((caf_arr, [resultdisc, resultdet]))
    
    # Store eigenvalues for each C_alpha_f
    all_eigenvalues = []
    
    for caf in caf_arr:
        solver = BicycleSolver(
            m=1400,
            a=1.14,
            b=1.33,
            C_alpha_f=25000,
            C_alpha_r=caf,
            I_z=2420,
            u=75 * 1000 / 3600
        )
        eigenvalues = np.linalg.eig(solver.A).eigenvalues
        all_eigenvalues.append(eigenvalues)
    
    # Set up colormap from red to blue
    cmap = cm.get_cmap('cool')
    norm = mcolors.Normalize(vmin=min(caf_arr), vmax=max(caf_arr))
    
    # Plot setup
    plt.figure(figsize=(8, 6))
    
    
    if (b == False):
        for i, caf in enumerate(caf_arr):
            color = cmap(norm(caf))
            for ev in all_eigenvalues[i]:
                plt.scatter(ev.real, ev.imag, color=color)

    # Plot the special extra point in a distinct color (e.g., red or hot)
    else:
        special_color = "red"
        
        plt.scatter(all_eigenvalues[-2].real, all_eigenvalues[-2].imag, color="red", edgecolors='black', s=100, label=f'Special Point C_alpha_r = {resultdisc}')

        # Plot each C_alpha_f's eigenvalues with color gradient
        for i, caf in enumerate(caf_arr[:-2]):
            color = cmap(norm(caf))
            for ev in all_eigenvalues[i]:
                plt.scatter(ev.real, ev.imag, color=color)
        plt.scatter(all_eigenvalues[-1].real, all_eigenvalues[-1].imag, color="black", edgecolors='black', s=100, label=f'Special Point C_alpha_r = {resultdet}')


    # Plot the special extra point in a distinct color (e.g., red or hot)

    plt.axvline(0, color='red', linestyle='--', label='Stability Boundary (Re = 0)')
    plt.xlabel('Real Part of Eigenvalue')
    plt.ylabel('Imaginary Part of Eigenvalue')
    plt.title('Eigenvalues vs. C_alpha_f')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():

    known_values = {
        "C_alpha_f": 25000,
        # "C_alpha_r": unknown (we want to solve for this)
        "m": 1400,
        "u": 75/3.6,
        "a": 1.14,
        "b": 1.33,
        "k": 2420
    }

    unknown_variable = "C_alpha_r"

    explicit_osc()
    reduced_mass()
    under_over_unbound(unknown_variable, known_values)
    plot_eigenvalues_vs_caf(BicycleSolver, unknown_variable, known_values)
    plot_eigenvalues_vs_caf(BicycleSolver, unknown_variable, known_values, b=True)

if __name__ == "__main__":
    main()