import utils
import numpy as np
import matplotlib.pyplot as plt

def levenberg_marquardt_method(P, x, X, num_iters=10, lamb=1e-3):
    
    rms_history = []

    for k in range(num_iters):
        
        # Linearize reprojection error
        r, J = utils.linearize_reprojection_error(P, x, X)

        # Current RMS error
        current_rms = utils.compute_reprojection_rms_error(P, x, X)

        #Computes the LM update.
        delta = np.linalg.solve(J.T @ J + lamb * np.eye(J.shape[1]), -J.T @ r)

        # Update solution
        P_new, X_new = utils.update_solution(P, X, delta)

        # Compute new RMS
        new_rms = utils.compute_reprojection_rms_error(P_new, x, X_new)

        # Accept / reject step
        if new_rms < current_rms:
            P, X = P_new, X_new
            lamb *= 0.1   # decrease damping (trust model more)
            print(f"Iteration {k+1}: RMS = {new_rms:.6f}, lambda = {lamb:.2e} (accepted)")
            rms_history.append(new_rms)
        else:
            lamb *= 10    # increase damping (trust model less)
            print(f"Iteration {k+1}: step rejected, lambda = {lamb:.2e}")
            rms_history.append(current_rms)

    return P, X, rms_history


if __name__ == "__main__":
    # Load data from Exercise 5
    data = np.load("ex5_results.npz")

    P1 = data["P1"]
    P2 = data["P2"]
    X  = data["X"]
    x1 = data["x1"]
    x2 = data["x2"]

    # Run steepest descent
    P_opt, X_opt, rms_history = levenberg_marquardt_method([P1, P2], [x1, x2], X, num_iters=10)

    # Plot RMS vs iterations
    plt.figure()
    plt.plot(range(1, len(rms_history) + 1), rms_history, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("RMS reprojection error")
    plt.title("Levenberg-Marquardt Optimization")
    plt.grid()
    plt.show()

    # Final RMS
    print("\nFinal RMS error:", rms_history[-1])