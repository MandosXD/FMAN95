import utils
import numpy as np
import matplotlib.pyplot as plt

def steepest_descent(P, x, X, num_iters=10):
    
    rms_history = []
    
    for k in range(num_iters):
        
        # Linearize
        r, J = utils.linearize_reprojection_error(P, x, X)
        
        # Compute gradient: 2*J^T r  (factor 2 not necessary for direction)
        gradient = J.T @ r
        
        # Current RMS error
        current_rms = compute_reprojection_rms_error(P, x, X)
        rms_history.append(current_rms)
        
        # Descent direction
        delta = -gradient
        
        # Backtracking line search
        gamma = 1e-3        # initial step size (may need smaller!)
        success = False
        
        while gamma > 1e-12:
            
            delta_scaled = gamma * delta
            
            # Update solution
            P_new, X_new = utils.update_solution(P, X, delta_scaled)
            
            # Compute new RMS
            new_rms = compute_reprojection_rms_error(P_new, x, X_new)
            
            if new_rms < current_rms:
                P, X = P_new, X_new
                success = True
                break
            else:
                gamma *= 0.5
        
        if not success:
            print("Line search failed at iteration", k)
            break
        
        print(f"Iteration {k+1}: RMS = {new_rms:.6f}, gamma = {gamma:.2e}")
    
    return P, X, rms_history