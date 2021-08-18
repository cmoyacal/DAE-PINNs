import argparse
import numpy as np
import os
import torch
from tqdm import tqdm

import deepxde as dde
from matplotlib.pyplot import plot
from scipy.integrate import solve_ivp

from data.DAE import dae_data
import events
from metrics import l2_relative_error
from models.DAEnn import three_bus_PN
from supervisor import supervisor
from utils.utils import dotdict

def scipy_integrate(func, X0, args, IRK_times, N=0):
    """
    Integrates stiff power network DAEs using scipy for N time steps of size h 
    """
    V0 = 0.7   # we fix the voltage initial condition
    t_span = [0.0, args.h * N]
    t_sim = np.array([t_span[0]])
    for k in range(1, N +1):
        temp = (k - 1) * args.h + IRK_times * args.h
        t_sim  = np.vstack((t_sim, temp))
        t_next = np.array([k * args.h])
        t_sim = np.vstack((t_sim, t_next))
        del temp, t_next
    sol = solve_ivp(func, t_span, [X0[0], X0[1], X0[2], X0[3], V0], method=args.method, t_eval=t_sim.reshape(-1,))
    y_test = sol.y
    return t_sim[1:,:], y_test[:, 1:]

# pinn function
def power_net_dae(model, y_n, h, IRK_weights):
    
    # parameters
    M_1, M_2, D, D_d, b = .052, .0531, .05, .005, 10.
    V_1, V_2, P_g, P_l, Q_l = 1.02, 1.05, -2.0, 3.0, .1

    # pinn
    yn = y_n.clone()
    
    w1, w2, d2, d3, v3 = model(yn)
    w1 = w1.to(y_n.device)
    w2 = w2.to(y_n.device)
    d2 = d2.to(y_n.device)
    d3 = d3.to(y_n.device)
    v3 = v3.to(y_n.device)

    xi_w1 = w1[...,:-1].to(y_n.device)
    xi_w2 = w2[...,:-1].to(y_n.device)
    xi_d2 = d2[...,:-1].to(y_n.device)
    xi_d3 = d3[...,:-1].to(y_n.device)
    zeta_v3 = v3[...,:-1].to(y_n.device)

    f_1 = b * V_1 * V_2 * torch.sin(xi_d2) + b * V_2 * zeta_v3 * torch.sin(xi_d2 - xi_d3) + P_g
    f_2 = b * V_1 * zeta_v3 * torch.sin(xi_d3) + b * V_2 * zeta_v3 * torch.sin(xi_d3 - xi_d2) + P_l

    # compute dynamic residuals
    F0 = T * (1 / M_1) * (- D * xi_w1 + f_1 + f_2)
    F1 = T * (1 / M_2) * (- D * xi_w2 - f_1)
    F2 = T * (xi_w2 - xi_w1)
    F3 = T * (- xi_w1 - (1 / D_d) * f_2)

    f0 = yn[...,0:1] -  (w1 - h*F0.mm(IRK_weights.T))
    f1 = yn[...,1:2] -  (w2 - h*F1.mm(IRK_weights.T))
    f2 = yn[...,2:3] -  (d2 - h*F2.mm(IRK_weights.T))
    f3 = yn[...,3:4] -  (d3 - h*F3.mm(IRK_weights.T))

    # compute algebrtaic residuals
    G = 2 * b * (v3 ** 2) - b * v3 * V_1 * torch.cos(d3) - b * v3 * V_2 * torch.cos(d3 -d2) + Q_l
    g = - (T/v3) * G

    return [f0, f1, f2, f3], [g]

def power_net_dae_plot(t, x):
    eps = 0.0001
        
    # parameters
    m_1, m_2, d, d_d, b = .052, .0531, .05, .005, 10.
    v_1, v_2, p_g, p_l, q_l = 1.02, 1.05, -2.0, 3.0, .1

    w1, w2, d2, d3, v3 = x

    f_1 = b * v_1 * v_2 * np.sin(d2) + b * v_2 * v3 * np.sin(d2 - d3) + p_g 
    f_2 = b * v_1 * v3 * np.sin(d3) + b * v_2 * v3 * np.sin(d3 - d2) + p_l
    g = 2 * b * (v3 ** 2) - b * v3 * v_1 * np.cos(d3) - b * v3 * v_2 * np.cos(d3 - d2) + q_l
        
    F0 = (1 / m_1) * (- d * w1 + f_1 + f_2)
    F1 = (1 / m_2) * (- d * w2 - f_1)
    F2 = (w2 - w1)
    F3 = (- w1 - (1 / d_d) * f_2)
    F4 = (- (1 / (eps * v3)) * g)
        
    return F0, F1, F2, F3, F4

def main(args):
    print("starting...\n")
    log_dir = "./logs/dae-pinns-best/" 

    # cuda training
    device = torch.device("cuda")
    print("using...", device)
    torch.cuda.set_device(1)
    print("using GPU device number...", torch.cuda.current_device())

    # construct best nns
    dynamic = dotdict()
    dynamic.num_IRK_stages = args.num_IRK_stages
    dynamic.state_dim = 4
    dynamic.activation = "sin"
    dynamic.initializer = "Glorot normal"
    dynamic.dropout_rate = 0.0
    dynamic.type = "attention"
    dim_out = dynamic.state_dim * (dynamic.num_IRK_stages + 1)
    dynamic.layer_size = [dynamic.state_dim] + [100] * 4 + [dim_out]
    
    def alg_output_feature_layer(x):
        return torch.nn.functional.softplus(x)
    
    algebraic = dotdict()
    algebraic.num_IRK_stages = args.num_IRK_stages
    dim_out_alg = algebraic.num_IRK_stages + 1
    algebraic.layer_size = [dynamic.state_dim] + [40] * 2 + [dim_out_alg]
    algebraic.activation = "sin"
    algebraic.initializer = "Glorot normal"
    algebraic.dropout_rate = 0.0
    algebraic.type = "attention"

    nn = three_bus_PN(
        dynamic, 
        algebraic, 
        alg_out_transform=alg_output_feature_layer,
        stacked=False, 
        ).to(device)

    # data for evaluation 
    geom = dde.geometry.Hypercube([-.25, -.25, -.001, -.13],[.25, .25, 0.0, -.12])
    np.random.seed(1234)
    X_train = geom.random_points(10)
    np.random.seed(3456)
    X_test = geom.random_points(args.num_eval)
    data = dae_data(X_train, X_test, args, device=device, func=power_net_dae)

    # start the supervisor
    super = supervisor(data, nn, device=device)

    # restore the model
    model_name = 'model.pth'
    restore_path = os.path.join(log_dir, model_name)

    # we integrate num-eval initial conditions from the test dataset
    N = args.N
    print("size of dataset...", X_test.shape)
    l2_rel_errors = np.empty((args.num_eval, 5))
    l2_rel_errors_vs_N_alg = np.empty((args.num_eval, N+1))
    for k in tqdm(range(args.num_eval)):
        X0 = X_test[k]
        X0_npy = np.array(X0)
        # predict
        y_pred = super.integrate(X0_npy, N=N, dyn_state_dim=4, model_restore_path=restore_path)
        # compute true trajectory
        _, y_eval = scipy_integrate(power_net_dae_plot, X0, args, super.data.IRK_times, N=N)
        # collect errors
        for j in range(y_eval.shape[0]):
            l2_rel_errors[k,j] = l2_relative_error(y_pred[j,...], y_eval[j,...])
        # collect errors vs N for the algebraic variable
        for n in range(1, N+1):
            y_pred_n = super.integrate(X0_npy, N=n, dyn_state_dim=4, model_restore_path=restore_path)
            _, y_eval_n = scipy_integrate(power_net_dae_plot, X0, args, super.data.IRK_times, N=n)
            l2_rel_errors_vs_N_alg[k, n-1] = l2_relative_error(y_pred_n[-1,...], y_eval_n[-1,...]) 
        del y_pred, y_eval, y_pred_n, y_eval_n

    N_vec = np.arange(1, N + 0.1)
    np.savez(os.path.join(log_dir, "errors-best-model.npz"), N=N_vec, l2_rel_errors=l2_rel_errors, l2_rel_errors_vs_N_alg=l2_rel_errors_vs_N_alg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dae-pinns")

    # general args
    parser.add_argument('--num-IRK-stages', type=int, default=100, help="number of RK stages")
    parser.add_argument('--num-eval', type=int, default=100, help="number of evaluation test examples")

    # integration args
    parser.add_argument('--h', type=float, default=.1, help="step size")
    parser.add_argument('--N', type=int, default=100, help="number of time steps")
    parser.add_argument('--method', type=str, default='BDF', help="integration method")

    args = parser.parse_args()
    main(args)
