import os
import argparse

import torch
import numpy as np
import deepxde as dde
from scipy.integrate import solve_ivp

from utils.utils import dotdict
from models.DAEnn import three_bus_PN
from data.other_DAE_solvers import dae_data_other
from supervisor import supervisor
import events
from utils.plots import plot_loss_history, plot_three_bus, plot_regression, plot_L2relative_error
from metrics import l2_relative_error

def scipy_integrate(func, X0, args, N=0):
    """
    integrates stiff power network dynamics using scipy for N time steps of size args.h 
    """
    V0 = 0.7   # we fix the volatge initial condition
    t_span = [0.0, args.h * N]
    t_sim = np.linspace(0.0, args.h * N, N+1)
    sol = solve_ivp(func, t_span, [X0[0], X0[1], X0[2], X0[3], V0], method=args.method, t_eval=t_sim.reshape(-1,))
    y_test = sol.y
    t_sim = t_sim.reshape(-1, 1)
    return t_sim[1:,:], y_test[:, 1:]


def power_net_dae(model, y_n, h):
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

    f_1 = b * V_1 * V_2 * torch.sin(d2) + b * V_2 * v3 * torch.sin(d2 - d3) + P_g
    f_2 = b * V_1 * v3 * torch.sin(d3) + b * V_2 * v3 * torch.sin(d3 - d2) + P_l

    F0 = (1 / M_1) * (- D * w1 + f_1 + f_2)
    F1 = (1 / M_2) * (- D * w2 - f_1)
    F2 = (w2 - w1)
    F3 = (- w1 - (1 / D_d) * f_2)

    f0 = yn[...,0:1] -  (w1 - h*F0)
    f1 = yn[...,1:2] -  (w2 - h*F1)
    f2 = yn[...,2:3] -  (d2 - h*F2)
    f3 = yn[...,3:4] -  (d3 - h*F3)

    # compute algebrtaic residuals
    G = 2 * b * (v3 ** 2) - b * v3 * V_1 * torch.cos(d3) - b * v3 * V_2 * torch.cos(d3 -d2) + Q_l
    g = - (1 / v3) * G

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

    # create log dir
    if os.path.isdir(args.log_dir) == False:
        os.makedirs(args.log_dir, exist_ok=True)

    # enabling gpu training
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using...", device)
    if use_cuda:
        torch.cuda.set_device(1)
        print("device number...", torch.cuda.current_device())

    # construct the neural nets
    dynamic = dotdict()
    dynamic.state_dim = 4
    
    def alg_output_feature_layer(x):
        return torch.nn.functional.softplus(x)

    dynamic.activation = "sin"
    dynamic.initializer = "Glorot normal"
    dynamic.dropout_rate = 0.0
    dynamic.batch_normalization = None
    dynamic.layer_normalization = None
    dynamic.type = "attention"
    dim_out = dynamic.state_dim 
    dynamic.layer_size = [dynamic.state_dim] + [100] * 5 + [dim_out]
    dynamic.num_IRK_stages = 0
    
    algebraic = dotdict()
    dim_out_alg = 1
    algebraic.layer_size = [dynamic.state_dim] + [100] * 2 + [dim_out_alg]
    algebraic.activation = "sin"
    algebraic.initializer = "Glorot normal"
    algebraic.dropout_rate = 0.0
    algebraic.batch_normalization = None
    algebraic.layer_normalization = None
    algebraic.type = "attention"        

    nn = three_bus_PN(
        dynamic, 
        algebraic,  
        alg_out_transform=alg_output_feature_layer,
        stacked=False, 
        ).to(device)

    # Data for training
    geom = dde.geometry.Hypercube([-.5, -.5, -.5, -.5],[.5, .5, .5, .5])
    np.random.seed(1234)
    num_train = 2000
    X_train = geom.random_points(num_train)
    np.random.seed(3456)
    num_test = 1000
    X_test = geom.random_points(num_test)
    data = dae_data_other(X_train, X_test, args, device=device, func=power_net_dae)

    # start the supervisor
    super = supervisor(data, nn, device=device)
    
    # compile the supervisor
    optimizer = torch.optim.Adam(nn.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                'min',
                patience=1000,
                verbose=True,
                factor=0.8,
            )
        
    super.compile(
        optimizer,
        loss_weights=[args.dyn_weight, args.alg_weight],
        scheduler=scheduler,
        scheduler_type= "plateau",
    )

    # train the DAE surrogate model
    model_name = 'model.pth'
    save_path = os.path.join(args.log_dir, model_name)
    checker = events.ModelCheckPoint(save_path, save_better_only=True, every=1000) 
    restore_path = save_path if args.start_from_best else None

    loss_history, state = super.train(
        epochs=args.epochs,
        batch_size=2000, 
        test_every=1000,
        num_val=100,
        events=[checker],
        model_restore_path=restore_path,
        use_tqdm=True,
    )

    # Results

    # plot loss history
    print("plotting train and test loss...\n")
    plot_loss_history(loss_history, fname=os.path.join(args.log_dir, 'loss.png'))
    # save loss history for future use
    np.savez(
        os.path.join(args.log_dir, 'loss-history'), 
        steps = np.array(loss_history.steps), 
        loss_train = np.array(loss_history.loss_train),
        loss_test = np.array(loss_history.loss_test)
        )

    # Test one trajectory
    X0 = [0., 0., .1, .1]
    X0_npy = np.array(X0)
    y_pred = super.integrate(X0_npy, N=args.N, dyn_state_dim=4, model_restore_path=save_path)
    
    t, y_eval = scipy_integrate(power_net_dae_plot, X0, args, N=args.N)
    print("plotting trajectory...\n")
    plot_three_bus(t, y_eval, y_pred, fname=os.path.join(args.log_dir, 'trajectories.png'), size=25, figsize=(16,24))

    # compute metrics for long-time integration
    l2_error = []
    for i in range(y_eval.shape[0]):
        l2_error.append(l2_relative_error(y_pred[i,...], y_eval[i,...]))
        print("L2relative error:", l2_error[i])

    # compute the L_2 relative error as a function of the number of time steps
    error_data = np.empty((args.N, 5))
    for k in range(1, args.N+1):
        y_pred_k = super.integrate(X0_npy, N=k, dyn_state_dim=4, model_restore_path=save_path)
        _, y_eval_k = scipy_integrate(power_net_dae_plot, X0, args, N=k)
        for i in range(5):
            error_data[k-1, i] = l2_relative_error(y_pred_k[i,...], y_eval_k[i,...]) 

    # plot L2 relative error for dynamic and algebraic variables
    N_vec = np.arange(1, args.N + 0.1)
    for k in range(5):
        fname_k = 'L2relative_error_' + str(k) + '.png'
        fname = os.path.join(args.log_dir, fname_k)
        plot_L2relative_error(N_vec, error_data[:, k], fname=fname, size=20, figsize=(8,6))

    # save data for future use
    np.savez(os.path.join(args.log_dir, "L2Relative_error"), N=N_vec, error=error_data)

    # regression plot for voltage
    x_line = [-.5, .5]
    y_line = [-.5, .5]
    plot_regression(y_pred[-2,...], y_eval[-2,...], fname=os.path.join(args.log_dir, 'regression-voltage.png'), size=20, figsize=(8,6), x_line=x_line, y_line=y_line)

    # saving data for future use
    np.savez(os.path.join(args.log_dir, "prediction-data"), y_pred=y_pred, y_eval=y_eval, time=t)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dae-pinns-example")

    # general
    parser.add_argument('--log-dir', type=str, default="./logs/dae-pinns-example-2/", help="log dir")
    parser.add_argument('--no-cuda', action='store_true', default=False, help="disable cuda training")
    parser.add_argument('--num-plot', type=int, default=1, help="number of ICs for plotting")

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--epochs', type=int, default=100000, help="number of epochs")
    parser.add_argument('--start-from-best', action='store_true', default=False, help='start from best model so far')
    
    # neural nets
    parser.add_argument('--dyn-weight', type=float, default=1.0, help="weight for dynamic residual loss")
    parser.add_argument('--alg-weight', type=float, default=1.0, help="weight for algebraic residual loss")

    # integration 
    parser.add_argument('--h', type=float, default=.25, help="step size")
    parser.add_argument('--N', type=int, default=20, help="number of steps")
    parser.add_argument('--method', type=str, default='BDF', help="integration method")

    args = parser.parse_args()
    main(args)
