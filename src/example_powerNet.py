import os
import argparse

import torch
import numpy as np
import deepxde as dde
from scipy.integrate import solve_ivp
from matplotlib.pyplot import plot

from utils.utils import dotdict
from models.DAEnn import three_bus_PN
from data.DAE import dae_data
from supervisor import supervisor
import events
from utils.plots import plot_loss_history, plot_three_bus, plot_regression, plot_L2relative_error
from metrics import l2_relative_error

def scipy_integrate(func, X0, args, IRK_times, N=0):
    """
    integrates stiff power network dynamics using scipy for N time steps of size args.h 
    """
    V0 = 0.7   # we fix the volatge initial condition
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


def main(args):
    print("starting...\n")

    # create log dir
    if os.path.isdir(args.log_dir) == False:
        os.makedirs(args.log_dir, exist_ok=True)

    # enabling gpu training
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    #cuda_str = "cuda:" + str(args.gpu_number)
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.cuda.set_device(1)
    print(torch.cuda.current_device())
    print("using...", device)
    print(torch.cuda.current_device())

    # construct the neural nets
    dynamic = dotdict()
    dynamic.num_IRK_stages = args.num_IRK_stages
    dynamic.state_dim = 4
    def dyn_input_feature_layer(x):
        return torch.cat((x,torch.cos(np.pi * x), torch.sin(np.pi * x), torch.cos(2 * np.pi * x), torch.sin(2 * np.pi * x)), dim=-1)

    def alg_output_feature_layer(x):
        return torch.nn.functional.softplus(x)

    dynamic.activation = args.dyn_activation
    dynamic.initializer = "Glorot normal"
    dynamic.dropout_rate = args.dropout_rate
    dynamic.batch_normalization = None if args.dyn_bn == "no-bn" else args.dyn_bn
    dynamic.layer_normalization = None if args.dyn_ln == "no-ln" else args.dyn_ln
    dynamic.type = args.dyn_type

    if args.unstacked:
        dim_out = dynamic.state_dim * (dynamic.num_IRK_stages + 1)
    else:
        dim_out = dynamic.num_IRK_stages + 1
    
    if args.use_input_layer:
        dynamic.layer_size = [dynamic.state_dim * 5] + [args.dyn_width] * args.dyn_depth + [dim_out]
    else:
        dynamic.layer_size = [dynamic.state_dim] + [args.dyn_width] * args.dyn_depth + [dim_out]
        dyn_input_feature_layer = None

    algebraic = dotdict()
    algebraic.num_IRK_stages = args.num_IRK_stages
    dim_out_alg = algebraic.num_IRK_stages + 1
    algebraic.layer_size = [dynamic.state_dim] + [args.alg_width] * args.alg_depth + [dim_out_alg]
    algebraic.activation = args.alg_activation
    algebraic.initializer = "Glorot normal"
    algebraic.dropout_rate = args.dropout_rate
    algebraic.batch_normalization = None if args.alg_bn == "no-bn" else args.alg_bn
    algebraic.layer_normalization = None if args.alg_ln == "no-ln" else args.alg_ln
    algebraic.type = args.alg_type        

    nn = three_bus_PN(
        dynamic, 
        algebraic, 
        dyn_in_transform=dyn_input_feature_layer, 
        alg_out_transform=alg_output_feature_layer,
        stacked=not args.unstacked, 
        ).to(device)

    # Data for training

    # pinn function
    def power_net_dae(model, y_n, h, IRK_weights):
        T = 1.0

        # parameters
        M_1, M_2, D, D_d, b = .052, .0531, .05, .005, 10.
        V_1, V_2, P_g, P_l, Q_l = 1.02, 1.05, -2.0, 3.0, .1

        # pinn
        yn = y_n.clone()
        # TO DO: fourier and exponential features

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

    # get data for the problem
    geom = dde.geometry.Hypercube([-.5, -.5, -.5, -.5],[.5, .5, .5, .5])
    np.random.seed(1234)
    X_train = geom.random_points(args.num_train)
    np.random.seed(3456)
    X_test = geom.random_points(args.num_test)
    data = dae_data(X_train, X_test, args, device=device, func=power_net_dae)

    # start the supervisor
    super = supervisor(data, nn, device=device)
    
    # compile the supervisor
    optimizer = torch.optim.Adam(nn.parameters(), lr=args.lr)
    if args.use_scheduler:
        if args.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                'min',
                patience=args.patience,
                verbose=True,
                factor=args.factor,
            )
        elif args.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.patience, gamma=args.factor
            )
        else:
            scheduler = None
    else:
        scheduler = None
    
    super.compile(
        optimizer,
        loss_weights=[args.dyn_weight, args.alg_weight],
        scheduler=scheduler,
        scheduler_type= args.scheduler_type,
    )

    # train the DAE surrogate model
    model_name = 'model.pth' if args.model_name == 'no-name' else ('model_' + args.model_name + '.pth')
    save_path = os.path.join(args.log_dir, model_name)
    chcker = events.ModelCheckPoint(save_path, save_better_only=True, every=1000) 
    restore_path = save_path if args.start_from_best else None

    loss_history, state = super.train(
        epochs=args.epochs,
        batch_size=args.batch_size, 
        test_every=args.test_every,
        num_val=args.num_val,
        events=[chcker],
        model_restore_path=restore_path,
        use_tqdm=args.use_tqdm,
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
    t, y_eval = scipy_integrate(power_net_dae_plot, X0, args, super.data.IRK_times, N=args.N)
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
        _, y_eval_k = scipy_integrate(power_net_dae_plot, X0, args, super.data.IRK_times, N=k)
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
    parser.add_argument('--num-IRK-stages', type=int, default=100, help="number of RK stages")
    parser.add_argument('--log-dir', type=str, default="./logs/dae-pinns-example-2/", help="log dir")
    parser.add_argument('--no-cuda', action='store_true', default=False, help="disable cuda training")
    parser.add_argument('--gpu-number', type=int, default=0, help="GPU device number")
    parser.add_argument('--num-train', type=int, default=1000, help="number of training examples")
    parser.add_argument('--num-val', type=int, default=200, help="number of validation examples")
    parser.add_argument('--num-test', type=int, default=400, help="number of test examples")
    parser.add_argument('--num-plot', type=int, default=1, help="number of ICs for plotting")

    # scheduler
    parser.add_argument('--use-scheduler', action='store_true', default=False, help='use lr scheduler')
    parser.add_argument('--scheduler-type', type=str, default="plateau", help="scheduler type")
    parser.add_argument('--patience', type=int, default=3000, help="patience for scheduler")
    parser.add_argument('--factor', type=float, default=.8, help="factor for scheduler")

    # optimizer
    parser.add_argument('--use-tqdm', action='store_true', default=False, help="disable tqdm for training")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--epochs', type=int, default=100000, help="number of epochs")
    parser.add_argument('--batch-size', type=int, default=100, help="batch size")
    parser.add_argument('--test-every', type=int, default=1000, help="test and log every * steps")
    parser.add_argument('--start-from-best', action='store_true', default=False, help='start from best model so far')
    parser.add_argument('--model-name', type=str, default="no-name", help="model_ + model-name + .pth")

    # neural nets
    parser.add_argument('--dropout-rate', type=float, default=0.0, help="dropout rate")
    
    parser.add_argument('--dyn-bn', type=str, default="no-bn", help="enable dyn-vars batch normalization \in {before, after} activation")
    parser.add_argument('--dyn-ln', type=str, default="no-ln", help="enable dyn-vars layer normalization \in {before, after} activation")
    parser.add_argument('--dyn-type', type=str, default="attention", help="type of dyn-vars net \in {fnn, attention, Conv1D}")
    parser.add_argument('--unstacked', action='store_true', default=False, help="use unstaked neural nets for the dynamic variables")
    parser.add_argument('--use-input-layer', action='store_true', default=False, help="use input feature layer for the dynamic variables")
    parser.add_argument('--dyn-width', type=int, default=100, help="width of hidden layers - dynamic variables")
    parser.add_argument('--dyn-depth', type=int, default=5, help="depth of hidden layers - dynamic variables")
    parser.add_argument('--dyn-activation', type=str, default="sin", help="dynamic variables activation function")
    parser.add_argument('--dyn-weight', type=float, default=1.0, help="weight for dynamic residual loss")

    parser.add_argument('--alg-bn', type=str, default="no-bn", help="enable alg-vars batch normalization \in {before, after} activation")
    parser.add_argument('--alg-ln', type=str, default="no-ln", help="enable alg-vars layer normalization \in {before, after} activation")
    parser.add_argument('--alg-type', type=str, default="attention", help="type of alg-vars net \in {fnn, attention, Conv1D}")
    parser.add_argument('--alg-width', type=int, default=40, help="width of hidden layers - algebraic variables")
    parser.add_argument('--alg-depth', type=int, default=2, help="depth of hidden layers - algebraic variables")
    parser.add_argument('--alg-activation', type=str, default="sin", help="algebraic variables activation function")
    parser.add_argument('--alg-weight', type=float, default=1.0, help="weight for algebraic residual loss")

    # integration 
    parser.add_argument('--h', type=float, default=.25, help="step size")
    parser.add_argument('--N', type=int, default=20, help="number of steps")
    parser.add_argument('--method', type=str, default='BDF', help="integration method")

    args = parser.parse_args()
    main(args)
