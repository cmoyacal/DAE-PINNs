import os
import argparse

import numpy as np
import torch
import deepxde as dde

from utils.plots import plot_depth_analysis
from utils.utils import dotdict
from models.DAEnn import three_bus_PN
from data.DAE import dae_data
from supervisor import supervisor


def main(args):
    print("starting...\n")

    # create log dir
    if os.path.isdir(args.log_dir) == False:
        os.makedirs(args.log_dir, exist_ok=True)

    # enabling gpu training
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    cuda_str = "cuda:" + str(args.gpu_number)
    device = torch.device(cuda_str if use_cuda else "cpu")
    print("using...", device)

    # list of depths
    depth = [1, 2, 3, 4, 5]
    train_depth = np.empty((len(depth),))
    test_depth = np.empty((len(depth),))

    # pinn function
    def power_net_dae(model, y_n, h, IRK_weights):
        T = 1.0

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

    # get data for the problem
    geom = dde.geometry.Hypercube([-.5, -.5, -.5, -.5],[.5, .5, .5, .5])

    def alg_output_feature_layer(x):
        return torch.nn.functional.softplus(x)

    for k in range(len(depth)):
        X_train = geom.random_points(args.num_train)
        X_test = geom.random_points(args.num_test)
        data = dae_data(X_train, X_test, args, device=device, func=power_net_dae)

        # construct the neural nets
        dynamic = dotdict()
        dynamic.num_IRK_stages = args.num_IRK_stages
        dynamic.state_dim = 4

        dynamic.activation = args.dyn_activation
        dynamic.initializer = "Glorot normal"
        dynamic.dropout_rate = 0.0
        dynamic.type = args.dyn_type
        if args.unstacked:
        # if NNs for dynamic states are unstaked
            dim_out = dynamic.state_dim * (dynamic.num_IRK_stages + 1)
        else:
            dim_out = dynamic.num_IRK_stages + 1

            dynamic.layer_size = [dynamic.state_dim] + [args.dyn_width] * depth[k] + [dim_out]

        algebraic = dotdict()
        algebraic.num_IRK_stages = args.num_IRK_stages
        dim_out_alg = algebraic.num_IRK_stages + 1
        algebraic.layer_size = [dynamic.state_dim] + [args.alg_width] * depth[k] + [dim_out_alg]
        algebraic.activation = args.alg_activation
        algebraic.initializer = "Glorot normal"
        algebraic.dropout_rate = 0.0
        algebraic.type = args.alg_type

        nn = three_bus_PN(
            dynamic, 
            algebraic, 
            alg_out_transform=alg_output_feature_layer,
            stacked=not args.unstacked, 
        ).to(device)

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

        _, state = super.train(
            epochs=args.epochs,
            batch_size=args.batch_size, 
            test_every=args.test_every,
            num_val=args.num_val,
            use_tqdm=args.use_tqdm,
        )

        train_depth[k] = state.best_loss_train
        test_depth[k] = state.best_loss_test

    # plot results
    print("plotting results...\n")
    plot_depth_analysis(depth, train_depth, test_depth, fname=os.path.join(args.log_dir,'depth-analysis.png'))

    # save the results for future use
    np.savez(os.path.join(args.log_dir, "depth-data"), depth=depth, train=train_depth, test=test_depth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="depth-analysis")

    # general
    parser.add_argument('--num-IRK-stages', type=int, default=100, help="number of RK stages")
    parser.add_argument('--log-dir', type=str, default="./logs/dae-pinns-depth-analysis/", help="log dir")
    parser.add_argument('--no-cuda', action='store_true', default=False, help="disable cuda training")
    parser.add_argument('--gpu-number', type=int, default=0, help="GPU device number")
    parser.add_argument('--num-train', type=int, default=500, help="number of training examples")
    parser.add_argument('--num-val', type=int, default=100, help="number of validation examples")
    parser.add_argument('--num-test', type=int, default=500, help="number of test examples")

    # scheduler
    parser.add_argument('--use-scheduler', action='store_true', default=False, help='use lr scheduler')
    parser.add_argument('--scheduler-type', type=str, default="plateau", help="scheduler type")
    parser.add_argument('--patience', type=int, default=2500, help="patience for scheduler")
    parser.add_argument('--factor', type=float, default=.8, help="factor for scheduler")

    # optimizer
    parser.add_argument('--use-tqdm', action='store_true', default=False, help="disable tqdm for training")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--epochs', type=int, default=100000, help="number of epochs")
    parser.add_argument('--batch-size', type=int, default=100, help="batch size")
    parser.add_argument('--test-every', type=int, default=1000, help="test and log every * steps")

    # NNs
    parser.add_argument('--dyn-type', type=str, default="attention", help="type of dyn-vars net \in {fnn, attention, Conv1D}")
    parser.add_argument('--unstacked', action='store_true', default=False, help="use unstaked neural nets for the dynamic variables")
    parser.add_argument('--dyn-activation', type=str, default="sin", help="dynamic variables activation function")
    parser.add_argument('--dyn-weight', type=float, default=1.0, help="weight for dynamic residual loss")
    parser.add_argument('--dyn-width', type=int, default=100, help="width of hidden layers - dynamic variables")

    parser.add_argument('--alg-type', type=str, default="attention", help="type of alg-vars net \in {fnn, attention, Conv1D}")
    parser.add_argument('--alg-activation', type=str, default="sin", help="algebraic variables activation function")
    parser.add_argument('--alg-weight', type=float, default=1.0, help="weight for algebraic residual loss")
    parser.add_argument('--alg-width', type=int, default=100, help="width of hidden layers - algebraic variables")

    # integration 
    parser.add_argument('--h', type=float, default=.25, help="step size")

    args = parser.parse_args()
    main(args)

