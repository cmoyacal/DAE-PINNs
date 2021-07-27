import matplotlib.pyplot as plt
from matplotlib import rc

import numpy as np

# Set the global font and size 
rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans'],'size':25})
# Set the font used for math
rc('mathtext',**{'default':'regular'})

def stylize_axes(ax, size=25, legend=True, xlabel=None, ylabel=None, title=None, xticks=None, yticks=None, xticklabels=None, yticklabels=None, top_spine=True, right_spine=True):
    """
    stylizes the axes of our plots.
    """
    ax.spines['top'].set_visible(top_spine)
    ax.spines['right'].set_visible(right_spine)

    ax.xaxis.set_tick_params(top='off', direction='out', width=1)
    ax.yaxis.set_tick_params(right='off', direction='out', width=1)
    
    if title is not None:
        ax.set_title(title)
    
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)
    
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    if legend:
        ax.legend(fontsize=size)
    return ax


def custom_logplot(ax, x, y, label="loss", xlims=None, ylims=None, color='red', linestyle='solid', marker=None):
    """
    Customized plot with log scale on y axis.
    """
    if marker is None:
        ax.semilogy(x, y, color=color, label=label, linestyle=linestyle)
    else:
        ax.semilogy(x, y, color=color, label=label, linestyle=linestyle, marker=marker)
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    return ax

def custom_scatterplot(ax, x, y, xlims=None, ylims=None, error=1.0, color='green', markerscale=10):
    """
    Customized scatter plot where marker size is proportional to error measure.
    """ 
    markersize = error * markerscale
    ax.scatter(x, y, color=color, marker='o', s=markersize, alpha=0.5)
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    return ax

def custom_lineplot(ax, x, y, label=None, xlims=None, ylims=None, color="red", linestyle="solid", linewidth=2.0, marker=None):
    """
    Customized line plot.
    """
    if label is not None:
        if marker is None:
            ax.plot(x, y, color=color, label=label, linestyle=linestyle, linewidth=linewidth)
        else:
            ax.plot(x, y, color=color, label=label, linestyle=linestyle, linewidth=linewidth, marker=marker)
    else:
        if marker is None:
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth)
        else:
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, marker=marker)
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)  
    return ax   

def custom_barchart(ax, x, y, error, xlims=None, ylims=None, color='blue', width=1.0, label=None):
    """
    Customized bar chart with positive error bars only.
    """
    error_kw = {'capsize': 5, 'capthick': 1, 'ecolor': 'black'}
    error = [np.zeros(len(error)), error]
    
    ax.bar(x, y, color=color, width=width, yerr=error, error_kw=error_kw, align='center', label=label)
    
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    
    return ax

def custom_loglogplot(ax, x, y, label="loss", xlims=None, ylims=None, color='red', linestyle='solid', marker=None):
    """
    Customized plot with log scale on both axis
    """
    if marker is None:
        ax.loglog(x, y, color=color, label=label, linestyle=linestyle)
    else:
        ax.loglog(x, y, color=color, label=label, linestyle=linestyle, marker=marker)
    
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    return ax

def plot_loss_history(loss_history, fname="./logs/loss.png", size=25, figsize=(8,6)):
    """
    plots the loss history.
    """
    loss_train = np.array(loss_history.loss_train)
    loss_test = np.array(loss_history.loss_test)

    fig, ax = plt.subplots(figsize=figsize)
    custom_logplot(ax, loss_history.steps, loss_train, label="Train loss", color='blue', linestyle='solid')
    custom_logplot(ax, loss_history.steps, loss_test, label="Train loss", linestyle='dashed')
    stylize_axes(ax, size=size, xlabel="No. of iterations", ylabel="M.s.e.")

    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight', transparent=True)

def plot_three_bus(t, y_eval, y_pred, fname="./logs/test-trajectory.png", size=25, figsize=(8,6)):
    """
    plots the exact and predicted power network trajectories.
    """
    t = t.reshape(-1,)
    ylims = [None, None, None, None, (0.2, 1.2)]
    xlabel = [None, None, None, None, 'time (s)']
    ylabel = ['$\omega_1(t)$', '$\omega_2(t)$', '$\delta_2(t)$', '$\delta_3(t)$', '$V_3(t)$']
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=figsize)
    for i in range(5):
        custom_lineplot(ax[i], t, y_eval[i,...].reshape(-1,), label="Exact", ylims=ylims[i])
        custom_lineplot(ax[i], t, y_pred[i,...].reshape(-1,), color="blue", linestyle="dashed", label="Predicted", ylims=ylims[i])
        stylize_axes(ax[i], size=size, xlabel=xlabel[i], ylabel=ylabel[i])
    fig.savefig(fname, dpi=300, bbox_inches='tight', transparent=True)

def plot_regression(predicted, y, fname="./log/regression.png", size=20, figsize=(8,6), x_line=None, y_line=None):
    """
    Parity plot.
    """
    predicted = predicted.reshape(-1,)
    y = y.reshape(-1,)
    if x_line is None:
        x_line = [y.min(), y.max()]
        y_line = [y.min(), y.max()]

    fig, ax = plt.subplots(figsize=figsize)
    custom_lineplot(ax, x_line, y_line, color="yellow", linestyle="dashed", linewidth=3.0)
    custom_scatterplot(ax, predicted, y, color='blue', markerscale=10)
    stylize_axes(ax, size=size, xlabel="Predicted", ylabel="Exact", legend=False)
    
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight', transparent=True)

def plot_barchart(train, test, fname="./log/regression.png", size=20, figsize=(8,6)):
    """
    Plots a a bar chart.
    """
    # train \in [num, 2]
    # test \in [num, 2]

    train = train.reshape(-1,2)
    test = test.reshape(-1,2)

    mean_train = train.mean(axis=0)
    mean_test = test.mean(axis=0)
    error_train = train.std(axis=0)
    error_test = test.std(axis=0)
    width = .25

    x = np.arange(len(mean_train))
    fig, ax = plt.subplots(figsize=figsize)
    custom_barchart(ax, x, mean_train, error_train, color='blue', width=width, label='Train')
    custom_barchart(ax, x + width, mean_test, error_test, color='red', width=width, label='Test')
    xticks = x + width/2
    xticklabels = ['Stacked', 'Unstacked']
    stylize_axes(ax, size=size, ylabel="M.s.e.", xticks=xticks, xticklabels=xticklabels, legend=True)
    
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight', transparent=True)



def plot_width_analysis(width, train, test, fname="./log/width_analysis.png", size=20, figsize=(8,6)):
    """
    Plots losses as a function of the nn width.
    """
    train = train.reshape(-1,)
    test = test.reshape(-1,)

    fig, ax = plt.subplots(figsize=figsize)
    custom_loglogplot(ax, width, train, linestyle='dashed', marker="s", color='red', label='Train')
    custom_loglogplot(ax, width, test, linestyle='dashed', marker="o", color='blue', label='Test')
    stylize_axes(ax, size=size, xlabel="Width", ylabel="M.s.e")
    
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight', transparent=True)

def plot_depth_analysis(depth, train, test, fname="./log/width_analysis.png", size=20, figsize=(8,6)):
    """
    Plots losses as a function of the nn depth.
    """
    train = train.reshape(-1,)
    test = test.reshape(-1,)

    fig, ax = plt.subplots(figsize=figsize)
    custom_logplot(ax, depth, train, linestyle='dashed', marker="s", color='red', label='Train')
    custom_logplot(ax, depth, test, linestyle='dashed', marker="o", color='blue', label='Test')
    stylize_axes(ax, size=size, xlabel="Depth", ylabel="M.s.e")
    
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight', transparent=True)

def plot_num_train_analysis(num_train, train, test, fname="./log/num_train.png", size=20, figsize=(8,6)):
    """
    Plots losses as a function of the number of training examples.
    """
    train = train.reshape(-1,)
    test = test.reshape(-1,)

    fig, ax = plt.subplots(figsize=figsize)
    custom_loglogplot(ax, num_train, train, linestyle='dashed', marker="s", color='red', label='Train')
    custom_loglogplot(ax, num_train, test, linestyle='dashed', marker="o", color='blue', label='Test')
    stylize_axes(ax, size=size, xlabel="No. of training examples", ylabel="M.s.e")
    
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight', transparent=True)

def plot_L2relative_error(N, error, fname="./log/num_train.png", size=20, figsize=(8,6)):
    """
    Plots the L_2-relative error as a function of the number of integration steps.
    """
    error = error.reshape(-1,)

    fig, ax = plt.subplots(figsize=figsize)
    custom_lineplot(ax, N, error, color="blue", linestyle="dashed", linewidth=3.0, marker='s')
    stylize_axes(ax, size=size, xlabel="No. of time steps $N$", ylabel="L$_2$-relative error", legend=False)
    
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight', transparent=True)