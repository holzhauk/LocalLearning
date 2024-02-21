import os, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from LocalLearning import ModelFactory
from LocalLearning import FKHL3
from LocalLearning import StatisticGardener
from LocalLearning.Data import LpUnitCIFAR10, GaussianData, DeviceDataLoader
from LocalLearning.Statistics import cov_spectrum

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

# Unsupervised Training Hyperparameters
NO_EPOCHS = 1000
BATCH_SIZE = 1000

def plot_weight_variance_histogram(ax, model: FKHL3, 
                                   var_range=(1e-9, 3e-3), no_bins=50, clr='b', alpha=1.0,
                                  ):
    '''
    plots the variance histogram of a FKHL3 model in specified axis

    ARGS:
        ax (pyplot.axis):  pyplot axis to plot into
        model (FKHL3):  model object

    VALS:
        return (pyplot.axis):   pyplot axis that was plotted into
    '''
    with torch.no_grad():
        W = model.W.detach().cpu().clone().T
        vars_W = torch.var(W, dim=1, keepdim=True)
    counts, bins = np.histogram(vars_W.numpy(), bins=no_bins, range=var_range, density=False)
    ax.hist(bins[:-1], bins, weights=counts, color=clr, alpha=alpha)
    return ax

def NormalizeMinMax(x: torch.Tensor) -> torch.Tensor:
    mi = torch.amin(x)
    ma = torch.amax(x)
    if torch.abs(ma - mi) < 1e-8:
        return torch.zeros(x.shape)
    return (x - mi) / (ma - mi)

def plot_model_weights(ax, model: FKHL3, Kx=30, Ky=20, fig_width_in=12.9, fig_height_in=10.0, shuffle=False, x0=0, y0=0):
    with torch.no_grad():
        # extract weight matrix from model and convert to numpy array
        W = model.W.detach().cpu().clone()
        W = W.T
        
        if shuffle:  # randomly choose Kx*Ky hidden entries
            x0 = y0 = 0
            W = W[np.random.choice(W.shape[0], Kx*Ky, replace=False)]
    
    yy=y0 + Kx*x0
    HM=torch.zeros((32*Ky, 32*Kx, 3))
    for y in range(Ky):
        for x in range(Kx):
            HM[(y)*32:(y + 1)*32,x*32:(x + 1)*32, :]=NormalizeMinMax(W[yy].reshape(32, 32, 3))
            yy += 1
    
    #nc=torch.max(torch.abs(HM))
    im=ax.imshow(HM.detach().cpu().numpy(),cmap='bwr')#,vmin=-nc,vmax=nc)
    #fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    ax.axis('off')
    
    return ax

if __name__ == "__main__":
    '''
    Evaluates the distribution of variances in the weights of FKHL3.
    If the distribution exhibits a clear second mode outside the primary one,
    it is attributed to units that have not fully converged yet.
    The respective units are pruned, the pruned model is saved to the same location
    with the suffix "_pruned" attached to the file name.

    In addition, Figure 1 is created for illustration.

    ARGS:
        <modelpath> (str):  directory and file name of the FKHL3 layer model.
        <figurepath> (str): directory and file name the figure is supposed to be saved to
    '''

    if len(sys.argv) != 3:
        print("usage: python prune_and_plot_FKHL3_CIFAR.py <modelpath> <figurepath>")
        os._exit(os.EX_USAGE)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = Path(sys.argv[1])

    figure_path = Path(sys.argv[2])
    if figure_path.is_file():
        print("usage: figurepath is supposed to be a directory path, not a file!")
        os._exit(os.EX_USAGE)

    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    # load the FKHL3 model from data
    theFactory = ModelFactory()
    state_dict = torch.load(model_path)
    model = theFactory.build_from_state(state_dict)
    model.eval()
    model.to(device)

    ###############################################
    # PRUNING
    gardener = StatisticGardener()
    model_pruned, no_components, cutoff = gardener.prune(model)
    if cutoff is None:
        print(f"Only {no_components} detected, nothing to prune. Figure 1 is not generated.")
        os._exit(os.EX_OK)
    model_path_pruned = model_path.parent / Path(model_path.stem + "_pruned" + model_path.suffix)
    # save the pruned model
    torch.save(
        {
            "model_state_dict": model_pruned.state_dict(),
            "model_parameters": model_pruned.param_dict(),
            "device_type": device.type,
        },
        model_path_pruned,
    )

    ###############################################
    # GENERATE FIGURE 1
    plt.style.use(['seaborn-paper', "../stlsheets/IJoCV.mplstyle"]) # load corresponding style sheet

    # load test data sets
    cifar10Test = LpUnitCIFAR10(
        root="../data/CIFAR10",
        train=False,
        transform=ToTensor(),
        p=model.pSet["p"],
    )

    gauss_noise_params = {
        "mu": 0.0, # standard normal process
        "sigma": 1.0,
        "img_width_px": 32, # cifar10 parameters
        "img_height_px": 32,
        "img_ch_num": 3, 
    }
    GaussianNoisedSet = GaussianData(
        gauss_noise_params,
        train=False,
    )

    TestLoader = DeviceDataLoader(
            cifar10Test,
            device=device,
            batch_size=BATCH_SIZE,
            num_workers=4,
            shuffle=False,
        )
    GaussLoader = DataLoader(
            GaussianNoisedSet,
            batch_size=BATCH_SIZE,
            num_workers=4,
            shuffle=False,
        )

    # calculate spectra
    l_n_cifar10 = cov_spectrum(
        TestLoader, 
        model, 
        model.pSet["hidden_size"],
        )
    l_n_cifar10_pruned = cov_spectrum(
        TestLoader, 
        model_pruned, 
        model_pruned.pSet["hidden_size"],
        )
    l_n_gauss = cov_spectrum(
        GaussLoader, 
        model, 
        model.pSet["hidden_size"],
        )
    l_n_gauss_pruned = cov_spectrum(
        GaussLoader, 
        model_pruned, 
        model_pruned.pSet["hidden_size"],
        )

    # specify colour scheme    
    clr_map = {
        "cifar": '#01665e',
        "gauss": '#8c510a',
    }

    # plot Figure 1
    fig = plt.figure(constrained_layout=True)
    #fig.tight_layout()
    fig.set_figheight(3.3)
    width_ratios = [0.5, 0.1, 0.5, 0.05, 0.5]
    height_ratios = [1.0, 0.05, 1.0]
    gs = mpl.gridspec.GridSpec(3, 5, width_ratios=width_ratios, height_ratios=height_ratios)

    # plot the image collections
    img_ax = fig.add_subplot(gs[0, 0])
    img_ax = plot_model_weights(img_ax, model, Kx=4, Ky=4, x0=190, y0=1)
    img_ax.set_title(r"Raw")

    img_ax_pruned = fig.add_subplot(gs[2, 0])
    img_ax_pruned = plot_model_weights(img_ax_pruned, model_pruned, Kx=4, Ky=4, x0=190, y0=1)
    img_ax_pruned.set_title(r"Pruned")

    hist_ax = fig.add_subplot(gs[0, 2:])
    hist_ax = plot_weight_variance_histogram(hist_ax, model, clr=clr_map["cifar"], alpha=0.3)
    hist_ax.axvline(x = cutoff, color='#d8b365')
    hist_ax.set_xlim((0.0, 2.5e-3))
    hist_ax.set_xticks([0.0, 1e-3, 2e-3], labels=['0', '0.001', '0.002'])
    hist_ax.set_xlabel(r"Variance")
    hist_ax.set_ylabel(r"Frequency")
    hist_ax.set_title(r"Weight variance distribution")

    spec_ax_1 = fig.add_subplot(gs[2, 2])
    n = np.arange(1, len(l_n_cifar10) + 1)
    l_n_c10 = l_n_cifar10.detach().cpu().numpy().copy()
    #l_n_c10 /= l_n_c10[0]
    spec_ax_1.loglog(n, l_n_c10, linestyle=":", color=clr_map["cifar"])
    l_n_c10_pr = l_n_cifar10_pruned.detach().cpu().numpy().copy()
    #l_n_c10_pr /= l_n_c10_pr[0]
    spec_ax_1.loglog(n, l_n_c10_pr, linestyle="-", color=clr_map["cifar"])
    spec_ax_1.set_ylim((1e-6, 1.5e2))
    spec_ax_1.set_ylabel(r"$\lambda_{n}$")
    spec_ax_1.set_title(r"CIFAR10")

    spec_ax_2 = fig.add_subplot(gs[2, 4])
    n = np.arange(1, len(l_n_gauss) + 1)
    l_n_g = l_n_gauss.detach().cpu().numpy().copy()
    #l_n_g /= l_n_g.sum()
    spec_ax_2.loglog(n, l_n_g, linestyle=":", color=clr_map["gauss"])
    l_n_g_pr = l_n_gauss_pruned.detach().cpu().numpy().copy()
    #l_n_g_pr /= l_n_g.sum()
    spec_ax_2.loglog(n, l_n_g_pr, linestyle="-", color=clr_map["gauss"])
    spec_ax_2.set_ylim((1e-4, 1.5e4))
    spec_ax_2.set_title(r"$\xi(t)$")

    # legend
    lines = [
        mpl.lines.Line2D([0], [0], color="#000000", linestyle=":"),
        mpl.lines.Line2D([0], [0], color="#000000", linestyle="-"),
    ]
    spec_ax_2.legend(
        lines, 
        [r"Raw", r"Pruned"], 
        loc='upper right',
        handlelength=0.5,
    )

    fig.text(0.65, 0.04, r"Dimension $n$", ha='center')
    fig.text(0.1, 0.9, r"a)", ha='center')
    fig.text(0.35, 0.9, r"b)", ha='center')
    fig.text(0.35, 0.45, r"c)", ha='center')

    fig.savefig(figure_path / Path("FigureA1-FKHL3Spectra.eps"))

