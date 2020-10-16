import numpy as np
import matplotlib.pyplot as plt

def prediction_plotting(X_test, Y_test, Y_pred, model_name, set_name = 'Testing', fontname = "liberation sans"):

    # Creating a plot with two subplots: one for the ground truth and another for the predictions
    fig, axes = plt.subplots(figsize = (14, 7), nrows = 1, ncols = 2)

    fig.suptitle("{} - Ground Truth and Prediction ({} set)".format(model_name, set_name), fontsize = 20, fontname = fontname)

    # Colors for the different eigenvalue categories
    color = ['crimson', 'darkorange', 'yellowgreen', 'olive', 'darkolivegreen', 'lightgray']

    # Container for eigenvalues of each category
    eigs_labeled = {'1 - Stable' : [],'2 - Critical' : [], '3 - Acceptable' : [], '4 - Good' : [], '5 - Satisfactory' : [], '6 - Irrelevant' : []}
    eigs_label_pred = {'1 - Stable' : [],'2 - Critical' : [], '3 - Acceptable' : [], '4 - Good' : [], '5 - Satisfactory' : [], '6 - Irrelevant' : []}

    for n_ev, eigenvalue_label in enumerate(Y_test):
        if (eigenvalue_label == 1):
            eigs_labeled['1 - Stable'].append(X_test[n_ev,0] + 1j * X_test[n_ev,1])
        elif (eigenvalue_label == 2):
            eigs_labeled['2 - Critical'].append(X_test[n_ev,0] + 1j * X_test[n_ev,1])
        elif (eigenvalue_label == 3):
            eigs_labeled['3 - Acceptable'].append(X_test[n_ev,0] + 1j * X_test[n_ev,1])
        elif (eigenvalue_label == 4):
            eigs_labeled['4 - Good'].append(X_test[n_ev,0] + 1j * X_test[n_ev,1])
        elif (eigenvalue_label == 5):
            eigs_labeled['5 - Satisfactory'].append(X_test[n_ev,0] + 1j * X_test[n_ev,1])
        elif (eigenvalue_label == 6):
            eigs_labeled['6 - Irrelevant'].append(X_test[n_ev,0] + 1j * X_test[n_ev,1])

    for n_ev, eigenvalue_label_pred in enumerate(Y_pred):
        if (eigenvalue_label_pred == 1):
            eigs_label_pred['1 - Stable'].append(X_test[n_ev,0] + 1j * X_test[n_ev,1])
        elif (eigenvalue_label_pred == 2):
            eigs_label_pred['2 - Critical'].append(X_test[n_ev,0] + 1j * X_test[n_ev,1])
        elif (eigenvalue_label_pred == 3):
            eigs_label_pred['3 - Acceptable'].append(X_test[n_ev,0] + 1j * X_test[n_ev,1])
        elif (eigenvalue_label_pred == 4):
            eigs_label_pred['4 - Good'].append(X_test[n_ev,0] + 1j * X_test[n_ev,1])
        elif (eigenvalue_label_pred == 5):
            eigs_label_pred['5 - Satisfactory'].append(X_test[n_ev,0] + 1j * X_test[n_ev,1])
        elif (eigenvalue_label_pred == 6):
             eigs_label_pred['6 - Irrelevant'].append(X_test[n_ev,0] + 1j * X_test[n_ev,1])

    # Converting lists to numpy arrays
    for label in eigs_labeled:
        eigs_labeled[label] = np.array(eigs_labeled[label])
    for label in eigs_label_pred:
        eigs_label_pred[label] = np.array(eigs_label_pred[label])

    # Plotting the eigenvalues (ground truth)
    for n_label, label in enumerate(eigs_labeled):
        real_part = [eig.real for eig in eigs_labeled[label]]
        imag_part = [eig.imag for eig in eigs_labeled[label]]
        axes[0].scatter(real_part, imag_part, color = color[n_label], marker = 'x')

    axes[0].set_xlabel("Real Axis", fontsize = 16, fontname = fontname)
    axes[0].set_ylabel("Imaginary Axis", fontsize = 16, fontname = fontname)
    axes[0].set_title("Ground Truth (hard-coded label)", fontsize = 18, fontname = fontname)
    axes[0].legend(["Unstable", "Critical", "Acceptable", "Good", "Satisfactory"], loc = 'upper left', prop = {'family' : fontname, 'size' : 12}) 
    axes[0].axvline(x = 0, color = 'red', linestyle = '--') # Drawing stability boundary

    # Formatting ticks
    for tick in axes[0].get_xticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(13)
    for tick in axes[0].get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(13)

    # Plotting the eigenvalues (predictions)
    for n_label, label in enumerate(eigs_label_pred):
        real_part = [eig.real for eig in eigs_label_pred[label]]
        imag_part = [eig.imag for eig in eigs_label_pred[label]]
        axes[1].scatter(real_part, imag_part, color = color[n_label], marker = 'x')

    axes[1].set_xlabel("Real Axis", fontsize = 16, fontname = fontname)
    axes[1].set_ylabel("Imaginary Axis", fontsize = 16, fontname = fontname)
    axes[1].set_title("Classification Prediction (predicted label)", fontsize = 18, fontname = fontname)
    axes[1].legend(["Unstable", "Critical", "Acceptable", "Good", "Satisfactory"], loc = 'upper left', prop = {'family' : fontname, 'size' : 12}) 
    axes[1].axvline(x = 0, color = 'red', linestyle = '--') # Drawing stability boundary

    # Formatting ticks
    for tick in axes[1].get_xticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(15)
    for tick in axes[1].get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(15)

    fig.savefig("Fig_{}_{}_GroundTruth_Prediction.png".format(model_name, set_name), dpi = 150)