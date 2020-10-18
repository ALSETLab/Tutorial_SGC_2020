import os
import numpy as np
import pickle

def create_ground_truth(df_eigenvalues, current_wd, df_name):
    
    # Path of the target file for storing the ground truth data
    path_output = os.path.join(current_wd, "_preproc_data/{}_ground_truth_data.pkl".format(df_name))

    if not os.path.exists(path_output):
        
        # Number of scenarios
        n_scenarios = df_eigenvalues.shape[1]
        # Number of eigenvalues per scenario
        n_eigs = df_eigenvalues.shape[0]

        # Name of each scenario
        scenarios = list(df_eigenvalues.columns)

        # Creating label container
        label = np.empty((n_eigs, n_scenarios), dtype = object)
        # Creating damping ratio container
        damping_ratio = np.zeros((n_eigs, n_scenarios))
        # Creating tag (i.e., category) container
        tag_label = np.zeros((n_eigs, n_scenarios))

        for n_sc, sc in enumerate(scenarios):
            for n_ev in range(0, n_eigs):
                # Classification between real and complex conjugate eigenvalues
                if (np.imag(df_eigenvalues[sc][n_ev]) == 0):
                    label[n_ev][n_sc] = "Real,"
                else:
                    if (np.real(df_eigenvalues[sc][n_ev]) == 0):
                        label[n_ev][n_sc] = "Pure imaginary,"
                    else:
                        label[n_ev][n_sc] = "Complex conjugate," 

                # Classification between stable and unstable eigenvalues
                if (np.real(df_eigenvalues[sc][n_ev]) > 0):
                    label[n_ev][n_sc] = label[n_ev][n_sc] + " unstable"
                else:
                    label[n_ev][n_sc] = label[n_ev][n_sc] + " stable"

                # Computation of damping ratio
                if (label[n_ev][n_sc] == "Complex conjugate, unstable" or label[n_ev][n_sc] == "Complex conjugate, stable"):
                    damping_ratio[n_ev][n_sc] = -np.real(df_eigenvalues[sc][n_ev]) / np.sqrt(np.square(np.real(df_eigenvalues[sc][n_ev])) + np.square(np.imag(df_eigenvalues[sc][n_ev])))
                else:
                    if (label[n_ev][n_sc] == "Real, stable"):
                        damping_ratio[n_ev][n_sc] = 1.1
                    else:
                        damping_ratio[n_ev][n_sc] = -1.1

                # Creating labels based on the damping ratio
                if (damping_ratio[n_ev][n_sc] < 0):
                    tag_label[n_ev][n_sc] = 1 # 1 - Unstable operation
                elif (damping_ratio[n_ev][n_sc] < 0.05):
                    tag_label[n_ev][n_sc] = 2 # 2 - Stable but critical operation
                elif (damping_ratio[n_ev][n_sc] >= 0.05 and damping_ratio[n_ev][n_sc] < 0.1):
                    tag_label[n_ev][n_sc] = 3 # 3 - Acceptable operation
                elif (damping_ratio[n_ev][n_sc] >= 0.1 and damping_ratio[n_ev][n_sc] < 1.0):
                    tag_label[n_ev][n_sc] = 4 # Good operation
                elif (np.abs(np.real(df_eigenvalues[sc][n_ev])) < 0.5 and damping_ratio[n_ev][n_sc] >= 1.0):
                    if np.abs(np.real(df_eigenvalues[sc][n_ev])) < 0.01:
                        tag_label[n_ev][n_sc] = 6 # Not relevant
                    else:
                        tag_label[n_ev][n_sc] = 2 # Critical operation; real eigenvalue close to the imaginary axis
                else:
                    tag_label[n_ev][n_sc] = 5 # Satisfactory operation

        # Saving ground truth data of eigenvalues    
        ground_truth_data = {'text_label' : label,
                          'damping_ratio' : damping_ratio,
                          'tag_label' : tag_label}

        with open(path_output, 'wb') as f:
            pickle.dump(ground_truth_data, f, pickle.HIGHEST_PROTOCOL)
        print("Ground truth data saved!\n")
        
        print("{} Eigenvalues (shape): {} (eigs) x {} (scenarios)".format(df_name, df_eigenvalues.shape[0], df_eigenvalues.shape[1]))
        print("{} Text labels (shape): {} (eigs) x {} (scenarios)".format(df_name, label.shape[0], label.shape[1]))
        print("{} Damping ratio (shape): {} (eigs) x {} (scenarios)".format(df_name, damping_ratio.shape[0], damping_ratio.shape[1]))
        print("{} Tag label (shape): {} (eigs) x {} (scenarios)\n".format(df_name, tag_label.shape[0], tag_label.shape[1]))
    else:
        print("Ground truth already exists!\n")