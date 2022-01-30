
import dataset_utilities as d_utils
import training_utilities as t_utils

import matplotlib.pyplot as plt
import numpy as np

def svg_dataset_analyze():
    _, keys1, _, vals1 = d_utils.analyze_dataset_disparity_coverage(use_file="sceneflow_analyze_precalculated.npy")
    _, keys2, _, vals2 = d_utils.analyze_dataset_disparity_coverage(compute_zeros_seperately=True,
                                                 use_file="kitti_analyze_precalculated.npy")
    #plt.figure(figsize=(15,5))
    plt.figure()
    plt.plot(keys1, vals1, label="Scene Flow")
    plt.plot(keys2, vals2, label="KITTI 2015")
    plt.xlim([-10, 256])
    plt.ylabel("Coverage")
    plt.xlabel("Max Disparity")
    plt.xticks(np.arange(16+1)*16)
    plt.yticks(np.arange(20+1)/20)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/dataset_analyze.svg")