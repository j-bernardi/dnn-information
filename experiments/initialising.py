"""Run experiments to test what parameters should be used."""
import os, sys
import matplotlib.pyplot as plt
import numpy as np

def dummy(params, i):
    """Pretend run"""

    return tm.construct_file(params), i

def run_experiment(params, experiment_folder, number_samples=-1, save_reps=False):

    # Get save file name for this given experiment
    fn = tm.construct_file(params, experiment_folder)

    ## LOAD DATA ##
    if params["torch"]:
        trainloader, testloader, classes = ts.load_torch_data(params, number_samples=number_samples)
    else:
        trainloader, testloader, train_id, test_id, classes = ts.load_h5_data(params, number_samples=number_samples)

    # LOAD MODEL with saving reps to folder#
    unet = ts.load_model(params, experiment_folder=fn, save_reps=save_reps)

    # Do training
    shape, numel, epoch_mean_loss, accuracy_mean_val, training_order =\
        ts.train(unet, trainloader, params, fake=False, 
                 experiment_folder=fn, torch_data=params["torch"])

    print("Training complete. Saving graphs")

    # can plot epoch_mean_loss on epoch for loss change:
    plt.plot(np.arange(len(epoch_mean_loss)), epoch_mean_loss)
    plt.savefig(fn + "epoch_loss.png")
    plt.clf()
    
    # Plot the accuracy on epoch of the validation set
    plt.plot(np.arange(len(accuracy_mean_val)), accuracy_mean_val)
    plt.savefig(fn + "epoch_accuracy.png")
    plt.clf()

    # Do test - TODO - implement graph printing here
    acc = ts.test(unet, testloader, params, shape, numel, classes, 
                  experiment_folder=fn, torch_data=params["torch"], save_graph=True)

    return fn, acc

## TODOs - run train segment 1 more time, then run this (on dummy data) to check it works 
# Then try running on cluster (maybe subset first?)

if __name__ == "__main__":

    print("appending", os.sep.join(os.path.realpath(__file__).split(os.sep)[:-2] + ["models","unet"]))
    sys.path.append(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-2] + ["models","unet"]))
    import train_segment2d as ts
    import training_metadata as tm

    meta_results_file = "data/initialising/metaresults.txt"
    experiment_folder = "data/initialising/"
    
    # Copy over default parameters        
    params = tm.get_params()
    print("using data from", params["scan_location"], params["torch"])
    #params["scan_location"] = "data/input_tensors/segmentation_data/datasets/"
    #params["torch"] = False

    #dummy_i = 0.555
    
    ordered_accuracy = []

    lr_bs_eps = [(0.01, 4, 12), (0.01, 8, 12), (0.01, 16, 12), 
                                (0.005, 8, 60),
                                (0.001, 8, 120)               ]

    number_samples = -1 # e.g. all

    ### TEMP - for local testing ####
    do_this = False
    if do_this:
        print("************************")
        print("TEMP number of samples 5")
        print("************************")
        lr_bs_eps = [(0.01, 1, 3)]
        number_samples = 5
    #############

    for tup in lr_bs_eps:
        
        lr, bs, e = tup[0], tup[1], tup[2]

        params["epochs"] = e
        params["batch_size"] = bs
        params["lr_0"] = lr

        # Run the experiment
        filename, test_accuracy = run_experiment(params, experiment_folder, number_samples=number_samples)
        
        # Dummy version to check exp is working
        """
        dummy_i *= lr
        filename, test_accuracy = dummy(params, dummy_i)
        """

        ordered_accuracy.append((filename, test_accuracy))

    # Sort ordered accuracy
    ordered_accuracy = sorted(ordered_accuracy, key=lambda x: x[1])

    # Print it to a file line by line
    print("Writing output")

    with open(meta_results_file, "w+") as mrf:
        mrf.write("Accuracy, Experiment")
        for l in ordered_accuracy:
            mrf.write("\n" + str(l[1]) + "," + str(l[0]))

    print("Successful completion")