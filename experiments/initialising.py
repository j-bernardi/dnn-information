"""Run experiments to test what parameters should be used."""
import os, sys, time
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

    # Do training - means are per cel
    out_shape, epoch_mean_loss, accuracy_mean_val, training_order =\
        ts.train(unet, trainloader, params, fake=False, 
                 experiment_folder=fn)

    print("Training complete. Saving graphs")

    # can plot epoch_mean_loss on epoch for loss change:
    plt.figure()
    plt.plot(np.arange(len(epoch_mean_loss)), epoch_mean_loss)
    plt.savefig(fn + "epoch_loss.png")
    plt.close()

    # Plot the accuracy on epoch of the validation set
    plt.figure()
    plt.plot(np.arange(len(accuracy_mean_val)), accuracy_mean_val)
    plt.savefig(fn + "epoch_accuracy.png")
    plt.close()

    # Do test
    acc = ts.test(unet, testloader, params, out_shape, classes, 
                  experiment_folder=fn, save_graph=True)

    return fn, acc

## TODOs - run train segment 1 more time, then run this (on dummy data) to check it works 
# Then try running on cluster (maybe subset first?)

if __name__ == "__main__":

    print("appending", os.sep.join(os.path.realpath(__file__).split(os.sep)[:-2] + ["models","unet"]))
    sys.path.append(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-2] + ["models","unet"]))
    import train_segment2d as ts
    import training_metadata as tm
    
    # Copy over default parameters        
    params = tm.get_params()
    #params["scan_location"] = "data/input_tensors/segmentation_data/datasets/"
    #params["torch"] = False

    print("using data from", params["scan_location"], params["torch"])

    # Time = 
    lr_bs_eps = [(0.002 , 2, 120 ),
                 (0.0005, 8, 240 ),
                 (0.001 , 4, 120 )]

    number_samples = -1 # e.g. all

    ### TEMP - for local testing ####
    train_local = False
    test_small_slurm = False
    if train_local:
        print("************************")
        print("TEMP number of samples 5")
        print("************************")
        lr_bs_eps = [(0.01, 1, 1)]#2)]
        number_samples = 5
    if test_small_slurm:
        print("*******************")
        print("TESTING SMALL SCALE")
        print("*******************")
        lr_bs_eps = [(0.01, 2, 3), (0.005, 4, 2)]
    #############
    for cln_type in ["loss", "no_clean"]:

        params["clean"] = cln_type

        print("Running clean type", cln_type)
        
        experiment_folder = "data/initialising_" + cln_type + "/"
        meta_results_file = experiment_folder + "metaresults.txt"
        running_file =  experiment_folder + "running.txt"
        
        lrs, bss, eps = [], [], [] # for tracking
        accuracies, accuracies_info = [], []
        #dummy_i = 0.555

        if not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)
        
        for tup in lr_bs_eps:
            
            lr, bs, e = tup[0], tup[1], tup[2]
            print("Running (lr, bs, e) :", tup)
            params["epochs"] = e
            params["batch_size"] = bs
            params["lr_0"] = lr

            lrs.append(lr)
            bss.append(bs)
            eps.append(eps)

            # Run the experiment
            t_start = time.time()
            filename, test_accuracy = run_experiment(params, experiment_folder, number_samples=number_samples)
            t_end = time.time()

            # Dummy version to check exp is working
            """
            dummy_i *= lr
            filename, test_accuracy = dummy(params, dummy_i)
            """

            accuracies.append(test_accuracy)
            accuracies_info.append((filename, test_accuracy, (t_end-t_start)/(60**2)))

            # Keep running order
            with open(running_file, "a+") as rf:
                rf.write(str(test_accuracy) + "," + str(filename) + "," + str((t_end-t_start)/(60**2)) + "\n")

        # Sort ordered accuracy
        ordered_accuracy = sorted(accuracies_info, key=lambda x: x[1])

        # Do the random search
        plt.figure()
        plt.plot(bss, accuracies)
        plt.savefig(experiment_folder + "accuracy_on_bs.png")
        plt.close()
        plt.figure()
        plt.plot(lrs, accuracies)
        plt.savefig(experiment_folder + "accuracy_on_lr.png")
        plt.close()

        # Print it to a file line by line
        print("Writing output")

        with open(meta_results_file, "w+") as mrf:
            mrf.write("Accuracy, Experiment, Time")
            for l in ordered_accuracy:
                mrf.write("\n" + str(l[1]) + "," + str(l[0]) + "," + str(l[2]))

    print("Successful completion")