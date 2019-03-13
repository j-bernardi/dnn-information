"""Run experiments to test what parameters should be used."""
import os, sys

lr_0s = [0.001]
batch_sizes = [4]
epochs = [120]
#lr_0s = [0.01, 0.001, 0.0001]
#batch_sizes = [4, 8, 16]
#epochs = [80, 120, 160]


def dummy(params, i):
    return tm.construct_file(params), i

def run_experiment(params, experiment_folder):

    # Get save file name
    fn = tm.construct_file(params, experiment_folder)

    ## LOAD DATA ##
    trainloader, testloader, classes = ts.load_data(params)

    # LOAD MODEL with saving reps to folder#
    unet = ts.load_model(params, experiment_folder=fn)

    # Do training
    shape, numel, epoch_mean_loss, accuracy_mean_val, training_order =\
        ts.train(unet, trainloader, params, fake=False, experiment_folder=fn)

    print("Training complete")

    # Do test - TODO - implement graph printing here
    acc = ts.test(unet, testloader, params, shape, numel, classes, experiment_folder=fn)

    return fn, acc

ordered_accuracy = []

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
    params = tm.params
    #dummy_i = 0.555

    for lr in lr_0s:
        for bs in batch_sizes:
            for e in epochs:

                # Update parameters with above
                params["epochs"] = e
                params["batch_size"] = bs
                params["lr_0"] = lr

                # Run the experiment
                filename, test_accuracy = run_experiment(params, experiment_folder)
                
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