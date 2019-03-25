"""Run experiments to test what parameters should be used."""
import os, sys, time, torch, datetime, pickle
import matplotlib.pyplot as plt
import numpy as np

def initialise_model(params):
    """If no model exists, create one and save it for all."""

    if torch.cuda.device_count() > 1:
        
        # Multi gpu - check if multi file exists
        if os.path.isfile('models/unet/saved_models/initialisation_multi.pth'):
            pass # Will load this later
        
        else:
            print("Making new model (multi gpu)") # Make the model and save it
            model = ts.load_model(params, experiment_folder="no", save_reps=False)
            torch.save(model.module.state_dict(), "models/unet/saved_models/initialisation_multi.pth")
    else:

        # single gpu - check if initialisiation exists
        if os.path.isfile('models/unet/saved_models/initialisation.pth'):
            pass  # Will load later
        else:
            print("Making new model")
            model = ts.load_model(params, experiment_folder="no", save_reps=False)
            torch.save(model.state_dict(), "models/unet/saved_models/initialisation.pth")

def define_experiment(test_small_slurm=False):

    # DEFINE
    cln_type = "loss"
    
    smoothing_types = ["none", "uniform_fixed_eps", "uniform_vary_eps", "weighted_fixed_eps", "weighted_vary_eps"]
    
    lr_bs_eps = (0.001, 4, 240)

    number_samples = -1 # e.g. all

    # For local testing:
    train_local = (torch.cuda.device_count() <= 1)
    
    if train_local:
        
        print("************************")
        print("TEMP number of samples 5")
        print("************************")
        
        lr_bs_eps = (0.02, 1, 1)
        number_samples = 5
    
    # If doing small slurm
    if test_small_slurm and torch.cuda.device_count() > 1:
        
        print("*******************")
        print("TESTING SMALL SCALE")
        print("*******************")

        lr_bs_eps = (0.005, 4, 2)

    return lr_bs_eps, number_samples, cln_type, smoothing_types

def load_fresh_model(params, experiment_folder="no", save_reps=False, save_model=False):

    # Initialise model
    unet = ts.load_model(params, experiment_folder="no", save_reps=False)

    if torch.cuda.device_count() > 1 and save_model:
        unet.module.load_state_dict(torch.load("models/unet/saved_models/initialisation_multi.pth"))
    elif save_model:
        unet.load_state_dict(torch.load("models/unet/saved_models/initialisation.pth"))

    return unet

def run_experiment(unet, params, trainloader, testloader, classes, experiment_folder, number_samples=-1, save_reps=False):

    # Get save file name for this given experiment
    fn = tm.construct_file(params, experiment_folder)

    # Do training - means are per cel
    out_shape, epoch_mean_loss, accuracy_mean_val, central_accuracy_mean_val, training_order =\
        ts.train(unet, trainloader, params, fake=False, 
                 experiment_folder=fn)

    print("\nTraining complete. Saving graphs")

    # can plot epoch_mean_loss on epoch for loss change:
    plt.figure()
    np.array(epoch_mean_loss).dump(fn + "epoch_loss.pkl")
    plt.plot(np.arange(len(epoch_mean_loss)), epoch_mean_loss)
    plt.title("Loss on epoch")
    plt.ylabel("Loss / pixel")
    plt.xlabel("Epochs")
    plt.savefig(fn + "epoch_loss.png")
    plt.close()

    # Plot the accuracy on epoch of the validation set
    plt.figure()
    np.array(accuracy_mean_val).dump(fn + "epoch_accuracy.pkl")
    plt.plot(np.arange(len(accuracy_mean_val)), accuracy_mean_val)
    plt.title("Accuracy on epoch")
    plt.ylabel("Accuracy %")
    plt.xlabel("Epochs")
    plt.savefig(fn + "epoch_accuracy.png")
    plt.close()

    # Plot the central on epoch of the validation set
    plt.figure()
    np.array(central_accuracy_mean_val).dump(fn + "epoch_central_accuracy.pkl")
    plt.plot(np.arange(len(central_accuracy_mean_val)), central_accuracy_mean_val)
    plt.title("Class 1-7 accuracy on epoch")
    plt.ylabel("Accuracy %")
    plt.xlabel("Epochs")
    plt.savefig(fn + "epoch_central_accuracy.png")
    plt.close()

    # Do test
    acc, central_acc = ts.test(unet, testloader, params, out_shape, classes, 
                               experiment_folder=fn, save_graph=True)

    return fn, acc, central_acc

if __name__ == "__main__":

    # Track time for whole script
    print("Script started at", datetime.datetime.now())
    TIME_TOTAL = time.time()

    ## FILE HANDLING ##

    # Set the experiment number
    experiment_number = "1"

    # Set the meta folder
    experiment_folder = "data/experiment_" + experiment_number + "/"
    meta_results_file = experiment_folder + "metaresults.txt"
    running_file =  experiment_folder + "running.txt"

    # Keep local results separate
    if (torch.cuda.device_count() <= 1):
        experiment_folder = experiment_folder.replace("experiment_", "LOCAL_experiment_")
    
    # Make file if doesn't exist
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    # Initialise the running file for the experiment
    with open(running_file, "w+") as rf:
        rf.write("Central, Accuracy, Smoothing_Type,   Time\n")
    
    # Set up system path
    print("appending", os.sep.join(os.path.realpath(__file__).split(os.sep)[:-2] + ["models","unet"]))
    sys.path.append(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-2] + ["models","unet"]))
    import train_segment2d as ts
    import training_metadata as tm
    
    # Copy over default parameters        
    params = tm.get_params()

    # Load up the experiment we want to run
    lr_bs_ep, number_samples, cln_type, smoothing_types = define_experiment()

    # Set the parameters
    (lr, bs, ep) = lr_bs_ep
    params["epochs"], params["batch_size"], params["lr_0"] = lr, bs, ep
    params["clean"] = cln_type

    print("Initialisation time %.3f secs" % (time.time() - TIME_TOTAL))
    
    ## RUN EXPERIMENT PER SMOOTHING TYPE ##
    for smoothing_type in smoothing_types:

        # Time the run
        print("\nRunning smoothing type", smoothing_type, "at", datetime.datetime.now())
        TIME_RUN = time.time()

        # Set the smoothing type
        params["smoothing_type"] = smoothing_type
        
        print("This smoothing type initialisation time %.3f secs" % (time.time() - t_clean))

        ## LOAD DATA ##
        if params["torch"]:
            trainloader, testloader, classes = ts.load_torch_data(params, number_samples=number_samples)
        else:
            trainloader, testloader, train_id, test_id, classes = ts.load_h5_data(params, number_samples=number_samples)

        ## LOAD a freshly initialised model ##
        unet = load_fresh_model(params)

        ####################
        # Run the experiment
        t_start = time.time()
        
        # Filename for this smoothing type only
        filename, test_accuracy, central_acc =\
            run_experiment(unet, params, trainloader, testloader, classes, 
                           experiment_folder, number_samples=number_samples)

        t_end = time.time()

        print("Training complete in %.3f hrs" % ((t_end-t_start)/(60**2)))
        ####################

        # Load in most recent tracking lists
        if not os.path.isfile(filename + "pickled_record.pickle"):
            
            print("No previous experiments found.")
            
            results_dict = {smoothing_type: {"accuracies":[], "central_accuracies":[], "accuracies_info":[]}}
        
        else:
            
            print("Reading in previous experiments.")
            
            with open(experiment_folder + "pickled_record.pickle", 'rb') as f:
                results_dict = pickle.load(f)
            
            if smoothing_type not in results_dict:
                results_dict[smoothing_type] = {"accuracies":[], "central_accuracies":[], "accuracies_info":[]}

        # Record outputs for graph plotting
        results_dict[smoothing_type]["accuracies"].append(test_accuracy)
        results_dict[smoothing_type]["central_accuracies"].append(central_acc)
        results_dict[smoothing_type]["accuracies_info"].append((central_acc, test_accuracy, filename, (t_end-t_start)/(60**2)))

        # Keep a running list of the results for plotting
        with open(experiment_folder + "pickled_record.pickle", 'wb') as f:
            pickle.dump((lrs, bss, eps, accuracies, accuracies_info, central_accuracies), f)

        # Record info to file
        print("\nWriting output to running results")
        with open(running_file, "a+") as rf:
            rf.write("%.3f, %.3f, %s, %.3f\n" % (central_acc, test_accuracy, filename, ((t_end-t_start)/(60**2))))

        ## Make running plots - overwrite old ones ##

        make_plot(bss, accuracies, "Total", "Batch Size", experiment_folder)
        make_plot(bss, central_accuracies, "Central", "Batch Size", experiment_folder)
        make_plot(lrs, accuracies, "Total", "Learning Rate_0", experiment_folder)
        make_plot(lrs, central_accuracies, "Central", "Learning Rate_0", experiment_folder)

        # PLOT PARAMETER SEARCH GRID
        make_heat_map(lrs, bss, central_accuracies, "central accuracy", experiment_folder)
        make_heat_map(lrs, bss, accuracies, "total accuracy", experiment_folder)

        print("Completed run at", datetime.datetime.now())
        print("Time for run %.3f hrs" % ((time.time() - TIME_RUN) / (60**2)))

    ## ANALYSIS ##

    # Sort ordered accuracy
    ordered_accuracy = sorted(accuracies_info, key=lambda x: x[1])

    # Print it to a file line by line
    print("\nWriting output to metaresults")
    with open(meta_results_file, "w+") as mrf:
        mrf.write("Central, Accuracy, Experiment,   Time (hrs)\n")
        for l in ordered_accuracy:
            mrf.write("%.3f, %.3f, %s, %.3f\n" % (l[0], l[1], l[2], l[3]))
        #mrf.write("\nTIME TO COMPLETION: %.3f" % (TIME_TOTAL / (60**2)))

    print("Successful completion")
    print("Completed in %.3f hours" % ((time.time() - TIME_TOTAL) / (60**2)))