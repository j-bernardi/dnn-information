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
    
    smoothing_types = ["weighted_vary_eps"]
    # Choose from:
    # ["none", "uniform_fixed_eps", "uniform_vary_eps", "weighted_fixed_eps", "weighted_vary_eps"]
    
    N_SO_FAR = 1
    N_REPEATS = 2

    ## TODO : fix h params
    lr_bs_eps = (0.0001, 8, 180)

    number_samples = -1 # e.g. all

    # For local testing:
    train_local = (torch.cuda.device_count() <= 1)
    
    if train_local:
        
        print("************************")
        print("TEMP number of samples 5")
        print("************************")
        #smoothing_types = ["none", "uniform_fixed_eps"]    
        smoothing_types = ["uniform_vary_eps"]
        lr_bs_eps = (0.02, 1, 1)
        number_samples = 5
    
    # If doing small slurm
    if test_small_slurm and torch.cuda.device_count() > 1:
        
        print("*******************")
        print("TESTING SMALL SCALE")
        print("*******************")
        #smoothing_types = ["uniform_vary_eps", "weighted_fixed_eps", "weighted_vary_eps"]
        lr_bs_eps = (0.0001, 8, 2)

    return lr_bs_eps, number_samples, cln_type, smoothing_types, (N_SO_FAR, N_REPEATS)

def load_fresh_model(params, experiment_folder="no", save_reps=False, save_model=False):

    # Initialise model
    unet = ts.load_model(params, experiment_folder="no", save_reps=False)

    if torch.cuda.device_count() > 1 and save_model:
        unet.module.load_state_dict(torch.load("models/unet/saved_models/initialisation_multi.pth"))
    elif save_model:
        unet.load_state_dict(torch.load("models/unet/saved_models/initialisation.pth"))

    return unet

def run_experiment(unet, params, trainloader, testloader, classes, experiment_folder, number_samples=-1, save_reps=False, nth_repeat=-1):

    # Get save file name for this given experiment
    fn = tm.construct_file(params, experiment_folder, append=("_"+str(nth_repeat)))

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
    acc, central_acc, confusion, norm_confusion = ts.test(unet, testloader, params, out_shape, classes, 
        experiment_folder=fn, save_graph=True, return_confusion=True)

    return fn, acc, central_acc, accuracy_mean_val, central_accuracy_mean_val, confusion, norm_confusion

def plot_experiment(results_dict, experiment_folder):

    ## TOTAL ##

    plt.figure()
    leg = []

    for st in results_dict:
        
        leg.append(st)

        # Take average
        res_lst = np.array(results_dict[st]["accuracy_epochs"])
        #print("res_lst", res_lst)

        avg_res = np.sum(res_lst, axis=0) / res_lst.shape[0]

        # Plot
        plt.plot(range(len(avg_res)), avg_res)
    
    plt.legend(leg, loc='lower right')
    plt.title("Total Accuracy on epochs")
    plt.ylabel("Accuracy %")
    plt.xlabel("Epochs")
    plt.savefig(experiment_folder + "accuracy_total_epochs.png")
    plt.close()

    ## CENTRAL ##

    plt.figure()
    leg = []
    for st in results_dict:
        
        leg.append(st)

        # Take average
        res_lst = np.array(results_dict[st]["central_accuracy_epochs"])
        
        avg_res = np.sum(res_lst, axis=0) / res_lst.shape[0]
        
        plt.plot(range(len(avg_res)), avg_res)
    
    plt.legend(leg, loc='lower right')
    plt.title("Class 1-7 accuracy on epochs")
    plt.ylabel("Central accuracy %")
    plt.xlabel("Epochs")
    plt.savefig(experiment_folder + "accuracy_central_epochs.png")
    plt.close()

def plot_test_results(results_dict, experiment_folder, acc_type="accuracies"):

    nms, avg_accs, stds = [], [], []

    for st in results_dict:
        
        acc_lst = np.array(results_dict[st][acc_type])
        
        nms.append(st + " (" + str(len(acc_lst)) + ")")
        avg_accs.append(np.sum(acc_lst) / len(acc_lst))
        stds.append(np.std(acc_lst))

    x_pos = np.arange(len(nms))
        
    plt.bar(x_pos, avg_accs, yerr=stds, align='center', alpha=0.5, ecolor='black')

    try:
        # Untested
        plt.xticks(x_pos, "\n".join(nms.split("_")))
    except:
        plt.xticks(x_pos, nms)

    plt.ylabel("Accuracy achieved %")
    plt.title("Accuracy achieved for each smoothing type")
    plt.savefig(experiment_folder + "comparison_" + acc_type + ".png")
    plt.close()

def plot_confusion_matrices(results_dict, file_to):
    """Save average confusion matrices"""
    
    classes = ["class" + str(i) for i in range(9)]

    for st in results_dict:
        
        if st in file_to:
            
            conf_mat = np.array(results_dict[st]["confusion_matrix"]) 
            norm_conf_mat = np.array(results_dict[st]["norm_confusion_matrix"])

            avg_confusion = np.floor(conf_mat.sum(axis=0) / conf_mat.shape[0]).astype(int)
            avg_norm_confusion = norm_conf_mat.sum(axis=0) / conf_mat.shape[0]
        
            # TODO - decide how I'm going to display uncertainty in the confusion matrix
            ts.save_confusion(avg_confusion, classes, "/".join(file_to.split("/")[:-2]) + "/avg_confusion_" + st)
            ts.save_confusion(avg_norm_confusion, classes, "/".join(file_to.split("/")[:-2]) + "/avg_norm_confusion_" + st)

def remove_from_dict(experiment_folder):

    new_dict = {}

    print("Reading in previous experiments.")
    with open(experiment_folder + "pickled_record.pickle", 'rb') as f:
        (results_dict, meta_results) = pickle.load(f)

    # Types and indices wewant to remove from the saved dict
    idxs = {"uniform_fixed_eps": [0], "weighted_vary_eps": [0]} # these didn't work
    for typ in results_dict:
        
        # If removing
        if typ in idxs:

            # Set up an empty dict for this type
            new_dict[typ] = {"accuracies":[], 
                             "central_accuracies":[], 
                             "accuracy_epochs": [], 
                             "central_accuracy_epochs": [],
                             "confusion_matrix": [],
                             "norm_confusion_matrix":[]}

            # Copy over the results we wanted
            for accuracy_metric in results_dict[typ]:

                for item_idx in range(len(results_dict[typ][accuracy_metric])):
                    
                    if item_idx in idxs[typ]:
                        pass
                    
                    else:
                        new_dict[typ][accuracy_metric].append(results_dict[typ][accuracy_metric][item_idx])


        else:

            # If not removing just copy
            new_dict[typ] = results_dict[typ]

        print("")
        print("TYPE BELOW", typ)
        print("old results\n", results_dict[typ])
        print("new results\n", new_dict[typ])
        print("TYPE ABOVE", typ)
        print("")

    # Dump the newly added results
    with open(experiment_folder + "pickled_record_new.pickle", 'wb') as f:
        pickle.dump((results_dict, meta_results), f)

    print("DUMPED to new file - be sure to change name for running.")


if __name__ == "__main__":

    # Track time for whole script
    print("Script started at", datetime.datetime.now())
    TIME_TOTAL = time.time()

    ## FILE HANDLING ##

    # Set the experiment number
    experiment_number = "1"

    # Set the meta folder
    experiment_folder = "data/experiment_" + experiment_number + "/"

    ############ TEMP #################
    #remove_from_dict(experiment_folder)
    #sys.exit()
    ###################################

    # Keep local results separate
    if (torch.cuda.device_count() <= 1):
        experiment_folder = experiment_folder.replace("experiment_", "LOCAL_experiment_")
    
    # Make file if doesn't exist
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    
    meta_results_file = experiment_folder + "metaresults.txt"
    running_file =  experiment_folder + "running.txt"

    # Initialise the running file for the experiment
    with open(running_file, "w+") as rf:
        rf.write("Central, Accuracy, Smoothing_Type,   Time\n")

    # Set up system path
    print("appending", os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3] + ["models","unet"]))
    sys.path.append(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3] + ["models","unet"]))
    import train_segment2d as ts
    import training_metadata as tm
    
    # Copy over default parameters        
    params = tm.get_params()

    # Load up the experiment we want to run
    lr_bs_ep, number_samples, cln_type, smoothing_types, (N_SO_FAR, N_TO) = define_experiment()

    # Set the parameters
    (lr, bs, ep) = lr_bs_ep
    params["lr_0"], params["batch_size"], params["epochs"] = lr, bs, ep
    params["clean"] = cln_type
    params["shuffle"] = True

    print("Initialisation time %.3f secs" % (time.time() - TIME_TOTAL))
    
    for n in range(N_SO_FAR, N_TO):

        ## RUN EXPERIMENT PER SMOOTHING TYPE ##
        for smoothing_type in smoothing_types:

            # Time the run
            print("\nRunning smoothing type", smoothing_type, "at", datetime.datetime.now(), "\n")
            TIME_RUN = time.time()

            # Set the smoothing type
            params["smoothing_type"] = smoothing_type

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
            filename, test_accuracy, central_acc, accuracy_epochs,\
            central_accuracy_epochs, confusion, norm_confusion =\
                run_experiment(unet, params, trainloader, testloader, classes, 
                               experiment_folder, number_samples=number_samples, nth_repeat=n)
            
            t_end = time.time()

            print("Training complete in %.3f hrs" % ((t_end-t_start)/(60**2)))
            ####################

            ## RESULTS DICT ##

            # Load in most recent tracking lists
            if not os.path.isfile(experiment_folder + "pickled_record.pickle"):
                
                print("No previous experiments found. Creating results dict")
                results_dict = {smoothing_type :  {"accuracies": [], 
                                                   "central_accuracies": [],
                                                   "accuracy_epochs": [],
                                                   "central_accuracy_epochs": [],
                                                   "confusion_matrix": [],
                                                   "norm_confusion_matrix": []
                                                  }
                                }
                meta_results = []
            else:
                
                print("Reading in previous experiments.")
                with open(experiment_folder + "pickled_record.pickle", 'rb') as f:
                    (results_dict, meta_results) = pickle.load(f)
                
                # Initialise if required
                if smoothing_type not in results_dict:
                    results_dict[smoothing_type] = {"accuracies":[], 
                                                    "central_accuracies":[], 
                                                    "accuracy_epochs": [], 
                                                    "central_accuracy_epochs": [],
                                                    "confusion_matrix": [],
                                                    "norm_confusion_matrix":[]}

            # Record outputs for graph plotting
            results_dict[smoothing_type]["accuracies"].append(test_accuracy)
            results_dict[smoothing_type]["central_accuracies"].append(central_acc)
            results_dict[smoothing_type]["accuracy_epochs"].append(np.array(accuracy_epochs))
            results_dict[smoothing_type]["central_accuracy_epochs"].append(np.array(central_accuracy_epochs))
            results_dict[smoothing_type]["confusion_matrix"].append(confusion)
            results_dict[smoothing_type]["norm_confusion_matrix"].append(norm_confusion)
            meta_results.append((central_acc, test_accuracy, filename, (t_end-t_start)/(60**2)))

            # Dump the newly added results
            with open(experiment_folder + "pickled_record.pickle", 'wb') as f:
                pickle.dump((results_dict, meta_results), f)

            # Record info to file
            print("\nWriting output to running results")
            with open(running_file, "a+") as rf:
                rf.write("%.3f, %.3f, %s, %.3f\n" % (central_acc, test_accuracy, filename, ((t_end-t_start)/(60**2))))

            ## Make running plots - overwrite old ones ##
            plot_experiment(results_dict, experiment_folder)
            plot_test_results(results_dict, experiment_folder, acc_type="accuracies")
            plot_test_results(results_dict, experiment_folder, acc_type="central_accuracies")
            plot_confusion_matrices(results_dict, filename)

            print("Completed run at", datetime.datetime.now())
            print("Time for run %.3f hrs" % ((time.time() - TIME_RUN) / (60**2)))

    ## ANALYSIS ##

    # Sort ordered accuracy
    ordered_accuracy = sorted(meta_results, key=lambda x: x[1])

    # Print it to a file line by line
    print("\nWriting output to metaresults")
    with open(meta_results_file, "w+") as mrf:
        mrf.write("Central, Accuracy, Experiment,   Time (hrs)\n")
        for l in ordered_accuracy:
            mrf.write("%.3f, %.3f, %s, %.3f\n" % (l[0], l[1], l[2], l[3]))

    print("Successful completion")
    print("Completed in %.3f hours" % ((time.time() - TIME_TOTAL) / (60**2)))