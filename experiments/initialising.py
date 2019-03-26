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

def remake_graph(experiment_folder="data/initialising_loss/"):

    with open(experiment_folder + "pickled_record.pickle", 'rb') as f:
        (lrs, bss, eps, accuracies, accuracies_info, central_accuracies) = pickle.load(f)

    make_heat_map(np.array(lrs) * 1000, bss, accuracies, "total accuracy", experiment_folder)
    make_heat_map(np.array(lrs) * 1000, bss, central_accuracies, "central accuracy", experiment_folder)
    make_plot(np.array(lrs) * 1000, central_accuracies, "Central", "Learning Rate_0", experiment_folder, xlim=(0,2))
    make_plot(np.array(lrs) * 1000, accuracies, "Total", "Learning Rate_0", experiment_folder, xlim=(0,2))

def define_experiment(test_small_slurm=False):

    # Too computationally expensive to go higher
    lr_2_eps = [(0.0015, 2, 70)]  # 2.5 hr

    lr_4_eps = [(0.0002, 4, 180), # 2   hr 
                (0.0010, 4, 100)] # 1.5 hr
 
    lr_6_eps = [(0.0005, 6, 120)] # 1.4 hr

    lr_8_eps = [(0.0001, 8, 220), # 1.8 hr
                (0.0013, 7, 80 )] # 1.3 hr

    more_8s = [(0.00005, 8, 220), 
               (0.00009, 8, 160)]

    a_12 = [(0.0001, 12, 180)]

    test_8s = [(0.0001, 8, 60),
               (0.0001, 8, 90),
               (0.0001, 8, 120),
               (0.0001, 8, 180),
               (0.0001, 8, 220)]

    # DEFINE
    cln_types = ["loss"]#, "no_clean"]
    lr_bs_eps = [test_8s[4]]
    #lr_bs_eps = lr_4_eps + lr_6_eps
    #lr_bs_eps = lr_2_eps + lr_4_eps

    number_samples = -1 # e.g. all

    # For local testing:
    train_local = (torch.cuda.device_count() <= 1)
    
    if train_local:
        
        print("************************")
        print("TEMP number of samples 5")
        print("************************")

        lr_bs_eps = [(0.0001    , 1, 1), (0.0001, 1, 2)]#, (0.002, 1, 1)]#2)]
        number_samples = 5
        cln_types = ["loss"]

    # If doing small slurm
    if test_small_slurm and torch.cuda.device_count() > 1:

        print("*******************")
        print("TESTING SMALL SCALE")
        print("*******************")

        lr_bs_eps = [(0.01, 2, 2), (0.005, 4, 2)]

    return lr_bs_eps, number_samples, cln_types

def load_fresh_model(params, experiment_folder="no", save_reps=False):

    # Initialise model
    unet = ts.load_model(params, experiment_folder="no", save_reps=False)

    if torch.cuda.device_count() > 1:
        unet.module.load_state_dict(torch.load("models/unet/saved_models/initialisation_multi.pth"))
    else:
        unet.load_state_dict(torch.load("models/unet/saved_models/initialisation.pth"))

    return unet

def run_experiment(unet, params, trainloader, testloader, classes, experiment_folder, number_samples=-1, save_reps=False, total_number_images=88):

    # Get save file name for this given experiment
    fn = tm.construct_file(params, experiment_folder)

    # Do training - means are per cel
    out_shape, epoch_mean_loss, accuracy_mean_val, central_accuracy_mean_val, training_order =\
        ts.train(unet, trainloader, params, fake=False, 
                 experiment_folder=fn, total_number_images=total_number_images)

    print("\nTraining complete. Saving graphs")

    # can plot epoch_mean_loss on epoch for loss change:
    plt.figure()
    np.array(epoch_mean_loss).dump(fn + "epoch_loss.pkl")
    plt.plot(np.arange(len(epoch_mean_loss)), epoch_mean_loss)
    plt.title("Loss on epoch, bs=%d lr_0=%s" % (params["batch_size"], str(params["lr_0"])))
    plt.ylabel("Loss / pixel")
    plt.xlabel("Epochs")
    plt.savefig(fn + "epoch_loss.png")
    plt.close()

    # Plot the accuracy on epoch of the validation set
    plt.figure()
    np.array(accuracy_mean_val).dump(fn + "epoch_accuracy.pkl")
    plt.plot(np.arange(len(accuracy_mean_val)), accuracy_mean_val)
    plt.title("Accuracy on epoch, bs=%d lr_0=%s" % (params["batch_size"], str(params["lr_0"])))
    plt.ylabel("Accuracy %")
    plt.xlabel("Epochs")
    plt.savefig(fn + "epoch_accuracy.png")
    plt.close()

    # Plot the central on epoch of the validation set
    plt.figure()
    np.array(central_accuracy_mean_val).dump(fn + "epoch_central_accuracy.pkl")
    plt.plot(np.arange(len(central_accuracy_mean_val)), central_accuracy_mean_val)
    plt.title("Class 1-7 accuracy on epoch, bs=%d lr_0=%s" % (params["batch_size"], str(params["lr_0"])))
    plt.ylabel("Accuracy %")
    plt.xlabel("Epochs")
    plt.savefig(fn + "epoch_central_accuracy.png")
    plt.close()

    # Do test
    acc, central_acc = ts.test(unet, testloader, params, out_shape, classes, 
                               experiment_folder=fn, save_graph=True)

    return fn, acc, central_acc

def make_plot(x, y, acc_type, title, experiment_folder, xlim=(0,14)):
    """Makes an ordered plot of y on x."""
    
    ordered_y = [i for _, i in sorted(zip(x, y))]
    ordered_x = sorted(x)

    # PLOT BS
    plt.figure()
    if "epoch" in title.lower():
        plt.plot(ordered_x, ordered_y, 'bx')
        plt.title(acc_type + " accuracy on " + title + " - lr=1e-4, bs=8")
    else:
        plt.scatter(ordered_x, ordered_y, marker='x')
        plt.title(acc_type + " accuracy on " + title)
    
    plt.xlim(xlim)
    
    if "learning" in title.lower():
        plt.xlabel("Learning rate_0 (10^-3)")
    else:
        plt.xlabel(title)
    
    plt.ylabel("Accuracy")
    plt.savefig(experiment_folder + acc_type+ "_accuracy_on_" + title.lower().replace(" ", "_") + ".png")
    plt.close()

def make_heat_map(lrs, bss, accuracies, title, experiment_folder):

    plt.figure()
    sc = plt.scatter(lrs, bss, c=accuracies, s=250, 
                     cmap=plt.cm.get_cmap('RdYlBu'), 
                     clim=(min(accuracies), max(accuracies)),
                     edgecolors='black',
                     linewidths=1.0)

    plt.colorbar(sc)
    plt.title("Parameter grid search - " + title.lower())
    plt.xlabel("LR_0 (10^-3)")
    plt.xlim((0, 2))
    plt.ylabel("Batch Size")
    plt.ylim((0, 14))
    plt.savefig(experiment_folder + title.lower().replace(" ", "_") + "_grid.png")
    plt.close()

if __name__ == "__main__":

    """
    remake_graph(experiment_folder="data/initialising_loss/")
    remake_graph(experiment_folder="data/initialising_no_clean/")
    sys.exit()
    """

    # Track time for whole script
    TIME_TOTAL = time.time()

    print("Script started at", datetime.datetime.now())
    
    # Set up system path
    print("appending", os.sep.join(os.path.realpath(__file__).split(os.sep)[:-2] + ["models","unet"]))
    sys.path.append(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-2] + ["models","unet"]))
    import train_segment2d as ts
    import training_metadata as tm
    
    # Copy over default parameters        
    params = tm.get_params()

    # Load up the experiment we want to run
    lr_bs_eps, number_samples, cln_types = define_experiment()

    # Save a model if one doesn't already exist
    initialise_model(params)

    print("Initialisation time %.3f secs" % (time.time() - TIME_TOTAL))
    
    ## RUN EXPERIMENT ##
    for cln_type in cln_types:
        
        t_clean = time.time()

        print("\nRunning clean type", cln_type)
        
        params["clean"] = cln_type

        experiment_folder = "data/initialising_" + cln_type + "/"

        # Keep local results separate
        if (torch.cuda.device_count() <= 1):
            experiment_folder = experiment_folder.replace("initialising_", "LOCAL_initialising_")
        
        # Make file if doensn't exist
        if not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)
        
        meta_results_file = experiment_folder + "metaresults.txt"
        running_file =  experiment_folder + "running.txt"

        # Initialise the running file for the experiment
        with open(running_file, "w+") as rf:
            rf.write("Central, Accuracy, Experiment,   Time\n")
        
        print("This clean type initialisation time %.3f secs" % (time.time() - t_clean))

        # Do the hyperparameter search
        for (lr, bs, e) in lr_bs_eps:

            print("\nStarting at", datetime.datetime.now(), "with lr", lr, "- bs", bs, "- epochs", e, "\n")
            TIME_RUN = time.time()

            # Define params
            params["epochs"], params["batch_size"], params["lr_0"] = e, bs, lr

            ## LOAD DATA - same order for all ##
            if params["torch"]:
                trainloader, testloader, classes = ts.load_torch_data(params, number_samples=number_samples)
            else:
                trainloader, testloader, train_id, test_id, classes = ts.load_h5_data(params, number_samples=number_samples)

            # Load fresh model
            unet = load_fresh_model(params)

            ####################
            # Run the experiment
            t_start = time.time()
            
            filename, test_accuracy, central_acc =\
                run_experiment(unet, params, trainloader, testloader, classes, 
                               experiment_folder, number_samples=number_samples)

            t_end = time.time()

            print("Training complete in %.3f hrs" % ((t_end-t_start)/(60**2)))
            ####################

            # Load in most recent tracking lists
            if not os.path.isfile(experiment_folder + "pickled_record.pickle"):
                print("No previous experiments found.")
                lrs, bss, eps, accuracies, accuracies_info, central_accuracies = [],[],[],[],[],[]    
            else:
                print("Reading in previous experiments.")
                with open(experiment_folder + "pickled_record.pickle", 'rb') as f:
                    (lrs, bss, eps, accuracies, accuracies_info, central_accuracies) = pickle.load(f)

            # Record outputs for graph plotting
            lrs.append(lr), bss.append(bs), eps.append(e)

            accuracies.append(test_accuracy)
            central_accuracies.append(central_acc)
            accuracies_info.append((central_acc, test_accuracy, filename, (t_end-t_start)/(60**2)))

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
            make_plot(np.array(lrs) * 1000, accuracies, "Total", "Learning Rate_0", experiment_folder, xlim=(0,2))
            make_plot(np.array(lrs) * 1000, central_accuracies, "Central", "Learning Rate_0", experiment_folder, xlim=(0,2))
            
            try:
                make_plot(np.array(eps), accuracies, "Total", "Epochs", experiment_folder, xlim=(40, 240))
                make_plot(np.array(eps), central_accuracies, "Central", "Epochs", experiment_folder, xlim=(40, 240))
            except:
                print("UNTESTED failure.")

            # PLOT PARAMETER SEARCH GRID
            make_heat_map(np.array(lrs) * 1000, bss, central_accuracies, "central accuracy", experiment_folder)
            make_heat_map(np.array(lrs) * 1000, bss, accuracies, "total accuracy", experiment_folder)

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