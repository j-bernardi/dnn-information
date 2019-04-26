"""Run experiments to test what parameters should be used."""
import os, sys, time, torch, datetime, pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

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

    make_plot(bss, accuracies, "Total", "Batch Size", experiment_folder)
    make_plot(bss, central_accuracies, "Central", "Batch Size", experiment_folder)
    
    make_plot(np.array(eps), accuracies, "Total", "Epochs", experiment_folder, xlim=(40, 320))
    make_plot(np.array(eps), central_accuracies, "Central", "Epochs", experiment_folder, xlim=(40, 320))
    
    make_heat_map(np.array(lrs), bss, accuracies, "total accuracy", experiment_folder)
    make_heat_map(np.array(lrs), bss, central_accuracies, "central accuracy", experiment_folder)
    make_plot(np.array(lrs), central_accuracies, "Central", "Learning Rate_0", experiment_folder, xlim=(0.00001, 0.1))
    make_plot(np.array(lrs), accuracies, "Total", "Learning Rate_0", experiment_folder, xlim=(0.00001, 0.1))

    plot_both(eps, accuracies, central_accuracies, experiment_folder)

def define_experiment(test_small_slurm=False):

    # Too computationally expensive to go higher than 2, 60

    lr_4_eps = [(0.005,  4, 180),
                (0.001, 4, 100)]

    lr_8_eps = [(0.0001, 8, 180),
                (0.0013, 8, 60 )]

    lr_16_eps= [(0.00005, 16, 220), 
                (0.0008, 16, 160)]

    lr_8_new = [(0.0005,  8, 130),
                (0.00004, 8, 220),
                (0.00009, 8, 190)]
    
    test_8s = [(0.0001, 8, 60), #0
               (0.0001, 8, 90),
               (0.0001, 8, 120),
               (0.0001, 8, 180),
               (0.0001, 8, 220),
               (0.0001, 8, 150),
               (0.0001, 8, 135),
               (0.0001, 8, 165),
               (0.0001, 8, 200), #8

               (0.0001, 8, 143),
               (0.0001, 8, 158),
               (0.0001, 8, 240),
               (0.0001, 8, 260),
               (0.0001, 8, 300)]

    # DEFINE
    cln_types = ["no_clean"]#, "no_clean"]
    lr_bs_eps = [test_8s[13]]

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

        lr_bs_eps = [(0.01, 4, 4)]#, (0.005, 4, 2)]

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

    return fn, acc, central_acc, accuracy_mean_val[-1], central_accuracy_mean_val[-1]

def make_plot(x, y, acc_type, title, experiment_folder, xlim=(0,20)):
    """Makes an ordered plot of y on log x."""
    
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
    
    if "learn" in title.lower():
        plt.xlabel("Initial learning rate")
        plt.xscale("log")
    else:
        plt.xlabel(title)
        plt.xscale("linear")

    if "batch" in title.lower():
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.ylabel("Accuracy")
    plt.savefig(experiment_folder + acc_type+ "_accuracy_on_" + title.lower().replace(" ", "_") + ".png")
    plt.close()

def plot_both(epochs, y1, y2, experiment_folder, xlim=(40,320)):

    ordered_y1 = [i for _, i in sorted(zip(epochs, y1))]
    ordered_y2 = [i for _, i in sorted(zip(epochs, y2))]

    y3, y4 = [], []
    these_epochs = []

    for e in [60,90,120,135,150,165,180,200,220, 240, 260, 300]:
        
        fn = "uniform_fixed_eps_lr0001_ep"+ str(e) + "_bs8/"
        
        try:
            y3.append(np.load(experiment_folder + fn + "epoch_accuracy.pkl")[-1])
            y4.append(np.load(experiment_folder + fn + "epoch_central_accuracy.pkl")[-1])
            these_epochs.append(e)
        except:
            print(experiment_folder + fn, "not found (yet)")

    ordered_y3 = [i for _, i in sorted(zip(these_epochs, y3))]
    ordered_y4 = [i for _, i in sorted(zip(these_epochs, y4))]
    ordered_x = sorted(epochs)
    ordered_x2 = sorted(these_epochs)

    # PLOT BS
    plt.figure()
    
    plt.plot(ordered_x, ordered_y1, 'bx-')
    plt.plot(ordered_x2, ordered_y3, 'b+--')
    plt.plot(ordered_x, ordered_y2, 'rx-')
    plt.plot(ordered_x2, ordered_y4, 'r+--')
    plt.legend(["Total accuracy", "Total training accuracy", 
                "Central accuracy", "Central training accuracy"], 
                loc='center right')

    plt.title("Accuracies on epochs - lr=1e-4, bs=8")
    
    plt.xlim(xlim)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    
    plt.savefig(experiment_folder +"compare_accuracy_epochs.png")
    
    plt.close()

    max_test = max(y2)
    max_idx = y2.index(max_test)
    print("max of central accuracies", max_test, "at epochs", epochs[max_idx])
    print("accuracy at 150 epochs", y2[epochs.index(150)])

def make_heat_map(lrs, bss, accuracies, title, experiment_folder):
    """LOG xscale"""

    plt.figure()
    sc = plt.scatter(lrs, bss, c=accuracies, s=250, 
                     cmap=plt.cm.get_cmap('RdYlBu'), 
                     clim=(min(accuracies), max(accuracies)),
                     edgecolors='black',
                     linewidths=1.0)

    plt.colorbar(sc)
    plt.title("Parameter grid search - " + title.lower())
    plt.xlabel("Initial learning rate)")
    plt.xscale('log')
    plt.xlim((0.00001, 0.1))
    plt.ylabel("Batch Size")
    plt.ylim((0, 20))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(experiment_folder + title.lower().replace(" ", "_") + "_grid.png")
    plt.close()

def remove_experiment_from_dict(experiment_folder="data/initialising_loss/"):

    remove_lr = 0.01
    remove_ep = 4
    remove_bs = 4
    new_lrs, new_bss, new_eps, new_accuracies, new_info, new_central =\
    [], [], [], [], [], []
    # Open
    with open(experiment_folder + "pickled_record.pickle", 'rb') as f:
        (lrs, bss, eps, accuracies, accuracies_info, central_accuracies) = pickle.load(f)

    for i in range(len(lrs)):
        
        if (lrs[i] == remove_lr) and\
           (eps[i] == remove_ep) and\
           (bss[i] == remove_bs):
            
            print("REMOVING")
            continue
        
        else:
            new_lrs.append(lrs[i])
            new_bss.append(bss[i])
            new_eps.append(eps[i])
            new_accuracies.append(accuracies[i])
            new_info.append(accuracies_info[i])
            new_central.append(central_accuracies[i])

    try:
        assert len(lrs) - 1 == len(new_lrs)
    except:
        print("No changes were made!")
        return

    # Resave If changes were made
    with open(experiment_folder + "pickled_record_old.pickle", 'wb') as f:
        pickle.dump((lrs, bss, eps, accuracies, accuracies_info, central_accuracies), f)

    # Resave before messing
    with open(experiment_folder + "pickled_record.pickle", 'wb') as f:
        pickle.dump((new_lrs, new_bss, new_eps, new_accuracies, new_info, new_central), f)    

if __name__ == "__main__":

    """
    #remake_graph(experiment_folder="data/initialising_loss/")
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
            
            filename, test_accuracy, central_acc, train_accuracy, central_train_accuracy =\
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

            remake_graph(experiment_folder=experiment_folder)
            """            
            make_plot(bss, accuracies, "Total", "Batch Size", experiment_folder)
            make_plot(bss, central_accuracies, "Central", "Batch Size", experiment_folder)
            make_plot(np.array(lrs), accuracies, "Total", "Learning Rate_0", experiment_folder, xlim=(0,2))
            make_plot(np.array(lrs), central_accuracies, "Central", "Learning Rate_0", experiment_folder, xlim=(0,2))
            
            make_plot(np.array(eps), accuracies, "Total", "Epochs", experiment_folder, xlim=(40, 240))
            make_plot(np.array(eps), central_accuracies, "Central", "Epochs", experiment_folder, xlim=(40, 240))
            
            try:
                plot_both(eps, accuracies, central_accuracies, experiment_folder)
            except:
                print("UNTESTED epochs failure.")
            

            # PLOT PARAMETER SEARCH GRID
            make_heat_map(np.array(lrs), bss, central_accuracies, "central accuracy", experiment_folder)
            make_heat_map(np.array(lrs), bss, accuracies, "total accuracy", experiment_folder)
            """

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