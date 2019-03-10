"""Run experiments to test what parameters should be used."""

from models.unet import train_segment2d as ts
from models.unet import training_metadata as tm

lr_0s = [0.01, 0.001, 0.0001]
batch_sizes = [4, 8, 16]
epochs = [80, 120, 160]


def run_experiment(params, save_dir):

    # Get save file name

    # Make the model
    
    # Do training

    # Save trained model and results

    # Do test

    # Save trained model and results

    # return end test accuracy


ordered_accuracy = []

if __name__ == "__main__":
    
    for lr in lr_0s:
        for bs in batch_sizes:
            for e in epochs:

                # Update parameters with above

                # Run the experiment
                test_accuracy = run_experiment(params)

                ordered_accuracy.append(("file_string", test_accuracy))

    # Sort ordered accuracy
    # Print it to a file line by line