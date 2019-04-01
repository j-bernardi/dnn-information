"""
USAGE - specify a loss type and a suffix for the file to be sent
e.g.
	python create_scripts.py uniform_fixed_eps _0
"""

import sys, os, io

if __name__ == "__main__":

	loss_type = sys.argv[-2]
	suffix = "_" + loss_type + "_" + sys.argv[-1]

	print("Loss type:", loss_type)
	print("Suffix read:", suffix)

	exp_folder = "experiments/experiment_" + sys.argv[-1] + "/" 

	if not os.path.exists(exp_folder):
		print("Creating folder", exp_folder)
		os.makedirs(exp_folder)

	# 1 - create the run-experiment script (simple copy)
	old_experiment_script = "experiments/run_experiment.py"
	new_experiment_script = exp_folder + "run_experiment" + suffix + ".py"

	print("Creating experiment script", new_experiment_script)

	with open(old_experiment_script, 'r') as original:

		# CHECK this line
		fl_string = original.read()
	
	# Change the loss type
	fl_string = fl_string.replace("LOSS_HERE", loss_type)
	# Make one deeper
	fl_string = fl_string.replace(".split(os.sep)[:-2]", ".split(os.sep)[:-3]")
	
	with io.open(new_experiment_script, 'w', newline="\n") as nw:

		nw.write(fl_string)


	# 1 - create the slurm script with the correct suffix
	old_slurm_script = "experiments/slurm_submit_experiment" 
	new_slurm_script = exp_folder + "slurm_submit_experiment" + suffix
	print("Creating slurm script", new_slurm_script)

	with open(old_slurm_script, 'r') as original:

		slurm_fl_string = original.read()

	slurm_fl_string = slurm_fl_string.replace("experiments/run_experiment.py", new_experiment_script)

	with io.open(new_slurm_script, 'w', newline="\n") as nw:

		nw.write(slurm_fl_string)

	print("Experiment files created.")