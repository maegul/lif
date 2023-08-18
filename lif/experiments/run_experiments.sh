#! /usr/bin/bash

export PYTHONUNBUFFERED="unbufferred"

exp_dir_path="experiments/spat_freq_ori_tun_sweeps/scripts"

echo "not done" > experiment_status


# Run all scripts in directory

echo "Running all scripts in $exp_dir_path" >> experiment_log_file

for f in "$exp_dir_path"/*.py;
do
	echo "Running $f ($(date))" >> experiment_log_file
	python "$f"  >> experiment_log_file 2>&1
done



echo "done" > experiment_status
