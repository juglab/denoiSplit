# BioSR
python notebooks/evaluate_notebook.py --ckpt_dir=/group/jug/ashesh/training/denoisplit/2502/D16-M3-S0-L0/5 --override_kwargs='{"MIXING_WEIGHT":0.5, "k_forward_pass":2, "data_dir":"/group/jug/ashesh/data/BioSR/"}'

python notebooks/evaluate_notebook.py --ckpt_dir=/group/jug/ashesh/training/denoisplit/2502/D16-M3-S0-L0/5 --data_dir=/group/jug/ashesh/data/BioSR/  --MIXING_WEIGHT=0.5 --k_forward_pass=1
