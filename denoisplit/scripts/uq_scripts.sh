# BioSR
uv run python notebooks/evaluate_notebook.py --ckpt_dir=/group/jug/ashesh/training/denoisplit/2502/D16-M3-S0-L0/5 --override_kwargs='{"MIXING_WEIGHT":0.5, "k_forward_pass":2, "data_dir":"/group/jug/ashesh/data/BioSR/"}'

uv run python notebooks/evaluate_notebook.py --ckpt_dir=/group/jug/ashesh/training/denoisplit/2502/D16-M3-S0-L0/5 --override_kwargs='{"MIXING_WEIGHT":0.5, "k_forward_pass":1, "data_dir":"/group/jug/ashesh/data/BioSR/"}'

# Hagen
uv run python notebooks/evaluate_notebook.py --ckpt_dir=/group/jug/ashesh/training/denoisplit/2503/D7-M3-S0-L0/1 --override_kwargs='{"MIXING_WEIGHT":0.5, "k_forward_pass":2, "data_dir":"/group/jug/ashesh/data/ventura_gigascience/"}'

uv run python notebooks/evaluate_notebook.py --ckpt_dir=/group/jug/ashesh/training/denoisplit/2503/D7-M3-S0-L0/1 --override_kwargs='{"MIXING_WEIGHT":0.5, "k_forward_pass":1, "data_dir":"/group/jug/ashesh/data/ventura_gigascience/"}'

# HTLIF (synthetic)
uv run python notebooks/evaluate_notebook.py --ckpt_dir=/group/jug/ashesh/training/denoisplit/2502/D26-M3-S0-L0/3 --override_kwargs='{"MIXING_WEIGHT":0.5, "k_forward_pass":2, "data_dir":"/group/jug/ashesh/data/scsplit_HT_LIF/500ms/Ch_B-Ch_D-Ch_BD/"}'

uv run python notebooks/evaluate_notebook.py --ckpt_dir=/group/jug/ashesh/training/denoisplit/2502/D26-M3-S0-L0/3 --override_kwargs='{"MIXING_WEIGHT":0.5, "k_forward_pass":1, "data_dir":"/group/jug/ashesh/data/scsplit_HT_LIF/500ms/Ch_B-Ch_D-Ch_BD/"}'

# HTLIF (real)
uv run python notebooks/evaluate_notebook.py --ckpt_dir=/group/jug/ashesh/training/denoisplit/2502/D26-M3-S0-L0/3 --override_kwargs='{"MIXING_WEIGHT":0.5, "k_forward_pass":2, "data_dir":"/group/jug/ashesh/data/scsplit_HT_LIF/500ms/Ch_B-Ch_D-Ch_BD/", "use_real_input": true}'

uv run python notebooks/evaluate_notebook.py --ckpt_dir=/group/jug/ashesh/training/denoisplit/2502/D26-M3-S0-L0/3 --override_kwargs='{"MIXING_WEIGHT":0.5, "k_forward_pass":1, "data_dir":"/group/jug/ashesh/data/scsplit_HT_LIF/500ms/Ch_B-Ch_D-Ch_BD/", "use_real_input": true}'