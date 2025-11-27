import argparse
import papermill as pm
from datetime import datetime
import os
import json
from denoisplit.core.data_split_type import DataSplitType


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a notebook")
    parser.add_argument("--notebook", type=str, help="Notebook to run", default="notebooks/CoveragePlot.ipynb")
    parser.add_argument(
        "--outputdir",
        type=str,
        help="Output notebook directory",
        default="/group/jug/ashesh/UQResults/notebook_results/",
    )
    parser.add_argument(
        "--ckpt_dir", type=str, help="Checkpoint to use. eg. /group/jug/ashesh/training/disentangle/2406/D25-M3-S0-L8/4"
    )
    parser.add_argument("--data_split_type", type=str, help="Data split type: Train/Val/Test", default="Test")
    parser.add_argument("--tag_time_flag", type=bool, help="Tag time flag", default=False)
    parser.add_argument(
        "--override_kwargs",
        type=json.loads,
        default="{}",
    )
    args = parser.parse_args()
    assert args.data_split_type in ["Train", "Val", "Test"]

    param_dict = args.override_kwargs
    keys = sorted(param_dict.keys())
    param_str = "_".join([f"{k}-{param_dict[k]}" for k in keys if k != "data_dir"])
    param_str += f"_{args.data_split_type}"
    ckpt_dir = args.ckpt_dir
    model_token = "-".join(ckpt_dir.strip("/").split("/")[-3:])
    outputdir = os.path.join(args.outputdir, model_token)
    fname = os.path.basename(args.notebook)
    fname = fname.replace(".ipynb", "")
    if args.tag_time_flag:
        now = datetime.now().strftime("%Y%m%d.%H.%M")
        fname = f"{fname}_{param_str}_{now}.ipynb"
    else:
        fname = f"{fname}_{param_str}.ipynb"

    param_dict["ckpt_dir"] = args.ckpt_dir
    output_fpath = os.path.join(outputdir, fname)
    output_config_fpath = os.path.join(outputdir, "config", fname.replace(".ipynb", ".txt"))
    output_results_fpath = os.path.join(outputdir, "results", fname.replace(".ipynb", ".pkl"))
    os.makedirs(os.path.dirname(output_config_fpath), exist_ok=True)
    os.makedirs(os.path.dirname(output_results_fpath), exist_ok=True)
    # save the configuration
    # convert args to dict
    args_dict = vars(args)
    # save as json
    with open(output_config_fpath, "w") as f:
        f.write(str(args_dict))

    if args.data_split_type == "Test":
        calibration_params_fpath = output_results_fpath.replace("_Test_", "_Val_").replace("_Test.", "_Val.")
        if os.path.exists(calibration_params_fpath):
            param_dict["calibration_params_fpath"] = calibration_params_fpath
            print("Calibration Params:", calibration_params_fpath)
        output_results_fpath = None

    param_dict["eval_datasplit_type"] = DataSplitType.from_name(args.data_split_type)
    param_dict["notebook_output_fpath"] = output_results_fpath
    print(output_fpath, "\n", output_config_fpath, "\n Data Split Evaluated:", args.data_split_type)
    pm.execute_notebook(args.notebook, output_fpath, parameters=param_dict)
    # python notebooks/evaluate_notebook.py --ckpt_dir=/group/jug/ashesh/training/denoisplit/2502/D16-M3-S0-L0/5 --data_dir=/group/jug/ashesh/data/BioSR/ --mmse_count=10 --MIXING_WEIGHT=0.1
    # python notebooks/evaluate_notebook.py --ckpt_dir=/group/jug/ashesh/training/disentangle/2507/D21-M3-S0-L0/12 --mmse_count=10 --data_dir=/group/jug/ashesh/data/diffsplit_HT_T24/ --notebook=/home/ashesh.ashesh/code/denoiSplit/notebooks/EvaluateRealInput.ipynb --custom_image_size=512 --image_size_for_grid_centers=256
    # parser = argparse.ArgumentParser(description='Run a notebook')
    # parser.add_argument('--notebook', type=str, help='Notebook to run', default='/home/ashesh.ashesh/code/denoiSplit/notebooks/EvaluateWithDifferentMixingWeights.ipynb')
    # parser.add_argument('--outputdir', type=str, help='Output notebook directory', default='/group/jug/ashesh/indiSplit/notebook_results_baselines/')
    # # parser.add_argument('parameters', type=str, help='Parameters for the notebook')
    # parser.add_argument('--ckpt_dir', type=str, help='Checkpoint to use. eg. /home/ashesh.ashesh/paper_models/Hagen/MitoVsAct/DeepLC/')
    # parser.add_argument('--data_dir', type=str, help='Data directory', default='/group/jug/ashesh/data/ventura_gigascience/')
    # parser.add_argument('--mmse_count', type=int, help='Number of mmse values to generate', default=50)
    # parser.add_argument('--MIXING_WEIGHT', type=float, help='Mixing parameter for input generation', default=0.5)
    # parser.add_argument('--image_size_for_grid_centers', type=int, help='Image size for grid centers', default=256)
    # parser.add_argument('--custom_image_size', type=int, help='Custom image size', default=512)
    # parser.add_argument('--batch_size', type=int, help='Batch size', default=8)
    # args = parser.parse_args()

    # # get a year-month-day hour-minute formatted string
    # param_str = f"T-{args.MIXING_WEIGHT}_MMSE-{args.mmse_count}"
    # now = datetime.now().strftime("%Y%m%d.%H.%M")
    # ckpt_dir = args.ckpt_dir
    # if ckpt_dir[-1] == '/':
    #     ckpt_dir = ckpt_dir[:-1]

    # outputdir = os.path.join(args.outputdir, '_'.join(ckpt_dir.split('/')[-3:]))
    # fname = os.path.basename(args.notebook)
    # fname = fname.replace('.ipynb','')
    # fname = f"denoiSplit_{fname}_{param_str}_{now}.ipynb"
    # output_fpath = os.path.join(outputdir, fname)
    # output_config_fpath = os.path.join(outputdir,'config', fname.replace('.ipynb','.txt'))
    # os.makedirs(os.path.dirname(output_config_fpath), exist_ok=True)
    # # save the configuration
    # # convert args to dict
    # args_dict = vars(args)
    # # save as json
    # with open(output_config_fpath, 'w') as f:
    #     f.write(str(args_dict))

    # print(output_fpath, '\n', output_config_fpath)
    # pm.execute_notebook(
    #     args.notebook,
    #     output_fpath,
    #     parameters = {
    #         'ckpt_dir':args.ckpt_dir,
    #         'data_dir':args.data_dir,
    #         'mmse_count':args.mmse_count,
    #         'MIXING_WEIGHT':args.MIXING_WEIGHT,
    #         'image_size_for_grid_centers':args.image_size_for_grid_centers,
    #         'custom_image_size':args.custom_image_size,
    #         'batch_size':args.batch_size
    #     }
    # )
