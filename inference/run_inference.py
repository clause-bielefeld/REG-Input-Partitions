import sys
import os
from os.path import dirname, abspath, join, split
import torch
import argparse
import json
import pickle
from inference_utils import main_val_set_with_att, override_config_with_checkpoint

file_path = abspath(dirname(__file__))
sys.path.append(abspath(join(file_path, os.pardir, 'model')))

from configuration import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REG")

    parser.add_argument(
        "--split", type=str.lower, choices=["val", "testa", "testb"], default="val"
    )
    parser.add_argument("--path", type=str, help="path to image", default=None)
    parser.add_argument("--checkpoint", type=str, help="checkpoint path", default=None, required=True)
    parser.add_argument(
        "--device", type=str.lower, choices=["cuda", "cpu", "auto"], default="auto"
    )
    parser.add_argument("--print_samples", action="store_true")
    parser.add_argument("--override_config", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    config = Config()
    config.batch_size = 1  # override batch size

    if args.override_config:
        override_config_with_checkpoint(split(args.checkpoint)[-1], config)

    # decode dataset
    print(args)
    metrics, generated = main_val_set_with_att(args, config)

    print(metrics)

    assert args.checkpoint is not None
    model_name = os.path.split(args.checkpoint)[-1]
    outdir = os.path.abspath("../data/results")
    if not os.path.isdir(outdir):
        print(f"create output directory {outdir}")
        os.makedirs(outdir)

    # generated expressions
    outfile_name = model_name.replace(".pth", f"_{args.split}_generated.pkl")
    outfile_path = os.path.join(outdir, outfile_name)
    print(f"write generated expressions to {outfile_path}")
    with open(outfile_path, "wb") as f:
        pickle.dump(generated, f)

    # evaluation results
    outfile_name = model_name.replace(".pth", f"_{args.split}_eval.json")
    outfile_path = os.path.join(outdir, outfile_name)
    print(f"write evaluation results to {outfile_path}")
    with open(outfile_path, "w") as f:
        json.dump(metrics, f)