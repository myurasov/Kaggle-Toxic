import os
import shutil
from datetime import datetime
from pprint import pformat

from src.config import config


# tensorboard log dir
def create_tensorboard_run_dir(run):
    tb_log_dir = f"/app/.tensorboard/{run}"
    shutil.rmtree(tb_log_dir, ignore_errors=True)
    return tb_log_dir


# save trained model
def save_trained_model(model, run, info):

    # model
    output_dir = config["DATA_DIR"] + "/saved_models"
    model_path = output_dir + f"/model.{run}.tf"
    os.makedirs(output_dir, exist_ok=True)
    model.save(model_path, overwrite=True, save_format="tf", include_optimizer=True)
    print(f"* Model saved to {model_path}")

    # info
    info_path = model_path + "_info.txt"
    print(
        f"Date:\n\n{datetime.now()} UTC\n\nArguments:\n\n{pformat(info)}",
        file=open(info_path, "w"),
    )
