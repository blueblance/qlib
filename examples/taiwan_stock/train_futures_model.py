import sqlite3
from pathlib import Path
import subprocess
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord

DB_PATH = Path(__file__).resolve().parent / "futures.db"
CSV_PATH = Path(__file__).resolve().parent / "futures.csv"
QLIB_DIR = Path.home() / ".qlib" / "qlib_data" / "my_futures"


def export_to_csv():
    """Read futures.db and export to CSV for dumping."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM futures_kbars", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
    df["symbol"] = "TW_FUT"
    df["factor"] = 1.0
    df.to_csv(CSV_PATH, index=False)


def dump_to_qlib():
    """Convert CSV data into Qlib format."""
    cmd = [
        "python",
        "scripts/dump_bin.py",
        "dump_all",
        "--csv_path",
        str(CSV_PATH),
        "--qlib_dir",
        str(QLIB_DIR),
        "--date_field_name",
        "date",
        "--symbol_field_name",
        "symbol",
        "--freq",
        "1min",
        "--include_fields",
        "open,high,low,close,volume,factor",
    ]
    subprocess.run(cmd, check=True)


def train_model():
    """Initialize qlib and train a high-frequency LightGBM model."""
    qlib.init(provider_uri=str(QLIB_DIR), region=REG_CN)

    data_handler_config = {
        "start_time": "2020-03-02 00:00:00",
        "end_time": "2025-04-16 23:59:00",
        "fit_start_time": "2020-03-02 00:00:00",
        "fit_end_time": "2024-12-31 23:59:00",
        "instruments": "TW_FUT",
        "freq": "1min",
        "infer_processors": [
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": False}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ],
        "label": ["Ref($close, -2)/Ref($close, -1) - 1"],
    }

    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": ("2020-03-02 00:00:00", "2024-10-31 23:59:00"),
                "valid": ("2024-11-01 00:00:00", "2025-02-28 23:59:00"),
                "test": ("2025-03-01 00:00:00", "2025-04-16 23:59:00"),
            },
        },
    }

    model_config = {
        "class": "HFLGBModel",
        "module_path": "qlib.contrib.model.highfreq_gdbt_model",
        "kwargs": {
            "loss": "binary",
            "learning_rate": 0.01,
            "max_depth": 8,
            "num_leaves": 150,
            "lambda_l1": 1.5,
            "lambda_l2": 1.0,
            "num_threads": 8,
        },
    }

    model = init_instance_by_config(model_config)
    dataset = init_instance_by_config(dataset_config)

    with R.start(experiment_name="futures_hf_train"):
        R.log_params(**flatten_dict({"model": model_config, "dataset": dataset_config}))
        model.fit(dataset)
        R.save_objects(model=model)
        recorder = R.get_recorder()
        SignalRecord(model, dataset, recorder).generate()
        SigAnaRecord(recorder).generate()


if __name__ == "__main__":
    export_to_csv()
    dump_to_qlib()
    train_model()

