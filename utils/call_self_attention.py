from urllib.parse import urlparse

import allrank.models.losses as losses
import numpy as np
import os
import torch
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset_role
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device, CustomDataParallel
# from allrank.training.train_utils import fit
from allrank.utils.command_executor import execute_command
from allrank.utils.experiments import dump_experiment_result, assert_expected_metrics
from allrank.utils.file_utils import create_output_dirs, PathsContainer, copy_local_to_gs
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import dummy_context_mgr
from argparse import ArgumentParser, Namespace
from attr import asdict
from functools import partial
from pprint import pformat
from torch import optim
from torch.utils.data import DataLoader

from utils.train_self_attention import fit


class temp:
    def __init__(self):
        self.job_dir = ""
        self.config_file_name = ""
        self.run_id = ""

def parse_args() -> Namespace:
    parser = ArgumentParser("allRank")
    parser.add_argument("--job-dir", help="Base output path for all experiments", required=False)
    parser.add_argument("--run-id", help="Name of this run to be recorded (must be unique within output dir)",
                        required=False)
    parser.add_argument("--config-file-name", required=False, type=str, help="Name of json file with config")

    return parser.parse_args()


def load_libsvm_dataset(input_path: str, slate_length: int, validation_ds_role: str):
    """
    Helper function loading a train LibSVMDataset and a specified validation LibSVMDataset.
    :param input_path: directory containing the LibSVM files
    :param slate_length: target slate length of the training dataset
    :param validation_ds_role: dataset role used for valdation (file name without an extension)
    :return: tuple of LibSVMDatasets containing train and validation datasets,
        where train slates are padded to slate_length and validation slates to val_ds.longest_query_length
    """
    train_ds = load_libsvm_dataset_role("train", input_path, slate_length)

    val_ds = load_libsvm_dataset_role(validation_ds_role, input_path, slate_length)

    test_ds = load_libsvm_dataset_role(validation_ds_role.replace("vali", "test"), input_path, slate_length)

    return train_ds, val_ds, test_ds

def create_data_loaders(train_ds, val_ds, test_ds, num_workers: int, batch_size: int):
    """
    Helper function creating train and validation data loaders with specified number of workers and batch sizes.
    :param train_ds: LibSVMDataset train dataset
    :param val_ds: LibSVMDataset validation dataset
    :param num_workers: number of data loader workers
    :param batch_size: size of the batches returned by the data loaders
    :return: tuple containing train and validation DataLoader objects
    """
    # We are multiplying the batch size by the processing units count
    gpu_count = torch.cuda.device_count()
    total_batch_size = max(1, gpu_count) * batch_size

    # Please note that the batch size for validation dataloader is twice the total_batch_size
    train_dl = DataLoader(train_ds, batch_size=total_batch_size, num_workers=num_workers, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=total_batch_size * 2, num_workers=num_workers, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=total_batch_size * 2, num_workers=num_workers, shuffle=False)
    return train_dl, val_dl, test_dl

def run(run_id=None, job_dir=None, config_file_name=None):
    # reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    # args = parse_args()
    args = temp()
    if job_dir is None:
        args.job_dir = "results-new/"
    else:
        args.job_dir = job_dir
    if run_id is None:
        args.run_id = "0/"
    else:
        args.run_id = run_id
    if config_file_name is None:
        args.config_file_name = "local.json"
    else:
        args.config_file_name = config_file_name

    paths = PathsContainer.from_args(args.job_dir, args.run_id, args.config_file_name)

    create_output_dirs(paths.output_dir)

    logger = init_logger(paths.output_dir)
    logger.info(f"created paths container {paths}")

    # read config
    config = Config.from_json(paths.config_path)
    logger.info("Config:\n {}".format(pformat(vars(config), width=1)))

    output_config_path = os.path.join(paths.output_dir, "used_config.json")
    # execute_command("cp {} {}".format(paths.config_path, output_config_path))
    if os.name == 'nt':
        execute_command("copy \"{}\" \"{}\"".format(paths.config_path, output_config_path))
    else:
        execute_command("cp {} {}".format(paths.config_path, output_config_path))

    # train_ds, val_ds
    train_ds, val_ds, test_ds = load_libsvm_dataset(
        input_path=config.data.path,
        slate_length=config.data.slate_length,
        validation_ds_role=config.data.validation_ds_role,
    )

    n_features = train_ds.shape[-1]
    assert n_features == val_ds.shape[-1], "Last dimensions of train_ds and val_ds do not match!"

    train_dl, val_dl, test_dl = create_data_loaders(
        train_ds, val_ds, test_ds, num_workers=config.data.num_workers, batch_size=config.data.batch_size)

    # gpu support
    dev = get_torch_device()
    logger.info("Model training will execute on {}".format(dev.type))

    # instantiate model
    model = make_model(n_features=n_features, **asdict(config.model, recurse=False))
    if torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
        logger.info("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
    model.to(dev)

    # load optimizer, loss and LR scheduler
    optimizer = getattr(optim, config.optimizer.name)(params=model.parameters(), **config.optimizer.args)
    loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)
    if config.lr_scheduler.name:
        scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.name)(optimizer, **config.lr_scheduler.args)
    else:
        scheduler = None

    with torch.autograd.detect_anomaly() if config.detect_anomaly else dummy_context_mgr():  # type: ignore
        # run training
        model, result = fit(
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=val_dl,
            config=config,
            device=dev,
            output_dir=paths.output_dir,
            tensorboard_output_path=paths.tensorboard_output_path,
            **asdict(config.training)
        )

    # dump_experiment_result(args, config, paths.output_dir, result)

    # if urlparse(args.job_dir).scheme == "gs":
    #     copy_local_to_gs(paths.local_base_output_path, args.job_dir)

    # assert_expected_metrics(result, config.expected_metrics)
    return model, dev

