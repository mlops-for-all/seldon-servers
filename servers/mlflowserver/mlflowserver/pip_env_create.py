#
#  MAKINAROCKS CONFIDENTIAL
#  ________________________
#
#  [2017] - [2021] MakinaRocks Co., Ltd.
#  All Rights Reserved.
#
#  NOTICE:  All information contained herein is, and remains
#  the property of MakinaRocks Co., Ltd. and its suppliers, if any.
#  The intellectual and technical concepts contained herein are
#  proprietary to MakinaRocks Co., Ltd. and its suppliers and may be
#  covered by U.S. and Foreign Patents, patents in process, and
#  are protected by trade secret or copyright law. Dissemination
#  of this information or reproduction of this material is
#  strictly forbidden unless prior written permission is obtained
#  from MakinaRocks Co., Ltd.
import argparse
import json
import logging
import os
import shutil
import tempfile
from typing import Any

import yaml
from pip._internal.operations import freeze
from seldon_core import Storage
from seldon_core.microservice import PARAMETERS_ENV_NAME, parse_parameters

log = logging.getLogger()
log.setLevel("INFO")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--parameters",
    type=str,
    default=os.environ.get(PARAMETERS_ENV_NAME, "[]"),
)

# This is already set on the environment_rest and environment_grpc files, but
# we'll define a default just in case.
DEFAULT_CONDA_ENV_NAME = "mlflow"
BASE_REQS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "requirements.txt",
)


def setup_env(model_folder: str) -> None:
    """Sets up a pip environment.

    This methods creates the pip environment described by the `MLmodel` file.

    Parameters
    --------
    model_folder : str
        Folder where the MLmodel files are stored.
    """
    mlmodel = read_mlmodel(model_folder)

    flavours = mlmodel["flavors"]
    pyfunc_flavour = flavours["python_function"]
    env_file_name = pyfunc_flavour["env"]
    env_file_path = os.path.join(model_folder, env_file_name)
    env_file_path = copy_env(env_file_path)
    copy_pip(env_file_path)
    install_base_reqs()


def read_mlmodel(model_folder: str) -> Any:
    """Reads an MLmodel file.

    Parameters
    ---------
    model_folder : str
        Folder where the MLmodel files are stored.

    Returns
    --------
    obj
        Dictionary with MLmodel contents.
    """
    log.info("Reading MLmodel file")
    mlmodel_path = os.path.join(model_folder, "MLmodel")
    return _read_yaml(mlmodel_path)


def _read_yaml(file_path: str) -> Any:
    """Reads a YAML file.

    Parameters
    ---------
    file_path
        Path to the YAML file.

    Returns
    -------
    dict
        Dictionary with YAML file contents.
    """
    with open(file_path, "r") as file_reader:
        return yaml.safe_load(file_reader)


def copy_env(env_file_path: str) -> str:
    """Copy conda.yaml to temp dir
    to prevent the case where the existing file is on Read-only file system.

    Parameters
    ----------
    env_file_path : str
        env file path to copy.

    Returns
    -------
    str
        tmp file directory.
    """
    temp_dir = tempfile.mkdtemp()
    new_env_path = os.path.join(temp_dir, "conda.yaml")
    shutil.copy2(env_file_path, new_env_path)

    return new_env_path


def copy_pip(new_env_path: str) -> None:
    """Copy pip packages from conda.yaml to requirements.txt

    Parameters
    ----------
    new_env_path : str
        requirements.txt path.
    """
    conda = _read_yaml(new_env_path)
    pip_packages = conda["dependencies"][-1]["pip"]
    freezed = freeze.freeze()
    freezed = map(lambda x: x.split("==")[0], freezed)
    package_to_install = []
    for package in pip_packages:
        name = package.split("==")[0]
        if name not in freezed:
            package_to_install += [package]

    with open(BASE_REQS_PATH, "a") as file_writer:
        file_writer.write("\n".join(package_to_install))


def install_base_reqs() -> None:
    """Install additional requirements from requirements.txt.
    If the variable is not defined, it falls back to `mlflow`.
    """
    log.info("Install additional package from requirements.txt")
    cmd = (
        "pip install"
        " --trusted-host devpi.makinarocks.ai"
        " --extra-index-url http://devpi.makinarocks.ai/root/dev/+simple/"
        f" -r {BASE_REQS_PATH}"
    )
    os.system(cmd)


def main(arguments: argparse.Namespace) -> None:
    """main algorithm.

    Parameters
    ----------
    arguments : argparse.Namespace
    """
    parameters = parse_parameters(json.loads(arguments.parameters))
    model_uri = parameters.get("model_uri", "/mnt/model/")

    log.info("Downloading model from %s", model_uri)
    model_folder = Storage.download(model_uri)
    setup_env(model_folder)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
