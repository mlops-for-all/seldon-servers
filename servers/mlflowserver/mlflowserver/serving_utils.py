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

import importlib
import os
from typing import Any, Dict, List, Optional, Union

import mlflow
import numpy as np
import pandas
import yaml
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.pyfunc import (
    _enforce_schema,
    _warn_potentially_incompatible_py_version_if_necessary,
)
from mlflow.tracking.artifact_utils import _download_artifact_from_uri

FLAVOR_NAME = "python_function"
MAIN = "loader_module"
CODE = "code"
DATA = "data"
ENV = "env"
PY_VERSION = "python_version"

PyFuncInput = Union[pandas.DataFrame, np.ndarray, List[Any], Dict[str, Any]]
PyFuncOutput = Union[pandas.DataFrame, pandas.Series, np.ndarray, list]


class PyFuncModel:
    """
    MLflow 'python function' model.

    Wrapper around model implementation and metadata. This class is not meant to be constructed
    directly. Instead, instances of this class are constructed and returned from
    :py:func:`load_model() <mlflow.pyfunc.load_model>`.

    ``model_impl`` can be any Python object that implements the `Pyfunc interface
    <https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#pyfunc-inference-api>`_, and is
    returned by invoking the model's ``loader_module``.

    ``model_meta`` contains model metadata loaded from the MLmodel file.
    """

    # Customized from mlflow==1.16.0
    def __init__(
        self,
        model_meta: Model,
        model_impl: Any,
        predict_method: str,
        ignore_input_schema: bool,
    ):
        if not model_meta:
            raise MlflowException("Model is missing metadata.")
        self._model_meta = model_meta
        self._model_impl = model_impl
        self._predict_method = predict_method
        self._check_input_schema = not ignore_input_schema

    def _check_schema(self, data: PyFuncInput) -> PyFuncInput:
        input_schema = self.metadata.get_input_schema()
        if input_schema is not None:
            data = _enforce_schema(data, input_schema)
        return data

    # Customized from mlflow==1.16.0
    def predict(
        self, data: PyFuncInput, predict_method: Optional[str] = None, **kwargs: Any
    ) -> PyFuncOutput:
        """
        Generate model predictions.

        If the model contains signature, enforce the input schema first before calling the model
        implementation with the sanitized input. If the pyfunc model does not include model schema,
        the input is passed to the model implementation as is. See `Model Signature Enforcement
        <https://www.mlflow.org/docs/latest/models.html#signature-enforcement>`_ for more details."

        :param data: Model input as one of pandas.DataFrame, numpy.ndarray, or
                     Dict[str, numpy.ndarray]
        :return: Model predictions as one of pandas.DataFrame, pandas.Series, numpy.ndarray or list.
        """
        if self._check_input_schema:
            data = self._check_schema(data)
        if predict_method is None:
            predict_method = self._predict_method
        result = getattr(self._model_impl, predict_method)(data, **kwargs)
        return result

    @property
    def metadata(self) -> Model:
        """Model metadata."""
        if self._model_meta is None:
            raise MlflowException("Model is missing metadata.")
        return self._model_meta

    def __repr__(self) -> Any:
        info = {}
        if self._model_meta is not None:
            if (
                hasattr(self._model_meta, "run_id")
                and self._model_meta.run_id is not None
            ):
                info["run_id"] = self._model_meta.run_id
            if (
                hasattr(self._model_meta, "artifact_path")
                and self._model_meta.artifact_path is not None
            ):
                info["artifact_path"] = self._model_meta.artifact_path

            info["flavor"] = self._model_meta.flavors[FLAVOR_NAME]["loader_module"]

        return yaml.safe_dump(
            {"mlflow.pyfunc.loaded_model": info},
            default_flow_style=False,
        )


def load_model(
    model_uri: str,
    predict_method: str,
    suppress_warnings: bool = True,
    ignore_input_schema: bool = False,
) -> PyFuncModel:
    """
    Load a model stored in Python function format.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.
    :param suppress_warnings: If ``True``, non-fatal warning messages associated with the model
                              loading process will be suppressed. If ``False``, these warning
                              messages will be emitted.
    """
    local_path = _download_artifact_from_uri(artifact_uri=model_uri)
    model_meta = Model.load(os.path.join(local_path, MLMODEL_FILE_NAME))

    conf = model_meta.flavors.get(FLAVOR_NAME)
    if conf is None:
        raise MlflowException(
            f'Model does not have the "{FLAVOR_NAME}" flavor',
            RESOURCE_DOES_NOT_EXIST,
        )
    model_py_version = conf.get(PY_VERSION)
    if not suppress_warnings:
        _warn_potentially_incompatible_py_version_if_necessary(
            model_py_version=model_py_version,
        )
    if CODE in conf and conf[CODE]:
        code_path = os.path.join(local_path, conf[CODE])
        mlflow.pyfunc.utils._add_code_to_system_path(code_path=code_path)
    data_path = os.path.join(local_path, conf[DATA]) if (DATA in conf) else local_path
    model_impl = importlib.import_module(conf[MAIN])._load_pyfunc(data_path)  # type: ignore
    return PyFuncModel(
        model_meta=model_meta,
        model_impl=model_impl,
        predict_method=predict_method,
        ignore_input_schema=ignore_input_schema,
    )
