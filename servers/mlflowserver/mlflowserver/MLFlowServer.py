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
import json
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests
from serving_utils import load_model
from mlflow.types import Schema
from seldon_core import Storage
from seldon_core.user_model import SeldonComponent, SeldonResponse

# from sklearn import metrics as sklearn_metrics


MLFLOW_SERVER = "model"

logger = logging.getLogger()


class MLFlowServer(SeldonComponent):
    """Serving Model"""
    def __init__(
        self,
        model_uri: str,
        xtype: str = "ndarray",
        predict_method: str = "predict",
        metric_method: str = "",
        method_kwargs: str = "{}",
        ignore_input_schema: bool = False,
    ):
        super().__init__()

        logger.info(
            "__init__(model_uri='%s', xtype='%s', predict_method='%s', metric_method='%s')",
            model_uri,
            xtype,
            predict_method,
            metric_method,
        )

        self.model_uri = model_uri
        self.xtype = xtype
        self.predict_method = predict_method
        self.metric_method = metric_method
        self.method_kwargs = json.loads(method_kwargs)
        self.ignore_input_schema = ignore_input_schema

        self.ready = False
        self.load()

    def _forward(
        self,
        X: np.ndarray,
        predict_method: str,
        names: List[str] = [],
        meta: Optional[Dict[Any, Any]] = None,
    ) -> Any:
        # Union[np.ndarray, List, Dict, str, bytes]:
        logger.info(
            "model_uri=%s : _forward(X=%s, names=%s, predict_method='%s')",
            self.model_uri,
            X,
            names,
            predict_method,
        )

        if not self.ready:
            raise requests.HTTPError("Model not loaded yet")

        index = None
        if meta is not None:
            tags = meta.get("tags", {})
            if isinstance(tags, dict):
                index = tags.get("index", None)

        if predict_method is None:
            predict_method = self.predict_method

        if self.xtype == "ndarray":
            result = self._model.predict(
                X, predict_method=predict_method, **self.method_kwargs
            )
        else:
            if (names is not None) and (len(names) > 0):
                df = pd.DataFrame(data=X, columns=names, index=index)
            else:
                df = pd.DataFrame(data=X, index=index)
            result = self._model.predict(
                df, predict_method=predict_method, **self.method_kwargs
            )

        if isinstance(result, pd.DataFrame):
            result = result.values
        logger.info("model_uri=%s : _forward() : result=%s", self.model_uri, result)

        return result, {} if index is None else {"index": index}

    def _metric_method(self, method_name: Optional[str] = None) -> Any:
        available_metric_method: Dict[str, Any] = {
            # "accuracy": sklearn_metrics.accuracy_score,
            # "auroc": sklearn_metrics.roc_auc_score,
            # "f1_score": sklearn_metrics.f1_score,
            # "precision": sklearn_metrics.precision_score,
            # "recall": sklearn_metrics.recall_score,
        }

        if not method_name:
            return None
        else:
            metric_method = method_name.lower()
            if method_name not in available_metric_method.keys():
                return None

        return available_metric_method[metric_method]

    def class_names(self) -> List[str]:
        logger.info("model_uri=%s : class_name()", self.model_uri)

        output_schema = self._model.metadata.get_output_schema()
        if output_schema is not None and output_schema.has_input_names():
            columns = [schema["name"] for schema in output_schema.to_dict()]
            return columns

        return []

    def load(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        logger.info(
            "model_uri=%s : load() : Load model from '%s'",
            self.model_uri,
            self.model_uri,
        )

        model_folder = Storage.download(self.model_uri)
        self._model = load_model(
            model_folder,
            predict_method=self.predict_method,
            ignore_input_schema=self.ignore_input_schema,
        )
        self.ready = True

    def predict(
        self,
        X: np.ndarray,
        names: List[str] = [],
        meta: Optional[Dict[Any, Any]] = None,
    ) -> Any:
        logger.info(
            "model_uri=%s : predict(X=%s, names=%s, meta=%s)",
            self.model_uri,
            X,
            names,
            meta,
        )

        result, tags = self._forward(
            X=X,
            names=names,
            predict_method=self.predict_method,
            meta=meta,
        )
        result = result.reshape(X.shape[0], -1)

        logger.info("model_uri=%s : predict() : result=%s", self.model_uri, result)

        # If you want to use runtime metrics, define them here.
        runtime_metrics: List[Dict[str, Any]] = []

        return SeldonResponse(data=result, metrics=runtime_metrics, tags=tags)

    def transform_input(
        self,
        X: np.ndarray,
        names: List[str] = [],
        meta: Optional[Dict[Any, Any]] = None,
    ) -> Any:
        logger.info(
            "model_uri=%s : transform_input(X=%s, names=%s, meta=%s)",
            self.model_uri,
            X,
            names,
            meta,
        )

        result, tags = self._forward(
            X=X,
            names=names,
            predict_method="transform",
            meta=meta,
        )

        logger.info(
            "model_uri=%s : transform_input() : result=%s",
            self.model_uri,
            result,
        )

        return SeldonResponse(data=result, tags=tags)

    def transform_output(
        self,
        X: np.ndarray,
        names: List[str] = [],
        meta: Optional[Dict[Any, Any]] = None,
    ) -> Any:
        logger.info(
            "model_uri=%s : transform_output(X=%s, names=%s, meta=%s)",
            self.model_uri,
            X,
            names,
            meta,
        )

        result, tags = self._forward(
            X=X,
            names=names,
            predict_method="transform",
            meta=meta,
        )

        logger.info(
            "model_uri=%s : transform_output() : result=%s",
            self.model_uri,
            result,
        )

        return SeldonResponse(data=result, tags=tags)

    def metrics(self) -> List[Dict[str, Any]]:
        logger.info("model_uri=%s : metrics()", self.model_uri)

        # If you want to use runtime metrics, define them here.
        runtime_metrics: List[Dict[str, Any]] = []

        return runtime_metrics

    def send_feedback(
        self,
        features: Union[np.ndarray, str, bytes],
        feature_names: List[str],
        reward: float,
        truth: Union[np.ndarray, str, bytes],
        routing: Union[int, None],
    ) -> Union[np.ndarray, List, Dict, str, bytes, None, SeldonResponse]:  # type: ignore
        logger.info(
            "model_uri=%s : send_feedback(features=%s, feature_names=%s, reward=%s, truth=%s, routing=%s)",
            self.model_uri,
            features,
            feature_names,
            reward,
            truth,
            routing,
        )

        predicted: np.ndarray = self._forward(
            X=features,
            names=feature_names,
            predict_method="predict",
        )
        predicted = predicted.ravel()  # pylint: disable=no-member

        logger.info(
            "model_uri=%s : send_feedback() : predicted=%s",
            self.model_uri,
            predicted,
        )

        metric_function = self._metric_method(self.metric_method)
        if metric_function:
            score = metric_function(truth, predicted).item()

            logger.info(
                "model_uri=%s : send_feedback() : %s=%s",
                self.model_uri,
                self.metric_method.lower(),
                score,
            )

            # If you want to use runtime metrics, define them here.
            runtime_metrics: List[Dict[str, Any]] = []

            return SeldonResponse(data=predicted, metrics=runtime_metrics)

        return SeldonResponse(data=predicted)

    def aggregate(
        self,
        X: np.ndarray,
        names: List[str] = [],
        meta: Optional[Dict[Any, Any]] = None,
    ) -> Any:
        logger.info(
            "model_uri=%s : aggregate(X=%s, names=%s, meta=%s)",
            self.model_uri,
            X,
            names,
            meta,
        )

        if isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, np.ndarray) and X.ndim == 3:
            pass
        else:
            raise ValueError(f"Invalid Input for Combiner : {X}")

        result, tags = self._forward(
            X=X,
            names=names,
            predict_method="predict",
            meta=meta,
        )

        if result.ndim == 1:
            # ndim of result ndarray should be larger than 2 to return class names
            result = result.reshape(X.shape[1], -1)

        return SeldonResponse(data=result, tags=tags)

    def init_metadata(self) -> Any:
        def _parse_schema(schema: Schema) -> Dict[Any, Any]:
            ret = {}
            if schema.has_input_names():
                ret["names"] = schema.input_names()
            ret["shape"] = [len(schema.inputs)]
            return ret

        model_meta = self._model.metadata
        meta = {
            "name": model_meta.artifact_path,
            "versions": [
                f"run_id: {model_meta.run_id}",
                f"created_at: {model_meta.utc_time_created}",
            ],
            "platform": "seldon",
            "inputs": [
                {
                    # is it proper?
                    "messagetype": "tensor",
                    "schema": _parse_schema(model_meta.get_input_schema()),
                },
            ],
            "outputs": [
                {
                    # is it proper?
                    "messagetype": "tensor",
                    "schema": _parse_schema(model_meta.get_output_schema()),
                },
            ],
        }
        return meta
