import tempfile
import os
from typing import Optional

import flax.serialization
import mlflow


def mlflow_log_jax(
    pytree,
    artifact_name: str,
    artifact_path: Optional[str] = None,
):
    b = flax.serialization.msgpack_serialize(pytree)
    with tempfile.TemporaryDirectory() as td:
        fpath = os.path.join(td, artifact_path or "", artifact_name)
        with open(fpath, "wb") as f:
            f.write(b)

        mlflow.log_artifacts(local_dir=td)
