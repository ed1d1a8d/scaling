import cuml.neighbors.kneighbors_classifier
import numpy as np
import sklearn.neighbors

from src.pretrain.datasets.embedding import EmbeddingDataset


def run_experiment(
    ds: EmbeddingDataset,
    k: int,
    metric: str = "euclidean",
    use_gpu: bool = False,
    report_per_class_results: bool = False,
):
    param_dict = dict(
        k=k,
        metric=metric,
        use_gpu=use_gpu,
    )

    # Initialize kNN classifier
    if use_gpu:
        clf = cuml.neighbors.kneighbors_classifier.KNeighborsClassifier(
            n_neighbors=k,
            weights="uniform",
            metric=metric,
        )
    else:
        clf = sklearn.neighbors.KNeighborsClassifier(
            n_neighbors=k,
            weights="uniform",
            metric=metric,
        )

    # Fit the classifier to training data
    clf.fit(ds.xs_train, ds.ys_train)

    # Evaluate the classifier on the test data
    acc = clf.score(ds.xs_test, ds.ys_test)

    # Evaluate the classifier on the test data per class (if requested)
    per_class_accs: dict[str, float] = {}
    if report_per_class_results:
        for y in sorted(np.unique(ds.ys_test)):
            mask = ds.ys_test == y
            per_class_accs[f"acc_{y}"] = clf.score(  # type: ignore
                ds.xs_test[mask], ds.ys_test[mask]
            )

    return param_dict | dict(acc=acc) | per_class_accs
