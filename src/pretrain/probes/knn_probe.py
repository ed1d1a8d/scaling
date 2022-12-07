import cuml.neighbors.kneighbors_classifier
import sklearn.neighbors

from src.pretrain.datasets.embedding import EmbeddingDataset


def run_experiment(
    ds: EmbeddingDataset,
    k: int,
    metric: str = "euclidean",
    use_gpu: bool = False,
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

    return param_dict | dict(acc=acc)
