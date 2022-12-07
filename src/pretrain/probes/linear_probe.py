import cuml.linear_model
import numpy as np
import scipy.optimize
import scipy.special
import sklearn.linear_model
import sklearn.preprocessing

from src.pretrain.datasets.embedding import EmbeddingDataset


def get_min_xent(
    probs_train: np.ndarray,
    labs_train: np.ndarray,
    probs_test: np.ndarray,
    labs_test: np.ndarray,
) -> float:
    logits_train = np.log(probs_train)
    logits_test = np.log(probs_test)

    def get_xent_train(alpha: float) -> float:
        alogits = alpha * logits_train
        aprobs = scipy.special.softmax(alogits, axis=1)
        return -np.log(aprobs[np.arange(len(aprobs)), labs_train]).mean()

    def get_xent_test(alpha: float) -> float:
        alogits = alpha * logits_test
        aprobs = scipy.special.softmax(alogits, axis=1)
        return -np.log(aprobs[np.arange(len(aprobs)), labs_test]).mean()

    opt_res: scipy.optimize.OptimizeResult = scipy.optimize.minimize_scalar(
        get_xent_train, bounds=(0.01, 100)
    )

    return get_xent_test(opt_res.x)


def run_experiment(
    ds: EmbeddingDataset,
    c: float,
    max_iter: int = 10000,
    seed: int = 0,
    use_gpu: bool = False,
):
    param_dict = dict(
        c=c,
        max_iter=max_iter,
        seed=seed,
        use_gpu=use_gpu,
    )

    # Remap labels to 0, 1, 2, ... with sklearn
    train_encoder = sklearn.preprocessing.LabelEncoder()
    train_encoder.fit(ds.ys_train)

    # Handle case where we only have one class
    # linear_model.LogisticRegression cannot handle this case.
    if ds.ys_train.min() == ds.ys_train.max():
        acc = (ds.ys_test == ds.ys_train[0]).mean()
        return param_dict | dict(acc=acc, xent=np.infty)

    # Initialize logistic regression classifier
    if use_gpu:
        clf = cuml.linear_model.LogisticRegression(
            max_iter=max_iter,
            **({"C": c, "penalty": "l2"} if c > 0 else {"penalty": "none"}),
        )
    else:
        clf = sklearn.linear_model.LogisticRegression(
            random_state=seed,
            max_iter=max_iter,
            multi_class="multinomial",
            **({"C": c, "penalty": "l2"} if c > 0 else {"penalty": "none"}),
        )

    # Fit the classifier to the training data
    clf.fit(ds.xs_train, train_encoder.transform(ds.ys_train))

    # Evaluate the classifier on the test data
    pred_test: np.ndarray = train_encoder.inverse_transform(
        clf.predict(ds.xs_test),
    )  # type: ignore
    acc = (pred_test == ds.ys_test).mean()

    # Assert clf.classes_ is sorted
    assert np.all(clf.classes_ == np.arange(len(clf.classes_)))

    # Compute cross entropy
    xent = (
        get_min_xent(
            probs_train=clf.predict_proba(ds.xs_train),
            labs_train=train_encoder.transform(ds.ys_train),  # type: ignore
            probs_test=clf.predict_proba(ds.xs_test),
            labs_test=train_encoder.transform(ds.ys_test),  # type: ignore
        )
        if np.isin(ds.ys_test, clf.classes_).all()
        else np.infty
    )

    return param_dict | dict(acc=acc, xent=xent)
