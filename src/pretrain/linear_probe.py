import numpy as np
import scipy.optimize
import scipy.special
import sklearn.linear_model

import dataclasses


@dataclasses.dataclass
class Dataset:
    xs_train: np.ndarray
    ys_train: np.ndarray
    xs_test: np.ndarray
    ys_test: np.ndarray

    name: str = "unnamed"


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
    ds: Dataset,
    n_train: int,
    n_classes: int,
    C: float,
    max_iter: int,
    seed: int = 0,
    debug: bool = False,
):
    param_dict = {
        "n_classes": n_classes,
        "C": C,
        "max_iter": max_iter,
        "seed": seed,
    }

    # Get first n_classes classes
    classes = np.arange(n_classes)

    # Subset train data to chosen classes.
    mask_train = np.isin(ds.ys_train, classes)
    xs_train = ds.xs_train[mask_train]
    ys_train = ds.ys_train[mask_train]

    # Subset test data to chosen classes.
    mask_test = np.isin(ds.ys_test, classes)
    xs_test = ds.xs_test[mask_test]
    ys_test = ds.ys_test[mask_test]

    # Possibly reduce n_train if there are not enough train samples.
    n_train = min(n_train, len(xs_train))
    param_dict["n_train"] = n_train

    # Subset train data to size n_train
    idx = np.random.choice(len(xs_train), n_train, replace=False)
    xs_train = xs_train[idx]
    ys_train = ys_train[idx]

    # Handle case where we only have one class
    # sklearn.linear_model.LogisticRegression cannot handle this case.
    if ys_train.min() == ys_train.max():
        acc = (ys_test == ys_train[0]).mean()
        return param_dict | {"acc": acc, "xent": np.infty}

    # Train a logistic regression classifier on the training data
    clf = sklearn.linear_model.LogisticRegression(
        random_state=seed,
        max_iter=max_iter,
        multi_class="multinomial",
        **({"C": C, "penalty": "l2"} if C > 0 else {"penalty": "none"}),
    )
    clf.fit(xs_train, ys_train)

    # Evaluate the classifier on the test data
    acc = clf.score(xs_test, ys_test)

    # Compute cross-entropy loss on the test data
    ys_train_reordered = ys_train.copy()
    ys_test_reordered = ys_test.copy()
    for i, cls in enumerate(clf.classes_):
        ys_train_reordered[ys_train == cls] = i
        ys_test_reordered[ys_test == cls] = i

    xent = (
        get_min_xent(
            probs_train=clf.predict_proba(xs_train),
            labs_train=ys_train_reordered,
            probs_test=clf.predict_proba(xs_test),
            labs_test=ys_test_reordered,
        )
        if np.isin(ys_test, clf.classes_).all()
        else np.infty
    )

    ret_dict = param_dict | {"acc": acc, "xent": xent}
    if debug:
        ret_dict["probs"] = clf.predict_proba(xs_test)
        ret_dict["ys_test_reordered"] = ys_test_reordered

    return ret_dict
