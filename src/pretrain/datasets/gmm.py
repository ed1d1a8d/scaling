"""Support for working with Gaussian mixture models."""

from __future__ import annotations
import dataclasses

import torch
import numpy as np
from src.pretrain.datasets.embedding import EmbeddingDataset
import cuml.neighbors.kneighbors_classifier


@dataclasses.dataclass
class GMMDataset:
    ds: EmbeddingDataset

    idx_to_y: np.ndarray
    mus: np.ndarray
    covs: np.ndarray
    cov_eps: float

    # label_flip_probs[i, j] is the probability i flips to j
    # this is used for to estimate optimal error-rate later
    with_label_noise: bool
    label_flip_probs: np.ndarray

    @classmethod
    def construct_proxy(
        cls,
        ds: EmbeddingDataset,
        with_label_noise: bool,
        cov_eps: float = 1e-9,
    ) -> GMMDataset:
        xs = np.concatenate([ds.xs_train, ds.xs_test])
        ys = np.concatenate([ds.ys_train, ds.ys_test])

        idx_to_y = np.unique(ys)
        n_classes = len(idx_to_y)
        d = xs.shape[1]

        xs_train = ds.xs_train
        ys_train = ds.ys_train
        xs_test = ds.xs_test
        ys_test = ds.ys_test

        mus = np.zeros((n_classes, d))
        covs = np.zeros((n_classes, d, d))
        label_flip_probs = np.eye(d)

        for i, y in enumerate(idx_to_y):
            # Compute sample mean and covariance for class y
            mask = ys == y
            mu = xs[mask].mean(axis=0)
            cov = np.cov(xs[mask].T) + cov_eps * np.eye(d)

            # Save mu and cov
            mus[i] = mu
            covs[i] = cov

            # Sample from the Gaussian and save to ret_ds
            mask_train = ys_train == y
            mask_test = ys_test == y
            xs_train[mask_train] = np.random.multivariate_normal(
                mean=mu, cov=cov, size=mask_train.sum()
            )
            xs_test[mask_test] = np.random.multivariate_normal(
                mean=mu, cov=cov, size=mask_test.sum()
            )

        if with_label_noise:
            # Fit a 3-nearest neighbor classifier to (xs, ys)
            clf = cuml.neighbors.kneighbors_classifier.KNeighborsClassifier(
                n_neighbors=3,
                weights="uniform",
                metric="euclidean",
            )
            clf.fit(xs, ys)

            # Get knn predictions
            knn_preds = clf.predict(xs)
            knn_preds_train = knn_preds[: len(ys_train)]
            knn_preds_test = knn_preds[len(ys_train) :]

            # Resample points in ret_ds where we have a knn prediction mismatch
            for i, y in enumerate(idx_to_y):
                mask_train = (knn_preds_train == y) & (ys_train != y)
                mask_test = (knn_preds_test == y) & (ys_test != y)

                xs_train[mask_train] = np.random.multivariate_normal(
                    mean=mus[i], cov=covs[i], size=mask_train.sum()
                )
                xs_test[mask_test] = np.random.multivariate_normal(
                    mean=mus[i], cov=covs[i], size=mask_test.sum()
                )

            # Compute flip probabilities
            # For estimating optimal error later
            for i, y_true in enumerate(idx_to_y):
                denom = (knn_preds == y_true).sum()
                for j, y_lab in enumerate(idx_to_y):
                    label_flip_probs[i, j] = (
                        (knn_preds == y_true) & (ys == y_lab)
                    ).sum() / denom

        return cls(
            ds=dataclasses.replace(
                ds,
                xs_train=xs_train,
                ys_train=ys_train,
                xs_test=xs_test,
                ys_test=ys_test,
            ),
            idx_to_y=idx_to_y,
            mus=mus,
            covs=covs,
            cov_eps=cov_eps,
            with_label_noise=with_label_noise,
            label_flip_probs=label_flip_probs,
        )

    @classmethod
    def construct_proxy_gpu(
        cls,
        ds: EmbeddingDataset,
        with_label_noise: bool,
        cov_eps: float = 1e-9,
    ) -> GMMDataset:
        xs = torch.from_numpy(np.concatenate([ds.xs_train, ds.xs_test])).cuda()
        ys = torch.from_numpy(np.concatenate([ds.ys_train, ds.ys_test])).cuda()

        idx_to_y = torch.unique(ys)
        n_classes = len(torch.unique(ys))
        d = xs.shape[1]

        xs_train = torch.from_numpy(ds.xs_train).cuda()
        ys_train = torch.from_numpy(ds.ys_train).cuda()
        xs_test = torch.from_numpy(ds.xs_test).cuda()
        ys_test = torch.from_numpy(ds.ys_test).cuda()

        mus = torch.zeros((n_classes, d), device="cuda")
        covs = torch.zeros((n_classes, d, d), device="cuda")
        label_flip_probs = torch.eye(d, device="cuda")

        for i, y in enumerate(idx_to_y):
            # Compute sample mean and covariance for class y
            mask = ys == y
            mu = xs[mask].mean(dim=0)
            cov = torch.cov(xs[mask].T) + cov_eps * torch.eye(d, device="cuda")

            # Save mu and cov
            mus[i] = mu
            covs[i] = cov

            # Sample from the Gaussian and save to ret_ds
            mask_train = ys_train == y
            mask_test = ys_test == y
            xs_train[
                mask_train
            ] = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=mu, covariance_matrix=cov
            ).sample(
                (mask_train.sum(),)  # type: ignore
            )
            xs_test[
                mask_test
            ] = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=mu, covariance_matrix=cov
            ).sample(
                (mask_test.sum(),)  # type: ignore
            )

        if with_label_noise:
            # Fit a 3-nearest neighbor classifier to (xs, ys)
            clf = cuml.neighbors.kneighbors_classifier.KNeighborsClassifier(
                n_neighbors=3,
                weights="uniform",
                metric="euclidean",
            )
            clf.fit(xs, ys)

            # Get knn predictions
            knn_preds = torch.as_tensor(clf.predict(xs), device="cuda")
            knn_preds_train = knn_preds[: len(ys_train)]
            knn_preds_test = knn_preds[len(ys_train) :]

            # Resample points in ret_ds where we have a knn prediction mismatch
            for i, y in enumerate(idx_to_y):
                mask_train = (knn_preds_train == y) & (ys_train != y)
                mask_test = (knn_preds_test == y) & (ys_test != y)

                xs_train[
                    mask_train
                ] = torch.distributions.multivariate_normal.MultivariateNormal(
                    loc=mus[i], covariance_matrix=covs[i]
                ).sample(
                    (mask_train.sum(),)  # type: ignore
                )
                xs_test[
                    mask_test
                ] = torch.distributions.multivariate_normal.MultivariateNormal(
                    loc=mus[i], covariance_matrix=covs[i]
                ).sample(
                    (mask_test.sum(),)  # type: ignore
                )

            # Compute flip probabilities
            # For estimating optimal error later
            for i, y_true in enumerate(idx_to_y):
                denom = (knn_preds == y_true).sum()
                for j, y_lab in enumerate(idx_to_y):
                    label_flip_probs[i, j] = (
                        (knn_preds == y_true) & (ys == y_lab)
                    ).sum() / denom

        return cls(
            ds=dataclasses.replace(
                ds,
                xs_train=xs_train.cpu().numpy(),
                ys_train=ys_train.cpu().numpy(),
                xs_test=xs_test.cpu().numpy(),
                ys_test=ys_test.cpu().numpy(),
            ),
            idx_to_y=idx_to_y.cpu().numpy(),
            mus=mus.cpu().numpy(),
            covs=covs.cpu().numpy(),
            cov_eps=cov_eps,
            with_label_noise=with_label_noise,
            label_flip_probs=label_flip_probs.cpu().numpy(),
        )
