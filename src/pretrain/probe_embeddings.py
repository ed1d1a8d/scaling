"""
Analyze embeddings of a pre-trained models using probes.
"""

import dataclasses
from typing import Any

import cuml.decomposition
import cuml.manifold.umap
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects
import wandb
from simple_parsing import ArgumentParser, subgroups
from tqdm.auto import tqdm

from src.pretrain import gen_embeddings
from src.pretrain.datasets import BaseDatasetConfig, get_dataset_index
from src.pretrain.datasets.embedding import EmbeddingDataset
from src.pretrain.models import BaseEmbedderConfig, get_embedder_index
from src.pretrain.probes import knn_probe, linear_probe


@dataclasses.dataclass
class Config:
    """
    Configuration for analyzing embeddings.

    Analyses performed:

        1. Visualize embeddings.
            - Using PCA
            - Using UMAP

        2. For d in n_classes, study scaling on d-way
           classification. When d < total number of classes, use a contiguous
           subrange of classes. Choose up to mx_cont_rngs disjoint subranges
           for each d.

            - Look at data scaling for k-nearest neighbors. k in {1, 5, 10}
            - Look at data scaling for linear classification.
              C in {0.01, 1, 100}.

           Also enforce the same number of samples per-class, and just sampling
           completely randomly.
    """

    embedder_cfg: BaseEmbedderConfig = subgroups(get_embedder_index())
    dataset_cfg: BaseDatasetConfig = subgroups(get_dataset_index())

    n_classes: tuple[int, ...] = (2, 5, 10, 20, 50, 100)
    mx_cont_rngs: int = 3

    umap_min_dist: float = 0.1
    umap_n_neighbors: int = 10

    wandb_dir: str = "/home/gridsan/groups/ccg"
    tags: tuple[str, ...] = ("test",)

    @property
    def class_names(self):
        return self.dataset_cfg.class_names


def plot_class_frequencies(ds: EmbeddingDataset) -> matplotlib.figure.Figure:
    plt.figure()

    # Get frequencies of each class in the training set
    train_class_freqs = np.bincount(ds.ys_train)
    plt.bar(np.arange(len(train_class_freqs)), train_class_freqs, label="train")

    # Get frequencies of each class in the test set
    test_class_freqs = np.bincount(ds.ys_test)
    plt.bar(np.arange(len(test_class_freqs)), test_class_freqs, label="test")

    # Legend on outside top right
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.title("Class frequencies")
    plt.xlabel("Class")

    return plt.gcf()


def plot_pca(
    ds: EmbeddingDataset,
    cfg: Config,
) -> plotly.graph_objects.Figure:
    pca = cuml.decomposition.PCA(n_components=3, whiten=True)
    components = pca.fit_transform(ds.xs_test)

    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(3),
        color=[cfg.class_names[c] for c in ds.ys_test],
        width=700,
        height=600,
    )
    # Make marker size smaller
    fig.update_traces(
        diagonal_visible=False,
        marker=dict(opacity=0.8, size=3),
    )
    fig.update_layout(legend=dict(itemsizing="constant"))

    return fig


def plot_umap(
    ds: EmbeddingDataset,
    cfg: Config,
) -> plotly.graph_objects.Figure:

    xs_test_2d = cuml.manifold.umap.UMAP(
        n_components=2,
        min_dist=cfg.umap_min_dist,
        n_neighbors=cfg.umap_n_neighbors,
        random_state=0,
    ).fit_transform(ds.xs_test)

    df = pd.DataFrame(
        {
            "x": xs_test_2d[:, 0],
            "y": xs_test_2d[:, 1],
            "class": ds.ys_test,
            "class_name": [cfg.class_names[c] for c in ds.ys_test],
        }
    )

    fig = px.scatter(
        df.sort_values("class"),
        x="x",
        y="y",
        color="class_name",
        title=f"UMAP of test set (min_dist={cfg.umap_min_dist}, n_neighbors={cfg.umap_n_neighbors})",
        labels={"class_name": "Class"},
        width=800,
        height=500,
    )

    # Make alpha 0.5
    fig.update_traces(marker=dict(opacity=0.8))

    return fig


def measure_scaling(
    ds: EmbeddingDataset,
    per_class: bool,
    ks: tuple[int, ...] = (1, 3, 10),
    cs: tuple[float, ...] = (0.01, 1, 100),
    use_gpu: bool = True,
) -> list[dict[str, Any]]:
    def gen_n_trains():
        base = 1
        while True:
            for i in range(1, 10):
                yield base * i
            base *= 10

    results = []

    mx_n_train = ds.min_samples_per_class if per_class else len(ds.xs_train)
    for n_train in gen_n_trains():
        if n_train > mx_n_train:
            n_train = mx_n_train

        sub_ds = (
            ds.subsample_per_class(n_train_per_class=n_train)
            if per_class
            else ds.subsample(n_train=n_train)
        )

        # k-NN probe experiments
        for k in ks:
            if k >= len(sub_ds.xs_train):
                continue
            results.append(
                knn_probe.run_experiment(
                    ds=sub_ds,
                    k=k,
                    metric="euclidean",
                    use_gpu=use_gpu,
                )
                | dict(n_train=n_train, per_class=per_class, probe="knn")
            )

        # linear probe experiments
        for c in cs:
            results.append(
                linear_probe.run_experiment(
                    ds=sub_ds,
                    c=c,
                    use_gpu=use_gpu,
                )
                | dict(n_train=n_train, per_class=per_class, probe="linear")
            )

        if n_train == mx_n_train:
            break

    return results


def get_scaling_results(ds: EmbeddingDataset, cfg: Config) -> pd.DataFrame:
    data: list[dict[str, Any]] = []

    tot_classes = ds.n_classes
    for n_classes in tqdm(cfg.n_classes):
        for i in range(cfg.mx_cont_rngs):
            cls_start = i * n_classes
            cls_end = cls_start + n_classes
            if cls_end > tot_classes:
                break

            cds = ds.filter_classes(range(cls_start, cls_end))

            for per_class in [False, True]:
                data.extend(
                    [
                        d | dict(cls_start=cls_start, cls_end=cls_end)
                        for d in measure_scaling(ds=cds, per_class=per_class)
                    ]
                )

    return pd.DataFrame(data)


def plot_scaling_results(
    df: pd.DataFrame,
    n_classes: int,
    cls_start: int,
    cls_end: int,
    per_class: bool,
) -> plotly.graph_objects.Figure:
    cdf = (
        df.query(
            "&".join(
                [
                    f"cls_start == {cls_start}",
                    f"cls_end == {cls_end}",
                    f"per_class == {per_class}",
                ]
            )
        )
        .copy()
        .sort_values(by=["probe", "n_train"], ascending=[False, True])
    )

    min_err = 1 / cdf.n_train.max() / (n_classes if per_class else 1)
    cdf.loc[cdf.err == 0, "err"] = min_err / 10

    # Solid line for linear probe
    fig = px.line(
        cdf,
        x="n_train",
        y="err",
        color="hyper",
        line_dash="probe",
        title=f"# classes = {n_classes} ({cls_start} to {cls_end - 1})",
    )

    # Draw horizontal line for min nonzero error, behind all other lines
    # Make color black
    fig.add_scatter(
        x=[1, cdf.n_train.max()],
        y=[min_err, min_err],
        name="1 / max_n_train",
        line_dash="dash",
        line_color="black",
    )

    # Reorder lines
    fig.data = fig.data[-1:] + fig.data[:-1]  # type: ignore

    # Update axes
    fig.update_xaxes(
        title_text="Train samples / class"
        if per_class
        else "Total train samples",
        type="log",
    )
    fig.update_yaxes(
        type="log",
        range=[np.log10(min_err / 20), 0],
    )

    # Set height and width
    fig.update_layout(width=500, height=300)

    # Decrease margins
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    return fig


def main(cfg: Config):
    embedding_cfg = gen_embeddings.Config(
        dataset_cfg=cfg.dataset_cfg,
        embedder_cfg=cfg.embedder_cfg,
    )
    ds = EmbeddingDataset.load_from_file(embedding_cfg.full_save_path).astype(
        np.float32  # type: ignore
    )

    wandb.log({"class_frequencies": plot_class_frequencies(ds)})
    wandb.log({"pca": plot_pca(ds, cfg)})
    wandb.log({"umap": plot_umap(ds, cfg)})

    df = get_scaling_results(ds=ds, cfg=cfg)
    df["err"] = 1 - df.acc
    df["hyper"] = ""
    df.loc[df.probe == "linear", "hyper"] = "c=" + df[
        df.probe == "linear"
    ].c.astype(str)
    df.loc[df.probe == "knn", "hyper"] = "k=" + df[df.probe == "knn"].k.astype(
        str
    )

    # Save df to wandb
    wandb.log({"df_scaling": wandb.Table(dataframe=df)})

    # Plot scaling results
    tot_classes = ds.n_classes
    for n_classes in tqdm(cfg.n_classes):
        for i in range(cfg.mx_cont_rngs):
            for per_class in [False, True]:
                cls_start = i * n_classes
                cls_end = cls_start + n_classes
                if cls_end > tot_classes:
                    break

                fig = plot_scaling_results(
                    df=df,
                    n_classes=n_classes,
                    cls_start=cls_start,
                    cls_end=cls_end,
                    per_class=per_class,
                )

                tag: str = "_".join(
                    (
                        f"scaling_{n_classes}",
                        "per_class" if per_class else "uniform",
                        f"{i}",
                    )
                )
                wandb.log({tag: fig})


if __name__ == "__main__":
    # Parse config
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    cfg: Config = args.config

    # Initialize wandb
    wandb.init(
        entity="data-frugal-learning",
        project="probe-embeddings",
        dir=cfg.wandb_dir,
        tags=cfg.tags,
        config=dataclasses.asdict(cfg),
        save_code=True,
    )

    # Run main logic
    main(cfg)
