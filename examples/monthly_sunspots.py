"""Train DeSpaWN on the monthly sunspot series used by the original project.

This example is a PyTorch port of the published TensorFlow example at
https://github.com/MichauGabriel/DeSpaWN. It keeps data preparation, model
training, and plotting outside the despawn_pytorch library.

Run it from the repository root with:

    uv run --group examples --extra torch-cpu \
        python examples/monthly_sunspots.py

Use ``--epochs 1 --no-show`` for a quick check without opening a plot window.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import polars as pl
import torch

from despawn_pytorch import Despawn, DespawnLoss, get_num_levels

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from torch import Tensor

DATA_PATH = Path(__file__).parent / "data" / "monthly-sunspots.csv"
TRAIN_LENGTH = 2_000
DB4_KERNEL = [
    -0.010597401785069032,
    0.0328830116668852,
    0.030841381835560764,
    -0.18703481171909309,
    -0.027983769416859854,
    0.6308807679298589,
    0.7148465705529157,
    0.2303778133088965,
]

plt.style.use("bmh")


def positive_int(value: str) -> int:
    """Parse a positive integer for a command line argument."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=positive_int,
        default=1_000,
        help="number of training epochs (default: 1000)",
    )
    parser.add_argument(
        "--log-every",
        type=positive_int,
        default=10,
        help="print training loss every N epochs (default: 10)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="PyTorch device (choices: cpu or cuda, default: cuda when available)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional path where the figure is saved",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="do not open the Matplotlib window"
    )
    return parser.parse_args()


def prepare_plot_backend(*, no_show: bool) -> bool:
    """Select a usable backend and report whether the figure can be shown."""
    if no_show:
        plt.switch_backend("Agg")
        return False

    try:
        probe = plt.figure()
    except Exception as error:
        backend = plt.get_backend()
        print(
            f"Matplotlib backend {backend!r} is unavailable "
            f"({type(error).__name__}). Using the noninteractive Agg backend instead."
        )
        plt.switch_backend("Agg")
        return False

    plt.close(probe)
    return True


def load_signal(path: Path) -> Tensor:
    """Load and standardize the complete sunspot series."""
    sunspots = (
        pl.scan_csv(path)
        .select(
            (
                (pl.col("Sunspots") - pl.col("Sunspots").mean())
                / pl.col("Sunspots").std()
            )
            .cast(pl.Float32)
            .alias("Sunspots")
        )
        .collect()
        .get_column("Sunspots")
    )
    return torch.from_numpy(sunspots.to_numpy()).unsqueeze(0)


def train_model(
    signal: Tensor, *, epochs: int, log_every: int, device: torch.device
) -> Despawn:
    """Train a model with the settings from the original example."""
    model = Despawn(
        kernel_init=DB4_KERNEL,
        kernel_learnable=True,
        kernels_constraint="per_layer",
        n_levels=get_num_levels(signal.shape[-1]),
        loss_coeff="l1",
        threshold_init=0.3,
        threshold_learnable=True,
    ).to(device)
    criterion = DespawnLoss(sparsity_weight=1.0)
    optimizer = torch.optim.NAdam(
        model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-7
    )
    signal = signal.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        reconstruction, coefficient_penalty = model(signal)
        loss = criterion(signal, reconstruction, coefficient_penalty)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            reconstruction_loss = torch.mean(torch.abs(signal - reconstruction))
            sparsity_loss = coefficient_penalty.mean()
            print(
                f"epoch {epoch:4d}/{epochs}: loss={loss.item():.6f}, "
                f"reconstruction={reconstruction_loss.item():.6f}, "
                f"sparsity={sparsity_loss.item():.6f}"
            )

    return model


def plot_coefficient_distributions(
    ax: Axes, coefficients: list[Tensor], title: str
) -> None:
    """Plot one coefficient distribution for each decomposition level."""
    for level, coefficient in enumerate(coefficients):
        values = coefficient[0].detach().cpu().abs().flatten().numpy()
        if len(values) > 1:
            ax.boxplot(values, positions=[level], widths=0.8)
        else:
            ax.plot(level, values[0], "o", color="#2e3436", alpha=0.75)

    ax.set_title(title)
    ax.set_xlabel("Decomposition level")
    ax.set_ylabel("Absolute coefficient value")


def plot_results(
    complete_signal: Tensor,
    train_recon: Tensor,
    test_recon: Tensor,
    train_coefficients: list[Tensor],
    test_coefficients: list[Tensor],
) -> Figure:
    """Plot reconstruction and coefficient distributions."""
    signal = complete_signal[0].cpu()
    train_recon = train_recon[0].detach().cpu()
    test_recon = test_recon[0].detach().cpu()

    figure = plt.figure(figsize=(11, 8))
    recon_ax = figure.add_subplot(2, 1, 1)
    train_coeffs_ax = figure.add_subplot(2, 2, 3)
    test_coeffs_ax = figure.add_subplot(2, 2, 4)

    train_x = range(TRAIN_LENGTH)
    test_x = range(TRAIN_LENGTH, signal.numel())
    recon_ax.plot(train_x, signal[:TRAIN_LENGTH], label="Train original")
    recon_ax.plot(train_x, train_recon, label="Train reconstructed")
    recon_ax.plot(test_x, signal[TRAIN_LENGTH:], label="Test original")
    recon_ax.plot(test_x, test_recon, label="Test reconstructed")
    recon_ax.set_title("Monthly sunspots")
    recon_ax.set_xlabel("Month index")
    recon_ax.set_ylabel("Standardized sunspot count")
    recon_ax.legend(loc="upper left")

    # Set line transparency (alpha) for all plots in recon_ax
    for line in recon_ax.get_lines():
        line.set_alpha(0.75)
        line.set_linewidth(1.25)

    plot_coefficient_distributions(
        train_coeffs_ax, train_coefficients, "Train coefficients"
    )
    plot_coefficient_distributions(
        test_coeffs_ax, test_coefficients, "Test coefficients"
    )
    test_coeffs_ax.set_xlim(train_coeffs_ax.get_xlim())
    test_coeffs_ax.set_ylim(train_coeffs_ax.get_ylim())

    figure.tight_layout()
    return figure


def main() -> None:
    """Train the model and display its reconstruction and coefficients."""
    args = parse_args()
    show_figure = prepare_plot_backend(no_show=args.no_show)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    complete_signal = load_signal(DATA_PATH)
    train_signal = complete_signal[:, :TRAIN_LENGTH]
    test_signal = complete_signal[:, TRAIN_LENGTH:]
    model = train_model(
        train_signal, epochs=args.epochs, log_every=args.log_every, device=device
    )

    model.eval()
    with torch.no_grad():
        train_reconstruction, train_approximation, train_details = model.decompose(
            train_signal.to(device)
        )
        test_reconstruction, test_approximation, test_details = model.decompose(
            test_signal.to(device)
        )

    figure = plot_results(
        complete_signal,
        train_reconstruction,
        test_reconstruction,
        [train_approximation, *train_details],
        [test_approximation, *test_details],
    )
    output_path = args.output
    if not show_figure and not args.no_show and output_path is None:
        output_path = Path(tempfile.gettempdir()) / "despawn-monthly-sunspots.png"

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
        print(f"saved figure to {output_path}")
    if show_figure:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
