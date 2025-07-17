import os
from typing import Callable
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch import Tensor
from matplotlib.patches import Patch
import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import logging

# Set up logger
logger = logging.getLogger(__name__)

SAVE_DIR = "local/logs/train"

# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------

def set_visualization_save_dir(save_dir: str) -> None:
    """Set the global save directory for visualizations.

    Parameters
    ----------
    save_dir : str
        Directory path where visualization files will be saved
    """
    global SAVE_DIR
    SAVE_DIR = save_dir


def visualize_tokens(
    module: L.LightningModule,
    logits: Tensor,
    labels: Tensor,
    masks: Tensor,
    prefix: str,
    sample_index: int | None = None,
    sample_name: str | None = None,
    batch_idx: int | None = None,
) -> None:
    attempt_visualization(visualize_token_predictions)(
        module,
        logits=logits,
        labels=labels,
        masks=masks,
        prefix=prefix,
        sample_index=sample_index,
        sample_name=sample_name,
        batch_idx=batch_idx,
    )


def visualize_entities(
    module: L.LightningModule,
    true_entity_labels: np.ndarray,
    pred_entity_labels: np.ndarray,
    pred_entity_logits: np.ndarray,
    true_token_masks: np.ndarray,
    eval_intervals: pd.DataFrame,
    prefix: str,
    sample_index: int | None = None,
    sample_name: str | None = None,
    batch_name: str | None = None,
    batch_idx: int | None = None,
) -> None:
    attempt_visualization(visualize_entity_predictions)(
        module,
        true_entity_labels=true_entity_labels,
        pred_entity_labels=pred_entity_labels,
        pred_entity_logits=pred_entity_logits,
        true_token_masks=true_token_masks,
        prefix=prefix,
        sample_index=sample_index,
        sample_name=sample_name,
        batch_idx=batch_idx,
    )
    attempt_visualization(visualize_entity_intervals)(
        module,
        entity_intervals=eval_intervals,
        true_entity_labels=true_entity_labels,
        true_token_masks=true_token_masks,
        prefix=prefix,
        sample_index=sample_index,
        sample_name=sample_name,
        batch_idx=batch_idx,
    )
    attempt_visualization(visualize_entity_interval_performance)(
        module,
        entity_intervals=eval_intervals,
        prefix=prefix,
        batch_name=batch_name,
        batch_idx=batch_idx,
    )


def attempt_visualization(fn):
    @rank_zero_only
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            import traceback

            logger.warning(
                f"Non-critical visualization error occurred in {fn.__name__}: {e}"
            )
            logger.warning("Full traceback:")
            traceback.print_exc()
            return None

    return wrapper


def _resolve_sample_info(
    sample_index: int | None,
    sample_name: str | None,
    default_index: Callable[[], int] | None = None,
) -> tuple[int, str]:
    if sample_index is None:
        sample_index = default_index()
    if sample_name is None:
        sample_name = f"batch[{sample_index}]"
    return sample_index, sample_name


def get_visualization_path(
    prefix: str,
    visualization_type: str,
    global_step: int,
    batch_idx: int | None = None,
    extension: str = "png",
) -> str:
    """Generate a standard path for saving visualization files.

    Parameters
    ----------
    prefix : str
        Prefix for the file name (usually 'train', 'val', etc.)
    visualization_type : str
        Type of visualization (e.g., 'token_prediction', 'entity_prediction')
    global_step : int
        Current global step of the model
    batch_idx : int, optional
        Batch index, by default None
    extension : str, optional
        File extension, by default 'png'

    Returns
    -------
    str
        Full path to save the visualization
    """
    if batch_idx is not None:
        filename = f"{prefix}__{visualization_type}__step_{global_step:05d}__batch_{batch_idx:05d}.{extension}"
    else:
        filename = f"{prefix}__{visualization_type}__step_{global_step:05d}.{extension}"
    path = os.path.join(SAVE_DIR, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# -------------------------------------------------------------------------
# Token Visualization
# -------------------------------------------------------------------------


def visualize_token_predictions(
    module: L.LightningModule,
    logits: Tensor,
    labels: Tensor,
    masks: Tensor,
    prefix: str,
    sample_index: int | None = None,
    sample_name: str | None = None,
    batch_idx: int | None = None,
) -> None:
    # Select specific sample if index provided, otherwise find example with most non-zero labels
    sample_index, sample_name = _resolve_sample_info(
        sample_index=sample_index,
        sample_name=sample_name,
        default_index=lambda: int(torch.argmax((labels > 0).sum(dim=1))),
    )

    # Extract example
    example_logits = logits[sample_index].detach().cpu().numpy()
    example_labels = labels[sample_index].detach().cpu().numpy()
    example_mask = masks[sample_index].detach().cpu().numpy()

    # Create figure with proper layout
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, width_ratios=[20, 1], height_ratios=[2, 1, 0.5])

    # Create axes for plots and colorbar
    ax1 = fig.add_subplot(gs[0, 0])  # Top plot - logits heatmap
    ax2 = fig.add_subplot(gs[1, 0])  # Middle plot - true class logits
    ax3 = fig.add_subplot(gs[2, 0])  # Bottom plot - mask
    cax = fig.add_subplot(gs[:, 1])  # Colorbar axis

    # First subplot: heatmap and true labels
    im = ax1.imshow(
        example_logits.T, aspect="auto", cmap="RdYlGn", interpolation="none"
    )
    fig.colorbar(im, cax=cax, label="Logit values")

    # Mark positions with labels
    x_coords = np.where(example_labels > 0)[0]
    y_coords = example_labels[x_coords]

    # Connect points with a line
    ax1.plot(
        x_coords,
        y_coords,
        "-",
        color="black",
        alpha=0.7,
        linewidth=1.5,
        label="Label sequence",
    )

    # Plot scatter points over the line
    ax1.scatter(
        x_coords,
        y_coords,
        marker="s",
        s=20,
        edgecolor="black",
        facecolor="none",
        label="True labels",
    )

    # Add y-axis labels with class names
    y_ticks = range(module.num_labels)
    y_tick_labels = [module.config.token_label_name(i) for i in y_ticks]
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_tick_labels)

    # Add titles and labels for first subplot
    ax1.set_title(
        f"Sequence Logits and Labels (Step {module.global_step}, Sample {sample_name})"
    )
    ax1.set_ylabel("Label")
    ax1.legend(loc="best", framealpha=0.5, fancybox=True, shadow=True)

    # Second subplot: logits of true class at each position
    seq_len = example_logits.shape[0]
    x_positions = np.arange(seq_len)

    # Get true class logits for all positions, use NaN for invalid positions
    true_class_logits = np.full(seq_len, np.nan)
    true_class_logits[example_mask] = example_logits[
        x_positions[example_mask], example_labels[example_mask].astype(int)
    ]

    # Plot the true class logits with a continuous line
    ax2.plot(
        x_positions,
        true_class_logits,
        "-",
        color="black",
        alpha=0.7,
        linewidth=0.5,
        label="True class logits",
    )

    # Add red dots at positions where labels change
    label_change_positions = np.nonzero(np.diff(example_labels))[0] + 1
    ax2.scatter(
        label_change_positions,
        true_class_logits[label_change_positions],
        color="#d62728",
        s=50,
        zorder=3,
        label="Label changes",
    )

    # Add legend with better styling
    ax2.legend(loc="best", framealpha=0.5, fancybox=True, shadow=True)

    # Add titles and labels for second subplot
    ax2.set_title("Logits of True Class at Each Position")
    ax2.set_ylabel("Logit Value")

    # Third subplot: mask visualization
    ax3.imshow(
        example_mask.reshape(1, -1),
        aspect="auto",
        cmap="binary",
        vmin=0,
        vmax=1,
        interpolation="none",
    )
    ax3.set_title("Valid Mask")
    ax3.set_yticks([])
    ax3.set_xlabel("Position")

    # Share x-axis limits across all subplots
    ax1.set_xlim(0, seq_len - 1)
    ax2.set_xlim(0, seq_len - 1)
    ax3.set_xlim(0, seq_len - 1)

    # Adjust layout
    plt.tight_layout()

    # Save and log the figure
    path = get_visualization_path(
        prefix, "token_prediction", module.global_step, batch_idx
    )
    logger.info(f"Saving prediction example to {path}")
    plt.savefig(path)
    plt.close()


# -------------------------------------------------------------------------
# Entity Visualization
# -------------------------------------------------------------------------


def visualize_entity_predictions(
    module: L.LightningModule,
    true_entity_labels: np.ndarray,
    pred_entity_labels: np.ndarray,
    pred_entity_logits: np.ndarray,
    true_token_masks: np.ndarray,
    prefix: str,
    sample_index: int | None = None,
    sample_name: str | None = None,
    batch_idx: int | None = None,
) -> None:
    # Select specific sample if index provided, otherwise find example with most non-zero labels
    sample_index, sample_name = _resolve_sample_info(
        sample_index=sample_index,
        sample_name=sample_name,
        default_index=lambda: int(np.argmax((true_entity_labels > 0).sum(axis=1))),
    )

    # Extract example
    example_true = true_entity_labels[sample_index]
    example_pred = pred_entity_labels[sample_index]
    example_logits = pred_entity_logits[sample_index]
    example_mask = true_token_masks[sample_index]

    # Create figure with proper layout
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(4, 2, width_ratios=[20, 1], height_ratios=[1, 1, 1, 0.5])

    # Create axes for plots
    ax0 = fig.add_subplot(gs[0, 0])  # Predicted logits
    ax1 = fig.add_subplot(gs[1, 0])  # Predicted labels
    ax2 = fig.add_subplot(gs[2, 0])  # True labels
    ax3 = fig.add_subplot(gs[3, 0])  # Mask
    cax = fig.add_subplot(gs[:, 1])  # Colorbar axis

    entity_names = module.config.token_entity_names_with_background()
    num_entities = len(entity_names)

    def to_heatmap(labels: np.ndarray) -> np.ndarray:
        mat = np.eye(num_entities)
        np.fill_diagonal(mat, np.arange(num_entities))
        return mat[labels].T

    # First subplot: entity logits
    im0 = ax0.imshow(
        example_logits.T, aspect="auto", cmap="RdYlGn", interpolation="none"
    )
    ax0.set_title("Entity Logits")
    # Add y-axis labels with entity names
    ax0.set_yticks(np.arange(num_entities))
    ax0.set_yticklabels(entity_names)
    fig.colorbar(im0, cax=cax, label="Logit values")

    # Second subplot: predicted entity labels
    cmap = plt.cm.get_cmap("tab10", num_entities)
    ax1.imshow(
        to_heatmap(example_pred),
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=module.num_core_entities,
        interpolation="none",
    )
    ax1.set_title("Predicted Entity Labels")
    ax1.set_yticks(np.arange(num_entities))
    ax1.set_yticklabels(entity_names)

    # Third subplot: true entity labels
    ax2.imshow(
        to_heatmap(example_true),
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=module.num_core_entities,
        interpolation="none",
    )
    ax2.set_title("True Entity Labels")
    ax2.set_yticks(np.arange(num_entities))
    ax2.set_yticklabels(entity_names)

    # Fourth subplot: mask
    ax3.imshow(
        example_mask.reshape(1, -1),
        aspect="auto",
        cmap="binary",
        vmin=0,
        vmax=1,
        interpolation="none",
    )
    ax3.set_title("Valid Mask")
    ax3.set_yticks([])
    ax3.set_xlabel("Position")

    # Create legend
    legend_elements = [
        Patch(facecolor=cmap(i), edgecolor="black", label=entity_names[i])
        for i in range(num_entities)
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=len(legend_elements),
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Add title
    fig.suptitle(
        f"Entity Predictions (Step {module.global_step}, Sample {sample_name})",
        fontsize=16,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    # Save and log the figure
    path = get_visualization_path(
        prefix, "entity_prediction", module.global_step, batch_idx
    )
    logger.info(f"Saving entity prediction example to {path}")
    plt.savefig(path)
    plt.close()


def visualize_entity_intervals(
    module: L.LightningModule,
    entity_intervals: pd.DataFrame,
    true_entity_labels: np.ndarray,
    true_token_masks: np.ndarray,
    prefix: str,
    sample_index: int | None = None,
    sample_name: str | None = None,
    batch_idx: int | None = None,
) -> None:
    # Select specific sample if index provided, otherwise find sample with the most intervals
    sample_index, sample_name = _resolve_sample_info(
        sample_index=sample_index,
        sample_name=sample_name,
        default_index=lambda: int(
            entity_intervals["sample_index"].value_counts().idxmax()
        ),
    )

    # Filter intervals for the selected sample
    sample_intervals = entity_intervals[
        entity_intervals["sample_index"] == sample_index
    ]

    # Get the sample's true labels and mask
    example_labels = true_entity_labels[sample_index]
    example_mask = true_token_masks[sample_index]
    sequence_length = len(example_labels)

    # Get unique entities in this sample
    unique_entities = sorted(sample_intervals["entity"].unique())

    if len(unique_entities) == 0:
        logger.info(f"No entities found for sample {sample_name}")
        return

    # Create figure with GridSpec for better control of layout
    fig = plt.figure(figsize=(15, 2.5 * len(unique_entities) + 1))
    gs = GridSpec(
        len(unique_entities) + 1, 1, height_ratios=[2] * len(unique_entities) + [0.5]
    )

    # Get entity names from the module
    entity_names = module.config.interval_entity_names

    # Create a colormap
    cmap = plt.cm.get_cmap("tab10", len(entity_names))

    # Plot each entity in its own subplot with two tracks
    for i, entity in enumerate(unique_entities):
        ax = fig.add_subplot(gs[i, 0])
        entity_name = module.config.interval_entity_name(entity)
        entity_data = sample_intervals[sample_intervals["entity"] == entity]

        # Plot background rectangle for better visualization
        ax.axhspan(0, 2, color="lightgrey", alpha=0.3)

        # Define y-positions for the two tracks
        labeled_y = 1.5  # Upper track for labeled intervals
        predicted_y = 0.5  # Lower track for predicted intervals

        # Define colors for different prediction types
        correct_color = "#2ca02c"  # green
        incorrect_color = "#d62728"  # red

        has_labeled = False
        has_correct = False
        has_incorrect = False

        # Plot labeled (true) intervals on upper track
        for _, row in entity_data[entity_data["labeled"]].iterrows():
            ax.axhspan(
                labeled_y - 0.4,
                labeled_y + 0.4,
                xmin=row["start"] / sequence_length,
                xmax=(row["stop"] + 1) / sequence_length,
                alpha=0.6,
                color=cmap(entity),
            )
            has_labeled = True

        # Plot predicted intervals on lower track, colored by match status
        for _, row in entity_data[entity_data["predicted"]].iterrows():
            if row["labeled"]:
                # Correct prediction (also in labeled)
                ax.axhspan(
                    predicted_y - 0.4,
                    predicted_y + 0.4,
                    xmin=row["start"] / sequence_length,
                    xmax=(row["stop"] + 1) / sequence_length,
                    alpha=0.6,
                    color=correct_color,
                )
                has_correct = True
            else:
                # Incorrect prediction
                ax.axhspan(
                    predicted_y - 0.4,
                    predicted_y + 0.4,
                    xmin=row["start"] / sequence_length,
                    xmax=(row["stop"] + 1) / sequence_length,
                    alpha=0.6,
                    color=incorrect_color,
                )
                has_incorrect = True

        # Style the plot
        ax.set_title(f"{entity_name} (Entity {entity})")
        ax.set_yticks([predicted_y, labeled_y])
        ax.set_yticklabels(["Predicted", "Labeled"])
        ax.set_ylim(0, 2)
        ax.set_xlim(0, sequence_length - 1)

        # Add legend
        handles = []
        labels = []

        if has_labeled:
            handles.append(plt.Rectangle((0, 0), 1, 1, alpha=0.6, color=cmap(entity)))
            labels.append("Labeled Interval")

        if has_correct:
            handles.append(plt.Rectangle((0, 0), 1, 1, alpha=0.6, color=correct_color))
            labels.append("Correct Prediction")

        if has_incorrect:
            handles.append(
                plt.Rectangle((0, 0), 1, 1, alpha=0.6, color=incorrect_color)
            )
            labels.append("Incorrect Prediction")

        if handles:
            ax.legend(
                handles,
                labels,
                loc="upper right",
                framealpha=0.5,
                fancybox=True,
                shadow=True,
            )

        # Only add x-label for the bottom entity plot
        if i == len(unique_entities) - 1:
            ax.set_xlabel("Position")

    # Add mask indicator at the bottom in its own subplot
    mask_ax = fig.add_subplot(gs[-1, 0])
    mask_ax.imshow(
        example_mask.reshape(1, -1),
        aspect="auto",
        cmap="binary",
        vmin=0,
        vmax=1,
        interpolation="none",
    )
    mask_ax.set_title("Valid Mask")
    mask_ax.set_yticks([])
    mask_ax.set_xlim(0, sequence_length - 1)
    mask_ax.set_xlabel("Position")

    # Adjust layout and add title
    fig.suptitle(
        f"Entity Intervals Visualization (Step {module.global_step}, Sample {sample_name})",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save and log the figure
    path = get_visualization_path(
        prefix, "entity_interval", module.global_step, batch_idx
    )
    logger.info(f"Saving entity intervals visualization to {path}")
    plt.savefig(path)
    plt.close()


# -------------------------------------------------------------------------
# Performance Visualization
# -------------------------------------------------------------------------


def visualize_entity_interval_performance(
    module: L.LightningModule,
    entity_intervals: pd.DataFrame,
    prefix: str,
    batch_name: str | None = None,
    batch_idx: int | None = None,
) -> None:
    """Generate an HTML table visualization of entity interval performance metrics."""

    # Create a function to calculate metrics for a group
    def calc_metrics(group):
        true_intervals = group["labeled"].sum()
        pred_intervals = group["predicted"].sum()
        true_positives = group[group["labeled"] & group["predicted"]].shape[0]

        precision = true_positives / pred_intervals if pred_intervals > 0 else 0
        recall = true_positives / true_intervals if true_intervals > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return pd.Series(
            {
                "true_intervals": true_intervals,
                "predicted_intervals": pred_intervals,
                "true_positives": true_positives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    # Group by sample and entity to calculate per-sample metrics
    sample_metrics = (
        entity_intervals.groupby(["sample_index", "entity"], as_index=False)
        .apply(calc_metrics, include_groups=False)
        .reset_index(level=0, drop=True)  # Drop the extra level created by apply
        .reset_index()  # Convert the remaining index to columns
    )

    # Add entity names
    sample_metrics["entity_name"] = sample_metrics["entity"].apply(
        module.config.interval_entity_name
    )

    # Calculate overall metrics per entity
    overall_metrics = (
        entity_intervals.groupby("entity", as_index=False)
        .apply(calc_metrics, include_groups=False)
        .reset_index(level=0, drop=True)  # Drop the extra level created by apply
        .reset_index()  # Convert the remaining index to columns
    )
    overall_metrics["sample_index"] = "All Samples"
    overall_metrics["entity_name"] = overall_metrics["entity"].apply(
        module.config.interval_entity_name
    )

    # Combine and sort results
    all_metrics = pd.concat(
        [overall_metrics, sample_metrics], axis=0, ignore_index=True
    )

    all_metrics = all_metrics[
        [
            "sample_index",
            "entity",
            "entity_name",
            "true_intervals",
            "predicted_intervals",
            "true_positives",
            "precision",
            "recall",
            "f1",
        ]
    ]

    # Highlight the "All Samples" rows
    def highlight_all_samples(row):
        return [
            "font-weight: bold; border: 2px solid #4682B4"
            if row["sample_index"] == "All Samples"
            else ""
            for _ in row
        ]

    if batch_name is not None:
        caption = f"Entity Interval Performance (Step {module.global_step}, Batch {batch_name})"
    else:
        caption = f"Entity Interval Performance (Step {module.global_step})"

    # Format the table
    styled_table = (
        all_metrics.style.format(
            {
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1": "{:.3f}",
                "true_intervals": "{:.0f}",
                "predicted_intervals": "{:.0f}",
                "true_positives": "{:.0f}",
            }
        )
        .apply(highlight_all_samples, axis=1)
        .background_gradient(cmap="RdYlGn", subset=["precision", "recall", "f1"])
        .background_gradient(
            cmap="Blues",
            subset=["true_intervals", "predicted_intervals", "true_positives"],
        )
        .set_table_attributes('class="table table-bordered table-hover"')
        .set_caption(caption)
    )

    # Save table to HTML file
    path = get_visualization_path(
        prefix,
        "entity_interval_performance",
        module.global_step,
        batch_idx,
        extension="html",
    )
    logger.info(f"Saving entity performance table to {path}")

    with open(path, "w") as f:
        f.write(
            "<html><head><style>body{font-family:Arial,sans-serif;}"
            + ".table{border-collapse:collapse;width:100%;margin:20px 0;}"
            + ".table th,.table td{padding:8px;text-align:left;border:1px solid #ddd;}"
            + ".table th{background-color:#f2f2f2;}"
            + ".table tr:hover{background-color:#f5f5f5;}</style></head><body>"
        )
        f.write(styled_table.to_html())
        f.write("</body></html>")
