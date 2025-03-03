import time
from FileHandling import *
import numpy as np
from matplotlib.patches import Circle
from pathlib import Path
import seaborn as sns
import pingouin as pg

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# %% run
# Use a breakpoint in the code line below to debug your script.

summary = iterate_dataset(
    subjects=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    # subjects=[10],
    cursors=["Head", "MM"],
    selections=["Dwell", "Score"],
    repetitions=range(2, 10),
)
# %%
summary[["total_time", "contact_time", "selection_time"]] = summary.apply(
    time_analysis, axis=1, result_type="expand"
)
summary["target_entries"] = summary.apply(entries_analysis, axis=1)
# corner
# summary["total_time"] = summary.apply(total_time, axis=1)
results = summary.drop(["data"], axis=1).dropna()
by_subjects = results.groupby(
    [results.subject, results.cursor, results.selection]
).mean()
# %% Defining error trials


def filter_group(group):
    # Select numeric columns for percentile calculation.
    exclude_cols = [
        "subject",
        "target",
        "total_time",
    ]  # Columns to exclude from percentile calculation

    # Select only numeric columns except excluded ones
    numeric_cols = results.select_dtypes(include="number").columns.difference(
        exclude_cols
    )
    # Calculate the 95th percentile (upper 5% cutoff) for each numeric column in this group.
    thresholds = group[numeric_cols].quantile(0.95)
    # Create a mask: True for rows where all numeric values are below or equal to their threshold.
    mask = (group[numeric_cols] <= thresholds).all(axis=1)
    return group[mask]


# Group by both condition columns and apply the filtering function to each group.
print(len(results))
filtered_df = results.groupby(["cursor", "selection"], group_keys=False).apply(
    filter_group
)
print(len(filtered_df))
# %%
sns.barplot(data=by_subjects, x="cursor", y="total_time", hue="selection")
plt.ylim(0, 5)
plt.show()
# %%
sns.barplot(data=by_subjects, x="cursor", y="success", hue="selection")
plt.ylim(0, 1)
plt.show()
# %%
sns.barplot(data=by_subjects, x="cursor", y="target_entries", hue="selection")
# plt.ylim(0, 1)
plt.show()
# %%
for c in ["total_time", "success", "target_entries", "contact_time", "selection_time"]:
    sns.barplot(data=by_subjects, x="cursor", y=c, hue="selection")
    plt.title(c)
    plt.show()
# %% RM anova
# 'success'
# "total_time"
for c in ["total_time", "success", "target_entries", "contact_time", "selection_time"]:
    dataset = summary.dropna(subset=[c])
    dataset["cursor"] = dataset["cursor"].map({"Head": 0, "MM": 1})
    dataset["selection"] = dataset["selection"].map({"Dwell": 0, "Score": 1})
    dataset.loc[:, "cursor"] = dataset["cursor"].astype(int)
    dataset.loc[:, "selection"] = dataset["selection"].astype(int)
    aov = pg.rm_anova(
        dv=c,
        within=["cursor", "selection"],
        subject="subject",
        data=dataset,
        detailed=True,
        effsize="ng2",
        correction=True,
    )
    aov.round(3)
    print(
        c.upper(),
    )
    pg.print_table(aov)

    posthoc = pg.pairwise_tests(
        dv=c,
        within=["cursor", "selection"],
        subject="subject",
        data=dataset,
        padjust="bonferroni",
    )  # Adjust for multiple comparisons
    from tabulate import tabulate

    print(tabulate(posthoc, headers="keys", tablefmt="grid"))

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Compute means & standard errors for each condition
df_summary = (
    dataset.groupby(["cursor", "selection"])["selection_time"]
    .agg(["mean", "sem"])
    .reset_index()
)

# Create the interaction plot
plt.figure(figsize=(8, 6))
sns.pointplot(
    data=df_summary,
    x="cursor",
    y="mean",
    hue="selection",
    markers=["o", "s"],
    capsize=0.1,
    errwidth=1.2,
    dodge=True,
)

# Labels & formatting
plt.title("Interaction Effect of Cursor & Selection on Selection Time", fontsize=14)
plt.xlabel("Multimodal Cursor (0 = Off, 1 = On)", fontsize=12)
plt.ylabel("Mean Selection Time (s)", fontsize=12)
plt.xticks([0, 1], ["Off", "On"])
plt.legend(title="Scoring Method", labels=["Off", "On"])
plt.grid(True)

# Show plot
plt.show()

# %% draw gifs


import matplotlib.pyplot as plt


from matplotlib.animation import FuncAnimation, PillowWriter

# Example DataFrame
# Replace this with your actual data
# data = {"x": [0, 1, 2, 3, 4, 5, 6], "y": [0, 1, 0, -1, 0, 1, 0]}
# df = pd.DataFrame(data)
for i in range(len(summary)):
    # for i in [1]:
    df = summary.data[i]
    # Parameters
    tail_length = 10  # Number of points in the tail
    if summary.cursor[i] == "MM":
        useHead = True
    else:
        useHead = False
    # Initialize figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(df["horizontal_offset"].min() - 2, df["horizontal_offset"].max() + 2)
    ax.set_ylim(df["vertical_offset"].min() - 2, df["vertical_offset"].max() + 2)
    ax.set_aspect("equal")
    # (point,) = ax.plot([], [], "bo", label="Moving Point")  # Moving point
    tail = ax.scatter([], [], c="red", s=20, alpha=0.5, label="Tail")  # Tail
    scat = ax.scatter([], [], c="blue", s=50)  # Points only
    if useHead:
        head = ax.scatter([], [], c="green", s=30, alpha=0.5)

    # Initialization function
    def init():
        scat.set_offsets([])  # Empty the scatter
        tail.set_offsets([])  # Empty the scatter
        if useHead:
            head.set_offsets([])
        return (scat,)

    # Update function
    def update(frame):
        # Get all points up to the current frame
        circle = Circle(
            (0, 0),
            radius=1.5,
            edgecolor="green",
            facecolor="none",
            linestyle="--",
            linewidth=2,
        )
        plt.gca().add_patch(circle)
        x_data = df["horizontal_offset"][frame : frame + 1]
        y_data = df["vertical_offset"][frame : frame + 1]
        start = max(0, frame - tail_length)
        tail_x = df["horizontal_offset"][start : frame + 1]
        tail_y = df["vertical_offset"][start : frame + 1]
        if useHead:
            x = df["head_horizontal_offset"][frame : frame + 1]
            y = df["head_vertical_offset"][frame : frame + 1]
            head.set_offsets(list(zip(x, y)))
        scat.set_offsets(list(zip(x_data, y_data)))
        tail.set_offsets(list(zip(tail_x, tail_y)))
        plt.title(f"{summary.cursor[i]}_{summary.selection[i]}")

        # plt.scatter(x_data, y_data)
        # Update scatter plot with new points
        # scat.set_offsets(list(zip(x_data, y_data)))
        # return (scat,)

    # Create animation
    ani = FuncAnimation(fig, update, frames=range(len(df)), interval=200)

    # Show the animation
    # plt.legend()
    # plt.show()
    directory = Path(f"gifs/{summary.subject[i]}")
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    ani.save(
        f"gifs/{summary.subject[i]}/{summary.subject[i]}_{summary.cursor[i]}_{summary.selection[i]}_{summary.target[i]}_{summary.repetition[i]}_{summary.success[i]}_animation.gif",
        writer="imagemagick",
        fps=30,
    )
# %%


def run_analysis():
    # Use a breakpoint in the code line below to debug your script.

    summary = iterate_dataset(
        subjects=[98],
        cursors=["Head", "MM"],
        selections=["Dwell", "Score"],
        repetitions=[0],
    )


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    current_time = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    print(f"Analyzer started at, {current_time}")  # Press ⌘F8 to toggle the breakpoint.

    summary = run_analysis()

    current_time = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    print(f"Analyzer finished at, {current_time}")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
