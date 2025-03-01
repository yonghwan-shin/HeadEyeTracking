import itertools
import math
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def read_data(subject, cursor, selection, repetition):
    root = Path(__file__).resolve().parent / "Dataset" / str(subject)
    trial_detail = f"subject{str(subject)}_cursor{str(cursor)}_Selection{str(selection)}_repetition{str(repetition)}"
    files = root.rglob(trial_detail + "*.json")

    for file in files:
        if trial_detail in file.name:
            with open(file) as f:
                output = pd.read_json(f)
                target_position = pd.json_normalize(
                    output.target_position, sep="_"
                ).rename(
                    columns={
                        "x": "target_position_x",
                        "y": "target_position_y",
                        "z": "target_position_z",
                    }
                )
                head_origin = pd.json_normalize(output.head_origin, sep="_").rename(
                    columns={
                        "x": "head_origin_x",
                        "y": "head_origin_y",
                        "z": "head_origin_z",
                    }
                )
                head_forward = pd.json_normalize(output.head_forward, sep="_").rename(
                    columns={
                        "x": "head_forward_x",
                        "y": "head_forward_y",
                        "z": "head_forward_z",
                    }
                )
                head_rotation = pd.json_normalize(output.head_rotation, sep="_").rename(
                    columns={
                        "x": "head_rotation_x",
                        "y": "head_rotation_y",
                        "z": "head_rotation_z",
                    }
                )
                eyeRay_direction = pd.json_normalize(
                    output.eyeRayDirection, sep="_"
                ).rename(
                    columns={
                        "x": "eyeRay_direction_x",
                        "y": "eyeRay_direction_y",
                        "z": "eyeRay_direction_z",
                    }
                )
                target_plane_position = pd.json_normalize(
                    output.target_plane_position, sep="_"
                ).rename(
                    columns={
                        "x": "target_plane_position_x",
                        "y": "target_plane_position_y",
                        "z": "target_plane_position_z",
                    }
                )
                cursor = pd.json_normalize(output.cursorData, sep="_")
                output = pd.concat(
                    [
                        output,
                        target_position,
                        head_origin,
                        head_forward,
                        head_rotation,
                        eyeRay_direction,
                        target_plane_position,
                        cursor,
                    ],
                    axis=1,
                )

                output["cursor_rotation"] = output.apply(
                    lambda x: asSpherical(x.direction_x, x.direction_y, x.direction_z),
                    axis=1,
                )
                output["target_rotation"] = output.apply(
                    lambda x: asSpherical(
                        x.target_position_x - x.origin_x,
                        x.target_position_y - x.origin_y,
                        x.target_position_z - x.origin_z,
                    ),
                    axis=1,
                )
                output["head_rotation"] = output.apply(
                    lambda x: asSpherical(
                        x.head_forward_x, x.head_forward_y, x.head_forward_z
                    ),
                    axis=1,
                )
                output["eyeRay_rotation"] = output.apply(
                    lambda x: asSpherical(
                        x.eyeRay_direction_x, x.eyeRay_direction_y, x.eyeRay_direction_z
                    ),
                    axis=1,
                )
                output["obstacles"] = output.apply(
                    lambda x: calculate_surrounding_positions(
                        (x.head_origin_x, x.head_origin_y, x.head_origin_z),
                        (x.target_position_x, x.target_position_y, x.target_position_z),
                        (x.target_position_x, x.target_position_y, x.target_position_z),
                    ),
                    axis=1,
                )
                obstacles = pd.json_normalize(output.obstacles)
                output = pd.concat([output, obstacles], axis=1)

                for ob in ["up", "down", "left", "right"]:
                    output[ob + "_distance"] = output.apply(
                        lambda x: calculate_angular_distance(
                            (x.head_origin_x, x.head_origin_y, x.head_origin_z),
                            (x.direction_x, x.direction_y, x.direction_z),
                            (x[ob]),
                        ),
                        axis=1,
                    )
                    prev_score = 0
                    prev_end_num = 0
                    output[ob + "_score"] = 0
                    for idx, row in output.iterrows():
                        current_end_num = row["end_num"]
                        if prev_end_num != current_end_num:
                            prev_score = 0
                        prev_end_num = current_end_num
                        new_score = update_scores_for_targets(
                            row[ob + "_distance"], prev_score
                        )
                        output.at[idx, str(ob + "_score")] = new_score
                        prev_score = new_score

                prev_score = 0
                prev_end_num = 0
                output["target_score"] = 0
                for idx, row in output.iterrows():
                    current_end_num = row["end_num"]
                    if prev_end_num != current_end_num:
                        prev_score = 0
                    prev_end_num = current_end_num
                    new_score = update_scores_for_targets(
                        row["cursor_angular_distance"], prev_score
                    )
                    output.at[idx, "target_score"] = new_score
                    prev_score = new_score
                output["selected_target"] = output[
                    [
                        "left_score",
                        "right_score",
                        "up_score",
                        "down_score",
                        "target_score",
                    ]
                ].apply(lambda row: row.idxmax() if row.max() >= 0.5 else None, axis=1)
                output.at[0, "selected_target"] = None
                output.at[1, "selected_target"] = None

                output["head_horizontal_angle"] = output.apply(
                    lambda x: x.head_rotation[1], axis=1
                )
                output["head_vertical_angle"] = output.apply(
                    lambda x: x.head_rotation[0], axis=1
                )
                output["cursor_horizontal_angle"] = output.apply(
                    lambda x: x.cursor_rotation[1], axis=1
                )
                output["cursor_vertical_angle"] = output.apply(
                    lambda x: x.cursor_rotation[0], axis=1
                )
                output["eyeRay_horizontal_angle"] = output.apply(
                    lambda x: x.eyeRay_rotation[1], axis=1
                )
                output["eyeRay_vertical_angle"] = output.apply(
                    lambda x: x.eyeRay_rotation[0], axis=1
                )
                output["target_horizontal_angle"] = output.apply(
                    lambda x: x.target_rotation[1], axis=1
                )
                output["target_vertical_angle"] = output.apply(
                    lambda x: x.target_rotation[0], axis=1
                )
                output["horizontal_offset"] = (
                    output.target_horizontal_angle - output.cursor_horizontal_angle
                ).apply(correct_angle)
                output["vertical_offset"] = (
                    output.target_vertical_angle - output.cursor_vertical_angle
                ).apply(correct_angle)
                output["head_horizontal_offset"] = (
                    output.target_horizontal_angle - output.head_horizontal_angle
                ).apply(correct_angle)
                output["head_vertical_offset"] = (
                    output.target_vertical_angle - output.head_vertical_angle
                ).apply(correct_angle)
                output["eyeRay_horizontal_offset"] = (
                    output.target_horizontal_angle - output.eyeRay_horizontal_angle
                ).apply(correct_angle)
                output["eyeRay_vertical_offset"] = (
                    output.target_vertical_angle - output.eyeRay_vertical_angle
                ).apply(correct_angle)
                # output["success"] = output.target_name == "Target_" + str(output.endnum)
                success_record = f.name[-14:-5]
                return output, success_record


def update_scores_for_targets(alpha, prev_scores):
    """
    Update the scores for all targets in one frame.

    Parameters:
        observer_pos: [x, y, z] of the observer.
        cursor_dir: [x, y, z] direction vector of the cursor.
        targets: Dictionary of target positions, e.g.:
            { "original": [x,y,z], "up": [x,y,z], ... }
        prev_scores: Dictionary of previous scores for each target.

    Returns:
        new_scores: Dictionary of updated scores for each target.
    """
    # new_scores = {}
    # for name, pos in targets.items():
    #     # Calculate angular distance (in degrees)
    #     alpha = calculate_angular_distance(observer_pos, cursor_dir, pos)
    # Calculate contribution score (ensure non-negative)
    CS = 0.7  # previous score weight
    CG = 0.3  # contribution weight
    contribute_score = max(0, 1 - alpha / 6)
    # Update the score using the previous score and the contribution.
    new_scores = prev_scores * CS + contribute_score * CG
    return new_scores


def calculate_angular_distance(observer_pos, cursor_dir, target_pos):
    """
    Calculate the angular distance between the cursor direction and the target.

    Parameters:
    observer_pos (array-like): The (x, y, z) coordinates of the observer.
    cursor_dir (array-like): The (x, y, z) direction vector of the cursor.
    target_pos (array-like): The (x, y, z) coordinates of the target.

    Returns:
    float: The angular distance in degrees.
    """
    # Convert inputs to numpy arrays
    observer_pos = np.array(observer_pos)
    cursor_dir = np.array(cursor_dir)
    target_pos = np.array(target_pos)

    # Vector from observer to target
    observer_to_target = target_pos - observer_pos

    # Normalize vectors
    cursor_dir_norm = cursor_dir / np.linalg.norm(cursor_dir)
    observer_to_target_norm = observer_to_target / np.linalg.norm(observer_to_target)

    # Calculate dot product
    dot_product = np.dot(cursor_dir_norm, observer_to_target_norm)

    # Ensure the dot product is within the valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the angle in radians
    angle_radians = np.arccos(dot_product)

    # Convert angle to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def calculate_surrounding_positions(observer, plane_center, original_object):
    # Convert points to numpy arrays
    observer = np.array(observer)
    plane_center = np.array(plane_center)
    plane_center[1] = observer[1]
    original_object = np.array(original_object)

    # Compute normal vector (plane faces observer)
    normal_vector = observer - plane_center  # Direction from plane to observer
    normal_vector /= np.linalg.norm(normal_vector)  # Normalize

    # Up vector is fixed in Unity (0,1,0) since the plane is vertical
    up_vector = np.array([0, 1, 0])

    # Compute right vector (perpendicular to normal and up)
    right_vector = np.cross(normal_vector, up_vector)
    right_vector /= np.linalg.norm(right_vector)  # Normalize

    # Compute surrounding positions relative to the original object
    displacement = 0.15  # Distance from the original object
    up_pos = original_object + displacement * up_vector
    down_pos = original_object - displacement * up_vector
    left_pos = original_object - displacement * right_vector
    right_pos = original_object + displacement * right_vector

    return {
        "up": up_pos.tolist(),
        "down": down_pos.tolist(),
        "left": left_pos.tolist(),
        "right": right_pos.tolist(),
    }


def iterate_dataset(subjects, cursors, selections, repetitions, targets=range(9)):
    summary = pd.DataFrame(
        columns=[
            "subject",
            "cursor",
            "selection",
            "target",
            "repetition",
            "success",
            "data",
        ]
    )
    for subject, cursor, selection, repetition in itertools.product(
        subjects, cursors, selections, repetitions
    ):
        d, success = read_data(subject, cursor, selection, repetition)
        data = split_target(d)
        for t in targets:
            trial_summary = {
                "subject": subject,
                "cursor": cursor,
                "selection": selection,
                "target": t,
                "repetition": repetition,
                "success": success[t],
            }
            temp_data = data[t].reset_index()
            temp_data.timestamp -= temp_data.timestamp.values[0]
            trial_summary["data"] = temp_data
            summary.loc[len(summary)] = trial_summary
    summary["success"] = summary["success"].map({"X": 0, "O": 1})
    return summary


def distance_analysis(mainrow):
    selection = mainrow["selection"]
    data = mainrow["data"]
    target_num = mainrow["target"]
    if selection == "Dwell":
        success_only = data[data.target_name == "Target_" + str(target_num)]
    else:
        success_only = data[
            (data.selected_target == "target_score") & (data.target_score > 0.5)
        ]
    if len(success_only) <= 0:
        print("No success")
        return


def time_analysis(mainrow):
    selection = mainrow["selection"]
    data = mainrow["data"]
    target_num = mainrow["target"]
    if selection == "Dwell":
        success_only = data[data.target_name == "Target_" + str(target_num)]
    else:
        success_only = data[
            (data.selected_target == "target_score") & (data.target_score > 0.5)
        ]

    if len(success_only) <= 0:
        print("No success")
        return
    contact_time = success_only.timestamp.values[0]
    total_time = data.timestamp.values[-1] - data.timestamp.values[0]
    selection_time = total_time - contact_time
    return total_time, contact_time, selection_time


def entries_analysis(mainrow):
    data = mainrow["data"]
    endnum = mainrow["target"]
    selection = mainrow["selection"]
    if selection == "Dwell":
        data["success"] = data.target_name == "Target_" + str(endnum)
    else:
        data["success"] = (data.selected_target == "target_score") & (
            data.target_score > 0.5
        )
    all_success_dwells = []
    for k, g in itertools.groupby(data.iterrows(), key=lambda row: row[1]["success"]):
        if k == True:
            df = pd.DataFrame([r[1] for r in g])
            if len(df) <= 1:
                continue
            all_success_dwells.append(df)
    target_in_count = len(all_success_dwells)
    return target_in_count


def correct_angle(angle):
    if angle > 180:
        return angle - 360
    if angle < -180:
        return angle + 360
    return angle


def asSpherical(x, y, z):
    r = math.sqrt(x * x + y * y + z * z)
    if r == 0:
        return [0, 0]
    theta = math.degrees(math.acos(y / r))
    # phi  = math.degrees(math.atan(y/x))
    phi = math.degrees(math.atan2(x, z))
    return [theta, phi]


def split_target(data):
    output = []

    data = data[data["step_num"] != 0]
    # first_end_num = data['end_num'].values[0]
    # for i in range(len(data) - 1):
    #     if data['end_num'].values[i] == first_end_num:
    #         pass
    #         # data = data.drop(i)
    #         # print(first_end_num, 'drop', i, len(data))
    #     else:
    #         # print(i, data['timestamp'].values[i],data['end_num'].values[i],first_end_num)
    #         data=data.drop([x for x in range(i)])
    #         break

    for target_num in range(9):
        output.append(data[data["end_num"] == target_num])
    return output
