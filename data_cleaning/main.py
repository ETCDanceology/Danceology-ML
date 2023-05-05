'''
Danceology
Originally Developed by Team Danceology Spring 2023
Christine Jung, Xiaoying Meng, Jiacheng Qiu, Yiming Xiao, Xueying Yang, Angela Zhang

This script and all related assets fall under the CC BY-NC-SA 4.0 License
All future derivations of this code should contain the above attribution
'''

import argparse
import json
import numpy as np
from scipy import signal
from util import AcRDP, lerp

NUM_JOINTS = 33
MFILTER_KERSIZE, MFILTER_KERSIZE3D = 3, 15
MFILTER_OFFSET, MFILTER_OFFSET3D = MFILTER_KERSIZE // 2, MFILTER_KERSIZE3D // 2
WINDOW_LENGTH, POLYORDER = 11, 2

def clean_data(input_file, output_file):
    # Load JSON Data
    pose_file_name = f"{input_file}"
    file = open(pose_file_name)
    json_data = json.load(file)
    pose_data = json_data["levelData"]["poseData"]

    pose_3d_file_name = f"{input_file}"
    file = open(pose_3d_file_name)
    json_data = json.load(file)
    pose_data3d = json_data["levelData"]["poseData"]

    # Reformat frames into a readable form
    orig_keypoints = []
    orig_keypoints3D = []
    scores = []

    for i in range(len(pose_data)):
        frame_keypoints = pose_data[i]["keypoints"]
        frame_keypoints3D = pose_data3d[i]["keypoints3D"]

        parsed_frame_keypoints = []
        parsed_frame_keypoints3D = []
        parsed_scores = []
        for i in range(len(frame_keypoints)):
            joint_pos = [frame_keypoints[i]['x'], frame_keypoints[i]['y']]
            parsed_frame_keypoints.append(joint_pos)
            parsed_scores.append(frame_keypoints[i]['score'])

            joint_pos3D = [frame_keypoints3D[i]['x'], frame_keypoints3D[i]['y'], frame_keypoints3D[i]['z']]
            parsed_frame_keypoints3D.append(joint_pos3D)

        orig_keypoints.append(parsed_frame_keypoints)
        orig_keypoints3D.append(parsed_frame_keypoints3D)
        scores.append(parsed_scores)

    # Convert to numpy arrays
    keypoints = np.array(orig_keypoints)
    keypoints3D = np.array(orig_keypoints3D)
    scores = np.array(scores)

    cleaned_keypoints = np.copy(keypoints)
    cleaned_keypoints3D = np.copy(keypoints3D)

    # Run median filter on all points
    for i in range(NUM_JOINTS):
        cleaned_keypoints[MFILTER_OFFSET:-MFILTER_OFFSET,i,:] = signal.medfilt(keypoints[MFILTER_OFFSET:-MFILTER_OFFSET,i,:], kernel_size=[MFILTER_KERSIZE, 1])
        cleaned_keypoints3D[MFILTER_OFFSET3D:-MFILTER_OFFSET3D,i,:] = signal.medfilt(keypoints3D[MFILTER_OFFSET3D:-MFILTER_OFFSET3D,i,:], kernel_size=[MFILTER_KERSIZE3D, 1])

    # Run AcRDP algorithm on 2D points
    selected_frames = AcRDP(keypoints, 0.001, 85)
    selected_frames.sort()
    last_frame = 0

    for i in range(len(selected_frames)):
        next_frame = selected_frames[i]
        times = [last_frame, next_frame]

        for frame in range(last_frame + 1, next_frame):
            for j in range(NUM_JOINTS):
                points = [cleaned_keypoints[last_frame, j, :], cleaned_keypoints[next_frame, j, :]]
                cleaned_keypoints[frame, j, :] = lerp(frame, times, points)

        last_frame = next_frame

    # Run AcRDP algorithm on 3D points
    selected_frames = AcRDP(cleaned_keypoints3D, 0.001, 70)
    selected_frames.sort()
    last_frame = 0

    for i in range(len(selected_frames)):
        next_frame = selected_frames[i]
        times = [last_frame, next_frame]

        for frame in range(last_frame + 1, next_frame):
            for j in range(NUM_JOINTS):
                points = [cleaned_keypoints3D[last_frame, j, :], cleaned_keypoints3D[next_frame, j, :]]
                cleaned_keypoints3D[frame, j, :] = lerp(frame, times, points)

        last_frame = next_frame

    # Run Savgol Filter on all points
    for i in range(NUM_JOINTS):
        cleaned_keypoints[:,i,:] = signal.savgol_filter(cleaned_keypoints[:,i,:], WINDOW_LENGTH, POLYORDER, axis=0)
        cleaned_keypoints3D[:,i,:] = signal.savgol_filter(cleaned_keypoints3D[:,i,:], WINDOW_LENGTH, POLYORDER, axis=0)

    # Revert to JSON format
    for i in range(len(pose_data)):
        frame_cleaned_keypoints = cleaned_keypoints[i]
        frame_cleaned_keypoints3D = cleaned_keypoints3D[i]

        for j in range(len(frame_cleaned_keypoints)):
            pose_data[i]["keypoints"][j]['x'] = frame_cleaned_keypoints[j][0]
            pose_data[i]["keypoints"][j]['y'] = frame_cleaned_keypoints[j][1]

            pose_data[i]["keypoints3D"][j]['x'] = frame_cleaned_keypoints3D[j][0]
            pose_data[i]["keypoints3D"][j]['y'] = frame_cleaned_keypoints3D[j][1]
            pose_data[i]["keypoints3D"][j]['z'] = frame_cleaned_keypoints3D[j][2]

    # Serializing json
    json_data["levelData"]["poseData"] = pose_data
    json_object = json.dumps(json_data, indent=4)
    
    # Writing to output file
    with open(f"{output_file}", "w") as outfile:
        outfile.write(json_object)

# Main Function
parser = argparse.ArgumentParser()
parser.add_argument('-i', required=True, help='input .json file to use when reading 2D and 3D keypoints')
parser.add_argument('-o', required=True, help='out .json file to use for cleaned output')
args = parser.parse_args()

input_file = args.i
output_file = args.o
clean_data(input_file, output_file)
