import argparse
import cv2
import json
import progressbar
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def process_video(input_file_path, output_json_file):
    cap = cv2.VideoCapture(f'{input_file_path}')
    levelData = {}

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up the progressbar
    widgets = ["Analyzing Video: ", progressbar.Percentage(), " ",
        progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval = n_frames, widgets=widgets).start()
    p = 0

    poseData = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert colors for better results
            results = pose.process(converted_image)

            if results == None or results.pose_landmarks == None:
                poseData.append(poseData[-1])

            else:
                keypoints = []
                keypoints3D = []

                for keypoint in results.pose_landmarks.landmark:
                    keypoints.append({
                        "x": keypoint.x * size[0],
                        "y": keypoint.y * size[1],
                        "z": keypoint.z,
                        "score": keypoint.visibility,
                    })

                for keypoint3D in results.pose_world_landmarks.landmark:
                    keypoints3D.append({
                        "x": keypoint3D.x,
                        "y": keypoint3D.y,
                        "z": keypoint3D.z,
                        "score": keypoint3D.visibility,
                    })

                poseData.append({
                    "keypoints": keypoints,
                    "keypoints3D": keypoints3D
                })

            p += 1
            pbar.update(p)

    cap.release()

    levelData["poseData"] = poseData
    levelData = { "levelData": levelData }
    json_object = json.dumps(levelData, indent=4)
    
    # Writing to output file
    with open(f"{output_json_file}", "w") as outfile:
        outfile.write(json_object)

# Main Function
parser = argparse.ArgumentParser()
parser.add_argument('-i', required=True, help='input video file to use for video processing')
parser.add_argument('-o', required=True, help='out .json file to use for ML output')
args = parser.parse_args()

input_file = args.i
output_file = args.o
process_video(input_file, output_file)
