# Danceology ML
Python scripts for data cleaning and ML processing of video input

Originally Developed by Team Danceology Spring 2023

Christine Jung, Xiaoying Meng, Jiacheng Qiu, Yiming Xiao, Xueying Yang, Angela Zhang

# Tech Requirements
- [Python 3.10](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installation/)

# Setup
1. Clone this repository locally and use the Terminal to `cd` into the root folder of the repository
2. (Optional) Set up a [Python virtual environment](https://docs.python.org/3/library/venv.html)
3. Install all dependencies using
```
% pip install -r requirements.txt
```

# Running ML Model on Input Video
All scripts and logic related to ML-based video processing are within the `video_processing` folder. More details on how the video processing is done can be found on our [documentation wiki](https://etcdanceology.github.io/Danceology-Documentation/).

1. `cd` into the `video_processing` directory
2. Run the following command
```
% python main.py -i [path_to_input_video] -o [path_to_output_json]
```

The output json file from this can be directly used for data cleaning below.


# Data Cleaning
All data cleaning scripts and logic are within the `data_cleaning` folder. The `main.py` file contains all the main cleaning functions; the `util.py` file contains all utility functions that are used during the data cleaning process.

You can customize how data cleaning is done by modifying different parameters within the `main.py` file. More details on how the cleaning is done can be found on our [documentation wiki](https://etcdanceology.github.io/Danceology-Documentation/).

1. `cd` into the `data_cleaning` directory
2. Run the following command
```
% python main.py -i [path_to_input_json] -o [path_to_output_json]
```

The output file from data cleaning can be used within the Unity project.
