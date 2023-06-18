import copy
import numpy as np
import cv2
from glob import glob
import os
import argparse
import json

# video file processing setup
# from: https://stackoverflow.com/a/61927951
import argparse
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple
from tqdm import tqdm
import multiprocessing as mp
from typing import List, Tuple, Optional, NamedTuple, Union
import io
import tempfile


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class FFProbeResult(NamedTuple):
    return_code: int
    json: str
    error: str


def ffprobe(file_path) -> FFProbeResult:
    command_array = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        file_path,
    ]
    result = subprocess.run(
        command_array,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return FFProbeResult(
        return_code=result.returncode, json=result.stdout, error=result.stderr
    )


# openpose setup
from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body("model/body_pose_model.pth")


def process_frame(frame, body=True, hands=True):
    if body:
        candidate = body_estimation(frame)
    else:
        candidate = None
    return candidate


# writing video with ffmpeg because cv2 writer failed
# https://stackoverflow.com/questions/61036822/opencv-videowriter-produces-cant-find-starting-number-error
import ffmpeg


def process_video(video_file: Union[io.BytesIO, str]) -> str:
    temp_video = None
    if isinstance(video_file, io.BytesIO):
        # opencv doesn't support reading from bytesio, so we have to write to a temp file
        temp_video = tempfile.NamedTemporaryFile(suffix=".mp4")
        temp_video.write(video_file.read())
        temp_video.flush()
        video_file = temp_video.name
    print(video_file)

    cap = cv2.VideoCapture(video_file)
    # get video file info
    ffprobe_result = ffprobe(video_file)
    info = json.loads(ffprobe_result.json)
    videoinfo = [i for i in info["streams"] if i["codec_type"] == "video"][0]
    input_fps = videoinfo["avg_frame_rate"]
    # input_fps = float(input_fps[0])/float(input_fps[1])
    input_pix_fmt = videoinfo["pix_fmt"]
    input_vcodec = videoinfo["codec_name"]
    # define a writer object to write to a movidified file
    postfix = info["format"]["format_name"].split(",")[0]
    pbar = tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT))
    infer_data = []
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break

        posed_frame = process_frame(frame, body=True, hands=False)
        pbar.update(1)
        if posed_frame is not None:
            posed_frame = posed_frame[0]
        else:
            continue

        infer_data.append(json.dumps(posed_frame, cls=NumpyEncoder))

    cap.release()
    if temp_video is not None:
        temp_video.close()
    return "\n".join(infer_data)


def infer_worker(video_file, output_path):
    output = process_video(str(video_file))
    with open(str(output_path / (video_file.name + ".json")), "w") as f:
        f.write(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    files = list(Path(args.input_path).glob("*.mp4"))
    files.extend(list(Path(args.input_path).glob("*.mov")))
    files.sort()
    output_path = Path(args.output_path)
    output_dirs = [output_path] * len(files)
    with mp.Pool(processes=4) as pool:
        for _ in tqdm(
            pool.starmap(infer_worker, zip(files, output_dirs)), total=len(files)
        ):
            pass

    infer_data = process_video(args.input)
    with open(args.output, "w") as f:
        f.write(infer_data)


if __name__ == "__main__":
    main()
