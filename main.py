import torch
import torch.nn as nn
import torch.nn.functional as F
import io
from classifier import GaitClassifier
from vid2seq import process_video
from convert_infer_results import convert
import json
import threading
from pydantic import BaseModel
from typing import List, Tuple, Optional, NamedTuple, Union
from fastapi import FastAPI, Depends, UploadFile, File
from queue import Queue

app = FastAPI()
model = GaitClassifier(26, 4096, 2)
model.load_state_dict(torch.load("model/gait_classifier.pth", device="cuda"))
model.eval()
infer_queue = Queue()


def video_to_tensor(video: io.BytesIO) -> torch.Tensor:
    openpose_out = process_video(video)
    print(openpose_out)
    openpose_out = convert(openpose_out)
    print(openpose_out)
    data = []
    for line in openpose_out.split("\n"):
        if len(line) == 0:
            continue
        if len(line) < 13:
            continue
        try:
            data_obj = json.loads(line)
            if len(data_obj) < 13:
                continue
            else:
                data.append(data_obj)
        except:
            break

    tensor = torch.tensor(data, dtype=torch.float32)
    tensor = tensor.view(tensor.shape[0], -1)
    return tensor


def classify(model: nn.Module, data: torch.Tensor) -> bool:
    # data shape (seq_len, input_size)
    # out shape (num_classes)
    out = model(data)
    out = torch.softmax(out, dim=0)
    out = out.argmax(dim=0)
    if out == 0:
        return False
    else:
        return True


def inference(model: nn.Module, video: io.BytesIO) -> bool:
    data = video_to_tensor(video)
    return classify(model, data)


def infer_worker(infer_queue: Queue):
    while True:
        vid_obj = infer_queue.get()
        print("getting video from queue")
        vid_io = io.BytesIO(vid_obj.video)
        result = inference(model, vid_io)
        print(result)
        infer_queue.task_done()


infer_thread = threading.Thread(target=infer_worker, args=(infer_queue,), daemon=True)
infer_thread.start()


class VideoType(BaseModel):
    user_id: str
    video_id: str
    video: bytes


@app.get("/")
async def ok() -> dict[str, str]:
    return {"status": "ok"}


# upload video file
# fields:
#   video: file
#   user_id: str
#   video_id: str
@app.post("/classify")
async def classify_video(
    video: UploadFile = File(...), user_id: str = "", video_id: str = ""
) -> dict[str, Union[str, bool]]:
    video_bytes = await video.read()
    vid_obj = VideoType(user_id=user_id, video_id=video_id, video=video_bytes)
    infer_queue.put(vid_obj)
    return {"status": "queued"}
