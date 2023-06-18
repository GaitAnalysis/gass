import json
from dataclasses import dataclass
import math
import argparse
from pathlib import Path
from tqdm import tqdm

IGNORED_POINTS = [0, 14, 15, 16, 17, 18, 19]

NORMALIZATION_DICT = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 1,
    9: 1,
    10: 1,
    11: 1,
    12: 1,
    13: 1,
}


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float
    point: int

    def __repr__(self):
        return f"Keypoint({self.x}, {self.y}, {self.confidence}, {self.point})"


@dataclass
class Displacement:
    displacement: float
    confidence: float
    point: int

    def __repr__(self):
        return f"Displacement({self.displacement}, {self.confidence}, {self.point})"


def calc_normalization(frame: list[Keypoint]) -> tuple[float, float]:
    # dist(keyp3, keyp4)
    k3, k4, k8, k9 = None, None, None, None
    for k in frame:
        if k.point == 3:
            k3 = k
        if k.point == 4:
            k4 = k
        if k.point == 8:
            k8 = k
        if k.point == 9:
            k9 = k
    if k3 is None or k4 is None or k8 is None or k9 is None:
        raise Exception("keypoints not found")
    d1 = math.dist([k3.x, k3.y], [k4.x, k4.y])
    d2 = math.dist([k8.x, k8.y], [k9.x, k9.y])
    return d1, d2


def convert(openpose_output: str) -> str:
    keypoints = []
    for l in openpose_output.splitlines():
        line = json.loads(l)
        # schema: [[x, y, confidence, float(keypoint)]]
        curr_frame = []
        for keyps in line:
            keyp = Keypoint(keyps[0], keyps[1], keyps[2], int(keyps[3]))
            if keyp.point in IGNORED_POINTS:
                continue
            curr_frame.append(keyp)
        keypoints.append(curr_frame)

    displacements = []
    for i in range(len(keypoints) - 1):
        try:
            k1 = keypoints[i]
            k2 = keypoints[i + 1]
            d = []
            normalizations = calc_normalization(k1)
            for p1, p2 in zip(k1, k2):
                dist = math.dist([p1.x, p1.y], [p2.x, p2.y])
                # normalize
                dist = dist / normalizations[NORMALIZATION_DICT[p1.point]]
                d.append(Displacement(dist, p1.confidence, p1.point))
            displacements.append(d)
        except Exception as e:
            print(e)
            print(f"skipping frame {i}")
            continue

    # remove low confidence displacements
    # build blacklist
    blacklist = []
    for i in range(len(displacements)):
        for d in displacements[i]:
            if d.confidence < 0.7:
                blacklist.append((i, d.point))

    drop_zero = False
    drop_latest = False
    # algorithm: if a displacement is low confidence, remove it and replace it with the average of the previous and next displacements
    for i, p in blacklist:
        if i == 0:
            # drop 0
            drop_zero = True
            continue
        if i == len(displacements) - 1:
            # drop last
            drop_latest = True
            continue
        try:
            displacements[i][p - 1].displacement = (
                displacements[i - 1][p - 1].displacement
                + displacements[i + 1][p - 1].displacement
            ) / 2
        except IndexError:
            print(f"skipping frame {i}")
            continue

    if drop_zero:
        displacements = displacements[1:]
    if drop_latest:
        displacements = displacements[:-1]
    # save
    outputs = []
    for d in displacements:
        disp_list = []
        for p in d:
            disp_list.append((p.displacement, p.confidence))
        outputs.append(json.dumps(disp_list))
    return "\n".join(outputs) + "\n"


def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--input", type=str, help="json file/dir location to process.", required=True
    )
    args.add_argument(
        "--output", type=str, help="dir to save results to.", required=True
    )
    args = args.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    if input_path.is_dir():
        files = list(input_path.glob("*"))
    else:
        files = [input_path]
    for fpath in tqdm(files):
        file = open(fpath, "r")
        try:
            displacements = convert(file.read())
        except Exception as e:
            print(e)
            print(f"skipping {fpath}")
            continue
        print(f"Saving to {output_path}")
        with open(output_path / fpath.name, "w") as f:
            f.write(displacements)


if __name__ == "__main__":
    main()
