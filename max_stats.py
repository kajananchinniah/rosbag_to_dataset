import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal

base_dir = sys.argv[1]
files = glob.glob(f"{base_dir}/*.pt")

max_speed = -torch.inf
min_speed = torch.inf

data = torch.load(files[0])
observations = data["observation"]
# files = files[1:]
count = 0

mapping = {}

dims = {
    "imu": 6,
    "state": 6,
    "wheel_rpm": 4,
    "shock_travel": 4,
    "intervention": 1,
}
for observation in observations.keys():
    key = observation
    if observation in dims.keys():
        mapping[key] = {}
        mapping[key]["min"] = [10000000 for _ in range(dims[key])]
        mapping[key]["max"] = [-1000000 for _ in range(dims[key])]

for file in files:
    print(file)
    count += 1
    print(count / len(files))
    data = torch.load(file)["observation"]
    observations = data
    print(mapping)
    print("\n")
    for key in data.keys():
        observation = observations[key]
        if key == "state":
            for i in range(7, observation.size()[1]):
                mapping[key]["max"][i - 7] = max(
                    torch.max(observation[:, i]), mapping[key]["max"][i - 7]
                )
                mapping[key]["min"][i - 7] = min(
                    torch.min(observation[:, i]), mapping[key]["min"][i - 7]
                )
        if (
            key == "wheel_rpm"
            or key == "imu"
            or key == "shock_travel"
            or key == "intervention"
        ):
            if key == "intervention":
                observation = observation.unsqueeze(0)
            for i in range(observation.size()[2]):
                mapping[key]["max"][i] = max(
                    torch.max(observation[:, :, i]), mapping[key]["max"][i]
                )
                mapping[key]["min"][i] = min(
                    torch.min(observation[:, :, i]), mapping[key]["min"][i]
                )

import json

for key in mapping.keys():
    for key2 in mapping[key].keys():
        for i in range(len(mapping[key][key2])):
            if isinstance(mapping[key][key2][i], torch.Tensor):
                mapping[key][key2][i] = mapping[key][key2][i].detach().numpy().tolist()
print(json.dumps(mapping, indent=4))
