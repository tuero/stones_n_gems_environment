import os
import copy
import random
from multiprocessing import Pool, Manager
import argparse

from env_factory.gem_exit import create_env_gem_exit
from util.rnd_util import flatten_map_str

config_easy = {
    "size": 12,
    "num_gems": [2, 6],
    "num_rooms": [0, 0],
    "room_size": 6,
    "ratio_gems_in_room": [0, 0],
    "exit_in_room": False,
    "max_steps" : 9999
}

config_medium = {
    "size": 16,
    "num_gems": [4, 8],
    "num_rooms": [0, 2],
    "room_size": 6,
    "ratio_gems_in_room": [0, 0.5],
    "exit_in_room": True,
    "max_steps" : 9999
}

config_hard = {
    "size": 24,
    "num_gems": [8, 12],
    "num_rooms": [3, 4],
    "room_size": 8,
    "ratio_gems_in_room": [0.25, 0.75],
    "exit_in_room": True,
    "max_steps" : 9999
}

config_all = {
    "easy": config_easy,
    "medium": config_medium,
    "hard": config_hard,
}


def create_map(args):
    manager_dict, config_name, seed = args
    config = copy.deepcopy(config_all[config_name])
    config["seed"] = seed
    config["num_gems"] = random.randint(config["num_gems"][0], config["num_gems"][1])
    config["num_rooms"] = random.randint(config["num_rooms"][0], config["num_rooms"][1])
    config["ratio_gems_in_room"] = random.uniform(config["ratio_gems_in_room"][0], config["ratio_gems_in_room"][1])
    config["exit_in_room"] = config["exit_in_room"] and random.uniform(0, 1) > 0.5
    map_str = create_env_gem_exit(**config)
    manager_dict[seed] = flatten_map_str(map_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", help="Number of total samples", required=False, type=int, default=10000)
    parser.add_argument("--export_path", help="Export path for file", required=True, type=str)
    parser.add_argument("--difficulty", help="Difficulty of maps", required=True, type=str, choices=["easy", "medium", "hard"])
    args = parser.parse_args()

    manager = Manager()
    data = manager.dict()
    with Pool(16) as p:
        p.map(
            create_map,
            [(data, args.difficulty, i) for i in range(args.num_samples)],
        )

    # Parse and save to file
    export_dir = os.path.dirname(args.export_path)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    with open(args.export_path, "w") as file:
        for i in range(len(data)):
            file.write(data[i])
            file.write("\n")


if __name__ == "__main__":
    main()
