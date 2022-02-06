import os
import sys
import copy
import random
from multiprocessing import Pool, Manager
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from env_factory.gem_key_exit import create_gem_key_exit
from util.rnd_util import flatten_map_str

config_easy = {
    "size": 14,
    "num_gems": [2, 4],
    "num_rooms": [0, 1],
    "room_size": 6,
    "num_locked_doors": [0, 1],
    "num_keys_in_main": [0, 2],
    "ratio_gems_in_room": [0.25, 0.5],
    "keys_in_order": True,
    "max_steps" : 9999
}

config_medium = {
    "size": 18,
    "num_gems": [4, 8],
    "num_rooms": [1, 3],
    "room_size": 6,
    "num_locked_doors": [1, 3],
    "num_keys_in_main": [0, 2],
    "ratio_gems_in_room": [0.25, 0.5],
    "keys_in_order": True,
    "max_steps" : 9999
}

config_hard = {
    "size": 24,
    "num_gems": [6, 10],
    "num_rooms": [4, 4],
    "room_size": 8,
    "num_locked_doors": [3, 4],
    "num_keys_in_main": [0, 2],
    "ratio_gems_in_room": [0.5, 0.75],
    "keys_in_order": False,
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
    config["num_locked_doors"] = random.randint(config["num_locked_doors"][0], config["num_locked_doors"][1])
    config["num_keys_in_main"] = random.randint(config["num_keys_in_main"][0], config["num_keys_in_main"][1])
    config["ratio_gems_in_room"] = random.uniform(config["ratio_gems_in_room"][0], config["ratio_gems_in_room"][1])
    if not config["keys_in_order"] :
        config["keys_in_order"] = random.uniform(0, 1) > 0.5

    map_str = create_gem_key_exit(**config)
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
