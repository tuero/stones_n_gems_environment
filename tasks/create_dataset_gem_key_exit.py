from email.policy import default
import os
import sys
import copy
import random
from multiprocessing import Pool, Manager
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from env_factory.gem_key_exit import create_gem_key_exit
from util.rnd_util import flatten_map_str

config_veryeasy = {
    "size": 14,
    "num_gems": [0, 2],
    "num_rooms": [0, 1],
    "room_size": 5,
    "num_locked_doors": [0, 1],
    "num_keys_in_main": [0, 2],
    "ratio_gems_in_room": [0.25, 0.5],
    "keys_in_order": True,
    "max_steps" : 9999
}


config_easy = {
    "size": 18,
    "num_gems": [1, 3],
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
    "num_gems": [2, 8],
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
    "veryeasy" : config_veryeasy,
    "easy": config_easy,
    "medium": config_medium,
    "hard": config_hard,
}


# 0 gems, exit in main
scenario_1 = {
    "size": 14,
    "num_gems": [0, 0],
    "num_rooms": [0, 0],
    "room_size": 5,
    "num_locked_doors": [0, 0],
    "num_keys_in_main": [0, 2],
    "ratio_gems_in_room": [1, 1],
    "keys_in_order": True,
    "exit_in_open": True,
    "max_steps" : 9999
}

# 0 gems, exit in open room
scenario_2 = {
    "size": 14,
    "num_gems": [0, 0],
    "num_rooms": [0, 0],
    "room_size": 5,
    "num_locked_doors": [0, 0],
    "num_keys_in_main": [0, 2],
    "ratio_gems_in_room": [1, 1],
    "keys_in_order": True,
    "exit_in_open": False,
    "max_steps" : 9999
}

# 0 gems, exit in locked room
scenario_3 = {
    "size": 14,
    "num_gems": [0, 0],
    "num_rooms": [0, 0],
    "room_size": 5,
    "num_locked_doors": [1, 1],
    "num_keys_in_main": [0, 1],
    "ratio_gems_in_room": [1, 1],
    "keys_in_order": True,
    "exit_in_open": False,
    "max_steps" : 9999
}

# 1 gem in main, exit in main
scenario_4 = {
    "size": 14,
    "num_gems": [1, 1],
    "num_rooms": [0, 0],
    "room_size": 5,
    "num_locked_doors": [0, 0],
    "num_keys_in_main": [0, 2],
    "ratio_gems_in_room": [1, 1],
    "keys_in_order": True,
    "exit_in_open": True,
    "max_steps" : 9999
}

# 1 gem in main, exit in open room
scenario_5 = {
    "size": 14,
    "num_gems": [1, 1],
    "num_rooms": [0, 0],
    "room_size": 5,
    "num_locked_doors": [0, 0],
    "num_keys_in_main": [0, 2],
    "ratio_gems_in_room": [1, 1],
    "keys_in_order": True,
    "exit_in_open": False,
    "max_steps" : 9999
}

# 1 gem in open room, exit in main
scenario_6 = {
    "size": 14,
    "num_gems": [1, 1],
    "num_rooms": [1, 1],
    "room_size": 5,
    "num_locked_doors": [0, 0],
    "num_keys_in_main": [0, 2],
    "ratio_gems_in_room": [1, 1],
    "keys_in_order": True,
    "exit_in_open": True,
    "max_steps" : 9999
}

# 1 gem in open room, exit in another open room
scenario_7 = {
    "size": 14,
    "num_gems": [1, 1],
    "num_rooms": [4, 4],
    "room_size": 5,
    "num_locked_doors": [0, 0],
    "num_keys_in_main": [0, 2],
    "ratio_gems_in_room": [1, 1],
    "keys_in_order": True,
    "exit_in_open": False,
    "max_steps" : 9999
}


# 1 gem in locked room, exit in main
scenario_8 = {
    "size": 14,
    "num_gems": [1, 1],
    "num_rooms": [1, 1],
    "room_size": 5,
    "num_locked_doors": [1, 1],
    "num_keys_in_main": [0, 2],
    "ratio_gems_in_room": [1, 1],
    "keys_in_order": True,
    "exit_in_open": True,
    "max_steps" : 9999
}

# 1 gem in main, exit locked room
scenario_9 = {
    "size": 14,
    "num_gems": [1, 1],
    "num_rooms": [4, 4],
    "room_size": 5,
    "num_locked_doors": [1, 1],
    "num_keys_in_main": [0, 2],
    "ratio_gems_in_room": [0, 0],
    "keys_in_order": True,
    "exit_in_open": False,
    "max_steps" : 9999
}

# 1 gem in open room, exit locked room
scenario_10 = {
    "size": 14,
    "num_gems": [1, 1],
    "num_rooms": [4, 4],
    "room_size": 5,
    "num_locked_doors": [1, 1],
    "num_keys_in_main": [0, 2],
    "ratio_gems_in_room": [1, 1],
    "keys_in_order": True,
    "exit_in_open": False,
    "max_steps" : 9999
}

scenarios_map = {
    0 : scenario_1,
    1 : scenario_2,
    2 : scenario_3,
    3 : scenario_4,
    4 : scenario_5,
    5 : scenario_6,
    6 : scenario_7,
    7 : scenario_8,
    8 : scenario_9,
    9 : scenario_10,
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


def create_map_mixed(args):
    manager_dict, config_name, seed = args
    config = copy.deepcopy(scenarios_map[config_name])
    config["seed"] = seed
    config["num_gems"] = random.randint(config["num_gems"][0], config["num_gems"][1])
    config["num_rooms"] = random.randint(config["num_rooms"][0], config["num_rooms"][1])
    config["num_locked_doors"] = random.randint(config["num_locked_doors"][0], config["num_locked_doors"][1])
    config["num_keys_in_main"] = random.randint(config["num_keys_in_main"][0], config["num_keys_in_main"][1])
    config["ratio_gems_in_room"] = random.uniform(config["ratio_gems_in_room"][0], config["ratio_gems_in_room"][1])
    map_str = create_gem_key_exit(**config)
    manager_dict[seed] = flatten_map_str(map_str)


def runner_normal(args, data):
    with Pool(16) as p:
        p.map(
            create_map,
            [(data, args.difficulty, i + args.seed) for i in range(args.num_samples)],
        )


def runner_mixed(args, data):
    assert len(args.scenarios) > 0
    scenarios = []
    for i, num in enumerate(args.scenarios):
        scenarios += [i] * int(num)

    random.shuffle(scenarios)
    with Pool(16) as p:
        p.map(
            create_map_mixed,
            [(data, scenarios[i], i + args.seed) for i in range(len(scenarios))],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", help="Number of total samples", required=False, type=int, default=10000)
    parser.add_argument("--export_path", help="Export path for file", required=True, type=str)
    parser.add_argument("--difficulty", help="Difficulty of maps", required=False, type=str, choices=["veryeasy", "easy", "medium", "hard"], default="easy")
    parser.add_argument("--mixed", help="Mixed map pool", action="store_true", default=False)
    parser.add_argument("--scenarios", help="List of scenario ratios for mixed", nargs="+")
    parser.add_argument("--seed", help="Start seed", type=int, required=False, default=0)
    args = parser.parse_args()

    manager = Manager()
    data = manager.dict()

    random.seed(args.seed)

    if args.mixed:
        runner_mixed(args, data)
    else:
        runner_normal(args, data)
    # with Pool(16) as p:
    #     p.map(
    #         create_map,
    #         [(data, args.difficulty, i) for i in range(args.num_samples)],
    #     )

    # Parse and save to file
    export_dir = os.path.dirname(args.export_path)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    with open(args.export_path, "w") as file:
        for i in range(len(data)):
            file.write(data[i + args.seed])
            file.write("\n")


if __name__ == "__main__":
    main()
