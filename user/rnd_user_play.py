"""
Play a map with user control (keyboard input.)
"""
import os
import sys
import numpy as np

from ptu.util.render import Render

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from base_rnd import RNDBaseEnv
from rnd_simple_exit import RNDSimpleExit
from rnd_single_diamond import RNDSingleDiamond
from rnd_single_room import RNDSingleRoom
from rnd_double_room import RNDDoubleRoom


class _Getch:
    """Gets a single character from standard input.  Does not echo to the screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self):
        return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty
        import sys

    def __call__(self):
        import sys
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()


action_mapping = {
    5 : 0,
    8 : 1,
    6 : 2,
    2 : 3,
    4 : 4
}


def play(env: RNDBaseEnv):
    getch = _Getch()
    r = Render(600, 600)

    env.reset()
    img = env.render()
    done = False

    r.draw(img)

    while not done:
        # Check for quit events
        if r.check_for_exit():
            break

        char = getch()
        if char == "q":
            break
        action = action_mapping[int(char)]
        _, reward, done = env.step(action)
        if not done:
            r.draw(env.render())
        print(reward)

    r.close()


if __name__ == "__main__":
    print("Enter game mode: ")
    print("0: Custom map string")
    print("1: Simple Exit Open")
    print("2: Single Diamond")
    print("3: Single Room with Diamond")
    print("4: Double Room with Diamond")
    choice = input()

    if choice == 0:
        print("Paste map string: ")
        sentinel=''
        input_map = '\n'.join(iter(input, sentinel))
        map_id = np.array([list(map(int, s1.split(","))) for s1 in input_map.split("\n")], dtype=np.uint8)
        map_details = {"map_id": map_id}
        env = RNDBaseEnv(map_details=map_details, use_noop=True)
    elif choice == "1":
        env = RNDSimpleExit(use_noop=True, env_mode=2)
    elif choice == "2":
        env = RNDSingleDiamond(use_noop=True, env_mode=2)
    elif choice == "3":
        env = RNDSingleRoom(use_noop=True, env_mode=2)
    elif choice == "4":
        env = RNDDoubleRoom(use_noop=True, env_mode=2)
    else:
        raise ValueError
    play(env)
