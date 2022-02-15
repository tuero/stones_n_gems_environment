"""
Play a map with user control (keyboard input.)
"""
import os
import sys
import numpy as np

from ptu.util.render import Render

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tree_search.rnd_state_spiel import RNDState
from util.img_draw import rnd_state_to_img


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


def play(env: RNDState):
    getch = _Getch()
    r = Render(600, 600)
    done = False

    r.draw(rnd_state_to_img(env.get_image_representation()))

    while not done:
        # Check for quit events
        if r.check_for_exit():
            break

        char = getch()
        if char == "q":
            break
        action = action_mapping[int(char)]
        env.apply_action(action)
        done = env.is_solution() or len(env.successors()) == 0
        if not done:
            r.draw(rnd_state_to_img(env.get_image_representation()))

        print(action)

    r.close()


if __name__ == "__main__":
    print("Paste map string: ")
    sentinel=''
    map_str = '\n'.join(iter(input, sentinel))
    env = RNDState(map_str=map_str, reward_structure=2)
    play(env)
