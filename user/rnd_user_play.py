"""
Play a map with user control (keyboard input.)
"""
import numpy as np

from ptu.util.render import Render
from base_rnd import RNDBaseEnv


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
    print("Paste map string: ")
    sentinel=''
    input_map = '\n'.join(iter(input, sentinel))
    map_id = np.array([list(map(int, s1.split(","))) for s1 in input_map.split("\n")], dtype=np.uint8)

    map_details = {"map_id": map_id}
    env = RNDBaseEnv(map_details=map_details, use_noop=True)
    play(env)
