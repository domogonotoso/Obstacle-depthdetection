"""Play the two Windows warning patterns once, without models or a camera."""
import argparse
import time

from app.sound import SoundPlayer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("Warning", "Danger", "both"), default="both")
    args = parser.parse_args()
    with SoundPlayer() as player:
        if not player.enabled:
            print("Windows sound playback is unavailable on this platform.")
            return 1
        levels = ("Warning", "Danger") if args.level == "both" else (args.level,)
        for level in levels:
            print(f"Playing {level}; verify audibility on the PC speaker.", flush=True)
            player.play(level)
            time.sleep(0.6)  # Only this manual test waits; video uses the worker asynchronously.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
