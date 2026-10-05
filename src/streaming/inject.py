"""Drift injector CLI: set the live drift mode, or run a timed demo scenario."""
import argparse
import json
import pathlib
import time

CONTROL = pathlib.Path("control/drift_mode.json")


def set_mode(mode, frac):
    CONTROL.parent.mkdir(exist_ok=True)
    CONTROL.write_text(json.dumps({"mode": mode, "frac": frac}))
    print(f"[{time.strftime('%H:%M:%S')}] drift mode -> {mode} (frac={frac})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["A", "B", "C", "U"])
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--scenario", choices=["demo"])
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--hold", type=int, default=60, help="seconds each drift lasts (scenario)")
    ap.add_argument("--gap", type=int, default=40, help="seconds of normal traffic between drifts (scenario)")
    a = ap.parse_args()

    if a.clear:
        set_mode("normal", a.frac)
    elif a.type:
        set_mode(a.type, a.frac)
    elif a.scenario == "demo":
        for kind in ["A", "B", "C", "U"]:
            set_mode(kind, a.frac)
            time.sleep(a.hold)
            set_mode("normal", a.frac)
            time.sleep(a.gap)
        print("Scenario complete.")
    else:
        ap.print_help()