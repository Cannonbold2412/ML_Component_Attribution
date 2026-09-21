"""Command line: the one entry point for the whole Study 2 pipeline.

    python -m mbca study fetch       download raw data (skips files already on disk)
    python -m mbca study clean       raw -> clean + manifest
    python -m mbca study validate    Yahoo vs FRED cross-check
    python -m mbca study run         all pre-registered experiments (needs committed code)
    python -m mbca study analyze     trades -> every table / statistic (results/study2/analysis)
    python -m mbca study paper       analysis -> paper/generated (tables, figures, numbers.tex)
"""
import argparse


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m mbca")
    sub = ap.add_subparsers(dest="group", required=True)
    st = sub.add_parser("study")
    st.add_argument("step", choices=["fetch", "clean", "validate", "run", "analyze", "paper"])
    st.add_argument("--workers", type=int, default=11)
    st.add_argument("--only", nargs="*", help="strategy:market tasks to run, e.g. jma_trend:crypto")
    st.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    if a.step in ("fetch", "clean", "validate"):
        from . import datasets
        print(getattr(datasets, a.step)())
    elif a.step == "run":
        from .study import run
        print("run id:", run(a.workers, a.only, a.allow_dirty))
    elif a.step == "analyze":
        from .analysis import analyze
        analyze()
    else:
        from .paper_assets import build
        build()


if __name__ == "__main__":
    main()
