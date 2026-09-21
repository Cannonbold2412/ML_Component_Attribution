"""MBCA -- Matched-Budget Component Attribution for hybrid (rule + ML) trading systems."""
from .components import Combo, Entry, Exit, MetaLabel, PositionSizing, RegimeFilter, RuleGate
from .data import MARKETS, load_csv, load_market
from .protocol import BASELINE, MBCAResult, run_mbca
from .strategy import Pipeline, adx_gate, bracket_exit, crossover_signal, jma_signals, trailing_exit, walk_forward


def paper_components() -> list:
    """The paper's full grid: rule ADX gate, 5 ML components (3 regime labels), 4 combinations."""
    meta, entry, exit_ = MetaLabel(), Entry(), Exit()
    sizing = PositionSizing(meta)  # same frozen model as meta-labeling, consumed continuously
    return [
        RuleGate(adx_gate), RegimeFilter("linearity"), RegimeFilter("excursion"), RegimeFilter("move_size"),
        meta, sizing, entry, exit_,
        Combo(entry, exit_, name="Entry+Exit"),
        Combo(entry, sizing, name="Entry+Sizing"),
        Combo(exit_, sizing, name="Exit+Sizing"),
        Combo(meta, sizing, name="Best Combined (Meta-Label + Sizing)"),
    ]
