from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import signal_tl
import signal_tl.ast
from cmaes import CMA

from .signal_processing import SignalDict


@dataclass(frozen=True)
class FalsificationRun:
    x: np.ndarray
    trace: SignalDict
    robustness: float

    def to_dict(self):
        trace = {k: dict(times=np.array(v.times), values=np.array(v.values)) for k, v in self.trace.items()}
        return {
            "x": self.x,
            "trace": trace,
            "robustness": self.robustness,
        }


def compute_robustness(spec: signal_tl.ast.Proposition, trace: SignalDict) -> float:
    aug_trace = signal_tl.ast.augment_trace(spec, trace)
    robustness = signal_tl.compute_robustness(spec.to_evaluation_tree(), aug_trace).at(0)
    return robustness


class Falsifier:
    def __init__(
        self,
        spec: str,
        assumption: str,
        search_dim: int,
        input_signal_generator: Callable[[Sequence[float]], SignalDict],
        model: Callable[[SignalDict], SignalDict],
        seed: int | None = None,
        timeout_sec: float = 10.0,
    ) -> None:
        self.spec = spec
        self.assumption = assumption
        self.search_dim = search_dim
        self.input_signal_generator = input_signal_generator
        self.model = model
        self.optimizer = CMA(
            mean=np.zeros(search_dim),  # Initial mean
            sigma=1.3,  # Initial standard deviation. sigma=1.3 is OK for most purposes.
            bounds=None,  # Lower and upper boundaries can be specified
            seed=seed,
        )
        self.timeout_sec = timeout_sec
        self.falsified_solutions: list[FalsificationRun] = []

        self._run_before_tell_buffer: list[FalsificationRun] = []
        self._start_time = time.time()
        self._current_best = np.inf

    def reset_timer(self) -> None:
        self._start_time = time.time()

    def should_stop(self) -> bool:
        return (
            time.time() - self._start_time > self.timeout_sec or self._current_best < 0
        )

    def ask(self) -> np.ndarray:
        return self.optimizer.ask()

    def ask_for_solution(self) -> FalsificationRun | None:
        x = self.optimizer.ask()
        input_signal = self.input_signal_generator(x)
        try:
            trace = self.model(input_signal)
            ass_robustness = compute_robustness(self.assumption, trace)
            if ass_robustness <= 0:
                return None
            robustness = compute_robustness(self.spec, trace)
            return FalsificationRun(x, trace, robustness)
        except IndexError:
            raise RuntimeError(
                "Evaluation failed. Maybe some variables in spec are not in the trace?"
            )
        except Exception as e:
            print(f"Error in evaluating x={x}: {e}")
            return None

    def tell(self, run: FalsificationRun) -> None:
        self._run_before_tell_buffer.append(run)

        if run.robustness < self._current_best:
            self._current_best = run.robustness
        if run.robustness < 0:
            self.falsified_solutions.append(run)

        # Buffer solutions until population-size solutions are found
        if len(self._run_before_tell_buffer) >= self.optimizer.population_size:
            runs = self._run_before_tell_buffer[: self.optimizer.population_size]
            self.optimizer.tell([(run.x, run.robustness) for run in runs])
            self._run_before_tell_buffer = []
