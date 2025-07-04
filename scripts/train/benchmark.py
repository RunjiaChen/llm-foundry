# benchmark_speed.py  (lint-friendly)

import sys, time, torch
from typing import Optional
from composer.core import Callback, State, Time
from composer.loggers import Logger 
from composer.utils import dist
from llmfoundry.command_utils import train_from_yaml


# ---------------------------------------------------------------------------
class StageTimer(Callback):
    """Record forward/backward wall-clock time per micro-batch."""

    def __init__(self, warmup: int = 5):
        self.warmup = warmup
        self.step   = 0
        self._t_forward_start: Optional[float] = None
        self._t_after_fwd     : Optional[float] = None
        self.fwd_times, self.bwd_times = [], []

    # -- helpers ----------------------------------------------------------------
    @staticmethod
    def _now() -> float:
        torch.cuda.synchronize()
        return time.perf_counter()

    # -- hook implementations ----------------------------------------------------
    def before_forward(self, state: State, logger: Logger):
        self._t_forward_start = self._now()

    def after_forward(self, state: State, logger: Logger):
        if self._t_forward_start is None:
            return
        self._t_after_fwd = self._now()
        if self.step >= self.warmup:
            self.fwd_times.append(self._t_after_fwd - self._t_forward_start)

    def after_backward(self, state: State, logger: Logger):
        if self._t_after_fwd is None:
            return
        t_after_bwd = self._now()
        if self.step >= self.warmup:
            self.bwd_times.append(t_after_bwd - self._t_after_fwd)
        # advance step counter *after* timing so fwd/bwd belong to same step
        self.step += 1

    # Print simple stats at the very end (rank-0 only)
    def fit_end(self, state: State, logger: Logger):
        if dist.get_global_rank() != 0:
            return
        import math
        mean = lambda xs: float("nan") if not xs else sum(xs) / len(xs)
        print("\n===== TRAIN-TIME SPEED REPORT =====")
        print(f"steps counted      : {len(self.fwd_times)}")
        print(f"mean forward  (s)  : {mean(self.fwd_times):.6f}")
        print(f"mean backward (s)  : {mean(self.bwd_times):.6f}")
        print("===================================\n")


# ---------------------------------------------------------------------------
@torch.inference_mode()
def measure_decode(model, vocab_size: int,
                   prompt_len=128, gen_len=128, n_iters=20) -> float:
    device = torch.device("cuda")
    prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = model.generate(prompt, max_new_tokens=gen_len)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (gen_len * n_iters) / elapsed   # tokens / sec


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    yaml_path, cli_overrides = sys.argv[1], sys.argv[2:]

    # Build Trainer (but don’t launch fit yet)
    trainer = train_from_yaml(yaml_path, cli_overrides)

    # Attach our callback
    trainer.state.callbacks.append(StageTimer(warmup=5))

    # Optionally shorten run so we don’t train forever; change if you like.
    trainer.state.max_duration = Time.from_timestring("100ba")

    # ---- forward + backward benchmark ----------------------------------------
    trainer.fit()

    # ---- decode benchmark -----------------------------------------------------
    if dist.get_global_rank() == 0:
        model = trainer.state.model  # type: ignore[attr-defined]
        tps = measure_decode(
            model,
            vocab_size=getattr(model, "tokenizer", None).vocab_size
                      if hasattr(model, "tokenizer") else 50272   # fallback
        )
        print(f"Decode throughput : {tps:.1f} tokens / sec")

    dist.barrier()
