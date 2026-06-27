"""Rich-backed training logs (terminal + optional plain-text file)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from rich.console import Console
from rich.theme import Theme

_THEME = Theme(
    {
        "tag.train": "bold cyan",
        "tag.val": "bold magenta",
        "tag.done": "bold green",
        "ok": "green",
        "bad": "red",
        "warn": "yellow",
        "dim": "dim",
    }
)

# SB3 metric key -> compact label (all train losses included).
_TRAIN_LABELS = (
    ("train/loss", "loss"),
    ("train/policy_gradient_loss", "pg"),
    ("train/value_loss", "vf"),
    ("train/entropy_loss", "ent"),
    ("train/approx_kl", "kl"),
    ("train/clip_fraction", "clip"),
    ("train/clip_range", "clip_rng"),
    ("train/explained_variance", "ev"),
    ("train/learning_rate", "lr"),
)


class TrainingLog:
    """Mirror compact one-line Rich output to terminal and an optional log file."""

    def __init__(self, log_file: Optional[str] = None) -> None:
        self.console = Console(theme=_THEME, soft_wrap=True)
        self._file_console: Optional[Console] = None
        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file_console = Console(
                file=path.open("w", encoding="utf-8"),
                theme=_THEME,
                force_terminal=False,
                width=200,
                soft_wrap=True,
            )

    def _emit(self, text: str) -> None:
        self.console.print(text)
        if self._file_console is not None:
            self._file_console.print(text)

    def info(self, message: str) -> None:
        self._emit(f"[dim]{message}[/]")

    def ppo_iteration(self, iteration: int, metrics: Mapping[str, Any], timesteps: int) -> None:
        chunks = [
            "[tag.train]train[/]",
            f"iter {iteration}",
            f"{timesteps:,} steps",
        ]
        if "time/fps" in metrics:
            chunks.append(f"{_fmt(metrics['time/fps'])} fps")
        if "rollout/ep_rew_mean" in metrics:
            chunks.append(f"ep_rew {_fmt(metrics['rollout/ep_rew_mean'])}")
        if "rollout/ep_len_mean" in metrics:
            chunks.append(f"ep_len {_fmt(metrics['rollout/ep_len_mean'])}")

        for key, label in _TRAIN_LABELS:
            if key in metrics:
                chunks.append(f"{label} {_fmt(metrics[key])}")

        self._emit(" │ ".join(chunks))

    def validation(
        self,
        *,
        iteration: int,
        epoch: Optional[int],
        timesteps: int,
        metrics: Mapping[str, float],
    ) -> None:
        ret = metrics["cumulative_return"] * 100
        ret_txt = f"[{'ok' if ret >= 0 else 'bad'}]{ret:+.2f}%[/]"
        chunks = [
            "[tag.val]val[/]",
            f"iter {iteration}",
        ]
        if epoch is not None:
            chunks.append(f"epoch {epoch}")
        chunks.extend(
            [
                f"{timesteps:,} steps",
                f"return {ret_txt}",
                f"sharpe {_fmt(metrics['sharpe'])}",
                f"max_dd {metrics['max_drawdown'] * 100:.2f}%",
                f"equity ${metrics['final_equity']:,.0f}",
                f"trades {metrics['round_trips']}",
            ]
        )
        self._emit(" │ ".join(chunks))

    def metrics_report(
        self,
        title: str,
        reproduced: Mapping[str, float],
        paper: Mapping[str, float],
        extra_lines: Optional[list[str]] = None,
    ) -> None:
        self._emit(f"[tag.done]{title}[/]")
        self._emit(
            " │ ".join(
                [
                    f"return {reproduced['cumulative_return'] * 100:+.2f}% (paper {paper['cumulative_return'] * 100:.2f}%)",
                    f"CAGR {reproduced['cagr'] * 100:.2f}%",
                    f"Sharpe {_fmt(reproduced['sharpe'])}",
                    f"max_dd {reproduced['max_drawdown'] * 100:.2f}%",
                    f"win_rate {reproduced['trade_win_rate'] * 100:.2f}%",
                ]
            )
        )
        if extra_lines:
            for line in extra_lines:
                self._emit(f"[dim]{line}[/]")

    def close(self) -> None:
        if self._file_console is not None:
            self._file_console.file.close()


def _fmt(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(value) >= 1000 or (0 < abs(value) < 1e-3):
        return f"{value:.4g}"
    return f"{value:.4f}"


def silence_sb3_stdout() -> None:
    """Route SB3's default logger away from stdout (we print our own lines)."""
    from stable_baselines3.common import utils as sb3_utils

    _orig = sb3_utils.configure_logger

    def _configure_logger(verbose: int = 0, *args, **kwargs):
        return _orig(0, *args, **kwargs)

    sb3_utils.configure_logger = _configure_logger
