"""The two data types every part of the library passes around."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Step:
    """One moment of a task attempt: what the agent saw, what it did, and the
    reward it got. `state` and `action` are raw env values of any type;
    featurizers turn them into numbers. ICRL methods never read `reward`."""

    state: Any
    action: Any
    reward: float | None  # None when the source doesn't store it (Malik's expert files)


@dataclass(frozen=True)
class Trajectory:
    """One recorded task attempt. It carries no label: the pile (expert or
    nominal) it sits in is its label."""

    steps: list[Step]
