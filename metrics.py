"""
metrics.py —— 量化"飘忽"：精确细节存活率

一个编码会话里真正致命的信息，几乎都是精确串：
    端口号、环境变量名、文件路径、函数名、字段名、版本号、密码引用……

所以指标定义为：
    detail_survival = 打包后的上下文里仍能找到的精确串 / 该时刻已存在的全部精确串

摘要会把这些变成 "<细节>"；只留最近 N 轮会直接丢掉早期的；
只有"保真存储 + 按需调取"能让它们活到最后。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from context_packer import Packed


@dataclass
class Anchor:
    turn: int
    text: str          # 精确串，必须在上下文里原样出现


@dataclass
class Score:
    turn: int
    mode: str
    tokens: int
    alive: int
    total: int
    missing: List[str] = field(default_factory=list)

    @property
    def survival(self) -> float:
        return self.alive / self.total if self.total else 1.0


def score_pack(pack: Packed, anchors: Sequence[Anchor], turn: int) -> Score:
    known = [a for a in anchors if a.turn <= turn]
    missing = [a.text for a in known if a.text not in pack.text]
    return Score(turn=turn, mode=pack.mode, tokens=pack.tokens,
                 alive=len(known) - len(missing), total=len(known), missing=missing)


class Scoreboard:
    def __init__(self) -> None:
        self.rows: List[Score] = []

    def add(self, s: Score) -> None:
        self.rows.append(s)

    def by_mode(self) -> Dict[str, List[Score]]:
        out: Dict[str, List[Score]] = {}
        for r in self.rows:
            out.setdefault(r.mode, []).append(r)
        return out

    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for mode, rows in self.by_mode().items():
            rows_sorted = sorted(rows, key=lambda r: r.turn)
            tail = rows_sorted[len(rows_sorted) // 2 :] or rows_sorted
            out[mode] = {
                "final_survival": rows_sorted[-1].survival,
                "tail_survival": sum(r.survival for r in tail) / len(tail),
                "avg_tokens": sum(r.tokens for r in rows_sorted) / len(rows_sorted),
                "peak_tokens": max(r.tokens for r in rows_sorted),
                "min_survival": min(r.survival for r in rows_sorted),
            }
        return out
