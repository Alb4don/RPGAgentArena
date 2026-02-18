import math
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from core.llm_client import chat
from core.platform_utils import get_db_path

_lock = threading.Lock()

APE_EVAL_EVERY = 5
APE_CANDIDATES = 3
APE_MIN_GAMES = 3
APE_MAX_POOL = 12


@dataclass
class PromptCandidate:
    prompt_id: str
    agent_id: str
    text: str
    wins: int = 0
    losses: int = 0
    avg_dmg: float = 0.0
    avg_rounds: float = 0.0
    generation: int = 0
    created_at: float = field(default_factory=time.time)

    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    def ucb_score(self, total_evals: int) -> float:
        total = self.wins + self.losses
        if total == 0:
            return float("inf")
        exploit = self.win_rate()
        explore = math.sqrt(2.0 * math.log(max(1, total_evals)) / total)
        dmg_bonus = min(0.15, self.avg_dmg / 600.0)
        return exploit + explore + dmg_bonus


def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(get_db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _load_candidates(agent_id: str) -> List[PromptCandidate]:
    with _lock:
        conn = _db_conn()
        rows = conn.execute(
            "SELECT * FROM prompt_candidates WHERE agent_id = ? "
            "ORDER BY generation DESC, wins DESC",
            (agent_id,),
        ).fetchall()
        conn.close()
    return [
        PromptCandidate(
            prompt_id=r["prompt_id"],
            agent_id=r["agent_id"],
            text=r["text"],
            wins=r["wins"],
            losses=r["losses"],
            avg_dmg=r["avg_dmg"],
            avg_rounds=r["avg_rounds"],
            generation=r["generation"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


def _save_candidate(c: PromptCandidate) -> None:
    with _lock:
        conn = _db_conn()
        conn.execute(
            """
            INSERT INTO prompt_candidates
                (prompt_id, agent_id, text, wins, losses, avg_dmg, avg_rounds, generation, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(prompt_id) DO UPDATE SET
                wins=excluded.wins,
                losses=excluded.losses,
                avg_dmg=excluded.avg_dmg,
                avg_rounds=excluded.avg_rounds
            """,
            (
                c.prompt_id, c.agent_id, c.text, c.wins, c.losses,
                c.avg_dmg, c.avg_rounds, c.generation, c.created_at,
            ),
        )
        conn.commit()
        conn.close()


def _update_candidate_result(
    prompt_id: str,
    won: bool,
    dmg: int,
    rounds: int,
) -> None:
    with _lock:
        conn = _db_conn()
        row = conn.execute(
            "SELECT * FROM prompt_candidates WHERE prompt_id = ?",
            (prompt_id,),
        ).fetchone()
        if row:
            alpha = 0.3
            new_dmg = row["avg_dmg"] * (1.0 - alpha) + dmg * alpha
            new_rnd = row["avg_rounds"] * (1.0 - alpha) + rounds * alpha
            conn.execute(
                """
                UPDATE prompt_candidates
                SET wins = wins + ?,
                    losses = losses + ?,
                    avg_dmg = ?,
                    avg_rounds = ?
                WHERE prompt_id = ?
                """,
                (1 if won else 0, 0 if won else 1, new_dmg, new_rnd, prompt_id),
            )
            conn.commit()
        conn.close()


def _select_ucb1(candidates: List[PromptCandidate]) -> PromptCandidate:
    if not candidates:
        raise ValueError("No prompt candidates available")
    total = sum(c.wins + c.losses for c in candidates)
    return max(candidates, key=lambda c: c.ucb_score(total))


def _generate_variants(
    base_prompt: str,
    agent_name: str,
    char_class: str,
    win_rate: float,
    feedback: str,
    n: int = 3,
) -> List[str]:
    meta = (
        "You are an expert at writing AI agent system prompts for RPG combat games.\n\n"
        "Current prompt for {name} (a {cls}):\n"
        "<CURRENT_PROMPT>\n{prompt}\n</CURRENT_PROMPT>\n\n"
        "Win rate: {wr:.1%}\n"
        "Recent battle feedback: {fb}\n\n"
        "Generate {n} improved variants. Each must:\n"
        "- Keep the name ({name}) and class ({cls})\n"
        "- Sound like a real person under pressure, not a game bot\n"
        "- Try a different strategic emphasis or emotional angle\n"
        "- Stay under 600 words\n"
        "- End with: ACTION: <action_name>\n\n"
        "Return exactly {n} variants like this:\n"
        "<VARIANT>\n...prompt...\n</VARIANT>"
    ).format(
        name=agent_name,
        cls=char_class,
        prompt=base_prompt[:1200],
        wr=win_rate,
        fb=feedback,
        n=n,
    )

    try:
        resp = chat(
            system="Return only the variant prompts in the requested format. No preamble.",
            messages=[{"role": "user", "content": meta}],
            max_tokens=2200,
            temperature=0.93,
        )
        parts = re.findall(r"<VARIANT>(.*?)</VARIANT>", resp, re.DOTALL)
        return [p.strip() for p in parts if len(p.strip()) > 80][:n]
    except Exception:
        return []


class APEController:
    def __init__(self, agent_id: str, agent_name: str, char_class: str) -> None:
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.char_class = char_class
        self._candidates: List[PromptCandidate] = _load_candidates(agent_id)
        self._current: Optional[PromptCandidate] = None
        self._games_since_ape: int = 0
        self._lock = threading.Lock()

    def has_candidates(self) -> bool:
        return len(self._candidates) > 0

    def seed_initial(self, base_prompt: str) -> None:
        c = PromptCandidate(
            prompt_id="seed_{}".format(self.agent_id),
            agent_id=self.agent_id,
            text=base_prompt,
            generation=0,
            created_at=time.time(),
        )
        _save_candidate(c)
        self._candidates = [c]

    def get_active_prompt(self) -> Optional[str]:
        with self._lock:
            if not self._candidates:
                return None
            self._current = _select_ucb1(self._candidates)
            return self._current.text

    def record_game_result(
        self,
        won: bool,
        dmg_dealt: int,
        rounds_survived: int,
        battle_summary: str,
    ) -> None:
        with self._lock:
            if self._current:
                _update_candidate_result(
                    self._current.prompt_id,
                    won,
                    dmg_dealt,
                    rounds_survived,
                )
                for c in self._candidates:
                    if c.prompt_id == self._current.prompt_id:
                        if won:
                            c.wins += 1
                        else:
                            c.losses += 1
                        break

            self._games_since_ape += 1
            total = sum(c.wins + c.losses for c in self._candidates)
            if self._games_since_ape >= APE_EVAL_EVERY and total >= APE_MIN_GAMES:
                self._games_since_ape = 0
                self._evolve(battle_summary)

    def _evolve(self, feedback: str) -> None:
        if not self._candidates:
            return
        best = max(self._candidates, key=lambda c: c.win_rate())
        current_gen = max(c.generation for c in self._candidates)

        variants = _generate_variants(
            base_prompt=best.text,
            agent_name=self.agent_name,
            char_class=self.char_class,
            win_rate=best.win_rate(),
            feedback=feedback,
            n=APE_CANDIDATES,
        )

        for i, vtext in enumerate(variants):
            cid = "ape_{}_{}_{:d}".format(self.agent_id, int(time.time()), i)
            c = PromptCandidate(
                prompt_id=cid,
                agent_id=self.agent_id,
                text=vtext,
                generation=current_gen + 1,
                created_at=time.time(),
            )
            _save_candidate(c)
            self._candidates.append(c)

        if len(self._candidates) > APE_MAX_POOL:
            total_e = sum(c.wins + c.losses for c in self._candidates)
            self._candidates.sort(key=lambda c: c.ucb_score(total_e), reverse=True)
            to_prune = self._candidates[APE_MAX_POOL - 2:]
            self._candidates = self._candidates[: APE_MAX_POOL - 2]
            with _lock:
                conn = _db_conn()
                for c in to_prune:
                    conn.execute(
                        "DELETE FROM prompt_candidates "
                        "WHERE prompt_id = ? AND wins + losses < 2",
                        (c.prompt_id,),
                    )
                conn.commit()
                conn.close()
