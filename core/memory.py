import json
import math
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.platform_utils import get_db_path, embed_text, cosine_similarity

_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(get_db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with _lock:
        conn = _get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id     TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                char_class   TEXT NOT NULL,
                level        INTEGER DEFAULT 1,
                wins         INTEGER DEFAULT 0,
                losses       INTEGER DEFAULT 0,
                dmg_dealt    INTEGER DEFAULT 0,
                dmg_taken    INTEGER DEFAULT 0,
                pref_actions TEXT DEFAULT '{}',
                opp_models   TEXT DEFAULT '{}',
                ucb_stats    TEXT DEFAULT '{}',
                created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at   TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS games (
                game_id    TEXT PRIMARY KEY,
                agent1_id  TEXT NOT NULL,
                agent2_id  TEXT NOT NULL,
                winner_id  TEXT,
                rounds     INTEGER DEFAULT 0,
                env        TEXT,
                log        TEXT DEFAULT '[]',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS episodes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id    TEXT NOT NULL,
                situation   TEXT NOT NULL,
                embedding   TEXT NOT NULL,
                action      TEXT NOT NULL,
                outcome     REAL DEFAULT 0.0,
                opp_class   TEXT DEFAULT '',
                env         TEXT DEFAULT '',
                created_at  REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS prompt_candidates (
                prompt_id   TEXT PRIMARY KEY,
                agent_id    TEXT NOT NULL,
                text        TEXT NOT NULL,
                wins        INTEGER DEFAULT 0,
                losses      INTEGER DEFAULT 0,
                avg_dmg     REAL DEFAULT 0.0,
                avg_rounds  REAL DEFAULT 0.0,
                generation  INTEGER DEFAULT 0,
                created_at  REAL DEFAULT 0.0
            );

            CREATE INDEX IF NOT EXISTS idx_episodes_agent   ON episodes(agent_id);
            CREATE INDEX IF NOT EXISTS idx_games_agents     ON games(agent1_id, agent2_id);
            CREATE INDEX IF NOT EXISTS idx_prompt_agent     ON prompt_candidates(agent_id);
        """)
        conn.commit()
        conn.close()


@dataclass
class AgentMemory:
    agent_id: str
    name: str
    char_class: str
    level: int = 1
    wins: int = 0
    losses: int = 0
    dmg_dealt: int = 0
    dmg_taken: int = 0
    pref_actions: Dict[str, float] = field(default_factory=dict)
    opp_models: Dict[str, dict] = field(default_factory=dict)
    ucb_stats: Dict[str, dict] = field(default_factory=dict)

    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    def preferred_action_list(self) -> List[str]:
        ranked = sorted(self.pref_actions.items(), key=lambda x: x[1], reverse=True)
        return [a for a, _ in ranked[:3]]

    def record_action_outcome(self, action: str, success: bool) -> None:
        if action not in self.pref_actions:
            self.pref_actions[action] = 0.5
        cur = self.pref_actions[action]
        self.pref_actions[action] = cur * 0.85 + (1.0 if success else 0.0) * 0.15

    def update_ucb(self, action: str, reward: float) -> None:
        if action not in self.ucb_stats:
            self.ucb_stats[action] = {"total": 0.0, "plays": 0}
        self.ucb_stats[action]["total"] += reward
        self.ucb_stats[action]["plays"] += 1

    def ucb_best_action(self, candidates: List[str]) -> Optional[str]:
        if not self.ucb_stats:
            return None
        total_plays = sum(v["plays"] for v in self.ucb_stats.values())
        best_score = -1.0
        best_action: Optional[str] = None
        for action in candidates:
            stats = self.ucb_stats.get(action, {"total": 0.0, "plays": 0})
            plays = stats["plays"]
            if plays == 0:
                return action
            avg = stats["total"] / plays
            score = avg + math.sqrt(2.0 * math.log(max(1, total_plays)) / plays)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def ucb_summary(self) -> str:
        if not self.ucb_stats:
            return ""
        ranked = sorted(
            self.ucb_stats.items(),
            key=lambda x: x[1]["total"] / max(1, x[1]["plays"]),
            reverse=True,
        )
        parts = [
            "{}({:.2f})".format(a, v["total"] / max(1, v["plays"]))
            for a, v in ranked[:4]
        ]
        return "Data: " + ", ".join(parts)

    def update_opp_model(self, opp_id: str, action: str, effective: bool) -> None:
        if opp_id not in self.opp_models:
            self.opp_models[opp_id] = {"tendencies": {}}
        tend = self.opp_models[opp_id]["tendencies"]
        tend[action] = tend.get(action, 0) + (1 if effective else -1)

    def opp_insight(self, opp_id: str) -> str:
        model = self.opp_models.get(opp_id)
        if not model:
            return ""
        tend = model.get("tendencies", {})
        if not tend:
            return ""
        ranked = sorted(tend.items(), key=lambda x: x[1], reverse=True)
        effective = [a for a, v in ranked if v > 0][:2]
        weak = [a for a, v in ranked if v < 0][:2]
        parts: List[str] = []
        if effective:
            parts.append("effective: {}".format(", ".join(effective)))
        if weak:
            parts.append("less useful: {}".format(", ".join(weak)))
        return "; ".join(parts)


def load_agent(agent_id: str) -> Optional[AgentMemory]:
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM agents WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        conn.close()
    if not row:
        return None
    return AgentMemory(
        agent_id=row["agent_id"],
        name=row["name"],
        char_class=row["char_class"],
        level=row["level"],
        wins=row["wins"],
        losses=row["losses"],
        dmg_dealt=row["dmg_dealt"],
        dmg_taken=row["dmg_taken"],
        pref_actions=json.loads(row["pref_actions"]),
        opp_models=json.loads(row["opp_models"]),
        ucb_stats=json.loads(row["ucb_stats"]),
    )


def save_agent(mem: AgentMemory) -> None:
    with _lock:
        conn = _get_conn()
        conn.execute(
            """
            INSERT INTO agents
                (agent_id, name, char_class, level, wins, losses,
                 dmg_dealt, dmg_taken, pref_actions, opp_models, ucb_stats)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                level=excluded.level,
                wins=excluded.wins,
                losses=excluded.losses,
                dmg_dealt=excluded.dmg_dealt,
                dmg_taken=excluded.dmg_taken,
                pref_actions=excluded.pref_actions,
                opp_models=excluded.opp_models,
                ucb_stats=excluded.ucb_stats,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                mem.agent_id, mem.name, mem.char_class, mem.level,
                mem.wins, mem.losses, mem.dmg_dealt, mem.dmg_taken,
                json.dumps(mem.pref_actions),
                json.dumps(mem.opp_models),
                json.dumps(mem.ucb_stats),
            ),
        )
        conn.commit()
        conn.close()


def store_episode(
    agent_id: str,
    situation: str,
    action: str,
    outcome: float,
    opp_class: str = "",
    env: str = "",
) -> None:
    embedding = embed_text(situation)
    with _lock:
        conn = _get_conn()
        conn.execute(
            """
            INSERT INTO episodes
                (agent_id, situation, embedding, action, outcome, opp_class, env, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id, situation[:500], json.dumps(embedding),
                action, outcome, opp_class, env, time.time(),
            ),
        )
        conn.commit()
        conn.close()


def recall_episodes(
    agent_id: str,
    current_situation: str,
    top_k: int = 3,
    min_similarity: float = 0.25,
) -> List[dict]:
    query_emb = embed_text(current_situation)

    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT situation, embedding, action, outcome FROM episodes "
            "WHERE agent_id = ? ORDER BY created_at DESC LIMIT 120",
            (agent_id,),
        ).fetchall()
        conn.close()

    scored: List[dict] = []
    for row in rows:
        try:
            stored = json.loads(row["embedding"])
            sim = cosine_similarity(query_emb, stored)
            if sim >= min_similarity:
                scored.append({
                    "situation": row["situation"],
                    "action": row["action"],
                    "outcome": row["outcome"],
                    "similarity": sim,
                })
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


def save_game(game_state: object, agent1_id: str, agent2_id: str, winner_id: Optional[str]) -> None:
    with _lock:
        conn = _get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO games
                (game_id, agent1_id, agent2_id, winner_id, rounds, env, log)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                game_state.game_id,
                agent1_id,
                agent2_id,
                winner_id,
                game_state.round_number,
                game_state.environment,
                json.dumps(game_state.battle_log),
            ),
        )
        conn.commit()
        conn.close()


def head_to_head(agent1_id: str, agent2_id: str) -> dict:
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            """
            SELECT winner_id, COUNT(*) as cnt FROM games
            WHERE (agent1_id = ? AND agent2_id = ?)
               OR (agent1_id = ? AND agent2_id = ?)
            GROUP BY winner_id
            """,
            (agent1_id, agent2_id, agent2_id, agent1_id),
        ).fetchall()
        conn.close()

    result: dict = {"total": 0, agent1_id: 0, agent2_id: 0, "draws": 0}
    for row in rows:
        cnt = row["cnt"]
        result["total"] += cnt
        wid = row["winner_id"]
        if wid == agent1_id:
            result[agent1_id] += cnt
        elif wid == agent2_id:
            result[agent2_id] += cnt
        else:
            result["draws"] += cnt
    return result
