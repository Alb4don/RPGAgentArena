import random
import time
from typing import Dict, Optional

from agents.rpg_agent import RPGAgent
from core.key_manager import get_key_manager
from core.memory import save_game, head_to_head
from core.platform_utils import safe_print, hp_bar
from game.mechanics import (
    ActionType, Character, GameState,
    create_character, resolve_action, random_environment,
    CHARACTER_CLASSES,
)

_W = 62
_DIV  = "-" * _W
_HDIV = "=" * _W


def _header(game_state: GameState, char1: Character, char2: Character) -> None:
    safe_print("\n" + _HDIV)
    safe_print("  GAME #{}  |  {}".format(game_state.game_id, game_state.environment.upper()))
    safe_print("  {}".format(game_state.weather))
    safe_print(_HDIV)
    _status(char1, char2)


def _status(char1: Character, char2: Character) -> None:
    safe_print("\n  {:<24} {}".format(
        char1.name, hp_bar(char1.stats.hp, char1.stats.max_hp)
    ))
    safe_print("  {:<24} {}".format(
        char2.name, hp_bar(char2.stats.hp, char2.stats.max_hp)
    ))
    safe_print("  " + "." * (_W - 2))


def _round_header(n: int) -> None:
    safe_print("\n  " + _DIV)
    safe_print("  ROUND {}".format(n))
    safe_print("  " + _DIV)


def _narration(agent_name: str, action: ActionType, text: str, damage: int) -> None:
    safe_print("\n  [ {} -- {} ]".format(agent_name, action.value.upper()))
    for sentence in text.replace(". ", ".\n").split("\n"):
        s = sentence.strip()
        if s:
            safe_print("  {}".format(s))
    if damage > 0:
        safe_print("  -> {} damage".format(damage))


def _cost_summary() -> None:
    try:
        km = get_key_manager()
        total = km.total_cost_usd()
        if total > 0.0001:
            safe_print("\n  Cost this session: ~${:.5f} USD".format(total))
            for k in km.summary():
                if k["tokens_in"] > 0:
                    safe_print(
                        "    [{}] in={} out={} cost=${:.5f} budget_left=${:.5f}".format(
                            k["alias"], k["tokens_in"], k["tokens_out"],
                            k["cost_usd"], k["budget_remaining_usd"],
                        )
                    )
    except Exception:
        pass


def run_battle(
    agent1: RPGAgent,
    agent2: RPGAgent,
    verbose: bool = True,
    delay: float = 0.4,
) -> Optional[str]:
    env, weather = random_environment()
    game_state = GameState(environment=env, weather=weather)

    class1 = random.choice(list(CHARACTER_CLASSES.keys()))
    class2 = random.choice(list(CHARACTER_CLASSES.keys()))
    char1 = create_character(agent1.name, class1)
    char2 = create_character(agent2.name, class2)

    if verbose:
        _header(game_state, char1, char2)
        safe_print("\n  {} ({})  vs  {} ({})".format(
            agent1.name, class1, agent2.name, class2
        ))

    winner_id: Optional[str] = None
    dmg_by: Dict[str, int] = {agent1.agent_id: 0, agent2.agent_id: 0}

    while not game_state.is_over():
        game_state.advance_round()

        if verbose:
            _round_header(game_state.round_number)

        pairs = [
            (agent1, char1, char2, agent2.agent_id),
            (agent2, char2, char1, agent1.agent_id),
        ]
        if char2.stats.speed > char1.stats.speed:
            pairs = [pairs[1], pairs[0]]

        for cur_agent, cur_char, tgt_char, opp_id in pairs:
            if not cur_char.is_alive() or not tgt_char.is_alive():
                break

            if verbose:
                time.sleep(delay)

            action, narration = cur_agent.decide(cur_char, tgt_char, game_state)
            damage, _effect = resolve_action(cur_char, tgt_char, action)

            dmg_by[cur_agent.agent_id] = dmg_by.get(cur_agent.agent_id, 0) + damage
            cur_agent.record_turn_outcome(damage, tgt_char.char_class, game_state.environment)
            game_state.log_action(cur_agent.name, action.value, narration, damage)

            if verbose:
                _narration(cur_agent.name, action, narration, damage)
                _status(char1, char2)
                time.sleep(delay)

            if not tgt_char.is_alive():
                break

        if not char1.is_alive() and not char2.is_alive():
            winner_id = None
        elif not char1.is_alive():
            winner_id = agent2.agent_id
        elif not char2.is_alive():
            winner_id = agent1.agent_id

        if winner_id is not None:
            break

    if winner_id is None and game_state.round_number >= game_state.max_rounds:
        if char1.stats.hp > char2.stats.hp:
            winner_id = agent1.agent_id
        elif char2.stats.hp > char1.stats.hp:
            winner_id = agent2.agent_id

    winner_name = (
        agent1.name if winner_id == agent1.agent_id else
        agent2.name if winner_id == agent2.agent_id else
        None
    )

    if verbose:
        safe_print("\n" + _HDIV)
        if winner_name:
            safe_print("  WINNER: {}".format(winner_name.upper()))
        else:
            safe_print("  DRAW -- both fighters spent")
        safe_print(_HDIV + "\n")

    save_game(game_state, agent1.agent_id, agent2.agent_id, winner_id)

    ref1 = agent1.post_game_reflect(
        won=(winner_id == agent1.agent_id),
        opponent_id=agent2.agent_id,
        game_state=game_state,
        dmg_dealt_total=dmg_by.get(agent1.agent_id, 0),
    )
    ref2 = agent2.post_game_reflect(
        won=(winner_id == agent2.agent_id),
        opponent_id=agent1.agent_id,
        game_state=game_state,
        dmg_dealt_total=dmg_by.get(agent2.agent_id, 0),
    )

    if verbose:
        safe_print("  {}: {}\n".format(agent1.name, ref1))
        safe_print("  {}: {}\n".format(agent2.name, ref2))
        _cost_summary()

    return winner_id


def run_series(
    agent1: RPGAgent,
    agent2: RPGAgent,
    games: int = 5,
    verbose: bool = True,
    delay: float = 0.4,
) -> dict:
    results: Dict[str, int] = {
        agent1.agent_id: 0,
        agent2.agent_id: 0,
        "draws": 0,
    }

    safe_print("\n" + _HDIV)
    safe_print("  SERIES: {}  vs  {}  ({} games)".format(agent1.name, agent2.name, games))

    h2h = head_to_head(agent1.agent_id, agent2.agent_id)
    if h2h["total"] > 0:
        safe_print("  All-time H2H: {}-{}-{} (W-W-D)".format(
            h2h.get(agent1.agent_id, 0),
            h2h.get(agent2.agent_id, 0),
            h2h["draws"],
        ))
    safe_print(_HDIV)

    for i in range(games):
        safe_print("\n  -- Game {} of {} --".format(i + 1, games))
        wid = run_battle(agent1, agent2, verbose=verbose, delay=delay)
        if wid == agent1.agent_id:
            results[agent1.agent_id] += 1
        elif wid == agent2.agent_id:
            results[agent2.agent_id] += 1
        else:
            results["draws"] += 1

    safe_print("\n" + _HDIV)
    safe_print("  SERIES RESULT")
    safe_print("  {}: {}".format(agent1.name, results[agent1.agent_id]))
    safe_print("  {}: {}".format(agent2.name, results[agent2.agent_id]))
    safe_print("  Draws: {}".format(results["draws"]))
    safe_print(_HDIV + "\n")
    _cost_summary()

    return results
