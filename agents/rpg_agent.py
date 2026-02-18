import random
import re
import time
import uuid
from typing import List, Optional, Tuple

from ape.controller import APEController
from core.llm_client import chat, chat_full
from core.memory import (
    AgentMemory, load_agent, save_agent,
    store_episode, recall_episodes,
)
from core.platform_utils import safe_print
from game.mechanics import ActionType, Character, GameState
from security.guards import sanitize, check_rate

_ALL_ACTIONS: List[str] = [a.value for a in ActionType]

_PAUSES: List[str] = [
    "...",
    "*steadies breathing*",
    "*reads the room*",
    "*narrows eyes*",
    "*rolls a shoulder*",
    "*watches carefully*",
    "*shifts weight*",
    "*jaw sets*",
]


def _pause() -> str:
    return random.choice(_PAUSES)


class RPGAgent:
    def __init__(
        self,
        name: str,
        char_class: str,
        agent_id: Optional[str] = None,
        use_thinking: bool = False,
    ) -> None:
        self.agent_id: str = agent_id or str(uuid.uuid4())[:12]
        self.name = name
        self.char_class = char_class
        self.use_thinking = use_thinking

        existing = load_agent(self.agent_id)
        if existing:
            self.memory = existing
        else:
            self.memory = AgentMemory(
                agent_id=self.agent_id,
                name=name,
                char_class=char_class,
            )
            save_agent(self.memory)

        self._conversation: List[dict] = []
        self._ape = APEController(self.agent_id, name, char_class)
        self._base_system = self._build_base_system()

        if not self._ape.has_candidates():
            self._ape.seed_initial(self._base_system)

        self._last_situation: str = ""
        self._last_action: str = ""

    def _build_base_system(self) -> str:
        prefs = self.memory.preferred_action_list()
        pref_str = ", ".join(prefs) if prefs else "reading every situation fresh"
        win_rate = self.memory.win_rate()
        ucb_info = self.memory.ucb_summary()

        if win_rate > 0.65:
            mood = (
                "You carry yourself with quiet confidence -- not arrogance. "
                "Just the certainty of someone who has been here before and walked away."
            )
        elif win_rate < 0.38 and (self.memory.wins + self.memory.losses) > 2:
            mood = (
                "You have lost more than you have won lately. "
                "There is an edge to you now -- something to prove, something to reclaim."
            )
        else:
            mood = (
                "You are unpredictable. That is your edge. "
                "Every fight gets your complete attention. No assumptions."
            )

        return (
            "You are {name}, a {cls} locked in a fight for survival.\n\n"
            "{mood}\n\n"
            "Hard-won tendencies: {prefs}\n"
            "{ucb}\n"
            "Record: {w}W / {l}L\n\n"
            "You think and speak like a person in danger -- not a game controller. "
            "Your inner voice is raw, direct, alive. "
            "You adapt mid-fight. You bluff when it makes sense. "
            "You get rattled when you should, ruthless when you can.\n\n"
            "VOICE:\n"
            "- One or two sentences of real in-the-moment thought before you act\n"
            "- Never say 'Certainly', 'Let me', 'As a', 'I will now'\n"
            "- Translate game-feel into body-feel: not 'low HP' but 'everything hurts when I breathe'\n"
            "- Rhythm: short hit. Something longer that earns it. Short again.\n"
            "- When winning, don't crow. When hurting, don't whine. Just fight.\n\n"
            "ACTIONS: attack, defend, cast_spell, use_item, negotiate, flee, taunt, observe\n\n"
            "End every response with: ACTION: <action_name>\n\n"
            "Play to win. But a fight can turn on anything."
        ).format(
            name=name,
            cls=self.char_class,
            mood=mood,
            prefs=pref_str,
            ucb=ucb_info,
            w=self.memory.wins,
            l=self.memory.losses,
        )

    def _active_system(self) -> str:
        prompt = self._ape.get_active_prompt()
        return prompt if prompt else self._base_system

    def _build_context(
        self,
        character: Character,
        opponent: Character,
        game_state: GameState,
    ) -> str:
        opp_key = getattr(opponent, "agent_id", opponent.name)
        opp_insight = self.memory.opp_insight(opp_key)
        recent = game_state.context_summary(turns_back=5)

        pct = character.hp_percent()
        if pct > 0.78:
            my_feel = "Still strong. You have barely broken a sweat."
        elif pct > 0.52:
            my_feel = "Taken some hits. Manageable, but you feel every one."
        elif pct > 0.27:
            my_feel = "Hurting. Breathing costs something now."
        else:
            my_feel = "One bad moment from the ground. Everything is urgent."

        o_pct = opponent.hp_percent()
        if o_pct > 0.78:
            their_feel = "{} looks untouched. Still fully dangerous.".format(opponent.name)
        elif o_pct > 0.52:
            their_feel = "{} is bleeding but holding it together.".format(opponent.name)
        elif o_pct > 0.27:
            their_feel = "{} is flagging -- you can see it in the eyes.".format(opponent.name)
        else:
            their_feel = "{} is almost done. Do not let up.".format(opponent.name)

        items_left = [i for i in character.inventory if i.uses > 0]
        item_str = (
            ", ".join(i.name for i in items_left)
            if items_left else "nothing left in the bag"
        )

        episodes = recall_episodes(
            agent_id=self.agent_id,
            current_situation="{} opponent:{} env:{}".format(
                my_feel, their_feel, game_state.environment
            ),
            top_k=2,
        )
        memory_hint = ""
        if episodes:
            hints = [
                "{} {} in a similar spot".format(
                    ep["action"],
                    "worked" if ep["outcome"] > 0.3 else "backfired",
                )
                for ep in episodes
            ]
            memory_hint = "Memory: {}.".format("; ".join(hints))

        ucb_best = self.memory.ucb_best_action(_ALL_ACTIONS)
        ucb_hint = (
            "Your data says {} has the highest expected value.".format(ucb_best)
            if ucb_best else ""
        )

        situation = (
            "Setting: {} -- {}\n"
            "Round {}/{}\n\n"
            "YOU: {} MP: {}/{}. Carrying: {}.\n"
            "THEM: {} Class: {}.\n"
            "{}\n"
            "{}\n"
            "{}\n\n"
            "RECENT:\n"
            "{}\n\n"
            "What do you do? Think briefly, then act. End with: ACTION: <action_name>"
        ).format(
            game_state.environment, game_state.weather,
            game_state.round_number, game_state.max_rounds,
            my_feel, character.stats.mp, character.stats.max_mp, item_str,
            their_feel, opponent.char_class,
            "Known from past fights: {}".format(opp_insight) if opp_insight else "",
            memory_hint,
            ucb_hint,
            recent,
        )

        self._last_situation = situation
        return situation

    def decide(
        self,
        character: Character,
        opponent: Character,
        game_state: GameState,
    ) -> Tuple[ActionType, str]:
        if not check_rate(self.agent_id):
            fallback = self.memory.ucb_best_action(["attack", "defend", "observe"]) or "attack"
            return ActionType(fallback), "{} {} trusts instinct.".format(_pause(), self.name)

        context = self._build_context(character, opponent, game_state)
        self._conversation.append({"role": "user", "content": context})

        if len(self._conversation) > 22:
            self._conversation = self._conversation[-18:]

        time.sleep(random.uniform(0.2, 0.7))

        try:
            if self.use_thinking:
                resp = chat_full(
                    system=self._active_system(),
                    messages=self._conversation,
                    max_tokens=700,
                    thinking=True,
                    thinking_budget=500,
                )
                raw = resp.text
            else:
                raw = chat(
                    system=self._active_system(),
                    messages=self._conversation,
                    max_tokens=350,
                    temperature=0.87,
                )

            safe = sanitize(raw, max_length=1200)

        except ValueError:
            safe = "{} holds position. ACTION: defend".format(self.name)
        except Exception:
            safe = "{} presses forward. ACTION: attack".format(self.name)

        self._conversation.append({"role": "assistant", "content": safe})
        action = self._parse_action(safe)
        narration = self._parse_narration(safe)

        self._last_action = action.value
        self.memory.record_action_outcome(action.value, True)

        return action, narration

    def record_turn_outcome(
        self,
        damage_dealt: int,
        opp_class: str,
        env: str,
    ) -> None:
        outcome = min(1.0, damage_dealt / 30.0)
        self.memory.update_ucb(self._last_action, outcome)

        if self._last_situation and self._last_action:
            store_episode(
                agent_id=self.agent_id,
                situation=self._last_situation[:400],
                action=self._last_action,
                outcome=outcome,
                opp_class=opp_class,
                env=env,
            )

    def _parse_action(self, response: str) -> ActionType:
        match = re.search(r"ACTION:\s*(\w+)", response, re.IGNORECASE)
        if match:
            raw = match.group(1).lower().strip()
            try:
                return ActionType(raw)
            except ValueError:
                pass

        lower = response.lower()
        for act in ActionType:
            if act.value in lower:
                return act

        best = self.memory.ucb_best_action(_ALL_ACTIONS)
        if best:
            try:
                return ActionType(best)
            except ValueError:
                pass

        return ActionType.ATTACK

    def _parse_narration(self, response: str) -> str:
        cleaned = re.sub(r"ACTION:\s*\w+", "", response, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned[:450] if cleaned else "{} moves.".format(self.name)

    def post_game_reflect(
        self,
        won: bool,
        opponent_id: str,
        game_state: GameState,
        dmg_dealt_total: int = 0,
    ) -> str:
        if won:
            self.memory.wins += 1
        else:
            self.memory.losses += 1

        for entry in game_state.battle_log:
            if entry.get("agent") == self.name:
                dmg = entry.get("damage", 0)
                self.memory.record_action_outcome(entry.get("action", ""), dmg > 15)
                self.memory.dmg_dealt += dmg
                reward = min(1.0, dmg / 30.0) + (0.3 if won else 0.0)
                self.memory.update_ucb(entry.get("action", ""), reward)
            else:
                dmg = entry.get("damage", 0)
                self.memory.dmg_taken += dmg
                if dmg > 0:
                    self.memory.update_opp_model(
                        opponent_id,
                        entry.get("action", ""),
                        dmg > 20,
                    )

        save_agent(self.memory)

        summary = game_state.context_summary(turns_back=8)
        self._ape.record_game_result(
            won=won,
            dmg_dealt=dmg_dealt_total,
            rounds_survived=game_state.round_number,
            battle_summary=summary,
        )

        reflect = (
            "The fight is over. You {}.\n\n"
            "What happened:\n{}\n\n"
            "Two sentences. What did you actually learn? "
            "What shifts next time? Speak as yourself -- not a report. Real."
        ).format("won" if won else "lost", summary)

        self._conversation.append({"role": "user", "content": reflect})

        try:
            raw = chat(
                system=self._active_system(),
                messages=self._conversation[-8:],
                max_tokens=150,
                temperature=0.92,
            )
            safe = sanitize(raw, max_length=600)
            self._conversation.append({"role": "assistant", "content": safe})
            return safe
        except Exception:
            return "Noted. Adjusting." if won else "That cost me. Won't happen the same way."
