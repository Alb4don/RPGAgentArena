import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ActionType(str, Enum):
    ATTACK     = "attack"
    DEFEND     = "defend"
    CAST_SPELL = "cast_spell"
    USE_ITEM   = "use_item"
    NEGOTIATE  = "negotiate"
    FLEE       = "flee"
    TAUNT      = "taunt"
    OBSERVE    = "observe"


class GamePhase(str, Enum):
    OPENING = "opening"
    EARLY   = "early"
    MID     = "mid"
    LATE    = "late"
    FINAL   = "final"


@dataclass
class Stats:
    hp:      int = 100
    max_hp:  int = 100
    mp:      int = 50
    max_mp:  int = 50
    attack:  int = 15
    defense: int = 10
    speed:   int = 12
    luck:    int = 8


@dataclass
class Item:
    name:   str
    effect: str
    uses:   int = 1
    power:  int = 20


@dataclass
class Character:
    name:           str
    char_class:     str
    stats:          Stats            = field(default_factory=Stats)
    inventory:      List[Item]       = field(default_factory=list)
    status_effects: List[str]        = field(default_factory=list)
    xp:             int              = 0
    level:          int              = 1

    def is_alive(self) -> bool:
        return self.stats.hp > 0

    def hp_percent(self) -> float:
        if self.stats.max_hp <= 0:
            return 0.0
        return self.stats.hp / self.stats.max_hp

    def take_damage(self, amount: int) -> int:
        reduction = max(0, self.stats.defense - random.randint(0, 5))
        actual = max(1, amount - reduction)
        self.stats.hp = max(0, self.stats.hp - actual)
        return actual

    def heal(self, amount: int) -> int:
        gain = min(amount, self.stats.max_hp - self.stats.hp)
        self.stats.hp += gain
        return gain

    def phase(self) -> GamePhase:
        pct = self.hp_percent()
        if pct > 0.85:
            return GamePhase.OPENING
        if pct > 0.60:
            return GamePhase.EARLY
        if pct > 0.35:
            return GamePhase.MID
        if pct > 0.15:
            return GamePhase.LATE
        return GamePhase.FINAL


@dataclass
class GameState:
    game_id:     str       = field(default_factory=lambda: str(uuid.uuid4())[:8])
    round_number: int      = 0
    max_rounds:  int       = 20
    environment: str       = "ancient ruins"
    weather:     str       = "stormy night"
    battle_log:  List[dict] = field(default_factory=list)
    winner:      Optional[str] = None

    def advance_round(self) -> None:
        self.round_number += 1

    def is_over(self) -> bool:
        return self.winner is not None or self.round_number >= self.max_rounds

    def log_action(self, agent_name: str, action: str, narration: str, damage: int = 0) -> None:
        self.battle_log.append({
            "round":    self.round_number,
            "agent":    agent_name,
            "action":   action,
            "narration": narration,
            "damage":   damage,
        })

    def context_summary(self, turns_back: int = 5) -> str:
        recent = self.battle_log[-turns_back:] if self.battle_log else []
        if not recent:
            return "The battle has just begun."
        lines = [
            "Round {} -- {}: {}".format(e["round"], e["agent"], e["narration"])
            for e in recent
        ]
        return "\n".join(lines)


ENVIRONMENTS: List[Tuple[str, str]] = [
    ("ancient ruins",    "stormy night"),
    ("enchanted forest", "misty dawn"),
    ("volcanic crater",  "scorching noon"),
    ("frozen tundra",    "blizzard"),
    ("haunted castle",   "moonless midnight"),
    ("sky fortress",     "thunderstorm"),
    ("sunken temple",    "eerie calm"),
    ("desert canyon",    "blazing heat"),
]

CHARACTER_CLASSES: Dict[str, Stats] = {
    "Berserker": Stats(hp=130, max_hp=130, mp=20,  max_mp=20,  attack=22, defense=8,  speed=10, luck=6),
    "Mage":      Stats(hp=70,  max_hp=70,  mp=100, max_mp=100, attack=10, defense=6,  speed=14, luck=10),
    "Paladin":   Stats(hp=110, max_hp=110, mp=60,  max_mp=60,  attack=14, defense=18, speed=9,  luck=12),
    "Rogue":     Stats(hp=85,  max_hp=85,  mp=40,  max_mp=40,  attack=18, defense=9,  speed=20, luck=16),
    "Shaman":    Stats(hp=90,  max_hp=90,  mp=80,  max_mp=80,  attack=12, defense=11, speed=11, luck=14),
    "Knight":    Stats(hp=120, max_hp=120, mp=30,  max_mp=30,  attack=16, defense=20, speed=8,  luck=8),
}

STARTER_ITEMS: Dict[str, List[Item]] = {
    "Berserker": [Item("War Elixir", "attack_boost", uses=2, power=8),
                  Item("Bandage",    "heal",         uses=3, power=25)],
    "Mage":      [Item("Mana Crystal",   "mp_restore",   uses=3, power=30),
                  Item("Scroll of Fire", "spell_damage", uses=2, power=35)],
    "Paladin":   [Item("Holy Water",  "heal",          uses=3, power=35),
                  Item("Shield Charm", "defense_boost", uses=2, power=10)],
    "Rogue":     [Item("Smoke Bomb",  "evasion", uses=2, power=15),
                  Item("Poison Vial", "poison",  uses=2, power=20)],
    "Shaman":    [Item("Spirit Totem", "mp_restore", uses=2, power=25),
                  Item("Hex Powder",  "curse",      uses=2, power=18)],
    "Knight":    [Item("Iron Ration", "heal",          uses=2, power=20),
                  Item("War Banner",  "defense_boost", uses=1, power=15)],
}


def create_character(name: str, char_class: str) -> Character:
    stats = CHARACTER_CLASSES.get(char_class, Stats())
    items = [
        Item(i.name, i.effect, i.uses, i.power)
        for i in STARTER_ITEMS.get(char_class, [])
    ]
    return Character(name=name, char_class=char_class, stats=stats, inventory=items)


def resolve_action(
    attacker: Character,
    defender: Character,
    action: ActionType,
) -> Tuple[int, str]:
    damage = 0
    effect = ""

    if action == ActionType.ATTACK:
        crit = random.random() < (attacker.stats.luck / 100.0)
        mult = 1.8 if crit else 1.0
        damage = int(attacker.stats.attack * mult * random.uniform(0.85, 1.15))
        defender.take_damage(damage)
        effect = "critical hit" if crit else "hit"

    elif action == ActionType.CAST_SPELL:
        if attacker.stats.mp >= 10:
            attacker.stats.mp -= 10
            damage = int(attacker.stats.attack * 1.5 * random.uniform(0.9, 1.2))
            defender.take_damage(damage)
            effect = "spell"
        else:
            effect = "fizzle (no MP)"

    elif action == ActionType.DEFEND:
        attacker.stats.defense += 4
        effect = "defended"

    elif action == ActionType.USE_ITEM:
        usable = [i for i in attacker.inventory if i.uses > 0]
        if usable:
            item = random.choice(usable)
            item.uses -= 1
            if item.effect == "heal":
                gained = attacker.heal(item.power)
                effect = "used {}, healed {} HP".format(item.name, gained)
            elif item.effect in ("spell_damage", "poison", "curse"):
                damage = item.power
                defender.take_damage(damage)
                effect = "used {}".format(item.name)
            elif item.effect == "attack_boost":
                attacker.stats.attack += item.power
                effect = "used {} (+{} ATK)".format(item.name, item.power)
            elif item.effect == "defense_boost":
                attacker.stats.defense += item.power
                effect = "used {} (+{} DEF)".format(item.name, item.power)
            elif item.effect == "mp_restore":
                attacker.stats.mp = min(attacker.stats.max_mp, attacker.stats.mp + item.power)
                effect = "used {} (+{} MP)".format(item.name, item.power)
            else:
                effect = "used {}".format(item.name)
        else:
            effect = "no items remaining"

    elif action == ActionType.TAUNT:
        effect = "taunted"

    elif action == ActionType.NEGOTIATE:
        effect = "attempted negotiation"

    elif action == ActionType.FLEE:
        effect = "attempted to flee"

    elif action == ActionType.OBSERVE:
        effect = "observed carefully"

    if action != ActionType.DEFEND and random.random() < 0.04:
        attacker.stats.defense = max(5, attacker.stats.defense - 3)

    return damage, effect


def random_environment() -> Tuple[str, str]:
    return random.choice(ENVIRONMENTS)
