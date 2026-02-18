import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from core.platform_utils import configure_console, safe_print, get_data_dir
from core.memory import init_db
from core.key_manager import get_key_manager
from agents.rpg_agent import RPGAgent
from game.engine import run_battle, run_series
from game.mechanics import CHARACTER_CLASSES


def _validate_env() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        safe_print("ERROR: ANTHROPIC_API_KEY is not set.")
        safe_print("  Add it to a .env file next to main.py, or set it as an environment variable.")
        safe_print("  For key rotation, also set ANTHROPIC_API_KEY_1 through ANTHROPIC_API_KEY_8.")
        sys.exit(1)


def _show_key_status() -> None:
    try:
        km = get_key_manager()
        keys = km.summary()
        safe_print("\n  API keys loaded: {}".format(len(keys)))
        for k in keys:
            status = "ready" if k["available"] else "unavailable"
            safe_print("    [{}] health={:.2f}  budget_left=${:.2f}  status={}".format(
                k["alias"], k["health"], k["budget_remaining_usd"], status
            ))
    except Exception as exc:
        safe_print("  Key status unavailable: {}".format(exc))


def _show_data_dir() -> None:
    d = get_data_dir()
    safe_print("  Data directory: {}".format(d))


def main() -> None:
    configure_console()
    _validate_env()
    init_db()

    class_names = list(CHARACTER_CLASSES.keys())

    parser = argparse.ArgumentParser(
        description=(
            "RPG Agent Arena -- AI agents fight, learn, and improve across sessions.\n"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples (Windows):\n"
            "  python main.py\n"
            "  python main.py --games 5\n"
            "  python main.py --name1 Kira --class1 Rogue --name2 Vorn --class2 Berserker\n"
            "  python main.py --id1 abc123 --id2 def456 --games 3\n"
            "  python main.py --thinking --games 2\n"
            "  python main.py --quiet --games 20\n"
            "  python main.py --status\n"
            "  python main.py --data-dir\n"
        ),
    )

    parser.add_argument("--name1",  default="Kira", help="Agent 1 name (default: Kira)")
    parser.add_argument("--name2",  default="Vorn", help="Agent 2 name (default: Vorn)")
    parser.add_argument("--class1", default="Rogue",     choices=class_names, help="Agent 1 class")
    parser.add_argument("--class2", default="Berserker", choices=class_names, help="Agent 2 class")
    parser.add_argument("--id1",    default=None, metavar="ID", help="Resume agent 1 by saved ID")
    parser.add_argument("--id2",    default=None, metavar="ID", help="Resume agent 2 by saved ID")
    parser.add_argument("--games",  type=int,   default=1,   help="Number of games to play (default: 1)")
    parser.add_argument("--delay",  type=float, default=0.4, help="Seconds between turns (default: 0.4)")
    parser.add_argument("--quiet",     action="store_true", help="Suppress per-turn output")
    parser.add_argument("--thinking",  action="store_true", help="Extended thinking mode for both agents")
    parser.add_argument("--thinking1", action="store_true", help="Extended thinking for agent 1 only")
    parser.add_argument("--thinking2", action="store_true", help="Extended thinking for agent 2 only")
    parser.add_argument("--status",   action="store_true", help="Show API key status and exit")
    parser.add_argument("--data-dir", action="store_true", dest="data_dir",
                        help="Show data directory path and exit")

    args = parser.parse_args()

    if args.data_dir:
        _show_data_dir()
        return

    if args.status:
        _show_key_status()
        return

    _show_key_status()

    agent1 = RPGAgent(
        name=args.name1,
        char_class=args.class1,
        agent_id=args.id1,
        use_thinking=args.thinking or args.thinking1,
    )
    agent2 = RPGAgent(
        name=args.name2,
        char_class=args.class2,
        agent_id=args.id2,
        use_thinking=args.thinking or args.thinking2,
    )

    verbose = not args.quiet

    if args.games == 1:
        run_battle(agent1, agent2, verbose=verbose, delay=args.delay)
    else:
        run_series(agent1, agent2, games=args.games, verbose=verbose, delay=args.delay)

    safe_print("\n  Agent IDs (use --id1 / --id2 to resume in future sessions):")
    safe_print("    {}: {}".format(agent1.name, agent1.agent_id))
    safe_print("    {}: {}".format(agent2.name, agent2.agent_id))


if __name__ == "__main__":
    main()
