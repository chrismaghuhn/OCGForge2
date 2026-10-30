"""K1 of the combo teacher: replay community combos in our engine (docs/combo-teacher-design.md).

    python -m ocgforge.combo_check --deck decks/elfnote.ydk --deck-type Elfnote --checkpoint runs/x/latest

Each combo becomes a solo puzzle: we go first with its starting cards (filled up to 5 from the
decklist), the opponent holds a hand but never acts. A beam search over the engine's answers follows the combo's
steps. A step is done when its card got where the step's action text says (summoned, added to hand,
sent to the GY, ...; or, for a bare "use", named by an answer), checked on the board after each
answer; the model ranks everything the steps do not decide (materials, zones, targets), and the
book's end-board cards on the field rank lines too. A combo counts as checked when every step is
done in order and at least 80 % of the book's end-board cards are on our field. The
end board and the line (semantic keys) go into the combo book. Several deck shuffles per combo
(--shuffles) keep draws and excavations honest: a combo only counts if it works in all of them.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import combos as cb
from . import features, rollout, worlds

TYPE_EXTRA = 0x40 | 0x2000 | 0x800000 | 0x4000000  # Fusion, Synchro, Xyz, Link
PHASE_END = re.compile(r"^idle\.[6-9]\.\d+$")  # to Battle Phase / End Phase (no card)
# Monsters that want the center Main Monster Zone (summoned there, or stronger while there).
CENTER = re.compile(r"(this card|Summoned|it)[^.]{0,40}\b(to|in) (the|your) center Main Monster Zone"
                    r"|while in the center Main Monster Zone")
# Answers that move the duel on and cannot loop: the board stays the same over several chain windows,
# so the loop guard (same answer on the same board) must not take them away.
PASSIVE = ("chain.pass", "yes_no.", "effect_yes_no.")
POS_ATTACK = 0x1
MAX_SKIPPED = 2  # steps a checked combo may pass over (once-per-turn effects, steps the board cannot show)
EMZ = (5, 6)


def wants_center(text: str) -> bool:
    return bool(CENTER.search(text or ""))


def scenario_script(combo: cb.Combo, decklist: tuple[list[int], list[int]], opponent: list[int], is_extra,
                    rng: random.Random, hand_size: int = 5, hand: list[int] | None = None) -> str:
    """The puzzle: our hand (the combo's starting cards, filled up from the decklist), our deck and
    Extra Deck (the decklist plus the combo's cards it lacks), an opponent with a deck and a hand it
    never plays (effects like PSY-Framelord Omega's need one)."""
    main, extra = list(decklist[0]), list(decklist[1])
    for code in combo.cards():  # the combo's cards we do not run
        target = extra if is_extra(code) else main
        if code not in target:
            target.append(code)
    wanted = list(hand) if hand is not None else [e.code for e in combo.starting if e.code is not None]
    hand = []
    for code in wanted:
        if code in main:
            main.remove(code)
        hand.append(code)
    rng.shuffle(main)
    while len(hand) < hand_size and main:
        hand.append(main.pop())
    lines = ["Debug.ReloadFieldBegin(0,5)", "Debug.SetPlayerInfo(0,8000,0,0)", "Debug.SetPlayerInfo(1,8000,0,0)"]
    for i, code in enumerate(main):
        lines.append(f"Debug.AddCard({code},0,0,{worlds.LOCATION_DECK},{i},8)")
    for code in extra:
        lines.append(f"Debug.AddCard({code},0,0,{worlds.LOCATION_EXTRA},0,8)")
    for i, code in enumerate(hand):
        lines.append(f"Debug.AddCard({code},0,0,{worlds.LOCATION_HAND},{i},10)")
    opponent = list(opponent)
    rng.shuffle(opponent)
    for i, code in enumerate(opponent[hand_size:]):
        lines.append(f"Debug.AddCard({code},1,1,{worlds.LOCATION_DECK},{i},8)")
    for i, code in enumerate(opponent[:hand_size]):  # it never plays them, but effects may target a hand
        lines.append(f"Debug.AddCard({code},1,1,{worlds.LOCATION_HAND},{i},10)")
    lines.append("Debug.ReloadFieldEnd()")
    return "\n".join(lines) + "\n"


@torch.no_grad()
def candidate_logprobs(model, rows: list[dict], device: str) -> list[list[float]]:
    model.eval()
    inputs = features.collate(rows, device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
        scores, _ = model(inputs)
    logprobs = torch.log_softmax(scores.float(), dim=-1).cpu()
    return [logprobs[i, : len(r["candidate_kind"])].tolist() for i, r in enumerate(rows)]


# What a step's action text says happened to its card, as the location the card has to reach. None
# means "the card was used" (its effect activated, it was selected, ...): any answer naming it.
VERBS = [
    (("synchro", "xyz", "link", "fusion", "ritual", "summon", "revive", "special"), worlds.LOCATION_MZONE),
    (("add", "search", "hand", "draw", "return"), worlds.LOCATION_HAND),
    (("send", "gy", "grave", "tribute", "discard", "material", "detach", "mill", "dump"), worlds.LOCATION_GRAVE),
    (("banish",), worlds.LOCATION_REMOVED),
    (("place", "set", "scale", "field", "szone"), worlds.LOCATION_SZONE),
]


def expected_location(action: str) -> int | None:
    text = action.lower()
    for words, location in VERBS:
        if any(w in text for w in words):
            return location
    return None


def board(duel) -> Counter:
    """(passcode, location) -> count of our cards, for telling what an answer did."""
    return Counter((c["code"], c["location"]) for c in duel.snapshot()["cards"] if c["controller"] == 0)


@dataclass
class Line:
    keys: list[str] = field(default_factory=list)  # every answer, both players
    progress: int = 0  # steps done so far
    score: float = 0.0  # the model's log-probabilities, plus bonuses
    board: Counter = field(default_factory=Counter)  # our cards after the last answer
    key_cards: int = 0  # book end-board cards on our field now
    tried: frozenset = frozenset()  # (board, answer) pairs this line already played: no loops
    skipped: int = 0  # steps passed over without being done
    duel: object = field(default=None, repr=False, compare=False)  # the engine at this line's end (a clone)


def advance(line: Line, targets: list[tuple[int, int | None]], last_key: str | None, now: Counter,
            lookahead: int) -> tuple[int, int]:
    """How many steps the last answer completed, and how many of them it skipped: a step with a
    location is done when its card arrived there; one without, when the answer named its card. A few
    steps may be skipped (the combo text names cards the engine never acts on, or an effect that is
    once per turn), but a skip costs the line rank."""
    arrived = {code_loc for code_loc, n in (now - line.board).items() if n > 0}
    named = worlds.card_ref(last_key) if last_key else None
    progress, skipped = line.progress, 0
    for _ in range(4):  # one answer can complete several steps (a summon that is also a material)
        for skip in range(lookahead):
            i = progress + skip
            if i >= len(targets):
                break
            code, location = targets[i]
            # A step without a location (the card was used) only as the very next step: the answer
            # summoning a card names it too, before the card arrives for the step before.
            if (location is not None and (code, location) in arrived) or (
                    location is None and skip == 0 and named is not None and int(named[1]) == code):
                progress, skipped = i + 1, skipped + skip
                if location is not None:
                    arrived.discard((code, location))
                break
        else:
            break
    return progress - line.progress, skipped


def placing(keys: list[str]) -> int | None:
    """The card a zone question is about: the last summon, activation or card choice of ours
    (materials and costs are chosen after it, so selections of several cards do not count)."""
    for key in reversed(keys):
        if key.startswith(("idle.", "card.", "chain.")):
            ref = worlds.card_ref(key)
            if ref is not None:
                return int(ref[1])
    return None


def forced(keys: list[str], history: list[str], upcoming_center: bool, center: set[int], is_extra) -> str | None:
    """The answer the rules fix without searching (docs/combo-teacher-design.md, K1 finding), or None.
    - Position: face-up attack.
    - Monster zone: the center for a monster that wants it; the Extra Monster Zone for another Extra
      Deck monster; else a side zone, keeping the center free when the plan still needs it."""
    if all(k.startswith("position.") for k in keys):
        attack = [k for k in keys if k.rsplit(".", 1)[-1].isdigit() and int(k.rsplit(".", 1)[-1]) & POS_ATTACK]
        return (attack or keys)[0]
    zones = {int(k.split(".")[3]): k for k in keys if k.startswith("place.0.4.") and len(k.split(".")) == 4}
    if zones and len(zones) == len(keys):
        code = placing(history)
        if code in center and 2 in zones:
            return zones[2]
        if code is not None and is_extra(code) and code not in center:
            for z in EMZ:
                if z in zones:
                    return zones[z]
        sides = [z for z in (1, 3, 0, 4) if z in zones]
        if sides:
            return zones[sides[0]]
        return zones[2] if 2 in zones and not upcoming_center else next(iter(zones.values()))
    return None


def plan_choice(keys: list[str], allowed: set[int], ahead: list[int]) -> set[int]:
    """In a choice of one card (`card.` answers: a search, a target, a monster to summon), only the
    card of the earliest coming step that is offered; the model must not swap it for another legal
    card (Welcome Home calling Fortuna where the plan wants Regina). Unchanged when none is offered."""
    if not all(k.startswith("card.") for k in keys):
        return allowed
    codes = {i: int(ref[1]) for i in allowed if (ref := worlds.card_ref(keys[i])) is not None}
    for code in ahead:
        chosen = {i for i, c in codes.items() if c == code}
        if chosen:
            return chosen
    return allowed


def grow(children: list[tuple[Line, object]]) -> list[Line]:
    """Play each chosen child's last answer on a clone of its parent's duel; the parent's last child
    takes the parent's duel itself. A child whose answer the engine rejects is dropped."""
    left = Counter(id(parent) for _, parent in children)
    grown = []
    for child, parent in children:
        left[id(parent)] -= 1
        try:
            duel = parent if left[id(parent)] == 0 else parent.clone()
            duel.act(child.keys[-1])
        except Exception:
            continue
        child.duel = duel
        grown.append(child)
    return grown


def check(combo: cb.Combo, start, model, card_index, device: str, beam: int = 12, branch: int = 4,
          max_decisions: int = 150, lookahead: int = 3, center: set[int] = frozenset(),
          is_extra=lambda code: False) -> tuple[Line, list[tuple[int, int | None]]]:
    """Beam search through the combo from a fresh puzzle duel (`start()`); returns the best line and
    the steps (card, expected location) it had to follow. Lines rank by steps done, then by how many
    of the book's end-board cards are on the field, then by the model's log-probabilities.

    Rules from playing Lucina 1cc by hand: positions and monster zones are fixed (`forced`); in a
    chain window only cards of the next steps may activate, else pass; in a choice of one card only the
    next step's card when offered (`plan_choice`); optional yes/no questions keep
    both answers in the beam (the board decides, not the model: a level change can break a later
    Synchro); end-board cards on the field are never chosen as material or cost."""
    targets = [(e.code, expected_location(e.action)) for step in combo.steps for e in step if e.code is not None]
    key_cards = Counter(combo.end_board)
    plan = {code for code, _ in targets} | set(key_cards)
    center = set(center) & plan
    # Steps first: an end-board card played early (a search taken too soon) must not beat the plan.
    rank = lambda l: (l.progress - l.skipped, l.key_cards, l.score)  # steps actually done  # noqa: E731
    first = start()
    if first is None:
        return Line(), targets
    # Each line holds its own duel: children clone their parent's (Duel.clone, ~5 ms) instead of
    # replaying the whole line into a fresh duel every decision. The keys stay the record: a line
    # replays from `start()` to the same position (main() replays the best one for the end board).
    lines = [Line(board=board(first), duel=first)]
    done: list[Line] = []
    for _ in range(max_decisions):
        expanded: list[tuple[Line, object]] = []
        seen: set = set()  # (board, question, steps) already in the beam: the rest are duplicates
        for line in lines:
            duel, line.duel = line.duel, None
            if duel is None:
                continue
            try:
                while not duel.finished and duel.player != 0:  # the opponent never acts
                    key = worlds.decline(duel.keys())
                    duel.act(key)
                    line.keys.append(key)
            except Exception:  # the engine failed on this line: drop it, not the whole run
                continue
            now = board(duel)
            mine = next((k for k in reversed(line.keys) if not k.startswith(("chain.pass", "yes_no.no"))), None)
            steps, skipped = advance(line, targets, mine, now, lookahead)
            line.progress += steps
            line.skipped += skipped
            line.board = now
            field_now = Counter({code: n for (code, loc), n in now.items()
                                 if loc in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE)})
            line.key_cards = sum((key_cards & field_now).values())
            # The board has no levels or ATK: yes/no answers (a level change, say) tell such states apart.
            choices = tuple(k for k in line.keys if k.startswith(("yes_no.", "effect_yes_no.", "option.")))
            identity = (hash(tuple(sorted(now.items()))), tuple(duel.keys()) if not duel.finished else (),
                        line.progress, line.skipped, choices)
            if identity in seen:
                continue
            seen.add(identity)
            if duel.finished or duel.turn > 1 or (line.progress == len(targets) and
                                                  line.key_cards >= sum(key_cards.values())):
                done.append(line)
            else:
                expanded.append((line, duel))
        if not expanded:
            break
        logprobs = candidate_logprobs(model, [d.encode(card_index) for _, d in expanded], device)
        children = []
        for (line, duel), lp in zip(expanded, logprobs):
            keys = duel.keys()
            state = hash(tuple(sorted(line.board.items())))
            ahead = [code for code, _ in targets[line.progress:line.progress + lookahead]]
            window = set(ahead)
            upcoming = window | set(key_cards)
            later_center = any(code in center for code, _ in targets[line.progress:])
            fixed = forced(keys, line.keys, later_center, center, is_extra)
            on_field = {code for (code, loc), n in line.board.items()
                        if n > 0 and loc in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE)}
            # An optional chain window (a mandatory one has no pass and is never filtered).
            chain_window = "chain.pass" in keys and all(k.startswith("chain.") for k in keys)
            if chain_window and not any((ref := worlds.card_ref(k)) and int(ref[1]) in window for k in keys):
                fixed = fixed or ("chain.pass" if "chain.pass" in keys else None)
            options = []
            for i, key in enumerate(keys):
                if fixed is not None:
                    if key == fixed:
                        options.append((0.0, i))
                    continue
                if (state, key) in line.tried and len(keys) > 1 and not key.startswith(PASSIVE):
                    continue  # same answer on the same board again: a summon cancelled and retried
                options.append((0.0, i))
            if fixed is None:
                allowed = {i for _, i in options} or set(range(len(keys)))  # all retried: undo still allowed
                options = []
                allowed = plan_choice(keys, allowed, ahead)
            for i, key in enumerate(keys):
                if fixed is not None or i not in allowed:
                    continue
                ref = worlds.card_ref(key)
                value = lp[i]
                if chain_window and ref is not None and int(ref[1]) not in window:
                    continue  # our own hand trap or negation on our own chain
                if (ref is not None and key.startswith(("unselect.", "select.", "card.")) and int(ref[1]) in key_cards
                        and int(ref[1]) in on_field and int(ref[1]) not in window and len(keys) > 1):
                    continue  # an end-board card as material or cost
                if key.startswith(("yes_no.", "effect_yes_no.")):
                    value = 0.5 if key.endswith(".yes") else 0.0  # both stay; the board decides
                if ref is not None and int(ref[1]) in window:
                    value += 6.0 - 2.0 * ahead.index(int(ref[1]))  # the next step's card first
                elif ref is not None and int(ref[1]) in upcoming:
                    value += 1.0  # an end-board card, but not yet
                elif ref is not None and int(ref[1]) not in plan and line.progress < len(targets):
                    value -= 8.0  # off-plan card (a hand trap fired at nothing, a wrong Synchro, ...)
                elif line.progress < len(targets) and PHASE_END.match(key):
                    value -= 20.0  # leaving Main Phase 1 with the combo unfinished
                options.append((value, i))
            if not options:  # every answer filtered out: stop here rather than break a rule
                done.append(line)
                continue
            for value, i in sorted(options, reverse=True)[:branch]:
                children.append((Line(line.keys + [keys[i]], line.progress, line.score + value, line.board,
                                      line.key_cards, line.tried | {(state, keys[i])}, line.skipped), duel))
        lines = grow(sorted(children, key=lambda c: rank(c[0]), reverse=True)[:beam])
    best = max(done + lines, key=rank, default=Line())
    return best, targets


def checked_line(combo: cb.Combo, line: Line, targets: list) -> bool:
    """Checked: every step done in order (at most MAX_SKIPPED passed over) and at least 80 % of the
    book's end-board cards on our field."""
    return bool(targets) and line.progress == len(targets) and line.skipped <= MAX_SKIPPED and (
        not combo.end_board or line.key_cards >= 0.8 * len(combo.end_board))


def end_board(duel) -> list[dict]:
    """Our field and the cards we added to hand: what the combo left us."""
    snapshot = duel.snapshot()
    return [{"code": c["code"], "location": c["location"], "sequence": c["sequence"], "position": c["position"]}
            for c in snapshot["cards"] if c["controller"] == 0
            and c["location"] in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE, worlds.LOCATION_HAND)]


def main(argv=None) -> None:
    from ._core import Duel
    from .cards import CardTable
    from .model import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--deck-type", required=True, help="masterduelmeta deck type, e.g. Elfnote")
    parser.add_argument("--opponent", type=Path, default=rollout.REPO / "decks" / "salamangreat.ydk")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shuffles", type=int, default=3, help="deck orders a combo must work in")
    parser.add_argument("--beam", type=int, default=12)
    parser.add_argument("--branch", type=int, default=4, help="answers tried per line and decision")
    parser.add_argument("--title", help="only combos whose title contains this")
    parser.add_argument("--precise", type=Path,
                        help="hand-written engine steps (JSON list of {title, hand, steps: [[card, action]], "
                             "end_board}): replace the community steps, starting hand and end board of those combos")
    parser.add_argument("--limit", type=int, default=0, help="check only the best-voted N combos")
    parser.add_argument("--workers", type=int, default=4, help="combos/shuffles searched at once (threads)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, help="combo book (JSON lines)")
    args = parser.parse_args(argv)

    table = CardTable.from_cdb()
    types = {card.code: card.type for card in table.cards}
    is_extra = lambda code: bool(types.get(code, 0) & TYPE_EXTRA)  # noqa: E731
    center = {card.code for card in table.cards if wants_center(card.text)}
    names = cb.name_index(table)
    model = load_checkpoint(args.checkpoint, args.device, dropout=0.0)
    card_index = model.card_features.index_map()
    decklist = worlds.read_decklist(args.deck)
    opponent = worlds.read_decklist(args.opponent)[0]
    combos = sorted((c for c in cb.load(args.deck_type, names) if c.usable and c.steps),
                    key=lambda c: -c.votes)
    hands: dict[str, list[int]] = {}
    if args.precise:
        code_of = lambda name: names.get(cb.normalize(name), names.get(cb.loose(name)))  # noqa: E731
        specs = {spec["title"]: spec for spec in json.loads(args.precise.read_text(encoding="utf-8"))}
        combos = [c for c in combos if c.title in specs]
        for combo in combos:
            spec = specs[combo.title]
            combo.steps = [[cb.Entry(code_of(card), card, action)] for card, action in spec["steps"]]
            combo.end_board = [code_of(card) for card in spec.get("end_board", [])]
            hands[combo.id] = [code_of(card) for card in spec["hand"]]
    if args.title:
        combos = [c for c in combos if args.title in c.title]
    if args.limit:
        combos = combos[: args.limit]
    cache = rollout.REPO / ".cache"
    verbs = Counter()  # action text -> [used, total]
    verbs_used = Counter()
    located = Counter()  # step texts mapped to a location (True) or only "card named" (False)

    def start_for(script: str, seed: int):
        def start():
            try:
                duel = Duel.world(str(cache / "cardscripts"), str(cache / "card_data.tsv"), str(args.deck),
                                  str(args.opponent), script, seed, 0)
                for _ in range(20):  # through the draw and standby phase to our first decision
                    if duel.finished or duel.player == 0:
                        break
                    duel.act(worlds.decline(duel.keys()))
            except Exception:  # a scenario the engine rejects
                return None
            return None if duel.finished else duel
        return start

    def one(combo: cb.Combo, shuffle: int):
        """One deck order of one combo: its own scenario and seed (`scenario(combo, shuffle)`)."""
        began = time.perf_counter()
        rng = random.Random(f"{combo.id}-{shuffle}")
        script = scenario_script(combo, decklist, opponent, is_extra, rng, hand=hands.get(combo.id))
        start = start_for(script, rng.getrandbits(32))
        line, targets = check(combo, start, model, card_index, args.device, beam=args.beam, branch=args.branch,
                              center=center, is_extra=is_extra)
        return line, targets, start, time.perf_counter() - began

    # Combos and shuffles run in threads: the engine releases the GIL (Duel.world, act, clone).
    pool = ThreadPoolExecutor(max(1, args.workers))
    jobs = [[pool.submit(one, combo, shuffle) for shuffle in range(args.shuffles)] for combo in combos]
    results = []
    for n, (combo, futures) in enumerate(zip(combos, jobs)):
        runs = [f.result() for f in futures]
        verdicts = [checked_line(combo, line, targets) for line, targets, _, _ in runs]
        checked = sum(verdicts)
        # The book keeps a line that is checked if there is one, else the furthest.
        shuffle = max(range(len(runs)), key=lambda i: (verdicts[i], runs[i][0].progress - runs[i][0].skipped,
                                                      runs[i][0].key_cards))
        line, targets, start, _ = runs[shuffle]
        seconds = sum(r[3] for r in runs)
        entries = [e for step in combo.steps for e in step if e.code is not None]
        for i, e in enumerate(entries):
            verb = e.action.strip().lower() or "(use)"
            verbs[verb] += 1
            verbs_used[verb] += i < line.progress
            located[expected_location(e.action) is not None] += 1
        duel = start()  # the line replays from the scenario alone (no clone involved)
        replays = duel is not None
        for key in line.keys if duel is not None else []:
            if duel.finished or key not in duel.keys():
                replays = False
                break
            duel.act(key)
        replays = replays and board(duel) == line.board
        final = end_board(duel) if duel is not None else []
        on_field = [c for c in final if c["location"] in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE)]
        result = {"combo": combo.id, "title": combo.title, "votes": combo.votes, "steps": len(targets),
                  "progress": line.progress, "skipped": line.skipped, "checked_shuffles": checked, "shuffles": args.shuffles,
                  "checked": checked == args.shuffles, "shuffle": shuffle, "replays": replays, "line": line.keys,
                  "end_board": final, "field": len(on_field), "book_end_board": combo.end_board,
                  "key_cards_reached": (sum(1 for c in combo.end_board if c in {b["code"] for b in on_field})
                                        / len(combo.end_board)) if combo.end_board else None,
                  "seconds": round(seconds, 1)}
        results.append(result)
        print(f"{n + 1}/{len(combos)} {'OK  ' if result['checked'] else 'part'} {line.progress}/{len(targets)} "
              f"steps ({line.skipped} skipped), {checked}/{args.shuffles} shuffles, field {len(on_field)}, "
              f"{seconds:.0f}s{'' if replays else ' NO REPLAY'}  {combo.title[:60]}", flush=True)
    pool.shutdown()
    passed = sum(r["checked"] for r in results)
    print(f"checked: {passed}/{len(results)} combos ({passed / max(1, len(results)):.0%}), mean field "
          f"{sum(r['field'] for r in results) / max(1, len(results)):.2f} (checked: "
          f"{sum(r['field'] for r in results if r['checked']) / max(1, passed):.2f}), search "
          f"{sum(r['seconds'] for r in results):.0f}s ({sum(r['seconds'] for r in results) / max(1, len(results)):.1f}s/combo)")
    print(f"steps with a location: {located[True]}, only named: {located[False]}")
    print("verb coverage (used/total):", {v: f"{verbs_used[v]}/{verbs[v]}" for v, _ in verbs.most_common(20)})
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")


if __name__ == "__main__":
    main()
