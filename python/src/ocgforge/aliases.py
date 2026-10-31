"""Community nicknames -> card names for Branded and Elfnote (player sheets, Discord posts, guides).

Keys are lower case, as `seeds.resolver` looks them up (`seeds.ALIASES`). Each name is a card of the
pinned database (tests/test_aliases.py), so a nickname resolves the same in every deck pool; without
an alias, a short name only resolves when one card of the deck's pool contains it ("Jade" is
Mirrorjade in Branded but ambiguous next to Jade Dragon Mech).

Found by running the resolver over the Branded sources (Discord sheet, masterduelmeta combos, pinned
posts) and the Elfnote sheets; nicknames still unresolved: "Elf"/"Elves", "Caesar", "Maximus",
"Sword" (meaning unclear from the sheet), and generic parts ("Dark monster", "A bystial", "Dragon").
"""

from __future__ import annotations

_BRANDED = {
    # starters and extenders
    "keeper": "Keeper of Dragon Magic",  # "Keeper + 2 Discards": searches Branded Fusion, discarding
    "aluber": "Aluber the Jester of Despia",
    "albaz": "Fallen of Albaz", "abaz": "Fallen of Albaz",
    "opening": "Branded Opening",
    "brafu": "Branded Fusion",
    "deployment": "Fusion Deployment",
    "nadir": "Nadir Servant",
    "sarc": "Gold Sarcophagus",
    "foolish": "Foolish Burial",
    "kitt": "Tri-Brigade Springans Kitt",
    "merc": "Tri-Brigade Mercourier", "mercourier": "Tri-Brigade Mercourier",
    "quem": "Guiding Quem, the Virtuous",
    "cart": "Blazing Cartesia, the Virtuous", "cartesia": "Blazing Cartesia, the Virtuous",
    "ecclesia": "Incredible Ecclesia, the Virtuous",
    "astellar": "Astellar of the White Forest",
    "saronir": "Bystial Saronir",
    "magnamhut": "Bystial Magnamhut", "magna": "Bystial Magnamhut",
    "lubellion": "Lubellion the Searing Dragon",
    "tragedy": "Despian Tragedy",
    "spoly": "Super Polymerization", "super poly": "Super Polymerization",
    "dupe": "Fusion Duplication",
    "ttt": "Triple Tactics Thrust",
    # spells / traps
    "bir": "Branded in Red",
    "lost": "Branded Lost",
    "banishment": "Branded Banishment",
    "retribution": "Branded Retribution",
    "high spirit": "Branded in High Spirits", "high spirits": "Branded in High Spirits",
    # Extra Deck
    "jade": "Mirrorjade the Iceblade Dragon", "mirrorjade": "Mirrorjade the Iceblade Dragon",
    "mirrorajde": "Mirrorjade the Iceblade Dragon",
    "gran": "Granguignol the Dusk Dragon", "granguignol": "Granguignol the Dusk Dragon",
    "granguinol": "Granguignol the Dusk Dragon", "granguniol": "Granguignol the Dusk Dragon",
    "ad lib": "Ad Libitum of Despia", "adlib": "Ad Libitum of Despia", "ad libitum": "Ad Libitum of Despia",
    "albion": "Albion the Branded Dragon",
    "shrouded": "Albion the Shrouded Dragon",
    "sanctifire": "Albion the Sanctifire Dragon",
    "rindbrum": "Rindbrumm the Striking Dragon", "rindbrumm": "Rindbrumm the Striking Dragon",
    "titaniklad": "Titaniklad the Ash Dragon",
    "dragostapelia": "Predaplant Dragostapelia",
    "masq": "Masquerade the Blazing Dragon",
    "ladd": "Light and Darkness Dragonlord",
    "furious": "Borreload Furious Dragon",
    "edge imp": "Edge Imp Chain",
}

_ELFNOTE = {
    "pp": "Elfnote Power Patron", "power patron": "Elfnote Power Patron",
    "lucina": "Elfnote Lucina",
    "tinia": "Elfnote Tinia",
    "regina": "Elfnote Regina",
    "fortuna": "Elfnote Fortuna",
    "june pride": "Elfnote June Pride", "june": "Elfnote June Pride",
    "strelitzia": "Elfnote Seraphim Strelitzia", "seraphim": "Elfnote Seraphim Strelitzia",
    "welcome home": "Elfnotes: Welcome Home",
    "rhapsodia": "Elfnotes: Rhapsodia of Madness",
    "terminus": "Prohibited Power Patron Portal - Terminus",
    "medius": "Medius the Pure",
    "theo medius": "Theorealize Medius",
    "past lull": "Theorealize Past Lull",
    "vidolium": "Vidolium the Unstable Power Patron of Unity",
    "junoldo": "Junoldo the Shadespirit Power Patron",
    "purification": "Purification Power Patron",
    "diactorus": "Artmage Diactorus",
    "baronne": "Baronne de Fleur",
    "dragster": "F.A. Dawn Dragster",
    "crystal wing": "Crystal Wing Synchro Dragon",
    "omega": "PSY-Framelord Omega",
    "b2b": "Kewl Tune B2B",
    "accel": "Accel Synchron",
    "jet": "Jet Synchron",
    "starjunk": "Starjunk Synchron",
    "ekhajar": "Ekhajar, Descendant Dragon of the Ice Barrier",
}

ALIASES: dict[str, str] = {**_BRANDED, **_ELFNOTE}


def with_aliases(resolve):
    """`resolve` (a `seeds.resolver`) that tries `ALIASES` first: for use before they are merged
    into `seeds.ALIASES`."""
    from .seeds import VERSION, WHERE  # here: seeds will import this module

    def wrapped(text: str):
        key = VERSION.sub("", WHERE.sub("", text.strip().rstrip("."))).strip().lower()
        return resolve(ALIASES.get(key, text))
    return wrapped
