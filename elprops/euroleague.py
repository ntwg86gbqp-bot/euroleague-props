"""Εισαγωγή προγράμματος και boxscores από τα δημόσια feeds της Euroleague."""
from __future__ import annotations
import time
from typing import Callable, Iterable
import requests

GAMES_URL = "https://api-live.euroleague.net/v2/competitions/E/seasons/E{season}/games"
BOX_URL = "https://live.euroleague.net/api/Boxscore"
PEOPLE_URL = "https://api-live.euroleague.net/v2/competitions/E/seasons/E{season}/clubs/{club}/people"
HEADERS = {"User-Agent": "Mozilla/5.0 (elprops personal use)", "Accept": "application/json"}


def parse_minutes(s) -> float:
    """'33:21' -> 33.35, 'DNP'/None -> 0."""
    if not s or ":" not in str(s):
        return 0.0
    m, sec = str(s).split(":")[:2]
    try:
        return int(m) + int(sec) / 60.0
    except ValueError:
        return 0.0


def parse_game(g: dict, season: int) -> tuple:
    loc, road = g["local"], g["road"]
    return (season, int(g["gameCode"]), int(g.get("round") or 0),
            g.get("utcDate") or g.get("date"),
            loc["club"]["code"], loc["club"]["name"],
            road["club"]["code"], road["club"]["name"],
            int(bool(g.get("played"))), int(loc.get("score") or 0), int(road.get("score") or 0))


def parse_boxscore(data: dict, season: int, game_code: int) -> list[tuple]:
    rows = []
    for side in data.get("Stats", []):
        for p in side.get("PlayersStats", []):
            mins = parse_minutes(p.get("Minutes"))
            if mins <= 0:
                continue
            g = lambda k: int(p.get(k) or 0)
            rows.append((season, game_code, str(p["Player_ID"]).strip(), str(p["Player"]).strip(),
                         str(p["Team"]).strip(), g("IsStarter"), mins, g("Points"),
                         g("FieldGoalsMade2"), g("FieldGoalsAttempted2"), g("FieldGoalsMade3"),
                         g("FieldGoalsAttempted3"), g("FreeThrowsMade"), g("FreeThrowsAttempted"),
                         g("OffensiveRebounds"), g("DefensiveRebounds"), g("TotalRebounds"),
                         g("Assistances"), g("Steals"), g("Turnovers"), g("BlocksFavour"),
                         g("FoulsCommited")))
    return rows


class RateLimited(RuntimeError):
    pass


def _request(s, url: str, params: dict, waits=(5, 15, 30)):
    """GET με υπομονή: στο 429/5xx περιμένει (Retry-After αν δίνεται) και ξαναδοκιμάζει."""
    for i in range(len(waits) + 1):
        r = s.get(url, headers=HEADERS, params=params, timeout=60)
        if r.status_code != 429 and r.status_code < 500:
            r.raise_for_status()
            return r
        if i == len(waits):
            break
        try:
            wait = min(float(r.headers.get("Retry-After", waits[i])), 60.0)
        except ValueError:
            wait = waits[i]
        time.sleep(max(wait, waits[i]))
    if r.status_code == 429:
        raise RateLimited("το API της Euroleague περιορίζει προσωρινά τα αιτήματα (429)")
    r.raise_for_status()
    return r


def _get_games(s, season: int, params: dict):
    js = _request(s, GAMES_URL.format(season=season), params).json()
    data = js["data"] if isinstance(js, dict) else js
    total = js.get("total") if isinstance(js, dict) else None
    return data, total


def fetch_games(season: int, session=None) -> list[tuple]:
    """Ολόκληρο το πρόγραμμα. Αν η απάντηση είναι σελιδοποιημένη, πάει ανά αγωνιστική."""
    s = session or requests
    data, total = _get_games(s, season, {})
    if total and total > len(data):
        data, empty = [], 0
        for rnd in range(1, 60):
            chunk, _ = _get_games(s, season, {"roundNumber": rnd})
            data += chunk
            empty = 0 if chunk else empty + 1
            if empty >= 3:
                break
    return list({(season, int(g["gameCode"])): parse_game(g, season) for g in data}.values())


def parse_people(js, season: int, club: str) -> list[tuple]:
    """Ρόστερ ομάδας -> (player_id, season, team, name, position, height) μόνο για παίκτες.
    Το person.code «003048» αντιστοιχεί στο Player_ID «P003048» των boxscores."""
    items = js.get("data", js) if isinstance(js, dict) else js
    out = []
    for it in items or []:
        if str(it.get("type")) != "J":
            continue
        per = it.get("person") or {}
        code = str(per.get("code") or "").strip()
        if not code:
            continue
        pos = it.get("positionName")
        h = per.get("height")
        out.append(("P" + code, season, (it.get("club") or {}).get("code") or club, per.get("name"),
                    pos if pos in ("Guard", "Forward", "Center") else None,
                    float(h) if isinstance(h, (int, float)) and h > 0 else None))
    return out


def update_rosters(con, seasons, sess=None, pause: float = 0.3) -> list[str]:
    """Κατεβάζει θέσεις και ύψη από τα ρόστερ. Επιστρέφει προειδοποιήσεις (δεν σταματά την ενημέρωση)."""
    sess = sess or requests.Session()
    warns = []
    for season in seasons:
        clubs = [r[0] for r in con.execute("SELECT DISTINCT home_code FROM games WHERE season=?", (season,))]
        for club in clubs:
            try:
                rows = parse_people(_request(sess, PEOPLE_URL.format(season=season, club=club), {}, (5, 15)).json(),
                                    season, club)
            except Exception as e:  # noqa: BLE001
                warns.append(f"Ρόστερ {club} {season}: {e}")
                if "429" in str(e):
                    return warns
                continue
            con.executemany("INSERT OR REPLACE INTO players VALUES (?,?,?,?,?,?)", rows)
            con.commit()
            time.sleep(pause)
    return warns


def update(con, seasons: Iterable[int], progress: Callable[[float, str], None] | None = None,
           pause: float = 0.6, waits=(5, 15, 30)) -> dict:
    """Φέρνει πρόγραμμα + όσα boxscores λείπουν. Επιστρέφει σύνοψη.

    Αν το πρόγραμμα δεν κατέβει (π.χ. 429), συνεχίζει με όσους αγώνες είναι ήδη στη βάση."""
    sess = requests.Session()
    summary = {"games": 0, "new_box": 0, "errors": [], "warnings": [], "missing": 0}
    todo = []
    for season in seasons:
        try:
            games = fetch_games(season, sess)
            con.executemany("INSERT OR REPLACE INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?)", games)
            con.commit()
            summary["games"] += len(games)
        except Exception as e:  # noqa: BLE001
            n = con.execute("SELECT COUNT(*) FROM games WHERE season=?", (season,)).fetchone()[0]
            (summary["warnings"] if n else summary["errors"]).append(
                f"Το πρόγραμμα του {season} δεν ανανεώθηκε ({e})."
                + (" Συνεχίζω με το αποθηκευμένο." if n else ""))
        have = {r[0] for r in con.execute(
            "SELECT DISTINCT game_code FROM box WHERE season=?", (season,))}
        todo += [(season, r[0]) for r in con.execute(
            "SELECT game_code FROM games WHERE season=? AND played=1", (season,)) if r[0] not in have]
    todo.sort()
    fails = 0
    for i, (season, code) in enumerate(todo):
        if progress:
            progress((i + 1) / len(todo), f"Boxscore {season} #{code} ({i + 1}/{len(todo)})")
        try:
            r = _request(sess, BOX_URL, {"gamecode": code, "seasoncode": f"E{season}"}, waits)
            rows = parse_boxscore(r.json(), season, code)
            fails = 0
        except Exception as e:  # noqa: BLE001
            fails += 1
            summary["errors"].append(f"Boxscore {season} #{code}: {e}")
            if fails >= 3:
                summary["errors"].append("Σταμάτησα μετά από 3 συνεχόμενες αποτυχίες. "
                                         "Περίμενε 10–15 λεπτά και ξαναπάτα «Ενημέρωση».")
                break
            continue
        if rows:
            con.executemany("INSERT OR REPLACE INTO box VALUES (" + ",".join("?" * 22) + ")", rows)
            con.commit()
            summary["new_box"] += 1
        time.sleep(pause)
    if fails < 3:
        summary["warnings"] += update_rosters(con, list(seasons), sess)
    summary["missing"] = con.execute(
        """SELECT COUNT(*) FROM games g WHERE played=1 AND NOT EXISTS
           (SELECT 1 FROM box b WHERE b.season=g.season AND b.game_code=g.game_code)""").fetchone()[0]
    return summary
