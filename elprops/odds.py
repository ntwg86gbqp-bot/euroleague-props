"""Αποδόσεις: OddsPapi API, χειροκίνητη εισαγωγή / CSV, αντιστοίχιση ονομάτων."""
from __future__ import annotations
import datetime as dt
import difflib
import json
import re
import time
import unicodedata
import pandas as pd
import requests

from .db import DATA_DIR

BASE = "https://api.oddspapi.io/v4"
EUROLEAGUE_TOURNAMENT = 138
BASKETBALL_SPORT = 11
CONFIG = DATA_DIR / "config.json"
MARKETS_CACHE = DATA_DIR / "oddspapi_markets.json"
DEFAULT_BOOKS = "pinnacle,bet365,betano,stoiximan,novibet,bwin"


# Εταιρείες όπου μπορεί να παίξει κάποιος από την Ελλάδα, όπως τις ονομάζει το OddsPapi (όσες υπάρχουν).
# Αλλάζουν από τη σελίδα «Αποδόσεις».
MY_BOOKS_DEFAULT: list[str] = []          # κενό = όλες οι εταιρείες (οι γραμμές συνήθως υπάρχουν και στη Stoiximan)
SUGGESTED_BOOKS = ["betano", "betsson", "bwin", "sportingbet", "bet365", "novibet", "stoiximan"]

_CON = None   # αν οριστεί (bind), οι ρυθμίσεις ζουν στη βάση, ώστε να συγχρονίζονται και στο cloud


def bind(con) -> None:
    """Οι ρυθμίσεις στη βάση. Την πρώτη φορά μεταφέρει ό,τι υπήρχε στο data/config.json."""
    global _CON
    con.execute("CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY, value TEXT)")
    if not con.execute("SELECT COUNT(*) FROM config").fetchone()[0] and CONFIG.exists():
        try:
            for k, v in json.loads(CONFIG.read_text()).items():
                con.execute("INSERT OR REPLACE INTO config VALUES (?, ?)", (k, json.dumps(v)))
        except Exception:  # noqa: BLE001
            pass
    con.commit()
    _CON = con


def load_config() -> dict:
    if _CON is not None:
        try:
            return {k: json.loads(v) for k, v in _CON.execute("SELECT key, value FROM config")}
        except Exception:  # noqa: BLE001
            return {}
    try:
        return json.loads(CONFIG.read_text())
    except Exception:  # noqa: BLE001
        return {}


def save_config(cfg: dict) -> None:
    if _CON is not None:
        old = load_config()
        for k, v in cfg.items():
            if old.get(k) != v:
                _CON.execute("INSERT OR REPLACE INTO config VALUES (?, ?)", (k, json.dumps(v)))
        for k in set(old) - set(cfg):
            _CON.execute("DELETE FROM config WHERE key=?", (k,))
        _CON.commit()
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2))


def my_books(cfg: dict | None = None) -> list[str]:
    cfg = load_config() if cfg is None else cfg
    return list(cfg.get("my_books", MY_BOOKS_DEFAULT))


# ---------- ονόματα ----------

_SUFFIX = {"JR", "SR", "II", "III", "IV"}


def norm_name(s: str) -> str:
    """'LARKIN, SHANE' και 'Shane Larkin' -> 'LARKIN SHANE'."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).upper()
    toks = [t for t in re.split(r"[^A-Z]+", s) if t and t not in _SUFFIX]
    return " ".join(sorted(toks))


class PlayerIndex:
    def __init__(self, players: pd.DataFrame):
        """players: index=player_id, στήλη 'player'."""
        self.by_norm: dict[str, list[str]] = {}
        for pid, name in players["player"].items():
            self.by_norm.setdefault(norm_name(name), []).append(pid)

    def match(self, raw: str) -> str | None:
        n = norm_name(raw)
        if n in self.by_norm:
            ids = self.by_norm[n]
            return ids[0] if len(ids) == 1 else None
        # αρχικό + επώνυμο (π.χ. 'S. Larkin') ή υποσύνολο tokens
        toks = set(n.split())
        cands = []
        for key, ids in self.by_norm.items():
            kt = set(key.split())
            if len(toks) >= 2 and (toks <= kt or kt <= toks):
                cands += ids
            elif len(toks) == 2:
                long = [t for t in toks if len(t) > 1]
                short = [t for t in toks if len(t) == 1]
                if len(long) == 1 and short and long[0] in kt and any(
                        k.startswith(short[0]) for k in kt - {long[0]}):
                    cands += ids
        if len(set(cands)) == 1:
            return cands[0]
        close = difflib.get_close_matches(n, list(self.by_norm), n=2, cutoff=0.86)
        if len(close) == 1 and len(self.by_norm[close[0]]) == 1:
            return self.by_norm[close[0]][0]
        return None


# ---------- αγορές ----------

_STAT_WORDS = [("pts", r"\bpoints?\b|\bpts\b"), ("reb", r"rebound|\breb\b"),
               ("ast", r"assist|\bast\b"), ("fg3m", r"three|3[- ]?point|\b3pt|threes|triples"),
               ("fg2m", r"\btwos?\b|two[- ]point|\b2pt"),
               ("stl", r"steal"), ("blk", r"block"), ("tov", r"turnover")]
_COMBO = {frozenset(["pts", "reb", "ast"]): "pra", frozenset(["pts", "reb"]): "pr",
          frozenset(["pts", "ast"]): "pa", frozenset(["reb", "ast"]): "ra",
          frozenset(["stl", "blk"]): "stl_blk"}


def stat_from_market_name(name: str) -> str | None:
    """Αντιστοιχίζει όνομα αγοράς σε εσωτερικό κλειδί. None αν δεν αναγνωρίζεται
    ή αν αφορά περίοδο/ημίχρονο/double-double κ.λπ."""
    n = name.lower()
    if re.search(r"quarter|half|1st|2nd|3rd|4th|double|first|last|most|race|odd|even", n):
        return None
    n = n.replace("3-point", "three").replace("3 point", "three")
    n = n.replace("2-point", "two").replace("2 point", "two")
    found = [k for k, pat in _STAT_WORDS if re.search(pat, n)]
    if ("fg3m" in found or "fg2m" in found) and "pts" in found and not re.search(r"\+|&| and ", n):
        found.remove("pts")  # 'three points made' είναι τρίποντα, όχι πόντοι
    if len(found) == 1:
        return found[0]
    return _COMBO.get(frozenset(found))


def side_from_outcome(name: str) -> str | None:
    n = str(name).lower()
    if "over" in n:
        return "over"
    if "under" in n:
        return "under"
    return None


# ---------- OddsPapi ----------

class OddsPapiError(RuntimeError):
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after          # δευτερόλεπτα που ζήτησε το API να περιμένουμε (όταν κόβει)


def _wait_of(r) -> float | None:
    """Πόσο ζητάει να περιμένουμε μια απάντηση 429: Retry-After, retryAfterSec ή X-RateLimit-Reset."""
    for h in ("Retry-After", "X-RateLimit-Reset"):
        try:
            v = float(r.headers.get(h, ""))
            return v - time.time() if v > 1e9 else v       # Reset μπορεί να είναι χρονοσφραγίδα
        except (TypeError, ValueError):
            pass
    try:
        return float((r.json() or {}).get("retryAfterSec"))
    except Exception:  # noqa: BLE001
        return None


def _human(sec: float) -> str:
    return f"{sec:.0f} δευτερόλεπτα" if sec < 90 else f"{sec / 60:.0f} λεπτά" if sec < 5400 else f"{sec / 3600:.1f} ώρες"


def _get(path: str, key: str, **params):
    q = {k: v for k, v in params.items() if v is not None}
    q["apiKey"] = key
    for attempt in range(4):
        r = requests.get(f"{BASE}/{path}", params=q, timeout=120)
        if r.status_code == 429:
            # όσο ζητάει το ίδιο το API, αλλιώς 3, 6, 9 δευτερόλεπτα· αν ζητάει πάνω από 20, σταματάμε εδώ
            asked = _wait_of(r)
            wait = asked if asked is not None else 3.0 * (attempt + 1)
            if wait > 20:
                raise OddsPapiError(f"το OddsPapi κόβει τα αιτήματα στο /{path} και ζητάει αναμονή {_human(wait)}.",
                                    retry_after=wait)
            time.sleep(max(wait, 1.0))
            continue
        if r.status_code == 404:
            return None
        if r.status_code in (401, 403):
            raise OddsPapiError(f"Το API απέρριψε το κλειδί ({r.status_code}): {r.text[:200]}")
        r.raise_for_status()
        time.sleep(1.1)  # cooldown ανά endpoint
        return r.json()
    raise OddsPapiError(f"το OddsPapi κόβει προσωρινά τα αιτήματα (/{path}). Ξαναδοκίμασε σε λίγα λεπτά.")


MARKETS_MAX_AGE = dt.timedelta(days=7)


def _markets_cache() -> tuple[dict | None, dt.datetime | None]:
    """Η αποθηκευμένη λίστα αγορών και πότε κατέβηκε. Στη βάση (σώζεται και στο cloud), αλλιώς σε αρχείο."""
    if _CON is not None:
        try:
            _CON.execute("CREATE TABLE IF NOT EXISTS api_cache(key TEXT PRIMARY KEY, value TEXT, at TEXT)")
            r = _CON.execute("SELECT value, at FROM api_cache WHERE key='oddspapi_markets'").fetchone()
            if r:
                return json.loads(r[0]), dt.datetime.fromisoformat(r[1])
        except Exception:  # noqa: BLE001
            pass
    if MARKETS_CACHE.exists():
        at = dt.datetime.fromtimestamp(MARKETS_CACHE.stat().st_mtime, dt.timezone.utc)
        try:
            return json.loads(MARKETS_CACHE.read_text()), at
        except Exception:  # noqa: BLE001
            pass
    return None, None


def _save_markets(out: dict) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if _CON is not None:
        try:
            _CON.execute("CREATE TABLE IF NOT EXISTS api_cache(key TEXT PRIMARY KEY, value TEXT, at TEXT)")
            _CON.execute("INSERT OR REPLACE INTO api_cache VALUES ('oddspapi_markets', ?, ?)", (json.dumps(out), now))
            _CON.commit()
        except Exception:  # noqa: BLE001
            pass
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MARKETS_CACHE.write_text(json.dumps(out))
    except Exception:  # noqa: BLE001
        pass


def prop_markets(key: str, refresh: bool = False) -> dict[str, dict]:
    """marketId -> {name, stat, line, outcomes{outcomeId: side}} για ειδικά παικτών μπάσκετ.

    Η λίστα σχεδόν δεν αλλάζει και το /markets είναι το βαρύτερο αίτημα του API: κρατιέται 7 ημέρες.
    Αν η ανανέωσή της αποτύχει (π.χ. όριο αιτημάτων), χρησιμοποιείται η παλιά."""
    cached, at = _markets_cache()
    fresh = cached is not None and at is not None and dt.datetime.now(dt.timezone.utc) - at < MARKETS_MAX_AGE
    if cached and fresh and not refresh:
        return cached
    try:
        out = _download_markets(key)
    except (OddsPapiError, requests.RequestException):
        if cached:
            return cached
        raise
    if out:
        _save_markets(out)
    return out or (cached or {})


def _download_markets(key: str) -> dict[str, dict]:
    out = {}
    for m in _get("markets", key, sportId=BASKETBALL_SPORT) or []:      # μόνο μπάσκετ: πολύ μικρότερο αίτημα
        if m.get("sportId") != BASKETBALL_SPORT or not m.get("playerProp"):
            continue
        out[str(m["marketId"])] = {
            "name": m.get("marketName", ""), "stat": stat_from_market_name(m.get("marketName", "")),
            "line": m.get("handicap"),
            "outcomes": {str(o["outcomeId"]): side_from_outcome(o.get("outcomeName", ""))
                         for o in m.get("outcomes", [])}}
    return out


def parse_fixture_odds(odds: dict, markets: dict[str, dict]) -> tuple[list[dict], set[str]]:
    """Επιστρέφει (γραμμές, ονόματα αγορών παικτών που δεν αναγνωρίστηκαν)."""
    rows, unknown = [], set()
    for book, board in (odds or {}).get("bookmakerOdds", {}).items():
        for mid, market in (board.get("markets") or {}).items():
            meta = markets.get(str(mid))
            if not meta:
                continue
            if not meta["stat"] or meta["line"] is None:
                unknown.add(meta["name"])
                continue
            for oid, leg in (market.get("outcomes") or {}).items():
                side = meta["outcomes"].get(str(oid))
                if not side:
                    continue
                for _pid, pl in (leg.get("players") or {}).items():
                    if not pl.get("active") or not pl.get("price") or not pl.get("playerName"):
                        continue
                    rows.append({"bookmaker": book, "player_raw": pl["playerName"],
                                 "stat": meta["stat"], "line": float(meta["line"]),
                                 "side": side, "price": float(pl["price"])})
    return rows, unknown


def fetch_oddspapi(key: str, bookmakers: str = DEFAULT_BOOKS, days: int = 7,
                   progress=None) -> dict:
    """Τραβάει ειδικά παικτών για τα επερχόμενα παιχνίδια Euroleague."""
    markets = prop_markets(key)
    now = dt.datetime.now(dt.timezone.utc)
    fixtures = _get("fixtures", key, tournamentId=EUROLEAGUE_TOURNAMENT,
                    **{"from": now.strftime("%Y-%m-%d"),
                       "to": (now + dt.timedelta(days=days)).strftime("%Y-%m-%d")}) or []
    iso = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    upcoming = [f for f in fixtures if f.get("hasOdds") and f.get("startTime", "") > iso]
    raw = []
    for i, f in enumerate(upcoming):
        if progress:
            progress((i + 1) / len(upcoming),
                     f"{f.get('participant1Name')} - {f.get('participant2Name')}")
        raw.append(_get("odds", key, fixtureId=f["fixtureId"], bookmakers=bookmakers or None))
    # αγορές που δεν ξέρουμε: ίσως πρόσθεσε νέες το API. Ανανέωση της λίστας, αλλά το πολύ μία φορά τη μέρα.
    seen = {str(mid) for o in raw for b in (o or {}).get("bookmakerOdds", {}).values() for mid in (b.get("markets") or {})}
    _, at = _markets_cache()
    if seen - set(markets) and (at is None or now - at > dt.timedelta(days=1)):
        markets = prop_markets(key, refresh=True)
    rows, unknown = [], set()
    for o in raw:
        r, u = parse_fixture_odds(o, markets)
        rows += r
        unknown |= u
    return {"rows": rows, "n_prop_markets": len(markets),
            "n_mapped_markets": sum(1 for m in markets.values() if m["stat"]),
            "n_fixtures": len(fixtures), "n_upcoming": len(upcoming), "unknown": sorted(unknown)}


# ---------- αποθήκευση ----------

def store(con, rows: list[dict], index: PlayerIndex, source: str) -> tuple[int, list[str]]:
    """Γράφει ένα νέο στιγμιότυπο. Επιστρέφει (πλήθος, ονόματα που δεν ταίριαξαν).

    Κρατάει μόνο την τελευταία τιμή: οι παλιότερες εγγραφές του ίδιου στοιχήματος σβήνονται (το ιστορικό τιμών
    δεν χρησιμοποιείται πουθενά· η κίνηση της αγοράς κρατιέται στο Αρχείο). Για το API, ένα νέο στιγμιότυπο
    αντικαθιστά ολόκληρο το προηγούμενο των ίδιων παικτών, ώστε να φεύγουν και οι γραμμές που αποσύρθηκαν."""
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    unmatched, n = set(), 0
    for r in rows:
        pid = r.get("player_id") or index.match(r["player_raw"])
        if not pid:
            unmatched.add(r["player_raw"])
            continue
        con.execute("INSERT INTO odds(fetched_at, source, bookmaker, player_raw, player_id, stat,"
                    " line, side, price) VALUES (?,?,?,?,?,?,?,?,?)",
                    (ts, source, r.get("bookmaker") or "—", r["player_raw"], pid, r["stat"],
                     float(r["line"]), r["side"], float(r["price"])))
        n += 1
    if n:
        con.execute("""DELETE FROM odds WHERE fetched_at < ? AND EXISTS (
                         SELECT 1 FROM odds o2 WHERE o2.fetched_at = ? AND o2.player_id = odds.player_id
                           AND o2.stat = odds.stat AND o2.line = odds.line AND o2.side = odds.side
                           AND o2.bookmaker = odds.bookmaker)""", (ts, ts))
        if source == "oddspapi":
            con.execute("""DELETE FROM odds WHERE source = 'oddspapi' AND fetched_at < ?
                             AND player_id IN (SELECT DISTINCT player_id FROM odds WHERE fetched_at = ? AND source = 'oddspapi')""",
                        (ts, ts))
    cut = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=10)).isoformat(timespec="seconds")
    con.execute("DELETE FROM odds WHERE fetched_at < ?", (cut,))      # πολύ παλιές: δεν χρησιμεύουν
    con.commit()
    return n, sorted(unmatched)


def latest(con, max_age_hours: float = 72) -> pd.DataFrame:
    """Τελευταία τιμή ανά (bookmaker, παίκτης, αγορά, γραμμή, πλευρά)."""
    df = pd.read_sql_query("SELECT * FROM odds", con)
    if df.empty:
        return df
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    cut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=max_age_hours)
    df = df[df["fetched_at"] >= cut]
    return (df.sort_values("fetched_at")
              .drop_duplicates(["bookmaker", "player_id", "stat", "line", "side"], keep="last"))


def wide_to_rows(df: pd.DataFrame) -> list[dict]:
    """Πίνακας (player, stat, line, over, under, bookmaker) -> γραμμές ανά πλευρά."""
    rows = []
    for r in df.to_dict("records"):
        if not r.get("player") or pd.isna(r.get("line")) or not r.get("stat"):
            continue
        for side in ("over", "under"):
            price = r.get(side)
            if price is not None and not pd.isna(price) and float(price) > 1:
                rows.append({"player_raw": str(r["player"]), "player_id": r.get("player_id"),
                             "stat": r["stat"], "line": float(r["line"]), "side": side,
                             "price": float(price), "bookmaker": r.get("bookmaker") or "χειροκίνητα"})
    return rows
