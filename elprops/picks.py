"""Αυτόματο αρχείο: κάθε στοίχημα που βγαίνει «Αξίζει» (ή «Οριακό») καταγράφεται μόνο του,
κλείνει μόνο του από τα boxscores και παίρνει χαρακτηρισμό για την έκβαση.

Κανόνες καταγραφής
- Μία εγγραφή ανά (παίκτης, αγορά, γραμμή, πλευρά, αγώνας).
- Καταγράφεται μόνο πριν από το τζάμπολ.
- Κρατιέται η καλύτερη ετυμηγορία που έφτασε: αν ένα «Οριακό» γίνει αργότερα «Αξίζει», τα στοιχεία
  (απόδοση, πιθανότητες) ενημερώνονται στη στιγμή που έγινε «Αξίζει». Αν πέσει, δεν σβήνεται.
- Μέχρι το τζάμπολ ενημερώνεται η τελευταία απόδοση που είδαμε (price_last). Η σύγκριση με την απόδοση
  της καταγραφής δείχνει προς τα πού κινήθηκε η αγορά (closing line value): αν η απόδοση έπεσε μετά,
  η αγορά συμφώνησε με το μοντέλο.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import model as M
from .bets import outcome

RECORD = ("strong", "marginal")
RANK = {"strong": 2, "marginal": 1}

SCHEMA = """
CREATE TABLE IF NOT EXISTS picks(
  id INTEGER PRIMARY KEY AUTOINCREMENT, first_seen TEXT, last_seen TEXT,
  player_id TEXT, player TEXT, team TEXT, opp TEXT, stat TEXT, line REAL, side TEXT,
  price REAL, bookmaker TEXT, p_model REAL, p_market REAL, p_final REAL, ev REAL, verdict TEXT,
  mu REAL, min_hat REAL, w_model REAL,
  price_last REAL, ev_last REAL, verdict_last TEXT,
  season INTEGER, game_code INTEGER, game_date TEXT,
  status TEXT DEFAULT 'open', result REAL, minutes REAL, tag TEXT, settled_at TEXT,
  UNIQUE(player_id, stat, line, side, season, game_code));
CREATE TABLE IF NOT EXISTS pick_tags(pick_id INTEGER, tag TEXT, set_at TEXT, PRIMARY KEY(pick_id, tag));
"""
DEFAULT_TAGS = ["Θα τα έπαιζα"]

# χαρακτηρισμοί έκβασης
TAGS = {
    "easy_win": "Κέρδισε άνετα",
    "close_win": "Κέρδισε στο νήμα",
    "close_loss": "Έχασε παρά τρίχα",
    "clear_loss": "Έχασε καθαρά",
    "low_min": "Έχασε: έπαιξε λίγο",
    "void": "Ακυρώθηκε: δεν αγωνίστηκε",
    "push": "Ισοπαλία",
}
TAG_MEANS = {
    "easy_win": "Πέρασε τη γραμμή με περισσότερες από 1 μονάδα.",
    "close_win": "Πέρασε τη γραμμή με το πολύ 1 μονάδα. Κέρδος, αλλά με τύχη.",
    "close_loss": "Έμεινε το πολύ 1 μονάδα από τη γραμμή. Χάθηκε στη λεπτομέρεια.",
    "clear_loss": "Έμεινε πάνω από 1 μονάδα μακριά από τη γραμμή, με κανονικά λεπτά: εδώ έκανε λάθος η εκτίμηση.",
    "low_min": "Έχασε και έπαιξε κάτω από το 60% των λεπτών που περίμενε το μοντέλο (τραυματισμός, φάουλ, σφαλιάρα).",
    "void": "Ο παίκτης δεν αγωνίστηκε. Στις εταιρείες το ποντάρισμα επιστρέφεται.",
    "push": "Έκανε ακριβώς τη γραμμή. Το ποντάρισμα επιστρέφεται.",
}
CLOSE = 1.0
LOW_MIN = 0.6


NEW_COLS = {"price_med": "REAL", "ev_med": "REAL", "n_books": "INTEGER",
            "fav": "INTEGER DEFAULT 0", "fav_at": "TEXT",          # «θα το έπαιζα» και πότε σημειώθηκε
            "excluded": "INTEGER DEFAULT 0", "excluded_at": "TEXT"}  # εξαίρεση από τα ποσοστά και πότε


def ensure(con) -> None:
    con.executescript(SCHEMA)
    have = {r[1] for r in con.execute("PRAGMA table_info(picks)")}
    for c, t in NEW_COLS.items():
        if c not in have:
            con.execute(f"ALTER TABLE picks ADD COLUMN {c} {t}")
    # παλιό «θα το έπαιζα» -> ετικέτα, με την ίδια ώρα σήμανσης
    con.execute("""INSERT OR IGNORE INTO pick_tags(pick_id, tag, set_at)
                   SELECT id, ?, fav_at FROM picks WHERE fav = 1""", (DEFAULT_TAGS[0],))
    con.execute("UPDATE picks SET fav = 0 WHERE fav = 1")
    con.commit()


def _now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")


def record(con, board: pd.DataFrame, nxt: pd.DataFrame, current: pd.DataFrame, w_model: float,
           now: pd.Timestamp | None = None, skip: set | None = None, force: bool = False) -> int:
    """Καταγράφει/ενημερώνει από τον πίνακα αξίας (B.build). Επιστρέφει πόσες νέες εγγραφές έγιναν."""
    ensure(con)
    skip = skip or set()
    if board is None or board.empty:
        return 0
    now = now or pd.Timestamp.now(tz="UTC")
    ts = now.isoformat(timespec="seconds")
    new = 0
    for r in board.to_dict("records"):
        team = r["Ομάδα"]
        if team not in nxt.index:
            continue
        g = nxt.loc[team]
        gdate = pd.Timestamp(g["date"])
        if pd.isna(gdate) or gdate <= now:
            continue                                   # μόνο πριν από το τζάμπολ
        key = (r["player_id"], r["stat"], float(r["Γραμμή"]), r["side"], int(g["season"]), int(g["game_code"]))
        if key in skip:
            continue
        row = con.execute("SELECT id, verdict, status FROM picks WHERE player_id=? AND stat=? AND line=? AND side=? "
                          "AND season=? AND game_code=?", key).fetchone()
        vals = dict(price=float(r["Απόδοση"]), bookmaker=r.get("Εταιρεία"), p_model=_f(r.get("P μοντέλου")),
                    p_market=_f(r.get("P αγοράς")), p_final=_f(r.get("P τελική")), ev=_f(r.get("EV")),
                    verdict=r["verdict"], mu=_f(r.get("Πρόβλεψη")), min_hat=_f(r.get("Λεπτά")), w_model=float(w_model),
                    price_med=_f(r.get("Τυπική απόδοση")), ev_med=_f(r.get("EV τυπικής")),
                    n_books=int(r.get("Εταιρείες αγοράς") or 0))
        if row is None:
            if r["verdict"] not in RECORD and not force:
                continue
            con.execute(
                "INSERT INTO picks(first_seen, last_seen, player_id, player, team, opp, stat, line, side, price, bookmaker,"
                " p_model, p_market, p_final, ev, verdict, mu, min_hat, w_model, price_last, ev_last, verdict_last,"
                " season, game_code, game_date, price_med, ev_med, n_books)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ts, ts, r["player_id"], r["Παίκτης"], team, r["Αντίπαλος"], r["stat"], key[2], r["side"],
                 vals["price"], vals["bookmaker"], vals["p_model"], vals["p_market"], vals["p_final"], vals["ev"],
                 vals["verdict"], vals["mu"], vals["min_hat"], vals["w_model"], vals["price"], vals["ev"], vals["verdict"],
                 key[4], key[5], gdate.isoformat(), vals["price_med"], vals["ev_med"], vals["n_books"]))
            new += 1
            continue
        pid_, old_verdict, status = row
        if status != "open":
            continue
        if RANK.get(r["verdict"], 0) > RANK.get(old_verdict, 0):
            con.execute("UPDATE picks SET first_seen=?, price=?, bookmaker=?, p_model=?, p_market=?, p_final=?, ev=?,"
                        " verdict=?, mu=?, min_hat=?, w_model=?, price_med=?, ev_med=?, n_books=? WHERE id=?",
                        (ts, vals["price"], vals["bookmaker"], vals["p_model"], vals["p_market"], vals["p_final"],
                         vals["ev"], vals["verdict"], vals["mu"], vals["min_hat"], vals["w_model"],
                         vals["price_med"], vals["ev_med"], vals["n_books"], pid_))
        con.execute("UPDATE picks SET last_seen=?, price_last=?, ev_last=?, verdict_last=? WHERE id=?",
                    (ts, vals["price"], vals["ev"], r["verdict"], pid_))
    con.commit()
    return new


def _f(x):
    try:
        return None if x is None or pd.isna(x) else float(x)
    except (TypeError, ValueError):
        return None


def characterize(value, minutes, line: float, side: str, min_hat) -> tuple[str, str]:
    """(status, tag) για ένα κλεισμένο στοίχημα."""
    if value is None:
        return "void", "void"
    status = outcome(value, line, side)
    if status == "push":
        return status, "push"
    margin = (value - line) if side == "over" else (line - value)
    if status == "won":
        return status, "close_win" if margin <= CLOSE else "easy_win"
    if min_hat and minutes is not None and minutes < LOW_MIN * min_hat:
        return status, "low_min"
    return status, "close_loss" if -margin <= CLOSE else "clear_loss"


def settle(con) -> int:
    """Κλείνει όσες ανοιχτές εγγραφές έχουν πλέον boxscore. Επιστρέφει πόσες έκλεισαν."""
    ensure(con)
    open_ = pd.read_sql_query("SELECT * FROM picks WHERE status='open'", con)
    n, ts = 0, _now()
    for p in open_.itertuples():
        if not con.execute("SELECT COUNT(*) FROM box WHERE season=? AND game_code=?", (p.season, p.game_code)).fetchone()[0]:
            continue
        cols = M.STATS[p.stat][1]
        row = con.execute(f"SELECT {'+'.join(cols)}, minutes FROM box WHERE season=? AND game_code=? AND player_id=?",
                          (p.season, p.game_code, p.player_id)).fetchone()
        value, minutes = (None, None) if row is None else (float(row[0]), float(row[1]))
        if minutes is not None and minutes <= 0:
            value, minutes = None, 0.0                # στη λίστα αλλά δεν μπήκε: όπως οι εταιρείες, άκυρο
        status, tag = characterize(value, minutes, p.line, p.side, p.min_hat)
        con.execute("UPDATE picks SET status=?, result=?, minutes=?, tag=?, settled_at=? WHERE id=?",
                    (status, value, minutes, tag, ts, p.id))
        n += 1
    con.commit()
    return n


def set_flag(con, ids, field: str, value: bool) -> None:
    """field: 'fav' ή 'excluded'. Κρατάει πότε έγινε η σήμανση, για να ξεχωρίζει «πριν» από «μετά» τον αγώνα."""
    assert field in ("fav", "excluded")
    ts = _now() if value else None
    con.executemany(f"UPDATE picks SET {field}=?, {field}_at=? WHERE id=?", [(int(value), ts, int(i)) for i in ids])
    con.commit()


FAV = DEFAULT_TAGS[0]


def pick_ids(con) -> dict:
    """(παίκτης, αγορά, γραμμή, πλευρά, σεζόν, αγώνας) -> id καταγραφής."""
    ensure(con)
    return {(r[1], r[2], float(r[3]), r[4], int(r[5]), int(r[6])): r[0] for r in
            con.execute("SELECT id, player_id, stat, line, side, season, game_code FROM picks")}


def keys_with_tag(con, tag: str = FAV) -> set:
    ensure(con)
    return {(r[0], r[1], float(r[2]), r[3], int(r[4]), int(r[5])) for r in con.execute(
        """SELECT p.player_id, p.stat, p.line, p.side, p.season, p.game_code FROM picks p
           JOIN pick_tags t ON t.pick_id = p.id WHERE t.tag = ?""", (tag,))}


def mark(con, rows: pd.DataFrame, nxt: pd.DataFrame, on: bool, tag: str = FAV, w_model: float = 0.5) -> int:
    """Βάζει/βγάζει την ετικέτα σε στοιχήματα του πίνακα αξίας. Όσα δεν έχουν καταγραφεί (π.χ. «Ύποπτο»)
    καταγράφονται πρώτα, για να μετρήσουν στο Αρχείο. Επιστρέφει πόσα άλλαξαν."""
    if rows.empty:
        return 0
    if on:
        record(con, rows, nxt, None, w_model, force=True)
    ids = pick_ids(con)
    n = 0
    for r in rows.to_dict("records"):
        team = r["Ομάδα"]
        if team not in nxt.index:
            continue
        k = (r["player_id"], r["stat"], float(r["Γραμμή"]), r["side"], int(nxt.at[team, "season"]), int(nxt.at[team, "game_code"]))
        i = ids.get(k)
        if i is None:
            continue
        cur = {x[0] for x in con.execute("SELECT tag FROM pick_tags WHERE pick_id=?", (i,))}
        set_tags(con, i, (cur | {tag}) if on else (cur - {tag}))
        n += 1
    return n


def set_tags(con, pick_id: int, tags) -> None:
    """Ορίζει τις ετικέτες ενός στοιχήματος. Όσες υπήρχαν κρατούν την αρχική ώρα σήμανσης."""
    tags = {t.strip() for t in (tags or []) if t and t.strip()}
    old = {r[0] for r in con.execute("SELECT tag FROM pick_tags WHERE pick_id=?", (int(pick_id),))}
    for t in old - tags:
        con.execute("DELETE FROM pick_tags WHERE pick_id=? AND tag=?", (int(pick_id), t))
    for t in tags - old:
        con.execute("INSERT INTO pick_tags VALUES (?,?,?)", (int(pick_id), t, _now()))
    con.commit()


def all_tags(con) -> list[str]:
    ensure(con)
    used = [r[0] for r in con.execute("SELECT tag FROM pick_tags GROUP BY tag ORDER BY MIN(set_at)")]
    return list(dict.fromkeys(DEFAULT_TAGS + used))


def load_tags(con, d: pd.DataFrame) -> pd.DataFrame:
    """Ένα row ανά (στοίχημα, ετικέτα) με before = σημειώθηκε πριν το τζάμπολ."""
    t = pd.read_sql_query("SELECT * FROM pick_tags", con)
    if t.empty or d.empty:
        return pd.DataFrame(columns=["pick_id", "tag", "set_at", "before"])
    t["set_at"] = pd.to_datetime(t["set_at"], utc=True, errors="coerce")
    t = t.merge(d[["id", "game_date"]], left_on="pick_id", right_on="id", how="inner")
    t["before"] = t["set_at"].notna() & (t["set_at"] < t["game_date"])
    return t[["pick_id", "tag", "set_at", "before"]]


def load(con) -> pd.DataFrame:
    ensure(con)
    try:
        d = pd.read_sql_query(
            """SELECT p.*, CASE WHEN h.player_id IS NULL THEN 0 ELSE 1 END AS hidden FROM picks p
               LEFT JOIN hidden_bets h ON h.player_id = p.player_id AND h.stat = p.stat AND h.line = p.line
                    AND h.side = p.side AND h.season = p.season AND h.game_code = p.game_code
               ORDER BY p.game_date DESC, p.id DESC""", con)
    except Exception:  # noqa: BLE001  — παλιά βάση χωρίς hidden_bets
        d = pd.read_sql_query("SELECT *, 0 AS hidden FROM picks ORDER BY game_date DESC, id DESC", con)
    if len(d):
        for c in ("first_seen", "game_date", "settled_at"):
            d[c] = pd.to_datetime(d[c], utc=True, errors="coerce")
        # τυπική απόδοση: ο διάμεσος των εταιρειών. Για παλιές εγγραφές (πριν μπει) εκτίμηση από την αγορά
        # με 6% γκανιότα, όσο περίπου έχει μια μεγάλη εταιρεία στα ειδικά παικτών.
        est = 1 / (d["p_market"] * 1.06)
        d["typ_approx"] = d["price_med"].isna()
        d["price_typ"] = d["price_med"].fillna(est.where(d["p_market"] > 0)).fillna(d["price"])
        d["price_typ"] = np.minimum(d["price_typ"], d["price"])
        won, lost = d["status"] == "won", d["status"] == "lost"
        d["units"] = np.select([won, lost], [d["price_typ"] - 1, -1.0], 0.0)         # στην τυπική απόδοση
        d["units_best"] = np.select([won, lost], [d["price"] - 1, -1.0], 0.0)        # στην καλύτερη
        d["exp_market"] = np.where(d["p_market"].notna(), d["p_market"], 1 / d["price_typ"])
        tg = load_tags(con, d)
        lists = tg.groupby("pick_id")["tag"].agg(list) if len(tg) else pd.Series(dtype=object)
        d["tags"] = [list(lists.get(i, [])) for i in d["id"]]
        after = set(tg.loc[~tg["before"], "pick_id"]) if len(tg) else set()
        d["tag_after"] = d["id"].isin(after)
        for f in ("fav", "excluded"):
            d[f] = d[f].fillna(0).astype(int)
            at = pd.to_datetime(d[f + "_at"], utc=True, errors="coerce")
            # «πριν»: σημειώθηκε πριν το τζάμπολ, άρα χωρίς να ξέρεις το αποτέλεσμα
            d[f + "_before"] = (d[f] == 1) & at.notna() & (at < d["game_date"])
        # κίνηση αγοράς: >0 σημαίνει ότι η απόδοση έπεσε μετά την καταγραφή (η αγορά ήρθε προς το μέρος μας)
        d["clv"] = d["price"] / d["price_last"] - 1
    return d


def summary(d: pd.DataFrame) -> dict:
    """Σύνοψη ανά 1 μονάδα. Το περιθώριο τύχης λογαριάζει ότι τα στοιχήματα του ίδιου αγώνα κινούνται μαζί
    (ένας αγώνας με χαμηλό σκορ κάνει πολλά Under να βγουν μαζί): διακύμανση από τα αθροίσματα ανά αγώνα."""
    s = d[d["status"].isin(["won", "lost"])]
    n = len(s)
    out = {"n": n, "n_open": int((d["status"] == "open").sum()), "n_void": int(d["status"].isin(["void", "push"]).sum())}
    if not n:
        return out
    win = (s["status"] == "won").astype(float)
    p_mod = s["p_model"].fillna(s["p_final"])
    p_mkt = s["exp_market"]

    def band(x: pd.Series) -> float:
        """90% περιθώριο για τον μέσο όρο του x, με συσταδοποίηση ανά αγώνα."""
        g = (x - x.mean()).groupby([s["season"], s["game_code"]]).sum()
        return float(1.64 * np.sqrt((g ** 2).sum()) / n) if len(g) > 1 else float("nan")
    games = s.groupby(["season", "game_code"]).ngroups
    moved = d[d["status"] != "void"].dropna(subset=["clv"])
    moved = moved[moved["price_last"] != moved["price"]]
    out.update(
        hit=float(win.mean()), exp_model=float(p_mod.mean()), exp_market=float(p_mkt.mean()),
        hit_band=band(win - p_mkt), games=games,
        units=float(s["units"].sum()), roi=float(s["units"].mean()), roi_band=band(s["units"]),
        units_best=float(s["units_best"].sum()), roi_best=float(s["units_best"].mean()),
        avg_price=float(s["price_typ"].mean()), avg_best=float(s["price"].mean()),
        approx=float(s["typ_approx"].mean()),
        exp_units=float(s["ev"].fillna(0).sum()),
        n_moved=len(moved), clv=float(moved["clv"].mean()) if len(moved) else None,
        clv_pos=float((moved["clv"] > 0).mean()) if len(moved) else None)
    return out
