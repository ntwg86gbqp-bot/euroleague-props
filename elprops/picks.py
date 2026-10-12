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
"""

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


def ensure(con) -> None:
    con.executescript(SCHEMA)


def _now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")


def record(con, board: pd.DataFrame, nxt: pd.DataFrame, current: pd.DataFrame, w_model: float,
           now: pd.Timestamp | None = None) -> int:
    """Καταγράφει/ενημερώνει από τον πίνακα αξίας (B.build). Επιστρέφει πόσες νέες εγγραφές έγιναν."""
    ensure(con)
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
        row = con.execute("SELECT id, verdict, status FROM picks WHERE player_id=? AND stat=? AND line=? AND side=? "
                          "AND season=? AND game_code=?", key).fetchone()
        vals = dict(price=float(r["Απόδοση"]), bookmaker=r.get("Εταιρεία"), p_model=_f(r.get("P μοντέλου")),
                    p_market=_f(r.get("P αγοράς")), p_final=_f(r.get("P τελική")), ev=_f(r.get("EV")),
                    verdict=r["verdict"], mu=_f(r.get("Πρόβλεψη")), min_hat=_f(r.get("Λεπτά")), w_model=float(w_model))
        if row is None:
            if r["verdict"] not in RECORD:
                continue
            con.execute(
                "INSERT INTO picks(first_seen, last_seen, player_id, player, team, opp, stat, line, side, price, bookmaker,"
                " p_model, p_market, p_final, ev, verdict, mu, min_hat, w_model, price_last, ev_last, verdict_last,"
                " season, game_code, game_date) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ts, ts, r["player_id"], r["Παίκτης"], team, r["Αντίπαλος"], r["stat"], key[2], r["side"],
                 vals["price"], vals["bookmaker"], vals["p_model"], vals["p_market"], vals["p_final"], vals["ev"],
                 vals["verdict"], vals["mu"], vals["min_hat"], vals["w_model"], vals["price"], vals["ev"], vals["verdict"],
                 key[4], key[5], gdate.isoformat()))
            new += 1
            continue
        pid_, old_verdict, status = row
        if status != "open":
            continue
        if RANK.get(r["verdict"], 0) > RANK.get(old_verdict, 0):
            con.execute("UPDATE picks SET first_seen=?, price=?, bookmaker=?, p_model=?, p_market=?, p_final=?, ev=?,"
                        " verdict=?, mu=?, min_hat=?, w_model=? WHERE id=?",
                        (ts, vals["price"], vals["bookmaker"], vals["p_model"], vals["p_market"], vals["p_final"],
                         vals["ev"], vals["verdict"], vals["mu"], vals["min_hat"], vals["w_model"], pid_))
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


def load(con) -> pd.DataFrame:
    ensure(con)
    d = pd.read_sql_query("SELECT * FROM picks ORDER BY game_date DESC, id DESC", con)
    if len(d):
        for c in ("first_seen", "game_date", "settled_at"):
            d[c] = pd.to_datetime(d[c], utc=True, errors="coerce")
        d["units"] = np.select([d["status"] == "won", d["status"] == "lost"], [d["price"] - 1, -1.0], 0.0)
        # κίνηση αγοράς: >0 σημαίνει ότι η απόδοση έπεσε μετά την καταγραφή (η αγορά ήρθε προς το μέρος μας)
        d["clv"] = d["price"] / d["price_last"] - 1
    return d


def summary(d: pd.DataFrame) -> dict:
    """Σύνοψη ανά 1 μονάδα ποντάρισμα, με περιθώριο τύχης για την απόδοση επένδυσης."""
    s = d[d["status"].isin(["won", "lost"])]
    n = len(s)
    out = {"n": n, "n_open": int((d["status"] == "open").sum()), "n_void": int(d["status"].isin(["void", "push"]).sum())}
    if not n:
        return out
    p = s["p_final"].fillna(1 / s["price"])
    var = (p * (1 - p) * s["price"] ** 2).sum()           # διασπορά κέρδους ανά στοίχημα του 1
    moved = d[d["status"] != "void"].dropna(subset=["clv"])
    moved = moved[moved["price_last"] != moved["price"]]
    out.update(units=float(s["units"].sum()), roi=float(s["units"].mean()), hit=float((s["status"] == "won").mean()),
               exp_hit=float(p.mean()), exp_units=float(s["ev"].fillna(0).sum()), exp_roi=float(s["ev"].fillna(0).mean()),
               roi_band=float(1.64 * np.sqrt(var) / n), avg_price=float(s["price"].mean()),
               n_moved=len(moved), clv=float(moved["clv"].mean()) if len(moved) else None,
               clv_pos=float((moved["clv"] > 0).mean()) if len(moved) else None)
    return out
