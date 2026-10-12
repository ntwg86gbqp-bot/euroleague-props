"""Ημερολόγιο στοιχημάτων: καταγραφή και αυτόματη εκκαθάριση από τα boxscores."""
from __future__ import annotations
import pandas as pd
from . import model as M


def place(con, *, player_id, player, team, opp, stat, line, side, price, stake, bookmaker=None,
          p_model=None, p_market=None, p_final=None, ev=None, verdict=None,
          season=None, game_code=None, game_date=None) -> int:
    now = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    cur = con.execute(
        """INSERT INTO bets(placed_at, player_id, player, team, opp, stat, line, side, price, stake, bookmaker,
                            p_model, p_market, p_final, ev, verdict, season, game_code, game_date, status)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'open')""",
        (now, player_id, player, team, opp, stat, float(line), side, float(price), float(stake), bookmaker,
         p_model, p_market, p_final, ev, verdict,
         None if season is None else int(season), None if game_code is None else int(game_code),
         None if game_date is None else str(game_date)))
    con.commit()
    return cur.lastrowid


def outcome(value: float, line: float, side: str) -> str:
    if value == line:
        return "push"
    return "won" if (value > line) == (side == "over") else "lost"


def pnl(status: str, stake: float, price: float) -> float:
    return {"won": stake * (price - 1), "lost": -stake}.get(status, 0.0)


def settle(con) -> int:
    """Κλείνει όσα ανοιχτά στοιχήματα έχουν πλέον αποτέλεσμα. Επιστρέφει πόσα έκλεισαν.

    Αν ο αγώνας παίχτηκε και ο παίκτης δεν αγωνίστηκε, το στοίχημα ακυρώνεται (void), όπως στις εταιρείες."""
    open_ = pd.read_sql_query("SELECT * FROM bets WHERE status='open'", con)
    n = 0
    now = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    for b in open_.itertuples():
        season, code = b.season, b.game_code
        if season is None or pd.isna(season) or code is None or pd.isna(code):
            # χωρίς γνωστό αγώνα: ο πρώτος αγώνας της ομάδας μετά το ποντάρισμα
            r = con.execute(
                """SELECT season, game_code FROM games WHERE played=1 AND date > ?
                   AND (home_code=? OR away_code=?) ORDER BY date LIMIT 1""",
                (b.placed_at, b.team, b.team)).fetchone()
            if not r:
                continue
            season, code = r
        season, code = int(season), int(code)
        n_box = con.execute("SELECT COUNT(*) FROM box WHERE season=? AND game_code=?", (season, code)).fetchone()[0]
        if not n_box:
            continue                      # δεν έχει παιχτεί ή δεν έχουν κατέβει τα στατιστικά
        cols = M.STATS[b.stat][1]
        row = con.execute(f"SELECT {'+'.join(cols)} FROM box WHERE season=? AND game_code=? AND player_id=?",
                          (season, code, b.player_id)).fetchone()
        if row is None:
            status, value = "void", None
        else:
            value = float(row[0])
            status = outcome(value, b.line, b.side)
        con.execute("UPDATE bets SET status=?, result=?, pnl=?, settled_at=?, season=?, game_code=? WHERE id=?",
                    (status, value, pnl(status, b.stake, b.price), now, season, code, b.id))
        n += 1
    con.commit()
    return n


def load(con) -> pd.DataFrame:
    d = pd.read_sql_query("SELECT * FROM bets ORDER BY placed_at DESC, id DESC", con)
    if len(d):
        d["placed_at"] = pd.to_datetime(d["placed_at"], utc=True)
    return d


def summary(d: pd.DataFrame) -> dict:
    """Σύνοψη των κλεισμένων στοιχημάτων (χωρίς void/push στα ποσοστά)."""
    s = d[d["status"].isin(["won", "lost"])]
    staked = float(s["stake"].sum())
    out = {"n_open": int((d["status"] == "open").sum()), "n_settled": int(len(s)),
           "n_void": int(d["status"].isin(["void", "push"]).sum()),
           "staked": staked, "pnl": float(s["pnl"].sum()),
           "roi": float(s["pnl"].sum() / staked) if staked else None,
           "hit": float((s["status"] == "won").mean()) if len(s) else None,
           "expected_hit": float(s["p_final"].mean()) if len(s) and s["p_final"].notna().all() else None,
           "expected_pnl": float((s["stake"] * s["ev"]).sum()) if len(s) and s["ev"].notna().all() else None}
    return out
