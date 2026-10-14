"""Πίνακας αξίας: ενώνει αποδόσεις, μοντέλο και ιστορικό."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import model as M


def next_games(con, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Επόμενος αγώνας κάθε ομάδας: index=team, στήλες opp, is_home, date, round."""
    g = pd.read_sql_query("SELECT * FROM games WHERE played=0", con)
    if g.empty:
        return pd.DataFrame(columns=["opp", "is_home", "date", "round", "season", "game_code"])
    g["date"] = pd.to_datetime(g["date"], utc=True, errors="coerce")
    now = now or pd.Timestamp.now(tz="UTC")
    g = g[g["date"] >= now - pd.Timedelta(hours=3)]
    a = g.rename(columns={"home_code": "team", "away_code": "opp"}).assign(is_home=True)
    b = g.rename(columns={"away_code": "team", "home_code": "opp"}).assign(is_home=False)
    cols = ["team", "opp", "is_home", "date", "round", "season", "game_code"]
    allg = pd.concat([a[cols], b[cols]]).sort_values("date")
    return allg.drop_duplicates("team").set_index("team")


def fresh_odds(odds: pd.DataFrame, current: pd.DataFrame, con, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Πετάει αποδόσεις που περάστηκαν πριν από το τζάμπολ του τελευταίου αγώνα της ομάδας του παίκτη.

    Οι αποδόσεις δεν ξέρουν σε ποιον αγώνα ανήκουν: αντιστοιχίζονται στον επόμενο αγώνα της ομάδας. Χωρίς αυτό
    το φίλτρο, σε διπλή αγωνιστική οι αποδόσεις της Τρίτης θα φαίνονταν την Τετάρτη σαν αποδόσεις της Πέμπτης."""
    if odds.empty:
        return odds
    now = now or pd.Timestamp.now(tz="UTC")
    g = pd.read_sql_query("SELECT home_code, away_code, date FROM games", con)
    g["date"] = pd.to_datetime(g["date"], utc=True, errors="coerce")
    g = g[g["date"] <= now]
    last = pd.concat([g[["home_code", "date"]].rename(columns={"home_code": "team"}),
                      g[["away_code", "date"]].rename(columns={"away_code": "team"})]).groupby("team")["date"].max()
    team = odds["player_id"].map(current["team"]) if len(current) else pd.Series(index=odds.index, dtype=object)
    cut = pd.to_datetime(team.map(last), utc=True)
    keep = cut.isna() | (pd.to_datetime(odds["fetched_at"], utc=True) > cut)
    return odds[keep.to_numpy()]


def absent_ids(con) -> set:
    return {r[0] for r in con.execute("SELECT player_id FROM absences")}


def set_absent(con, team_ids, absent) -> None:
    """Ορίζει ποιοι από τους `team_ids` λείπουν (οι υπόλοιποι της λίστας θεωρούνται διαθέσιμοι)."""
    now = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    for pid in team_ids:
        if pid in absent:
            con.execute("INSERT OR IGNORE INTO absences(player_id, set_at) VALUES (?,?)", (pid, now))
        else:
            con.execute("DELETE FROM absences WHERE player_id=?", (pid,))
    con.commit()


def clear_returned(con) -> int:
    """Σβήνει απουσίες παικτών που ξαναέπαιξαν μετά τη δήλωση."""
    n = con.execute(
        """DELETE FROM absences WHERE EXISTS (
             SELECT 1 FROM box b JOIN games g USING(season, game_code)
             WHERE b.player_id = absences.player_id AND g.date > absences.set_at)""").rowcount
    con.commit()
    return n


def minutes_for(pid: str, current: pd.DataFrame, ctx: dict) -> float:
    return float(M.absence_effect(pid, current, ctx, ctx.get("absent", set()))["min"])


def projection(pid: str, stat: str, current: pd.DataFrame, ctx: dict, nxt: pd.DataFrame):
    """(μ, αντίπαλος, εντός;) για τον επόμενο αγώνα, με τις δηλωμένες απουσίες συμπαικτών."""
    team = current.at[pid, "team"]
    opp = nxt.at[team, "opp"] if team in nxt.index else None
    is_home = bool(nxt.at[team, "is_home"]) if team in nxt.index else None
    mu = M.project(pid, stat, current, ctx, opp, is_home)
    eff = M.absence_effect(pid, current, ctx, ctx.get("absent", set()))
    return mu * eff["mult"][stat], opp, is_home


def k_for(pid: str, stat: str, current: pd.DataFrame, disp: dict, ctx: dict | None = None) -> float:
    m = minutes_for(pid, current, ctx) if ctx else float(current.at[pid, "min_hat"])
    return float(M.k_of(disp[stat], m))


VERDICTS = {"strong": "🟢 Αξίζει", "marginal": "🟡 Οριακό", "suspect": "🟠 Ύποπτο", "no": "⚪ Όχι"}


def verdict(ev: float, p_model: float, p_mkt, n_games: int, tail: bool = False, outlier: bool = False,
            n_books: int = 0) -> str:
    """Ετυμηγορία σε απλά λόγια.

    suspect : φαίνεται κερδοφόρο, αλλά το μοντέλο διαφωνεί με την αγορά πάνω από 18 μονάδες
              (συνήθως λείπει πληροφορία) ή η γραμμή είναι πολύ πάνω από την πρόβλεψη
    strong  : EV >= 6%, αρκετό ιστορικό, και το μοντέλο πάνω από την αγορά τουλάχιστον 4 μονάδες
    marginal: EV >= 2%
    """
    gap = None if p_mkt is None else p_model - p_mkt
    # όταν πολλές εταιρείες συμφωνούν, η αγορά είναι πιο αξιόπιστη: ανεχόμαστε μικρότερη διαφωνία
    max_gap = 0.12 if n_books >= 10 else 0.18
    if gap is not None and abs(gap) >= max_gap and ev >= 0.02:
        return "suspect"
    if outlier and ev >= 0.02:
        return "suspect"      # η απόδοση είναι πολύ πάνω από όλες τις άλλες εταιρείες: πιθανό λάθος/παλιά γραμμή
    if tail and ev >= 0.02:
        return "suspect"      # ψηλή γραμμή: εκεί το μοντέλο υπερεκτιμά το Over (βλ. is_tail)
    if ev >= 0.06 and n_games >= 10 and (gap is None or gap >= 0.04):
        return "strong"
    if ev >= 0.02:
        return "marginal"
    return "no"


def is_tail(line: float, mu: float, side: str) -> bool:
    """Over σε γραμμή πολύ πάνω από την πρόβλεψη. Στον έλεγχο του 2025-26 το μοντέλο υπερεκτιμούσε αυτές
    τις πιθανότητες (π.χ. πόντοι βασικών στο 1.5×μ: πρόβλεψη 19%, πραγματικό 17%· στο 2×μ: 7% έναντι 5%),
    και στις μεγάλες αποδόσεις μια τέτοια διαφορά φαίνεται σαν μεγάλο αναμενόμενο κέρδος."""
    return side == "over" and line >= 1.4 * max(mu, 0.5)


def explain(v: str, side: str, p_model: float, p_mkt, hits10, f_opp: float, stat_label: str,
            n_games: int, min_hat: float) -> str:
    parts = []
    if p_mkt is not None:
        parts.append(f"μοντέλο {p_model:.0%} έναντι αγοράς {p_mkt:.0%}")
    else:
        parts.append(f"μοντέλο {p_model:.0%}")
    if hits10[1]:
        parts.append(f"{hits10[0]}/{hits10[1]} στους τελευταίους")
    if abs(f_opp) >= 0.04:
        good = (f_opp > 0) == (side == "over")
        parts.append(f"ο αντίπαλος δίνει {f_opp:+.0%} ({'υπέρ' if good else 'κατά'})")
    if v == "suspect":
        parts.append("μεγάλη απόκλιση: έλεγξε απουσίες ή αλλαγή ρόλου")
    if n_games < 10:
        parts.append("λίγοι αγώνες στο ιστορικό")
    if min_hat < 15:
        parts.append("λίγα και ασταθή λεπτά")
    return " · ".join(parts)


OUTLIER = 1.25          # απόδοση > 1.25 × δίκαιη απόδοση της αγοράς -> ύποπτη (λάθος ή παλιά γραμμή)


def playable(odds: pd.DataFrame, books) -> pd.Series:
    """Ποιες αποδόσεις μπορείς να παίξεις: όσες πέρασες εσύ (screenshot/χειροκίνητα) πάντα, και από το API
    μόνο όσες είναι από τις εταιρείες σου. Χωρίς επιλογή εταιρειών, όλες."""
    if not books:
        return pd.Series(True, index=odds.index)
    src = odds["source"] if "source" in odds else pd.Series("manual", index=odds.index)
    return (src != "oddspapi") | odds["bookmaker"].str.lower().isin({b.lower() for b in books})


def build(odds: pd.DataFrame, df: pd.DataFrame, current: pd.DataFrame, ctx: dict,
          disp: dict, nxt: pd.DataFrame, w_model: float = 0.5,
          assumed_margin: float = 0.07, books=None) -> pd.DataFrame:
    """Πίνακας αξίας.

    books: οι εταιρείες όπου μπορείς να παίξεις (κενό = όλες). Η αγορά (πιθανότητα χωρίς γκανιότα) υπολογίζεται
    από ΟΛΕΣ τις εταιρείες (διάμεσος), η «Απόδοση» είναι η καλύτερη από τις δικές σου και η «Τυπική απόδοση»
    ο διάμεσος όλων των εταιρειών: αυτό περίπου βρίσκεις σε μια μεγάλη εταιρεία όπως η Stoiximan."""
    if odds.empty:
        return pd.DataFrame()
    absent = ctx.get("absent", set())
    o = odds[odds["player_id"].isin(current.index) & odds["stat"].isin(list(M.STATS))
             & ~odds["player_id"].isin(absent)].copy()
    if o.empty:
        return pd.DataFrame()
    o["_ok"] = playable(o, books).to_numpy()
    # μία τιμή ανά (στοίχημα, εταιρεία, πλευρά), και πίνακας over/under ανά εταιρεία — μία φορά για όλα
    agg = o.groupby(["player_id", "stat", "line", "bookmaker", "side"], sort=False).agg(
        price=("price", "max"), ok=("_ok", "max")).reset_index()
    wide = agg.pivot_table(index=["player_id", "stat", "line", "bookmaker"], columns="side",
                           values="price", aggfunc="max")
    for c in ("over", "under"):
        if c not in wide:
            wide[c] = np.nan
    wo, wu = wide["over"].to_numpy(float), wide["under"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        marg = 1 / wo + 1 / wu - 1
        pfair = (1 / wo) / (1 / wo + 1 / wu)
    good = np.isfinite(marg) & (wo > 1) & (wu > 1) & (marg > -0.02) & (marg < 0.25)
    wk = wide.index.droplevel("bookmaker")
    mk = pd.DataFrame({"p": np.where(good, pfair, np.nan), "m": np.where(good, marg, np.nan)}, index=wk)
    mk = mk.groupby(level=[0, 1, 2], sort=False).agg(p=("p", "median"), m=("m", "median"), n=("p", "count"))
    mkd = {k_: (r[0], r[1], r[2]) for k_, r in zip(mk.index, mk[["p", "m", "n"]].to_numpy())}
    med = agg.groupby(["player_id", "stat", "line", "side"], sort=False)["price"].median().to_dict()
    playable_ = agg[agg["ok"]]
    if playable_.empty:
        return pd.DataFrame()
    best_idx = playable_.groupby(["player_id", "stat", "line", "side"], sort=False)["price"].idxmax()
    best = agg.loc[best_idx.to_numpy(), ["player_id", "stat", "line", "side", "price", "bookmaker"]]

    cur_season = int(df["season"].max())
    hist = {pid: g for pid, g in df.groupby("player_id", sort=False)}
    per_pid = {}
    rows = []
    for (pid, stat, line), bs in best.groupby(["player_id", "stat", "line"], sort=False):
        if pid not in per_pid:
            team = current.at[pid, "team"]
            per_pid[pid] = (team, M.absence_effect(pid, current, ctx, absent),
                            nxt.at[team, "opp"] if team in nxt.index else None)
        team, eff, opp = per_pid[pid]
        is_home = bool(nxt.at[team, "is_home"]) if team in nxt.index else None
        mu = M.project(pid, stat, current, ctx, opp, is_home) * eff["mult"][stat]
        k = float(M.k_of(disp[stat], eff["min"]))
        po, pp, pu = (float(x) for x in M.probs(mu, k, line))
        key = (pid, stat, line)
        p_mkt, margin, n_books = None, None, 0
        r_mk = mkd.get(key)
        if r_mk and r_mk[2] > 0:
            p_mkt, margin, n_books = float(r_mk[0]), float(r_mk[1]), int(r_mk[2])
        else:
            mo, mu_ = med.get((pid, stat, line, "over")), med.get((pid, stat, line, "under"))
            if mo is not None or mu_ is not None:
                p_mkt = M.devig(mo, mu_, assumed_margin)
        live = 1 - pp
        cond_model = po / live if live > 0 else 0.5
        cond = w_model * cond_model + (1 - w_model) * p_mkt if p_mkt is not None else cond_model
        h = hist[pid]
        vals = h[stat].to_numpy(float)
        vals_season = vals[h["season"].to_numpy() == cur_season]
        n_g = int(current.at[pid, "n_games"])
        f_o = float(ctx["opp"][opp][M.KEYS.index(stat)] - 1) if opp in ctx["opp"] else 0.0
        for side, price, bookmaker in zip(bs["side"].tolist(), bs["price"].tolist(), bs["bookmaker"].tolist()):
            price = float(price)
            typ = float(med.get((pid, stat, line, side), price))
            p_win = live * (cond if side == "over" else 1 - cond)
            p_lose = live - p_win
            pm_side = po if side == "over" else pu
            l5, l10 = M.hit_rate(vals, line, side, 5), M.hit_rate(vals, line, side, 10)
            ss = M.hit_rate(vals_season, line, side)
            pk_side = None if p_mkt is None else (p_mkt if side == "over" else 1 - p_mkt)
            ev_ = M.ev(p_win, p_lose, price)
            tail = is_tail(line, mu, side)
            outlier = bool(pk_side and n_books >= 3 and price > OUTLIER / pk_side)
            vd = verdict(ev_, pm_side, pk_side, n_g, tail, outlier, n_books)
            last5 = vals[-5:]
            rows.append({
                "verdict": vd, "Ετυμηγορία": VERDICTS[vd],
                "Στοίχημα": f"{'Over' if side == 'over' else 'Under'} {line:g} {M.STATS[stat][0]}",
                "Γιατί": explain(vd, side, pm_side, pk_side, l10, f_o, M.STATS[stat][0], n_g, eff["min"])
                         + (" · λείπει: " + ", ".join(n.split(",")[0].title() for n in eff["out"]) if eff["out"] else "")
                         + (" · ψηλή γραμμή: το μοντέλο υπερεκτιμά τις μεγάλες βραδιές" if tail and vd == "suspect" else "")
                         + (f" · η απόδοση είναι πολύ πάνω από τις άλλες {n_books} εταιρείες (τυπική {typ:.2f}): "
                            "πιθανό λάθος ή παλιά γραμμή" if outlier and vd == "suspect" else ""),
                "p_mkt_side": pk_side,
                "Τελευταίοι 10": [float(x) for x in vals[-10:]],
                "Φόρμα": "".join("–" if x == line else ("✓" if (x > line) == (side == "over") else "✗")
                                 for x in last5),
                "f_opp": f_o, "side": side,
                "player_id": pid, "Παίκτης": current.at[pid, "player"], "Ομάδα": team,
                "Αντίπαλος": opp or "—", "stat": stat, "Αγορά": M.STATS[stat][0],
                "Γραμμή": line, "Πλευρά": "Over" if side == "over" else "Under",
                "Απόδοση": price, "Εταιρεία": bookmaker, "Τυπική απόδοση": typ,
                "EV τυπικής": M.ev(p_win, p_lose, typ),
                "Πρόβλεψη": mu, "Λεπτά": eff["min"],
                "P μοντέλου": pm_side, "P αγοράς": pk_side,
                "P τελική": p_win, "EV": ev_,
                "Kelly/4": M.kelly(p_win, p_lose, price) / 4,
                "L5": f"{l5[0]}/{l5[1]}", "L10": f"{l10[0]}/{l10[1]}",
                "Σεζόν": f"{ss[0]}/{ss[1]}", "hit10": l10[0] / l10[1] if l10[1] else np.nan,
                "Αγώνες": n_g, "Γκανιότα": margin, "Εταιρείες αγοράς": n_books,
            })
    out = pd.DataFrame(rows)
    return out.sort_values("EV", ascending=False).reset_index(drop=True) if len(out) else out
