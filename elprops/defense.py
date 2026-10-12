"""Τι δίνει κάθε άμυνα σε κάθε θέση (γκαρντ / φόργουορντ / σέντερ).

Μονάδα ανάλυσης: ένας αγώνας μιας άμυνας. Για κάθε αγώνα αθροίζεται η παραγωγή των αντιπάλων
ανά θέση (π.χ. «πόντοι που έβαλαν οι γκαρντ του αντιπάλου»), είτε ανά αγώνα είτε ανά 100 κατοχές.

Θόρυβος: με λίγους αγώνες ο μέσος όρος μιας ομάδας είναι κυρίως τύχη. Γι' αυτό η «διορθωμένη»
τιμή συρρικνώνεται προς τον μέσο όρο της λίγκας με empirical Bayes:
    τιμή = (n·μέσος_ομάδας + n0·μέσος_λίγκας) / (n + n0),   n0 = σ²_εντός / τ²_μεταξύ
όπου σ²_εντός η διακύμανση από αγώνα σε αγώνα και τ² η πραγματική διασπορά μεταξύ ομάδων.
Ένα n0 = 30 σημαίνει ότι χρειάζονται 30 αγώνες για να εμπιστευτείς τα δεδομένα της ομάδας κατά 50%.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .model import BASE_KEYS

DSTATS = ["pts", "reb", "ast", "fg3m", "fg2m", "stl", "blk", "tov", "pra"]
POS = (0, 1, 2)
ALL = 3                      # όλες οι θέσεις μαζί
N0_MIN, N0_MAX = 2.0, 400.0


def possessions(df: pd.DataFrame) -> pd.DataFrame:
    """Κατοχές ανά αγώνα (μέσος όρος των δύο ομάδων): FGA + 0.44·FTA − OREB + TOV."""
    t = df.groupby(["season", "game_code", "team"])[["fg2a", "fg3a", "fta", "oreb", "tov"]].sum()
    p = t["fg2a"] + t["fg3a"] + 0.44 * t["fta"] - t["oreb"] + t["tov"]
    return p.groupby(level=["season", "game_code"]).mean().rename("poss").reset_index()


def game_table(df: pd.DataFrame, roles: dict) -> pd.DataFrame:
    """Ένα row ανά (άμυνα, αγώνας, θέση): τι έκαναν οι αντίπαλοι αυτής της θέσης.
    Στήλες: team (η άμυνα), vs (ποιος επιτέθηκε), season, game_code, date, round, is_home, pos, poss,
    τα DSTATS και minutes. pos=3 είναι όλες οι θέσεις μαζί."""
    d = df.assign(pos=df["player_id"].map(roles).fillna(1).astype(int))
    keys = ["opp", "team", "season", "game_code", "date", "round", "is_home"]
    cols = [k for k in DSTATS if k in d.columns] + ["minutes"]
    g = d.groupby(keys + ["pos"])[cols].sum()
    tot = d.groupby(keys)[cols].sum()
    # κάθε αγώνας παίρνει και τις τρεις θέσεις (0 αν δεν έπαιξε κανείς αυτής της θέσης)
    idx = pd.MultiIndex.from_tuples([(*k, p) for k in tot.index for p in POS], names=keys + ["pos"])
    full = g.reindex(idx, fill_value=0).reset_index()
    tot = tot.reset_index().assign(pos=ALL)
    out = pd.concat([full, tot], ignore_index=True)
    out = out.rename(columns={"team": "vs"}).rename(columns={"opp": "team"})
    out["is_home"] = ~out["is_home"].astype(bool)          # έδρα από τη σκοπιά της άμυνας
    out = out.merge(possessions(df), on=["season", "game_code"], how="left")
    return out.sort_values(["date", "game_code", "pos"]).reset_index(drop=True)


def per_unit(G: pd.DataFrame, unit: str) -> pd.DataFrame:
    """unit: 'game' ή 'poss100'."""
    if unit == "game":
        return G
    G = G.copy()
    f = 100.0 / G["poss"].clip(lower=40)
    for k in DSTATS:
        G[k] = G[k] * f
    return G


def estimate_n0(G: pd.DataFrame, min_games: int = 10) -> dict:
    """n0 = σ²_εντός / τ²_μεταξύ, ανά (θέση, στατιστικό), από όλες τις σεζόν (ομάδα-σεζόν = μία ομάδα)."""
    out = {}
    for pos, gp in G.groupby("pos"):
        grp = gp.groupby(["team", "season"])
        n = grp.size()
        ok = n[n >= min_games].index
        for k in DSTATS:
            s = grp[k]
            mean, var = s.mean().loc[ok], s.var().loc[ok]
            nn = n.loc[ok]
            if len(ok) < 6:
                out[(pos, k)] = 20.0
                continue
            s2w = float(np.average(var, weights=nn - 1))
            # αφαιρείται η διαφορά επιπέδου μεταξύ σεζόν
            centred = mean - mean.groupby(level="season").transform("mean")
            tau2 = float(centred.var() - s2w * np.mean(1.0 / nn))
            out[(pos, k)] = float(np.clip(s2w / tau2, N0_MIN, N0_MAX)) if tau2 > 0 else N0_MAX
    return out


def table(G: pd.DataFrame, seasons, n0: dict, last: int | None = None) -> pd.DataFrame:
    """Ένα row ανά (ομάδα, θέση, στατιστικό) για τις σεζόν `seasons` (int ή λίστα).

    raw: μέσος όρος της ομάδας · league: μέσος όρος λίγκας · adj: διορθωμένη τιμή (shrinkage)
    pct: adj/league − 1 (αυτό χρωματίζεται) · rel: n/(n+n0), πόσο μετράνε τα δεδομένα της ομάδας
    rank: 1 = δίνει τα περισσότερα στη λίγκα (με βάση το adj)."""
    S = G[G["season"].isin(np.atleast_1d(seasons))]
    keep = set(S[S["season"] == S["season"].max()]["team"])       # μόνο όσες παίζουν στην τελευταία σεζόν
    if last:
        S = S.sort_values("date").groupby(["team", "pos"], group_keys=False).tail(last)
    rows = []
    for (team, pos), t in S.groupby(["team", "pos"]):
        if team not in keep:
            continue
        n = len(t)
        for k in DSTATS:
            rows.append((team, int(pos), k, n, float(t[k].mean())))
    T = pd.DataFrame(rows, columns=["team", "pos", "stat", "n", "raw"])
    if T.empty:
        return T
    lg = S.groupby("pos")[DSTATS].mean().stack().rename("league")
    lg.index.names = ["pos", "stat"]
    T = T.join(lg, on=["pos", "stat"])
    T["n0"] = [n0.get((p, k), 20.0) for p, k in zip(T["pos"], T["stat"])]
    T["rel"] = T["n"] / (T["n"] + T["n0"])
    T["adj"] = T["rel"] * T["raw"] + (1 - T["rel"]) * T["league"]
    safe = T["league"].where(T["league"] > 0, np.nan)
    T["pct"] = (T["adj"] / safe - 1).fillna(0.0)
    T["pct_raw"] = (T["raw"] / safe - 1).fillna(0.0)
    T["rank"] = T.groupby(["pos", "stat"])["adj"].rank(ascending=False, method="min").astype(int)
    T["n_teams"] = T.groupby(["pos", "stat"])["team"].transform("count")
    # πέρα από τη συνολική άμυνα: (1+pct_θέσης)/(1+pct_όλων) − 1. Το μοντέλο ήδη ξέρει τη συνολική.
    allp = T[T["pos"] == ALL].set_index(["team", "stat"])["pct"]
    T["extra"] = (1 + T["pct"]) / (1 + T.set_index(["team", "stat"]).index.map(allp).to_numpy(float)) - 1
    return T


def persistence(G: pd.DataFrame, min_games: int = 16) -> pd.DataFrame:
    """Έχει ουσία; Για κάθε ομάδα-σεζόν με ≥ min_games: μέσος όρος στο 1ο μισό των αγώνων έναντι 2ου μισού.
    Συσχέτιση μεταξύ ομάδων (αφού αφαιρεθεί ο μέσος όρος κάθε σεζόν).

    r_total   : «δίνει πολλά στους γκαρντ» στο 1ο μισό → και στο 2ο;
    r_specific: το ίδιο για το κομμάτι που ΔΕΝ εξηγείται από τη συνολική άμυνα
                (μερίδιο της θέσης στο σύνολο που δέχεται η ομάδα). Αυτό είναι που λείπει από το μοντέλο."""
    res = []
    g = G.sort_values("date")
    n_games = g[g["pos"] == ALL].groupby(["team", "season"]).size()
    ok = set(n_games[n_games >= min_games].index)
    if len(ok) < 8:
        return pd.DataFrame(columns=["pos", "stat", "r_total", "r_specific", "n_teams"])
    halves = {}
    for (team, season), t in g[g["pos"] == ALL].groupby(["team", "season"]):
        if (team, season) in ok:
            codes = t["game_code"].tolist()
            halves[(team, season)] = set(codes[: len(codes) // 2])
    g = g[[(a, b) in ok for a, b in zip(g["team"], g["season"])]].copy()
    g["half"] = [int(c not in halves[(a, b)]) for a, b, c in zip(g["team"], g["season"], g["game_code"])]
    m = g.groupby(["pos", "team", "season", "half"])[DSTATS].mean()
    for pos in POS:
        for k in DSTATS:
            a = m.loc[pos][k].unstack("half")
            tot = m.loc[ALL][k].unstack("half")
            share = a / tot.replace(0, np.nan)

            def r_of(x):
                x = x.dropna()
                x = x - x.groupby(level="season").transform("mean")
                return float(np.corrcoef(x[0], x[1])[0, 1]) if len(x) >= 8 and x[0].std() > 0 and x[1].std() > 0 else np.nan
            res.append((pos, k, r_of(a), r_of(share), len(a)))
    return pd.DataFrame(res, columns=["pos", "stat", "r_total", "r_specific", "n_teams"])


def hurt_list(df: pd.DataFrame, roles: dict, team: str, seasons, pos: int, stat: str, top: int = 8) -> pd.DataFrame:
    """Οι καλύτερες εμφανίσεις αντιπάλων της θέσης `pos` απέναντι στην `team` φέτος,
    με τον μέσο όρο του παίκτη στους υπόλοιπους αγώνες του για σύγκριση."""
    s = df[df["season"].isin(np.atleast_1d(seasons))]
    s = s.assign(pos=s["player_id"].map(roles).fillna(1).astype(int))
    vs = s[(s["opp"] == team) & (s["minutes"] > 0) & ((s["pos"] == pos) if pos != ALL else True)]
    if vs.empty:
        return vs
    other = s[(s["opp"] != team) & (s["minutes"] > 0)].groupby("player_id")[stat].mean()
    out = vs.assign(avg=vs["player_id"].map(other)).sort_values(stat, ascending=False).head(top)
    out["diff"] = out[stat] - out["avg"]
    return out[["player_id", "player", "team", "date", "round", "minutes", stat, "avg", "diff", "pos"]]
