"""Μοντέλο πιθανότητας για ειδικά παικτών.

X ~ NegBin(μ, k),   μ = r̂ · m̂ · f_opp · f_έδρας
  r̂   : ρυθμός ανά λεπτό, εκθετικά σταθμισμένος, με empirical-Bayes shrinkage προς τον μέσο
        όρο της λίγκας: r̂ = (Σx + κ·r0)/(Σmin + κ), κ ∝ r0/Var(ρυθμών μεταξύ παικτών) ανά αγορά
  m̂   : προβλεπόμενα λεπτά: εκθετικά σταθμισμένος μέσος, διορθωμένος γραμμικά για παλινδρόμηση
        προς τον μέσο (όσοι παίζουν πολύ παίζουν κατά μέσο όρο λίγο λιγότερο από το πρόσφατο)
  f_opp: πόσο πάνω/κάτω από τον μέσο όρο δίνει ο αντίπαλος στο συγκεκριμένο στατιστικό
  k    : διασπορά που εξαρτάται από τα λεπτά, 1/k = a + c/m̂² ανά αγορά (μέγιστη πιθανοφάνεια).
         Ο όρος c/m̂² είναι η αβεβαιότητα των λεπτών: για σ_min ≈ σταθερό, CV² ∝ 1/m̂².
         Με ενιαίο k οι βασικοί έβγαιναν με υπερβολικά φαρδιά κατανομή και οι αναπληρωματικοί
         με υπερβολικά στενή.

Όλες οι προβλέψεις είναι walk-forward: για κάθε αγώνα χρησιμοποιούνται μόνο προηγούμενοι.
Οι παράμετροι στο Params επιλέχθηκαν με χρονικό split (τελευταίο 40% ως validation).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import minimize

STATS: dict[str, tuple[str, list[str]]] = {
    "pts": ("Πόντοι", ["pts"]),
    "reb": ("Ριμπάουντ", ["reb"]),
    "ast": ("Ασίστ", ["ast"]),
    "fg3m": ("Τρίποντα", ["fg3m"]),
    "fg2m": ("Δίποντα", ["fg2m"]),
    "stl": ("Κλεψίματα", ["stl"]),
    "blk": ("Κοψίματα", ["blk"]),
    "tov": ("Λάθη", ["tov"]),
    "pra": ("Πόντοι+Ριμπ.+Ασίστ", ["pts", "reb", "ast"]),
    "pr": ("Πόντοι+Ριμπάουντ", ["pts", "reb"]),
    "pa": ("Πόντοι+Ασίστ", ["pts", "ast"]),
    "ra": ("Ριμπάουντ+Ασίστ", ["reb", "ast"]),
    "stl_blk": ("Κλεψίματα+Κοψίματα", ["stl", "blk"]),
}
KEYS = list(STATS)
# η διόρθωση ρόλου αφορά τη «χρήση» (σουτ, μπάλες), όχι τα ριμπάουντ/κλεψίματα/κοψίματα
ROLE_W = np.array([{"pts": 1, "ast": 1, "fg3m": 1, "fg2m": 1, "tov": 1, "pra": 0.75, "pr": 0.7, "pa": 1}.get(k, 0.0)
                   for k in STATS])


@dataclass(frozen=True)
class Params:
    hl_rate: float = 40.0     # ημιζωή (σε αγώνες) για τον ρυθμό ανά λεπτό
    hl_min: float = 4.0       # ημιζωή για τα λεπτά
    kappa_scale: float = 0.6  # πολλαπλασιαστής στο empirical-Bayes κ
    kappa_min: float = 12.0   # όρια για το κ (λεπτά «ψευδο-δείγματος» προς τον μέσο όρο λίγκας)
    kappa_max: float = 3000.0
    hl_opp: float = 15.0      # ημιζωή για τον αντίπαλο
    n0_opp: float = 6.0       # shrinkage του f_opp προς το 1
    season_decay: float = 0.3  # επιπλέον απόσβεση στην αλλαγή σεζόν (άμυνες αντιπάλων)
    rate_decay: float = 0.3    # ... για τον ρυθμό ανά λεπτό του παίκτη
    min_decay: float = 0.3     # ... για τα λεπτά του παίκτη
    move_decay: float = 1.0    # επιπλέον πολλαπλασιαστής όταν ο παίκτης άλλαξε ομάδα (1 = καμία διαφορά)
    role_beta: float = 0.5     # διόρθωση ρόλου: ρυθμός × (λεπτά φέτος / λεπτά πέρσι)^β
    role_n0: float = 3.0       # ... με βάρος n/(n+n0), n = φετινοί αγώνες μέχρι τότε
    role_clip: float = 1.2     # ... το πολύ ×1.2 ή ÷1.2 (το όριο σχεδόν δεν αλλάζει το σφάλμα· το συντηρητικό)
    level_n0: float = 400.0    # μετατόπιση λεπτών της τρέχουσας σεζόν (π.χ. φέτος οι ομάδες μοιράζουν τα λεπτά σε
                               # περισσότερους): μέσο σφάλμα λεπτών φέτος × n/(n+n0). 0 = χωρίς.
    min_prior: int = 5        # ελάχιστοι προηγούμενοι αγώνες για να βγει πρόβλεψη
    home: bool = True         # συντελεστής έδρας ανά αγορά
    min_recal: bool = True    # γραμμική διόρθωση των λεπτών


def load_frame(con) -> pd.DataFrame:
    """Ένα row ανά παίκτη-αγώνα, με αντίπαλο, έδρα και όλες τις αγορές ως στήλες."""
    df = pd.read_sql_query(
        """SELECT b.*, g.date, g.round, g.home_code, g.away_code
           FROM box b JOIN games g USING(season, game_code)""", con)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["is_home"] = df["team"] == df["home_code"]
    df["opp"] = np.where(df["is_home"], df["away_code"], df["home_code"])
    for k, (_, cols) in STATS.items():
        if len(cols) > 1:
            df[k] = df[cols].sum(axis=1)
    return df.sort_values(["date", "game_code", "player_id"]).reset_index(drop=True)


def _opp_factors(df: pd.DataFrame, P: Params):
    """f_opp πριν από κάθε αγώνα (χωρίς διαρροή) και η τρέχουσα τιμή ανά ομάδα."""
    tg = (df.groupby(["season", "game_code", "date", "opp"], sort=False)[KEYS].sum()
            .reset_index().rename(columns={"opp": "def_team"}).sort_values("date"))
    league = tg[KEYS].mean().to_numpy(float)
    league = np.where(league > 0, league, 1.0)
    lam = 0.5 ** (1 / P.hl_opp)
    prior, current = {}, {}
    for team, g in tg.groupby("def_team", sort=False):
        S, W, last_season = np.zeros(len(KEYS)), 0.0, None
        for row in g.itertuples(index=False):
            if last_season is not None and row.season != last_season:
                S, W = S * P.season_decay, W * P.season_decay
            prior[(row.season, row.game_code, team)] = (S / league + P.n0_opp) / (W + P.n0_opp)
            x = np.array([getattr(row, k) for k in KEYS], float)
            S, W, last_season = lam * S + x, lam * W + 1.0, row.season
        current[team] = (S / league + P.n0_opp) / (W + P.n0_opp)
    return prior, current


def shrinkage(df: pd.DataFrame, r0: np.ndarray, P: Params) -> np.ndarray:
    """κ ανά αγορά. Prior Gamma στους ρυθμούς: Var_between = Σ[(X-r0·M)² - X] / ΣM² (ροπές),
    κ = r0 / Var_between. Σπάνια γεγονότα (κοψίματα) παίρνουν πολύ ισχυρότερο shrinkage."""
    tot = df.groupby("player_id")[KEYS + ["minutes"]].sum()
    Mi = tot["minutes"].to_numpy(float)[:, None]
    X = tot[KEYS].to_numpy(float)
    var = np.sum((X - r0 * Mi) ** 2 - X, axis=0) / max(np.sum(Mi ** 2), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(var > 0, r0 / var, P.kappa_max)
    return np.clip(k * P.kappa_scale, P.kappa_min, P.kappa_max)


def load_positions(con) -> dict:
    """Επίσημη θέση κάθε παίκτη (η πιο πρόσφατη σεζόν): 0 γκαρντ, 1 φόργουορντ, 2 σέντερ."""
    try:
        rows = con.execute("SELECT player_id, position FROM players WHERE position IS NOT NULL ORDER BY season").fetchall()
    except Exception:  # noqa: BLE001
        return {}
    code = {"Guard": 0, "Forward": 1, "Center": 2}
    return {pid: code[pos] for pid, pos in rows if pos in code}


def load_only_current(con) -> set:
    """Παίκτες για τους οποίους ο χρήστης ζήτησε «μέτρα μόνο τη φετινή σεζόν»."""
    try:
        return {r[0] for r in con.execute("SELECT player_id FROM only_current")}
    except Exception:  # noqa: BLE001
        return set()


def role_note(pid: str, ctx: dict, current: pd.DataFrame | None = None) -> str | None:
    """Σύντομη περιγραφή αν ο παίκτης έχει «νέο ρόλο» φέτος (άλλη ομάδα ή ±6′ από πέρσι)."""
    r = ctx.get("role", {}).get(pid)
    if not r:
        return None
    if r.get("only_current"):
        return "μετράει μόνο η φετινή σεζόν (δική σου επιλογή)"
    parts = []
    team = None if current is None or pid not in current.index else current.at[pid, "team"]
    if r.get("prev_team") and team and r["prev_team"] != team:
        parts.append(f"νέα ομάδα (πέρσι {r['prev_team']})")
    pm, cm = r.get("prev_min"), r.get("cur_min")
    if pm and cm is not None and r.get("cur_n") and abs(cm - pm) >= 6:
        parts.append(f"{cm:.0f}′ φέτος έναντι {pm:.0f}′ πέρσι")
        f = r.get("f_pts", 1.0)
        if abs(f - 1) >= 0.03:
            parts.append(f"ρυθμός σκοραρίσματος ×{f:.2f}")
    return "νέος ρόλος: " + ", ".join(parts) if parts else None


def walk_forward(df: pd.DataFrame, P: Params = Params(), positions: dict | None = None,
                 only_current: set | None = None):
    """Επιστρέφει (preds, current, ctx).

    preds  : ένα row ανά παίκτη-αγώνα με mu_<stat> υπολογισμένο ΜΟΝΟ από προηγούμενους αγώνες
    current: κατάσταση κάθε παίκτη μετά τον τελευταίο του αγώνα (base_<stat> = r̂·m̂, ουδέτερη έδρα)
    ctx    : {"opp": f_opp ανά ομάδα, "home": (f_εντός, f_εκτός), "min_cal": (a, b), "r0": ...,
              "role": {παίκτης: αλλαγή ρόλου}}
    only_current: παίκτες για τους οποίους η προηγούμενη σεζόν αγνοείται εντελώς (επιλογή του χρήστη)

    Διόρθωση ρόλου: όποιος παίζει φέτος πολύ περισσότερα λεπτά από πέρσι, συνήθως παίρνει και περισσότερα
    σουτ/μπάλες ανά λεπτό (και το αντίθετο). Τα λεπτά τα πιάνει γρήγορα ο εκθετικός μέσος· ο ρυθμός ανά λεπτό
    όμως έχει μεγάλη μνήμη. Γι' αυτό ο ρυθμός πολλαπλασιάζεται με (m_φέτος/m_πέρσι)^β, σταθμισμένο με το πόσους
    φετινούς αγώνες έχουμε."""
    only_current = only_current or set()
    prior_f, opp_now = _opp_factors(df, P)
    tot_min = df["minutes"].sum()
    r0 = df[KEYS].sum().to_numpy(float) / max(tot_min, 1.0)
    kappa = shrinkage(df, r0, P)
    nk = len(KEYS)
    hf, af = np.ones(nk), np.ones(nk)
    if P.home and df["is_home"].any() and (~df["is_home"]).any():
        h, a = df[df["is_home"]], df[~df["is_home"]]
        safe = np.where(r0 > 0, r0, 1.0)
        hf = np.where(r0 > 0, h[KEYS].sum().to_numpy(float) / h["minutes"].sum() / safe, 1.0)
        af = np.where(r0 > 0, a[KEYS].sum().to_numpy(float) / a["minutes"].sum() / safe, 1.0)
    lr, lm = 0.5 ** (1 / P.hl_rate), 0.5 ** (1 / P.hl_min)
    pred_rows, cur_rows, roles_info = [], [], {}
    for pid, g in df.groupby("player_id", sort=False):
        X = g[KEYS].to_numpy(float)
        mins = g["minutes"].to_numpy(float)
        seasons, codes, opps = g["season"].to_numpy(), g["game_code"].to_numpy(), g["opp"].to_numpy()
        homes = g["is_home"].to_numpy()
        teams = g["team"].to_numpy()
        reset = pid in only_current
        Sx, Sm, Mm, Wm = np.zeros(nk), 0.0, 0.0, 0.0
        prev_min, cur_min, cur_n = None, 0.0, 0     # μέσα λεπτά προηγούμενης σεζόν · φετινά μέχρι τώρα
        season_min, season_n = 0.0, 0
        for t in range(len(g)):
            if t and (seasons[t] != seasons[t - 1] or teams[t] != teams[t - 1]):
                new_season = seasons[t] != seasons[t - 1]
                mv = P.move_decay if teams[t] != teams[t - 1] else 1.0
                rd = (P.rate_decay if new_season else 1.0) * mv
                md = (P.min_decay if new_season else 1.0) * mv
                if new_season and reset and seasons[t] == seasons[-1]:
                    rd = md = 0.0
                Sx, Sm = Sx * rd, Sm * rd
                Mm, Wm = Mm * md, Wm * md
                if new_season:
                    prev_min = season_min / season_n if season_n else None
                    season_min, season_n = 0.0, 0
            role = 1.0
            if P.role_beta and prev_min and prev_min >= 5 and season_n and not reset:
                ratio = max(season_min / season_n, 1.0) / prev_min
                w = season_n / (season_n + P.role_n0)
                role = np.clip(ratio ** (P.role_beta * w * ROLE_W), 1 / P.role_clip, P.role_clip)
            if Wm > 0 and (t >= P.min_prior or (reset and season_n >= 1 and seasons[t] == seasons[-1])):
                rate = (Sx + kappa * r0) / (Sm + kappa) * role
                f = prior_f.get((seasons[t], codes[t], opps[t]), np.ones(nk))
                pred_rows.append((g.index[t], t, Mm / Wm, *(rate * f * (hf if homes[t] else af))))
            Sx, Sm = lr * Sx + X[t], lr * Sm + mins[t]
            Mm, Wm = lm * Mm + mins[t], lm * Wm + 1.0
            season_min, season_n = season_min + mins[t], season_n + 1
        last = g.iloc[-1]
        role = 1.0
        if P.role_beta and prev_min and prev_min >= 5 and season_n and not reset:
            w = season_n / (season_n + P.role_n0)
            role = np.clip((max(season_min / season_n, 1.0) / prev_min) ** (P.role_beta * w * ROLE_W), 1 / P.role_clip, P.role_clip)
        roles_info[pid] = {"prev_team": teams[seasons != seasons[-1]][-1] if (seasons != seasons[-1]).any() else None,
                           "prev_min": prev_min if seasons[-1] != seasons[0] else None,
                           "cur_min": season_min / season_n if season_n else None, "cur_n": season_n,
                           "factor": float(np.max(role)) if np.ndim(role) else 1.0, "f_pts": float(np.ravel(role)[0]) if np.ndim(role) else 1.0,
                           "only_current": reset}
        cur_rows.append((pid, last["player"], last["team"], int(last["season"]), last["date"],
                         len(g), Mm / Wm, *((Sx + kappa * r0) / (Sm + kappa) * role)))
    mu_cols = [f"mu_{k}" for k in KEYS]
    preds = pd.DataFrame(pred_rows, columns=["idx", "n_prior", "min_raw"] + mu_cols).set_index("idx")
    current = pd.DataFrame(cur_rows, columns=["player_id", "player", "team", "season", "last_date",
                                               "n_games", "min_raw"] + [f"base_{k}" for k in KEYS]
                           ).set_index("player_id")
    preds = preds.join(df)
    # λεπτά: actual ≈ a + b·m̂ (παλινδρόμηση προς τον μέσο)
    a_, b_ = 0.0, 1.0
    if P.min_recal and len(preds) >= 200:
        b_, a_ = np.polyfit(preds["min_raw"].to_numpy(float), preds["minutes"].to_numpy(float), 1)
        b_ = float(np.clip(b_, 0.5, 1.0))
        a_ = float(preds["minutes"].mean() - b_ * preds["min_raw"].mean())
    preds["min_hat"] = (a_ + b_ * preds["min_raw"]).clip(lower=1.0)
    current["min_hat"] = (a_ + b_ * current["min_raw"]).clip(lower=1.0)
    cs = int(df["season"].max())
    lv = preds[preds["season"] == cs]
    shift = 0.0
    if P.level_n0 and len(lv):
        shift = len(lv) / (len(lv) + P.level_n0) * float(lv["minutes"].mean() - lv["min_hat"].mean())
        is_cs = current["season"] == cs
        current.loc[is_cs, "min_hat"] = (current.loc[is_cs, "min_hat"] + shift).clip(lower=1.0)
    for d, cols in ((preds, mu_cols), (current, [f"base_{k}" for k in KEYS])):
        d[cols] = d[cols].to_numpy(float) * d["min_hat"].to_numpy(float)[:, None]
    ctx = {"opp": opp_now, "home": (hf, af), "min_cal": (a_, b_), "r0": dict(zip(KEYS, r0)),
           "roles": {**player_roles(df), **(positions or {})}, "official": set(positions or {}),
           "team_avg": team_averages(df), "role": roles_info, "min_shift": shift}
    ctx["def_pos"] = defense_by_position(df, ctx["roles"])
    return preds, current, ctx


# ---------- απουσίες συμπαικτών ----------
# Εκτιμήθηκαν στη σεζόν 2025-26 από 351 αγώνες ομάδων με «φρέσκια» απουσία παίκτη του ροτέισον
# (σύγκριση πραγματικών λεπτών/στατιστικών των συμπαικτών με την walk-forward πρόβλεψη):
#   Δλεπτά = [A_ίδιου_ρόλου·(43.9 − 1.55·m̂) + A_άλλου_ρόλου·(25.4 − 0.93·m̂)] / 200,  ≥ 0
#   ρυθμός ανά λεπτό × (1 + 0.25·μερίδιο απόντων στο στατιστικό), για σκοράρισμα/δημιουργία
# όπου A = λεπτά των απόντων και m̂ τα προβλεπόμενα λεπτά του παίκτη. Οι αλλαγές ωφελούνται
# πολύ περισσότερο από τους βασικούς, και η απουσία στον ίδιο ρόλο μετράει ~1.75 φορές.
ABS_MIN = {"same": (43.9, -1.547), "diff": (25.4, -0.926)}
ABS_RATE = {"pts": 0.25, "ast": 0.25, "fg3m": 0.25, "fg2m": 0.25, "tov": 0.25, "blk": 0.25, "reb": 0.0, "stl": 0.0}
ROLE_NAMES = {0: "Γκαρντ", 1: "Φόργουορντ", 2: "Σέντερ"}
ROLE_PLURAL = {0: "γκαρντ", 1: "φόργουορντ", 2: "σέντερ"}
BASE_KEYS = [k for k in KEYS if len(STATS[k][1]) == 1]


def player_roles(df: pd.DataFrame) -> dict:
    """0=γκαρντ, 1=φόργουορντ, 2=ψηλός, από το προφίλ ανά λεπτό (ριμπάουντ, κοψίματα, ασίστ, τρίποντα)."""
    t = df.groupby("player_id")[["minutes", "reb", "ast", "fg3a", "blk", "oreb"]].sum()
    if len(t) < 10:
        return {pid: 1 for pid in t.index}
    m = t["minutes"].clip(lower=1)

    def z(s):
        sd = s.std()
        return (s - s.mean()) / sd if sd > 0 else s * 0.0
    big = z(t["reb"] / m) + z(t["oreb"] / m) + z(t["blk"] / m) - z(t["ast"] / m) - z(t["fg3a"] / m)
    q1, q2 = np.quantile(big, [0.4, 0.75])
    return dict(zip(t.index, np.where(big > q2, 2, np.where(big > q1, 1, 0)).tolist()))


def defense_by_position(df: pd.DataFrame, roles: dict, last: int = 20, n0: float = 8.0) -> dict:
    """Τι δέχεται κάθε άμυνα από κάθε θέση, σε σχέση με τον μέσο όρο της λίγκας (1.0 = μέσος όρος).

    Ανά αγώνα αθροίζεται η παραγωγή των αντιπάλων κάθε θέσης· μέσος όρος στους τελευταίους `last` αγώνες,
    με shrinkage n0 αγώνων προς το 1. Περιγραφικό: στον έλεγχο του μοντέλου η διάκριση ανά θέση δεν
    βελτίωσε τις προβλέψεις, οπότε δεν μπαίνει στο μ."""
    d = df.assign(pos=df["player_id"].map(roles).fillna(1).astype(int))
    g = d.groupby(["opp", "season", "game_code", "date", "pos"])[BASE_KEYS].sum().reset_index()
    full = g.pivot_table(index=["opp", "season", "game_code", "date"], columns="pos", values=BASE_KEYS, fill_value=0)
    league = full.mean()
    out = {}
    for team, t in full.groupby(level="opp"):
        t = t.sort_index(level="date").tail(last)
        n = len(t)
        res = {}
        for pos in (0, 1, 2):
            res[pos] = {}
            for k in BASE_KEYS:
                if (k, pos) not in t.columns or league[(k, pos)] <= 0:
                    res[pos][k] = 1.0
                    continue
                ratio = t[(k, pos)].mean() / league[(k, pos)]
                res[pos][k] = float((n * ratio + n0) / (n + n0))
        out[team] = res
    return out


def team_averages(df: pd.DataFrame, last: int = 10) -> dict:
    """Μέσος όρος ομάδας ανά αγώνα (τελευταίοι `last` αγώνες) για κάθε βασικό στατιστικό."""
    tg = df.groupby(["team", "season", "game_code", "date"])[BASE_KEYS].sum().reset_index().sort_values("date")
    return {t: g.tail(last)[BASE_KEYS].mean().to_dict() for t, g in tg.groupby("team")}


def absence_effect(pid: str, current: pd.DataFrame, ctx: dict, absent: set) -> dict:
    """Πώς αλλάζει η πρόβλεψη του παίκτη όταν λείπουν οι `absent` (ids) από την ομάδα του.

    Επιστρέφει {"min": νέα λεπτά, "mult": {stat: πολλαπλασιαστής του μ}, "out": [ονόματα]}."""
    team = current.at[pid, "team"]
    m = float(current.at[pid, "min_hat"])
    mates = [j for j in absent if j != pid and j in current.index and current.at[j, "team"] == team]
    if not mates:
        return {"min": m, "mult": {k: 1.0 for k in KEYS}, "out": []}
    roles, tavg = ctx.get("roles", {}), ctx.get("team_avg", {}).get(team, {})
    a_same = sum(float(current.at[j, "min_hat"]) for j in mates if roles.get(j, 1) == roles.get(pid, 1))
    a_diff = sum(float(current.at[j, "min_hat"]) for j in mates) - a_same
    d = (a_same * max(ABS_MIN["same"][0] + ABS_MIN["same"][1] * m, 0.0)
         + a_diff * max(ABS_MIN["diff"][0] + ABS_MIN["diff"][1] * m, 0.0)) / 200.0
    m_new = min(m + d, max(m, 36.0))
    base_mult = {}
    for k in BASE_KEYS:
        share = sum(float(current.at[j, f"base_{k}"]) for j in mates) / max(float(tavg.get(k, 0.0)), 1e-9) if tavg else 0.0
        base_mult[k] = (m_new / m) * (1.0 + ABS_RATE.get(k, 0.0) * min(share, 0.6))
    mult = dict(base_mult)
    for k in KEYS:
        if k not in mult:          # συνδυαστικά: σταθμισμένος μέσος των συνιστωσών
            comp = STATS[k][1]
            w = np.array([max(float(current.at[pid, f"base_{c}"]), 1e-9) for c in comp])
            mult[k] = float(np.sum(w * np.array([base_mult[c] for c in comp])) / np.sum(w))
    return {"min": m_new, "mult": mult, "out": [current.at[j, "player"] for j in mates]}


def project(pid: str, stat: str, current: pd.DataFrame, ctx: dict, opp: str | None = None,
            is_home: bool | None = None) -> float:
    """Μέση πρόβλεψη για τον επόμενο αγώνα του παίκτη."""
    i = KEYS.index(stat)
    f = ctx["opp"][opp][i] if opp in ctx["opp"] else 1.0
    h = 1.0 if is_home is None else ctx["home"][0 if is_home else 1][i]
    return float(current.at[pid, f"base_{stat}"] * f * h)


def inv_k(coef, min_hat):
    """1/k = a + c/m̂²."""
    m = np.maximum(np.asarray(min_hat, float), 3.0)
    return np.maximum(coef[0] + coef[1] / m ** 2, 1e-5)


def k_of(coef, min_hat):
    return 1.0 / inv_k(coef, min_hat)


def fit_dispersion(preds: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """(a, c) ανά αγορά με μέγιστη πιθανοφάνεια της αρνητικής διωνυμικής."""
    out = {}
    core = preds[preds["min_hat"] >= 12.0]      # εκεί που υπάρχουν γραμμές στις εταιρείες
    preds = core if len(core) >= 500 else preds
    mh = preds["min_hat"].to_numpy(float)
    for key in KEYS:
        mu = np.maximum(preds[f"mu_{key}"].to_numpy(float), 1e-9)
        x = preds[key].to_numpy(float)

        def nll(th):
            k = 1.0 / inv_k(np.exp(th), mh)
            return -sps.nbinom.logpmf(x, k, k / (k + mu)).mean()

        if len(x) < 50:
            out[key] = (0.15, 40.0)
            continue
        r = minimize(nll, [-2.0, 3.5], method="Nelder-Mead",
                     options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 600})
        out[key] = tuple(float(v) for v in np.exp(r.x))
    return out


def _dist(mu, k):
    mu = np.maximum(np.asarray(mu, float), 1e-9)
    k = np.asarray(k, float)
    return sps.nbinom(k, k / (k + mu))


def probs(mu, k, line):
    """(P(over), P(push), P(under)). Over κερδίζει όταν X > line."""
    line = np.asarray(line, float)
    mu = np.maximum(np.asarray(mu, float), 1e-9)
    k = np.asarray(k, float)
    pr = k / (k + mu)
    fl = np.floor(line)
    p_over = np.asarray(sps.nbinom.sf(fl, k, pr), float)        # P(X >= floor(line)+1)
    p_push = np.where(line == fl, sps.nbinom.pmf(fl, k, pr), 0.0)
    return p_over, p_push, 1.0 - p_over - p_push


def quantiles(mu: float, k: float, qs=(0.1, 0.5, 0.9)):
    return [float(_dist(mu, k).ppf(q)) for q in qs]


def pmf_table(mu: float, k: float) -> pd.DataFrame:
    d = _dist(mu, k)
    hi = int(max(d.ppf(0.999), 3))
    xs = np.arange(0, hi + 1)
    return pd.DataFrame({"x": xs, "p": d.pmf(xs)})


# ---------- αγορά / EV ----------

def devig(over: float | None, under: float | None, assumed_margin: float = 0.07) -> float | None:
    """Πιθανότητα Over που υπονοεί η αγορά, χωρίς γκανιότα."""
    io = 1 / over if over and over > 1 else None
    iu = 1 / under if under and under > 1 else None
    if io and iu:
        return io / (io + iu)
    if io:
        return min(io / (1 + assumed_margin), 0.999)
    if iu:
        return max(1 - iu / (1 + assumed_margin), 0.001)
    return None


def ev(p_win: float, p_lose: float, price: float) -> float:
    """Αναμενόμενο κέρδος ανά μονάδα πονταρίσματος (το push επιστρέφει το ποντάρισμα)."""
    return p_win * (price - 1) - p_lose


def kelly(p_win: float, p_lose: float, price: float) -> float:
    b = price - 1
    return max((p_win * b - p_lose) / b, 0.0) if b > 0 else 0.0


# ---------- εμπειρικά ποσοστά ----------

def hit_rate(values, line: float, side: str = "over", last: int | None = None):
    """(επιτυχίες, αγώνες) στους τελευταίους `last` αγώνες. Τα push δεν μετράνε."""
    v = np.asarray(values, float)
    if last:
        v = v[-last:]
    v = v[v != line]
    hits = int(np.sum(v > line) if side == "over" else np.sum(v < line))
    return hits, int(len(v))


def beta_interval(hits: int, n: int, level: float = 0.8):
    """Διάστημα αξιοπιστίας για το πραγματικό ποσοστό (Jeffreys prior)."""
    a, b = hits + 0.5, n - hits + 0.5
    lo, hi = sps.beta.ppf([(1 - level) / 2, 1 - (1 - level) / 2], a, b)
    return float(lo), float(hi)


# ---------- backtest ----------

def backtest(preds: pd.DataFrame, disp: dict[str, float], window: int = 10) -> pd.DataFrame:
    """Για κάθε παίκτη-αγώνα και 3 συνθετικές γραμμές γύρω από το μ:
    πιθανότητα μοντέλου, ποσοστό επιτυχίας τελευταίων `window` αγώνων, αποτέλεσμα."""
    out = []
    preds = preds.sort_values(["player_id", "date"])
    for k in KEYS:
        for _, g in preds.groupby("player_id", sort=False):
            if len(g) == 0:
                continue
            x = g[k].to_numpy(float)
            mu = g[f"mu_{k}"].to_numpy(float)
            kk = k_of(disp[k], g["min_hat"].to_numpy(float))
            for c in (0.6, 1.0, 1.4):
                line = np.floor(mu * c) + 0.5
                p, _, _ = probs(mu, kk, line)
                emp = np.full(len(x), np.nan)
                for t in range(window, len(x)):
                    emp[t] = np.mean(x[t - window:t] > line[t])
                out.append(pd.DataFrame({"stat": k, "p_model": p, "p_emp": emp,
                                         "hit": (x > line).astype(float),
                                         "min_hat": g["min_hat"].to_numpy(float)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def score_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    """Brier score ανά αγορά (μικρότερο = καλύτερο) σε κοινό δείγμα."""
    d = bt.dropna(subset=["p_emp"])
    rows = []
    for k, g in list(d.groupby("stat")) + [("ΣΥΝΟΛΟ", d)]:
        base = g["hit"].mean()
        rows.append({"stat": k, "n": len(g),
                     "brier_model": np.mean((g.p_model - g.hit) ** 2),
                     "brier_hitrate": np.mean((g.p_emp - g.hit) ** 2),
                     "brier_const": np.mean((base - g.hit) ** 2)})
    return pd.DataFrame(rows)


def calibration(bt: pd.DataFrame, col: str = "p_model", bins: int = 10) -> pd.DataFrame:
    d = bt.dropna(subset=[col]).copy()
    d["bin"] = np.clip((d[col] * bins).astype(int), 0, bins - 1)
    g = d.groupby("bin").agg(pred=(col, "mean"), obs=("hit", "mean"), n=("hit", "size")).reset_index()
    g["se"] = np.sqrt(g.obs * (1 - g.obs) / g.n.clip(lower=1))
    return g
