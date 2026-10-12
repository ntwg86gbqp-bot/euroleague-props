"""Ανάγνωση αποδόσεων από screenshot της Stoiximan. Δύο διατάξεις:
  σκάλα  (υπολογιστής): παίκτης αριστερά, κελιά «10+ / 1.47» δεξιά  -> Over (Ν − 0.5) για κάθε κελί
  O/U    (κινητό):      παίκτης | Line | Over | Under                -> Over και Under στην ίδια γραμμή
Κάθε γραμμή που επιστρέφεται: {"name", "line", "side", "price", "ok", "why"} (+ "n" στη σκάλα).

Η αναγνώριση κειμένου γίνεται από το macOS (Vision) μέσω του μικρού προγράμματος `ocr`· εδώ γίνεται
η ερμηνεία των κουτιών κειμένου. Κάθε κουτί: {"text", "x", "y", "w", "h"} σε κλάσματα της εικόνας, αρχή πάνω αριστερά.
"""
from __future__ import annotations
import json
import math
import re
import subprocess
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
ODDS_RE = re.compile(r"^\d{1,3}[.,]\d{2}$")
MS_EXACT = re.compile(r"^(\d{1,2})\+$")
SMOOTH_TOL = 0.055      # μέγιστη «καμπύλωση» του log(απόδοσης) σε γνήσια σκάλα είναι ~0.03
MS_LOOSE = re.compile(r"^(\d{1,2})[^\d\s]{1,2}$")       # «11t», «12±», «8+.»

STAT_WORDS = [("pra", r"πόντοι.*ριμπ.*ασίστ|points.*rebounds.*assists"), ("pr", r"πόντοι.*ριμπ|points.*rebounds"),
              ("pa", r"πόντοι.*ασίστ|points.*assists"), ("ra", r"ριμπ.*ασίστ|rebounds.*assists"),
              ("fg3m", r"τρίποντ|τριποντ|three|3-?point"), ("fg2m", r"δίποντ|διποντ|two|2-?point"),
              ("reb", r"ριμπ|rebound"), ("ast", r"ασίστ|ασιστ|assist"), ("stl_blk", r"κλεψ.*κοψ|steals.*blocks"),
              ("stl", r"κλεψ|steal"), ("blk", r"κοψ|μπλοκ|block"), ("tov", r"λάθ|λαθ|turnover"), ("pts", r"πόντ|ποντ|points")]


class OcrUnavailable(RuntimeError):
    pass


def ocr_mac(image_path: str) -> list[dict]:
    """Τρέχει την αναγνώριση κειμένου του macOS. Προτιμά το μεταγλωττισμένο `ocr`, αλλιώς `swift ocr.swift`."""
    import os
    if os.environ.get("ELPROPS_FAKE_OCR"):          # για δοκιμές εκτός macOS
        return json.loads(Path(os.environ["ELPROPS_FAKE_OCR"]).read_text())
    exe, src = HERE / "ocr", HERE / "elprops" / "ocr.swift"
    cmds = ([[str(exe), image_path]] if exe.exists() else []) + [["/usr/bin/swift", str(src), image_path]]
    last = "δεν βρέθηκε πρόγραμμα αναγνώρισης"
    for cmd in cmds:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as e:
            last = str(e)
            continue
        if r.returncode == 0 and r.stdout.strip().startswith("["):
            return json.loads(r.stdout)
        last = (r.stderr or r.stdout).strip()[:300]
    raise OcrUnavailable(last)


def detect_stat(boxes: list[dict]) -> str | None:
    """Μαντεύει την αγορά από τον τίτλο πάνω αριστερά (π.χ. «Πόντοι»)."""
    top = sorted(boxes, key=lambda b: (b["y"], b["x"]))[:4]
    text = " ".join(b["text"] for b in top).lower()
    for key, pat in STAT_WORDS:
        if re.search(pat, text):
            return key
    return None


def _tokens(boxes):
    """Σπάει κάθε κουτί σε λέξεις και μοιράζει τη θέση x αναλογικά (το OCR ενώνει μερικές φορές γειτονικά κελιά)."""
    out = []
    for b in boxes:
        text = re.sub(r"(\d)\s+\+", r"\1+", str(b["text"]).strip())
        if not text:
            continue
        n = len(text)
        for m in re.finditer(r"\S+", text):
            xc = b["x"] + b["w"] * ((m.start() + m.end()) / 2) / n
            out.append({"t": m.group(), "xc": xc, "x0": b["x"] + b["w"] * m.start() / n, "yc": b["y"] + b["h"] / 2, "h": b["h"]})
    return out


def _cluster(vals, tol):
    """Ομαδοποίηση 1-Δ τιμών: επιστρέφει τα κέντρα των ομάδων, ταξινομημένα."""
    centers = []
    for v in sorted(vals):
        if centers and v - centers[-1][-1] <= tol:
            centers[-1].append(v)
        else:
            centers.append([v])
    return [float(np.mean(c)) for c in centers]


def _fit_ladder(cols, exact, loose):
    """Τα κατώφλια μιας σειράς είναι αριθμητική πρόοδος (βήμα 1, 2, ...). Βρίσκει (αρχή, βήμα) που συμφωνεί
    με τα περισσότερα αναγνωσμένα κελιά και συμπληρώνει όσα το OCR διάβασε λάθος. cols: δείκτες στηλών."""
    best, best_score = None, 0.0
    for step in (1, 2, 3, 4, 5):
        for c0, n0 in list(exact.items()) + list(loose.items()):
            a = n0 - step * c0
            score = sum(2.0 for c, n in exact.items() if a + step * c == n) + \
                sum(1.0 for c, n in loose.items() if a + step * c == n and c not in exact)
            if score > best_score + 1e-9 and a + step * min(cols) >= 0:
                best, best_score = (a, step), score
    if best is None or best_score < 3.0:          # χρειάζονται τουλάχιστον δύο κελιά που συμφωνούν
        return None
    return {c: best[0] + best[1] * c for c in cols}


LINE_RE = re.compile(r"^\d{1,2}([.,]5)?$")


def parse_ou(boxes: list[dict]) -> list[dict]:
    """Διάταξη «παίκτης | Line | Over | Under». Δύο γραμμές ανά παίκτη (over, under)."""
    toks = _tokens(boxes)
    odds = [t for t in toks if ODDS_RE.match(t["t"])]
    if len(odds) < 2:
        return []
    h = float(np.median([t["h"] for t in odds]))
    cols = _cluster([t["xc"] for t in odds], 2.0 * h)
    if len(cols) != 2:
        return []
    out = []
    for ry in _cluster([t["yc"] for t in odds], 0.6 * h):
        row = [t for t in odds if abs(t["yc"] - ry) <= 0.6 * h]
        left = min(t["x0"] for t in row)
        lines = [t for t in toks if LINE_RE.match(t["t"]) and abs(t["yc"] - ry) <= 0.7 * h and t["xc"] < left]
        if not lines:
            continue
        lt = max(lines, key=lambda t: t["xc"])
        line = float(lt["t"].replace(",", "."))
        name_t = [t for t in toks if t["xc"] < lt["x0"] and abs(t["yc"] - ry) <= 1.3 * h
                  and re.search(r"[A-Za-zΑ-Ωα-ωά-ώ]{2,}", t["t"])]
        name = " ".join(t["t"] for t in sorted(name_t, key=lambda t: (round(t["yc"] / (0.7 * h)), t["xc"])))
        name = re.sub(r"(?<=[a-zα-ωά-ώ])(?=[A-ZΑ-Ω])", " ", name).strip()
        got = {}
        for t in row:
            got["over" if abs(t["xc"] - cols[0]) <= abs(t["xc"] - cols[1]) else "under"] = float(t["t"].replace(",", "."))
        why = "" if name else "δεν διάβασα όνομα"
        if not why and len(got) == 2 and not 1.0 < 1 / got["over"] + 1 / got["under"] < 1.2:
            why = "οι δύο αποδόσεις δεν ταιριάζουν μεταξύ τους"
        if not why and len(got) < 2:
            why = "διάβασα μόνο τη μία απόδοση"
        for side, price in got.items():
            out.append({"name": name, "line": line, "side": side, "price": price, "ok": not why, "why": why})
    return out


def parse(boxes: list[dict]) -> list[dict]:
    """Διαλέγει διάταξη: O/U αν υπάρχουν επικεφαλίδες Over/Under ή αν οι αποδόσεις πέφτουν σε δύο στήλες
    με μια γραμμή (π.χ. 15.5) αριστερά τους· αλλιώς σκάλα."""
    words = {str(b["text"]).strip().lower() for b in boxes}
    has_ms = any(MS_EXACT.match(t["t"]) for t in _tokens(boxes))
    if ({"over", "under"} <= words or not has_ms):
        ou = parse_ou(boxes)
        if ou:
            return ou
    rows = parse_ladder(boxes)
    for r in rows:
        r["line"], r["side"] = r["n"] - 0.5, "over"
    return rows


def parse_ladder(boxes: list[dict]) -> list[dict]:
    """Σκάλα: γραμμές {"name", "n" (κατώφλι Ν+), "price", "ok", "why"}.

    ok=False σημαίνει ότι το κελί θέλει έλεγχο: το κατώφλι δεν επιβεβαιώθηκε ή η απόδοση δεν ταιριάζει με τις διπλανές."""
    toks = _tokens(boxes)
    odds = [t for t in toks if ODDS_RE.match(t["t"])]
    if len(odds) < 2:
        return []
    h = float(np.median([t["h"] for t in odds]))
    for t in odds:
        t["price"] = float(t["t"].replace(",", "."))
    row_y = _cluster([t["yc"] for t in odds], 0.6 * h)
    col_x = _cluster([t["xc"] for t in odds], 1.2 * h)
    left_edge = min(t["x0"] for t in odds)
    rest = [t for t in toks if not ODDS_RE.match(t["t"])]
    out = []
    for ry in row_y:
        cells = {}
        for t in odds:
            if abs(t["yc"] - ry) <= 0.6 * h:
                cells[int(np.argmin([abs(t["xc"] - cx) for cx in col_x]))] = t
        if not cells:
            continue
        exact, loose = {}, {}
        for t in rest:
            if not (ry - 2.4 * h <= t["yc"] <= ry - 0.4 * h) or t["x0"] < left_edge - 2 * h:
                continue
            c = int(np.argmin([abs(t["xc"] - cx) for cx in col_x]))
            if abs(t["xc"] - col_x[c]) > 2.0 * h:
                continue
            m = MS_EXACT.match(t["t"])
            if m:
                exact[c] = int(m.group(1))
            else:
                m = MS_LOOSE.match(t["t"].replace("O", "0").replace("o", "0"))
                if m:
                    loose[c] = int(m.group(1))
        cols = sorted(cells)
        ladder = _fit_ladder(list(range(min(cols), max(cols) + 1)), exact, loose)
        name_t = [t for t in rest if t["xc"] < left_edge - 0.5 * h and ry - 2.6 * h <= t["yc"] <= ry + 0.9 * h
                  and re.search(r"[A-Za-zΑ-Ωα-ωά-ώ]{2,}", t["t"])]
        name = " ".join(t["t"] for t in sorted(name_t, key=lambda t: (round(t["yc"] / (0.7 * h)), t["xc"])))
        name = re.sub(r"(?<=[a-zα-ωά-ώ])(?=[A-ZΑ-Ω])", " ", name)      # «SylvainFrancisco» -> «Sylvain Francisco»
        for c in cols:
            n = ladder[c] if ladder else exact.get(c)
            if n is None:
                continue
            why = "" if ladder is not None else "το όριο δεν επιβεβαιώθηκε"
            if not name.strip():
                why = "δεν διάβασα όνομα"
            out.append({"name": name.strip(), "n": int(n), "price": cells[c]["price"], "ok": not why, "why": why})
    return check(out)


def check(rows: list[dict]) -> list[dict]:
    """Σημαδεύει αποδόσεις που δεν ταιριάζουν με τις διπλανές τους στην ίδια σκάλα: πρέπει να ανεβαίνουν
    και ο λογάριθμός τους να αλλάζει ομαλά (ένα λάθος ψηφίο, π.χ. 2.52 αντί για 2.72, σπάει την ομαλότητα)."""
    by = {}
    for r in rows:
        if "n" in r:
            by.setdefault(r["name"], []).append(r)
    for rs in by.values():
        rs.sort(key=lambda r: r["n"])
        lp = [math.log(r["price"]) for r in rs]
        for i, r in enumerate(rs):
            bad = (i > 0 and lp[i] <= lp[i - 1]) or (i + 1 < len(rs) and lp[i + 1] <= lp[i])
            if 0 < i < len(rs) - 1 and rs[i + 1]["n"] - r["n"] == r["n"] - rs[i - 1]["n"]:
                bad = bad or abs(lp[i] - (lp[i - 1] + lp[i + 1]) / 2) > SMOOTH_TOL
            if bad and r["ok"]:
                r["ok"], r["why"] = False, "δεν ταιριάζει με τις διπλανές αποδόσεις"
    return rows


def merge(a: list[dict], b: list[dict]) -> list[dict]:
    """Ενώνει δύο αναγνώσεις της ίδιας εικόνας. Όπου διαφωνούν κρατά την πρώτη και τη σημαδεύει."""
    def key(r):
        return (re.sub(r"[^a-zα-ωά-ώ]", "", r["name"].lower()), r.get("n"), r.get("line"), r.get("side"))
    out = {key(r): dict(r) for r in a}
    for r in b:
        k = key(r)
        if k not in out:
            out[k] = dict(r)
        elif abs(out[k]["price"] - r["price"]) > 1e-9:
            out[k]["ok"], out[k]["why"] = False, f"διάβασα και {r['price']:.2f}"
    return check(sorted(out.values(), key=lambda r: (r["name"], r.get("line", 0), r.get("side", ""))))


def _prep(img):
    """Γκρι, ανεστραμμένο (σκούρα γράμματα σε ανοιχτό φόντο), με τεντωμένη αντίθεση· μεγέθυνση αν είναι μικρό."""
    from PIL import Image, ImageOps
    g = img.convert("L")
    if np.asarray(g.resize((64, 64))).mean() < 128:      # σκούρο θέμα -> σκούρα γράμματα σε ανοιχτό φόντο
        g = ImageOps.invert(g)
    g = ImageOps.autocontrast(g)
    s = min(max(3200 / max(g.width, 1), 1.0), 3.0)
    return g.resize((int(g.width * s), int(g.height * s)), Image.LANCZOS) if s > 1.05 else g


def _rapid(img, engine) -> list[dict]:
    arr = np.array(img.convert("RGB"))[:, :, ::-1]
    res, _ = engine(arr)
    w, h = img.size
    out = []
    for box, text, conf in res or []:
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        out.append({"text": text, "x": min(xs) / w, "y": min(ys) / h, "w": (max(xs) - min(xs)) / w,
                    "h": (max(ys) - min(ys)) / h, "conf": float(conf)})
    return out


_ENGINE = None


def read_image(path: str) -> list[dict]:
    """Εικόνα -> γραμμές αποδόσεων. Διαβάζει την εικόνα δύο φορές (ολόκληρη και σε δύο λωρίδες) και
    σημαδεύει όπου οι δύο αναγνώσεις διαφωνούν. Αν δεν υπάρχει το rapidocr, δοκιμάζει το Vision του macOS."""
    global _ENGINE
    try:
        from PIL import Image
        from rapidocr_onnxruntime import RapidOCR
    except Exception:  # noqa: BLE001
        return parse(ocr_mac(path))
    if _ENGINE is None:
        _ENGINE = RapidOCR()
    img = Image.open(path)
    whole = parse(_rapid(_prep(img), _ENGINE))
    bands = []
    if img.height >= 500:
        cut, ov = img.height // 2, max(int(img.height * 0.12), 60)
        for top, bot in ((0, cut + ov), (cut - ov, img.height)):
            bands = merge(bands, parse(_rapid(_prep(img.crop((0, top, img.width, bot))), _ENGINE))) if bands else \
                parse(_rapid(_prep(img.crop((0, top, img.width, bot))), _ENGINE))
    return merge(whole, bands) if bands else whole
