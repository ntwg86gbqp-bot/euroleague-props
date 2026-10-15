"""Euroleague Props — ειδικά στοιχήματα παικτών: ιστορικό, μοντέλο, αξία.

Εκκίνηση:  streamlit run app.py
"""
from __future__ import annotations
import io
import math
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import importlib
import sys

# Στο Streamlit Cloud, μετά από νέα έκδοση ξανατρέχει μόνο το app.py: τα modules του elprops μένουν στη μνήμη
# όπως ήταν. Αν άλλαξε κάποιο αρχείο τους από την τελευταία φόρτωση, τα ξαναφορτώνουμε (με σειρά εξαρτήσεων).
_EL_MODS = ["db", "model", "picks", "bets", "board", "odds", "euroleague", "cloud", "defense", "shot"]


def _reload_changed():
    stamps = getattr(sys, "_elprops_stamps", None)
    here = os.path.dirname(os.path.abspath(__file__))
    now = {m: os.path.getmtime(os.path.join(here, "elprops", m + ".py")) for m in _EL_MODS
           if os.path.exists(os.path.join(here, "elprops", m + ".py"))}
    if stamps is not None and stamps != now:
        for m in _EL_MODS:
            mod = sys.modules.get("elprops." + m)
            if mod is not None:
                importlib.reload(mod)
    sys._elprops_stamps = now


_reload_changed()
from elprops import bets as BT, board as B, cloud as CL, db, euroleague as E, model as M, odds as O, picks as PK, shot as SH

import html as _html
import re

# --- παλέτα (σκούρο θέμα· οι ίδιες τιμές υπάρχουν στο .streamlit/config.toml)
BG, SURF, RAISED, HAIR = "#10151F", "#182030", "#212B3F", "#2C3850"
TEXT, MUTED = "#E8ECF3", "#93A0B8"
BLUE, GREY, INK = "#3987e5", "#6f7a90", "#93A0B8"
GOOD, WARN, ORANGE = "#0ca30c", "#c98500", "#d95926"
V_COLOR = {"strong": "#3CCB7F", "marginal": "#F2C14E", "suspect": "#F2785C", "no": "#6f7a90"}
V_NAME = {"strong": "Αξίζει", "marginal": "Οριακό", "suspect": "Ύποπτο", "no": "Όχι"}
V_MEANS = {
    "strong": "Αναμενόμενο κέρδος τουλάχιστον 6% και το μοντέλο αρκετά πάνω από την αγορά.",
    "marginal": "Μικρό θετικό αναμενόμενο κέρδος (2–6%). Εύκολα το εξαφανίζει ένα μικρό λάθος του μοντέλου.",
    "suspect": "Φαίνεται κερδοφόρο, αλλά το μοντέλο διαφωνεί πολύ με την αγορά ή η γραμμή είναι πολύ ψηλή. Συνήθως του λείπει πληροφορία: απουσία, αλλαγή ρόλου.",
    "no": "Η απόδοση δεν είναι αρκετά μεγάλη για την πιθανότητα που δίνει το μοντέλο.",
}
STAT_LABEL = {k: v[0] for k, v in M.STATS.items()}

# --- υπόμνημα όρων: μία πρόταση ο καθένας, χρησιμοποιείται παντού
GLOSS = {
    "Γραμμή": "Το όριο που βάζει η εταιρεία, π.χ. 14.5 πόντοι.",
    "Over / Under": "Στοίχημα ότι ο παίκτης θα κάνει περισσότερα (Over) ή λιγότερα (Under) από τη γραμμή.",
    "Απόδοση": "Τι παίρνεις πίσω για κάθε 1 € αν κερδίσεις. Στο 2.00 διπλασιάζεις.",
    "Πρόβλεψη": "Πόσο περιμένει το μοντέλο κατά μέσο όρο στον επόμενο αγώνα.",
    "Αναμενόμενα λεπτά": "Πόσο υπολογίζει το μοντέλο ότι θα παίξει. Είναι η πιο αβέβαιη εκτίμηση: σφάλμα περίπου 4–5 λεπτά.",
    "Πιθανότητα": "Πόσο συχνά θα κέρδιζε το στοίχημα αν ο αγώνας παιζόταν πολλές φορές.",
    "Πιθανότητα μοντέλου": "Η πιθανότητα μόνο από το μοντέλο, χωρίς να ληφθεί υπόψη η αγορά.",
    "Πιθανότητα αγοράς": "Η πιθανότητα που υπονοούν οι αποδόσεις της εταιρείας, αφού αφαιρεθεί η γκανιότα.",
    "Νεκρό σημείο": "Η πιθανότητα που χρειάζεται για να μη χάνεις μακροπρόθεσμα: 1 διά την απόδοση.",
    "Αναμενόμενο κέρδος": "Τι κερδίζεις κατά μέσο όρο ανά 1 € αν έπαιζες το ίδιο στοίχημα πολλές φορές. +5% σημαίνει 5 λεπτά ανά ευρώ.",
    "Γκανιότα": "Το περιθώριο της εταιρείας: οι αποδόσεις είναι λίγο χαμηλότερες από τις δίκαιες.",
    "Δίκαιη απόδοση": "Η απόδοση στην οποία το στοίχημα δεν θα είχε ούτε κέρδος ούτε ζημιά: 1 διά την πιθανότητα.",
    "Φόρμα": "Αν θα κέρδιζε αυτό το στοίχημα στους 5 τελευταίους αγώνες. ✓ ναι, ✗ όχι, ο πιο πρόσφατος δεξιά.",
    "Τελευταίοι 10": "Τι έκανε σε κάθε έναν από τους 10 τελευταίους αγώνες, ο παλιότερος αριστερά.",
    "Τι δίνει ο αντίπαλος": "Πόσο περισσότερο ή λιγότερο από τον μέσο όρο της λίγκας δέχεται η άμυνα του αντιπάλου σε αυτό το στατιστικό.",
    "Ποντάρισμα": "Προτεινόμενο ποσοστό του κεφαλαίου σου (το ¼ του κριτηρίου Kelly). Μικρό επίτηδες.",
    "Ετυμηγορία": "Αξίζει, Οριακό, Ύποπτο ή Όχι. Συνδυάζει αναμενόμενο κέρδος και διαφορά από την αγορά.",
    "Απουσίες": "Παίκτες που δηλώνεις ότι δεν θα παίξουν. Οι προβλέψεις των συμπαικτών τους ανεβαίνουν ανάλογα.",
    "Απόδοση επένδυσης": "Κέρδος ή ζημιά ως ποσοστό όσων ποντάρισες. +5% σημαίνει 5 € κέρδος για κάθε 100 € που έπαιξες.",
    "Ακυρώθηκε": "Ο παίκτης δεν αγωνίστηκε, οπότε το στοίχημα δεν μετράει και το ποντάρισμα επιστρέφεται.",
    "Νέος ρόλος ⚠️": "Άλλαξε ομάδα ή παίζει φέτος τουλάχιστον 6′ διαφορετικά από πέρσι. Το μοντέλο διορθώνει τον ρυθμό του, "
                    "αλλά με λίγους φετινούς αγώνες η πρόβλεψη είναι πιο αβέβαιη.",
}

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
st.set_page_config(page_title="Euroleague Props", layout="wide",
                   page_icon=os.path.join(_ASSETS, "favicon.png") if os.path.exists(os.path.join(_ASSETS, "favicon.png")) else "🏀")


@st.cache_data
def _touch_icon_b64() -> str | None:
    import base64
    try:
        with open(os.path.join(_ASSETS, "apple-touch-icon.png"), "rb") as f:
            return base64.b64encode(f.read()).decode()
    except OSError:
        return None


def home_screen_icon():
    """Εικονίδιο και όνομα για «Προσθήκη στην οθόνη Αφετηρίας» (iPhone) και για Android.

    Το iPhone διαβάζει το <link rel="apple-touch-icon"> της ΕΞΩΤΕΡΙΚΗΣ σελίδας. Στο Streamlit Cloud η εφαρμογή
    τρέχει μέσα σε iframe της ίδιας διεύθυνσης, οπότε το βάζουμε και στο parent και στο top."""
    b64 = _touch_icon_b64()
    if not b64:
        return
    import inspect
    html = f"""<script>
    (function () {{
      const href = "data:image/png;base64,{b64}";
      const wins = [window];
      try {{ if (window.parent !== window) wins.push(window.parent); }} catch (e) {{}}
      try {{ if (!wins.includes(window.top)) wins.push(window.top); }} catch (e) {{}}
      for (const w of wins) {{
        try {{
          const d = w.document;
          if (d.querySelector('link[data-elprops]')) continue;
          d.querySelectorAll('link[rel="apple-touch-icon"], link[rel="apple-touch-icon-precomposed"]').forEach(e => e.remove());
          const l = d.createElement('link'); l.rel = 'apple-touch-icon'; l.sizes = '180x180'; l.href = href;
          l.setAttribute('data-elprops', '1'); d.head.appendChild(l);
          const ic = d.createElement('link'); ic.rel = 'icon'; ic.type = 'image/png'; ic.href = href; d.head.appendChild(ic);
          const meta = (n, c) => {{ let m = d.querySelector('meta[name="' + n + '"]');
            if (!m) {{ m = d.createElement('meta'); m.name = n; d.head.appendChild(m); }} m.content = c; }};
          meta('apple-mobile-web-app-title', 'EL Props');
          meta('application-name', 'EL Props');
          meta('apple-mobile-web-app-capable', 'yes');
          meta('apple-mobile-web-app-status-bar-style', 'black-translucent');
          meta('theme-color', '#10151F');
        }} catch (e) {{}}
      }}
    }})();
    </script>"""
    # νέα Streamlit: το script τρέχει μέσα στη σελίδα· παλιά (π.χ. στο Mac): μέσα σε iframe ύψους 0
    if "unsafe_allow_javascript" in inspect.signature(st.html).parameters:
        st.html(html, unsafe_allow_javascript=True)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=0)


st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Commissioner:wght@400;500;600;700&display=swap');
:root {{ --bg:{BG}; --surf:{SURF}; --raised:{RAISED}; --hair:{HAIR}; --text:{TEXT}; --muted:{MUTED}; }}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, [data-testid="stMarkdownContainer"] {{
  font-family: 'Commissioner', -apple-system, 'Helvetica Neue', sans-serif; }}
.stApp h1 {{ font-weight: 700; letter-spacing: -0.02em; font-size: 2.1rem; padding-bottom: .2rem; }}
.stApp h3 {{ font-weight: 600; letter-spacing: -0.01em; }}
.block-container {{ padding-top: 2.2rem; max-width: 1380px; }}
[data-testid="stSidebar"] {{ border-right: 1px solid var(--hair); }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding: .35rem .5rem; border-radius: 8px; width: 100%; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: var(--raised); }}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{ background: var(--raised); box-shadow: inset 3px 0 0 {BLUE}; }}
[data-testid="stExpander"] details {{ border: 1px solid var(--hair); border-radius: 10px; background: transparent; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--hair); border-radius: 10px; overflow: hidden; }}
.el-brand {{ font-weight: 700; font-size: 1.15rem; letter-spacing: -0.01em; margin: .2rem 0 1rem; }}
.el-brand span {{ color: var(--muted); font-weight: 500; }}
.el-sub {{ color: var(--muted); margin: -.4rem 0 1.1rem; max-width: 70ch; }}
.el-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 10px; margin: .2rem 0 1.2rem; }}
.el-tile {{ background: var(--surf); border: 1px solid var(--hair); border-radius: 10px; padding: 12px 14px 11px; }}
.el-tile .l {{ color: var(--muted); font-size: .8rem; font-weight: 500; }}
.el-tile .v {{ font-size: 1.7rem; font-weight: 600; line-height: 1.25; font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }}
.el-tile .n {{ color: var(--muted); font-size: .76rem; line-height: 1.35; margin-top: 4px; }}
.el-tile .v small {{ font-size: .85rem; font-weight: 500; margin-left: 6px; }}
.el-card .row .v small {{ font-size: .8rem; font-weight: 600; margin-left: 4px; white-space: nowrap; }}
.el-verdict {{ background: var(--surf); border: 1px solid var(--hair); border-left: 5px solid var(--vc); border-radius: 10px;
  padding: 14px 18px; margin: .3rem 0 1rem; }}
.el-verdict .t {{ font-size: 1.15rem; font-weight: 600; }}
.el-verdict .t b {{ color: var(--vc); font-weight: 700; }}
.el-verdict .s {{ margin-top: 4px; line-height: 1.5; max-width: 95ch; }}
.el-verdict .m {{ color: var(--muted); font-size: .82rem; margin-top: 6px; }}
.el-card {{ background: var(--surf); border: 1px solid var(--hair); border-top: 4px solid var(--vc); border-radius: 12px;
  padding: 16px 18px 14px; height: 100%; }}
.el-card .chip {{ color: var(--vc); font-weight: 700; font-size: .85rem; }}
.el-card .who {{ color: var(--muted); font-size: .86rem; margin-top: 2px; }}
.el-card .bet {{ font-size: 1.45rem; font-weight: 700; letter-spacing: -0.015em; margin: 8px 0 12px; line-height: 1.2; }}
.el-card .row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; border-top: 1px solid var(--hair); padding-top: 10px; }}
.el-card .row .l {{ color: var(--muted); font-size: .74rem; }}
.el-card .row .v {{ font-size: 1.2rem; font-weight: 600; font-variant-numeric: tabular-nums; }}
.el-card .why {{ color: var(--muted); font-size: .8rem; margin-top: 10px; line-height: 1.4; }}
.el-h {{ font-weight: 600; font-size: 1rem; margin: .6rem 0 .1rem; }}
.el-leg {{ color: var(--muted); font-size: .8rem; line-height: 1.5; margin: 0 0 .3rem; }}
.el-leg i {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin: 0 5px 0 0; vertical-align: -1px; }}
.el-leg i.ln {{ height: 0; width: 16px; border-top: 2px solid; border-radius: 0; vertical-align: 3px; background: none !important; }}
.el-leg i.dash {{ border-top-style: dashed; }}
.el-leg span {{ margin-right: 12px; white-space: nowrap; }}
.el-gloss dt {{ font-weight: 600; margin-top: .5rem; font-size: .88rem; }}
.el-bet {{ background: var(--surf); border: 1px solid var(--hair); border-left: 4px solid var(--vc); border-radius: 10px;
  padding: 10px 12px; margin-bottom: 8px; }}
.el-bet .top {{ display: flex; justify-content: space-between; gap: 8px; align-items: baseline; }}
.el-bet .who {{ font-weight: 600; }}
.el-bet .chip {{ color: var(--vc); font-weight: 700; font-size: .8rem; white-space: nowrap; }}
.el-bet .bet {{ font-size: 1.05rem; font-weight: 600; margin: 2px 0 6px; }}
.el-bet .nums {{ display: flex; gap: 14px; font-variant-numeric: tabular-nums; font-size: .9rem; }}
.el-bet .nums span b {{ font-weight: 600; }}
.el-bet .nums span i {{ color: var(--muted); font-style: normal; font-size: .74rem; display: block; }}
.el-bet .why {{ color: var(--muted); font-size: .76rem; margin-top: 6px; line-height: 1.35; }}
@media (max-width: 640px) {{
  .block-container {{ padding: 3.2rem .9rem 5rem; }}
  .stApp h1 {{ font-size: 1.55rem; }}
  .el-sub {{ font-size: .9rem; margin-bottom: .8rem; }}
  .el-tiles {{ grid-template-columns: repeat(2, 1fr); gap: 8px; }}
  .el-tiles.counts {{ grid-template-columns: repeat(4, 1fr); }}
  .el-tiles.counts .n {{ display: none; }}
  .el-tiles.counts .el-tile {{ padding: 8px 10px; }}
  .el-tile .v {{ font-size: 1.35rem; }}
  .el-tile .n {{ font-size: .72rem; }}
  .el-card .bet {{ font-size: 1.2rem; }}
  .el-verdict {{ padding: 12px 14px; }}
  .el-verdict .t {{ font-size: 1rem; }}
  [data-testid="stHorizontalBlock"] {{ gap: .5rem; }}
}}
[data-testid="stElementContainer"]:has(iframe[height="0"]) {{ display: none; }}
.el-gloss dd {{ color: var(--muted); margin: 0; font-size: .84rem; line-height: 1.4; }}
</style>""", unsafe_allow_html=True)


home_screen_icon()


def _auto_mobile() -> bool:
    try:
        ua = st.context.headers.get("User-Agent", "")
    except Exception:  # noqa: BLE001
        ua = ""
    return bool(re.search(r"iPhone|iPod|Android.*Mobile|Mobile Safari|Windows Phone", ua))


def is_mobile() -> bool:
    """Διάταξη κινητού: αυτόματα από τη συσκευή, αλλάζει και με το χέρι από τις Ρυθμίσεις."""
    return bool(st.session_state.get("mobile_view", _auto_mobile()))


def short(name: str) -> str:
    """«MILLER-MCINTYRE, CODI» -> «Miller-Mcintyre C.» για στενές οθόνες."""
    parts = [p.strip() for p in str(name).split(",")]
    return parts[0].title() + (f" {parts[1][:1].upper()}." if len(parts) > 1 and parts[1] else "")


def lan_url() -> str | None:
    """Η διεύθυνση της εφαρμογής για άλλες συσκευές στο ίδιο Wi-Fi."""
    import socket
    try:
        sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sk.connect(("10.255.255.255", 1))
        ip = sk.getsockname()[0]
        sk.close()
        return None if ip.startswith("127.") else f"http://{ip}:{st.get_option('server.port')}"
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=300, show_spinner=False)
def remote_url() -> str | None:
    """Διεύθυνση μέσω Tailscale (ιδιωτικό δίκτυο 100.64.0.0/10), για το κινητό εκτός σπιτιού."""
    import ipaddress
    import shutil
    import subprocess
    port = st.get_option("server.port")
    for exe in (shutil.which("tailscale"), "/Applications/Tailscale.app/Contents/MacOS/Tailscale"):
        if exe and os.path.exists(exe):
            try:
                out = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=3).stdout.split()
                if out:
                    return f"http://{out[0]}:{port}"
            except Exception:  # noqa: BLE001
                pass
    try:
        txt = subprocess.run(["/sbin/ifconfig"], capture_output=True, text=True, timeout=3).stdout
    except Exception:  # noqa: BLE001
        return None
    net = ipaddress.ip_network("100.64.0.0/10")
    for ip in re.findall(r"inet (\d+\.\d+\.\d+\.\d+)", txt):
        if ipaddress.ip_address(ip) in net:
            return f"http://{ip}:{port}"
    return None


SERVICE = os.environ.get("ELPROPS_SERVICE") == "1"     # τρέχει μόνιμα ως υπηρεσία του macOS


POS_COLORS = ["#3987e5", "#d95926", "#199e70"]     # γκαρντ, φόργουορντ, σέντερ (παλέτα validated για σκούρο φόντο)


def pos_of(D, pid) -> int:
    return int(D["ctx"]["roles"].get(pid, 1))


def pos_name(D, pid, mark: bool = True) -> str:
    """Θέση παίκτη. Με «~» όταν είναι εκτίμηση από τα στατιστικά και όχι η επίσημη του ρόστερ."""
    n = M.ROLE_NAMES[pos_of(D, pid)]
    return n if (not mark or pid in D["ctx"].get("official", set())) else f"~{n}"


def esc(x) -> str:
    return _html.escape(str(x))


def tiles(target, items, cls: str = ""):
    """items: (ετικέτα, τιμή, επεξήγηση[, χρώμα τιμής])."""
    h = ""
    for it in items:
        col = f' style="color:{it[3]}"' if len(it) > 3 and it[3] else ""
        h += (f'<div class="el-tile"><div class="l">{esc(it[0])}</div><div class="v"{col}>{it[1]}</div>'
              f'<div class="n">{esc(it[2])}</div></div>')
    target.markdown(f'<div class="el-tiles {cls}">{h}</div>', unsafe_allow_html=True)


def heading(target, title: str, legend_html: str = ""):
    target.markdown(f'<div class="el-h">{esc(title)}</div><div class="el-leg">{legend_html}</div>',
                    unsafe_allow_html=True)


def sw(color: str, text: str, kind: str = "") -> str:
    style = f"border-color:{color}" if kind else f"background:{color}"
    return f'<span><i class="{kind}" style="{style}"></i>{esc(text)}</span>'


def gloss_html(keys) -> str:
    return '<dl class="el-gloss">' + "".join(f"<dt>{esc(k)}</dt><dd>{esc(GLOSS[k])}</dd>" for k in keys) + "</dl>"


def _secret(key: str):
    try:
        return st.secrets.get(key)
    except Exception:  # noqa: BLE001  — χωρίς secrets.toml (π.χ. στο Mac)
        return None


CLOUD = bool(_secret("GITHUB_TOKEN") and _secret("DATA_REPO"))     # τρέχει στο cloud με μόνιμη αποθήκευση
SEED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed", "elprops.sqlite.gz")


def _schema(c):
    from elprops.picks import SCHEMA as PICKS
    c.executescript(db.SCHEMA)
    c.executescript(PICKS)


@st.cache_resource(show_spinner="Φόρτωση δεδομένων…")
def get_con():
    """(σύνδεση, συγχρονιστής cloud ή None, μήνυμα εκκίνησης ή None)."""
    import sqlite3
    path = os.environ.get("ELPROPS_DB") or db.DB_PATH
    db.connect(path).close()
    con = sqlite3.connect(path, check_same_thread=False)
    if not CLOUD:
        return con, None, None
    store = CL.Store(_secret("GITHUB_TOKEN"), _secret("DATA_REPO"), _secret("DATA_BRANCH") or "data")
    try:
        data = store.download()
    except Exception as e:  # noqa: BLE001
        # ΔΕΝ αποθηκεύουμε τίποτα: μια άδεια βάση θα έσβηνε τα δεδομένα που είναι στο GitHub
        return con, None, ("error", f"Δεν φόρτωσαν τα αποθηκευμένα δεδομένα: {e} Η εφαρμογή δουλεύει, αλλά "
                                    "ό,τι κάνεις τώρα ΔΕΝ θα αποθηκευτεί. Κάνε επανεκκίνηση αργότερα.")
    msg, push = None, False
    if data:
        CL.restore_into(con, data)
    elif os.path.exists(SEED):
        with open(SEED, "rb") as f:
            CL.restore_into(con, f.read())
        msg, push = ("info", "Πρώτη εκκίνηση στο cloud: φορτώθηκαν τα δεδομένα από το Mac."), True
    _schema(con)
    con.commit()
    sync = CL.Syncer(store, con, every=float(os.environ.get("ELPROPS_SYNC_EVERY", 180)))
    if push:
        sync.maybe_push(force=True)
    return con, sync, msg


con, SYNC, BOOT = get_con()
O.bind(con)                     # ρυθμίσεις (κλειδί OddsPapi, εταιρείες) στη βάση, ώστε να σώζονται και στο cloud
if SYNC:
    SYNC.maybe_push()


def data_sig():
    return con.execute("SELECT (SELECT COUNT(*) FROM box), (SELECT COALESCE(SUM(pts),0) FROM box), "
                       "(SELECT COUNT(*) FROM players WHERE position IS NOT NULL), "
                       "(SELECT COALESCE(GROUP_CONCAT(player_id), '') FROM only_current)").fetchone()


@st.cache_resource(show_spinner="Υπολογισμός μοντέλου…")
def compute(sig):
    df = M.load_frame(con)
    if df.empty:
        return None
    preds, current, ctx = M.walk_forward(df, positions=M.load_positions(con), only_current=M.load_only_current(con))
    ctx["height"] = {pid: h for pid, h in con.execute(
        "SELECT player_id, height FROM players WHERE height IS NOT NULL ORDER BY season")}
    disp = M.fit_dispersion(preds) if len(preds) else {k: (0.15, 40.0) for k in M.KEYS}
    return {"df": df, "preds": preds, "current": current, "ctx": ctx, "disp": disp}


@st.cache_resource(show_spinner="Backtest…")
def compute_bt(sig):
    D = compute(sig)
    return M.backtest(D["preds"], D["disp"])


def show_chart(target, chart):
    """Γράφημα σε όλο το πλάτος, με το σκούρο θέμα της εφαρμογής."""
    chart = (chart.configure(background="transparent", font="Commissioner")
             .configure_view(stroke=None)
             .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=HAIR, domainColor=HAIR, tickColor=HAIR,
                             labelFontSize=11, titleFontSize=12, titleFontWeight="normal", titlePadding=8)
             .configure_legend(labelColor=TEXT, titleColor=MUTED, labelFontSize=12, symbolSize=90))
    try:
        target.altair_chart(chart, width="stretch", theme=None)
    except TypeError:
        target.altair_chart(chart, use_container_width=True, theme=None)


def pct(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}%"


def max_odds() -> float:
    """Πάνω από αυτή την τυπική απόδοση τίποτα δεν βγαίνει «Αξίζει»/«Οριακό» (ρύθμιση του χρήστη)."""
    try:
        return float(O.load_config().get("max_odds", 3.0))
    except (TypeError, ValueError):
        return 3.0


def live_odds(D):
    """Οι τρέχουσες αποδόσεις, χωρίς όσες ανήκουν σε αγώνα που έχει ήδη αρχίσει."""
    return B.fresh_odds(O.latest(con), D["current"], con)


def board_for(D, nxt, w_model):
    """(αποδόσεις, πίνακας αξίας), υπολογισμένα μία φορά ανά αλλαγή δεδομένων/αποδόσεων/ρυθμίσεων."""
    odds = live_odds(D)
    books = O.my_books()
    mx = max_odds()
    key = (mx, tuple(books), data_sig(), con.execute("SELECT COALESCE(MAX(id), 0) FROM odds").fetchone()[0], round(float(w_model), 3),
           tuple(sorted(D["ctx"].get("absent", set()))), tuple(nxt["game_code"].astype(str)) if len(nxt) else (), len(odds))
    hit = st.session_state.get("_board")
    if hit and hit[0] == key:
        return hit[1], hit[2]
    b = B.build(odds, D["df"], D["current"], D["ctx"], D["disp"], nxt, w_model, books=books, max_odds=mx) \
        if not odds.empty else pd.DataFrame()
    st.session_state["_board"] = (key, odds, b)
    return odds, b


# ------------------------------------------------------------------ παίκτης

def season_compare(D, pid: str, key: str):
    """Φέτος δίπλα σε πέρσι, σήμανση «νέος ρόλος» και επιλογή «μέτρα μόνο τη φετινή σεζόν»."""
    h = D["df"][D["df"]["player_id"] == pid]
    seasons = sorted(h["season"].unique())
    note = M.role_note(pid, D["ctx"], D["current"])
    only = pid in M.load_only_current(con)
    cs = int(D["df"]["season"].max())
    if note:
        st.markdown(f'<div class="el-leg" style="color:{WARN}">⚠️ {esc(note[0].upper() + note[1:])}. '
                    'Το μοντέλο το λαμβάνει υπόψη, αλλά με λίγους φετινούς αγώνες η πρόβλεψη είναι πιο αβέβαιη.</div>',
                    unsafe_allow_html=True)
    if len(seasons) >= 2 or only:
        with st.expander("Φέτος έναντι πέρσι" + (" · νέος ρόλος" if note and not only else ""), expanded=bool(note)):
            rows = []
            for s_ in seasons[-2:]:
                x = h[h["season"] == s_]
                m = x["minutes"].sum()
                rows.append({"Σεζόν": f"{s_}-{str(s_ + 1)[-2:]}", "Ομάδα": x["team"].iloc[-1], "Αγώνες": len(x),
                             "Λεπτά": x["minutes"].mean(), "Πόντοι": x["pts"].mean(), "Ριμπάουντ": x["reb"].mean(),
                             "Ασίστ": x["ast"].mean(), "Τρίποντα": x["fg3m"].mean(),
                             "Πόντοι / 36′": 36 * x["pts"].sum() / m if m else np.nan})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.1f")
                                        for c in ("Λεπτά", "Πόντοι", "Ριμπάουντ", "Ασίστ", "Τρίποντα", "Πόντοι / 36′")}
                         | {"Πόντοι / 36′": st.column_config.NumberColumn(format="%.1f", help="Πόντοι ανά 36 λεπτά: δείχνει αν άλλαξε ο ρόλος του στην επίθεση, ανεξάρτητα από τα λεπτά.")})
            new = st.toggle("Μέτρα μόνο τη φετινή σεζόν", value=only, key=f"{key}_only_{pid}",
                            help="Για όταν ξέρεις ότι ο ρόλος του άλλαξε ουσιαστικά (π.χ. έγινε βασικός). Το μοντέλο αγνοεί εντελώς "
                                 "τα περσινά του. Προσοχή: με λίγους φετινούς αγώνες η πρόβλεψη γίνεται πιο θορυβώδης. "
                                 "Στον έλεγχο, το να αγνοείται το πέρσι για όλους χειροτέρευε τις προβλέψεις.")
            if new != only:
                if new:
                    con.execute("INSERT OR REPLACE INTO only_current VALUES (?, ?)",
                                (pid, pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")))
                else:
                    con.execute("DELETE FROM only_current WHERE player_id=?", (pid,))
                con.commit()
                st.session_state.pop("_board", None)
                st.rerun()
            if (h["season"] == cs).sum() < 3 and new:
                st.caption(f"Έχει μόνο {(h['season'] == cs).sum()} φετινούς αγώνες: η πρόβλεψη στηρίζεται κυρίως στον μέσο όρο της λίγκας.")


def render_player(D, nxt, pid: str, stat: str, line: float, over=None, under=None, key="p",
                  side: str = "over", w_model: float = 0.5):
    df, current, disp = D["df"], D["current"], D["disp"]
    h = df[df["player_id"] == pid].sort_values("date")
    team = current.at[pid, "team"]
    mu, opp, is_home = B.projection(pid, stat, current, D["ctx"], nxt)
    kk = B.k_for(pid, stat, current, disp, D["ctx"])
    eff = M.absence_effect(pid, current, D["ctx"], D["ctx"].get("absent", set()))
    po, pp, pu = (float(x) for x in M.probs(mu, kk, line))
    q10, q50, q90 = M.quantiles(mu, kk)
    vals = h[stat].to_numpy(float)
    side_lab = "Over" if side == "over" else "Under"
    p_side = po if side == "over" else pu
    min_hat = float(eff["min"])
    f_opp = float(D["ctx"]["opp"][opp][M.KEYS.index(stat)] - 1) if opp in D["ctx"]["opp"] else 0.0

    st.subheader(current.at[pid, "player"].title())
    h_ = D["ctx"].get("height", {}).get(pid)
    h_txt = "" if not h_ else (f" · {h_ / 100:.2f} μ" if h_ > 3 else f" · {h_:.2f} μ")
    st.markdown(f'<div class="el-sub">{esc(team)} · {esc(pos_name(D, pid))}{h_txt}'
                + (f", επόμενος αγώνας {'εντός' if is_home else 'εκτός'} έδρας με {esc(opp)}" if opp else "")
                + "</div>", unsafe_allow_html=True)

    if pid in D["ctx"].get("absent", set()):
        st.warning("Τον έχεις δηλώσει απόντα. Αν τελικά παίζει, ξετσέκαρέ τον στη σελίδα «Αγώνας».")
    season_compare(D, pid, key)
    if eff["out"]:
        st.markdown(
            f'<div class="el-leg">Λείπει: <b>{esc(", ".join(n.title() for n in eff["out"]))}</b>. Γι\' αυτό η πρόβλεψη '
            f'ανέβηκε: λεπτά {current.at[pid, "min_hat"]:.0f}′ → {eff["min"]:.0f}′, {esc(STAT_LABEL[stat])} ×{eff["mult"][stat]:.2f}.</div>',
            unsafe_allow_html=True)

    # --- ετυμηγορία σε μία πρόταση
    price = over if side == "over" else under
    p_mkt = M.devig(over, under) if (over or under) else None
    pk_side = None if p_mkt is None else (p_mkt if side == "over" else 1 - p_mkt)
    if price and price > 1:
        live = 1 - pp
        cond_m = (po / live) if live > 0 else 0.5
        cond = w_model * cond_m + (1 - w_model) * p_mkt if p_mkt is not None else cond_m
        p_win = live * (cond if side == "over" else 1 - cond)
        e = M.ev(p_win, live - p_win, price)
        vd = B.verdict(e, p_side, pk_side, int(current.at[pid, "n_games"]), B.is_tail(line, mu, side))
        st.markdown(
            f'<div class="el-verdict" style="--vc:{V_COLOR[vd]}">'
            f'<div class="t"><b>{V_NAME[vd]}</b> · {side_lab} {line:g} {esc(STAT_LABEL[stat])} στο {price:.2f}</div>'
            f'<div class="s">Το μοντέλο δίνει {pct(p_side)}'
            + (f', η αγορά {pct(pk_side)}' if pk_side is not None else '')
            + f'. Για να μη χάνεις μακροπρόθεσμα χρειάζεται πάνω από {pct(1 / price)}. '
            f'Αναμενόμενο κέρδος <b style="color:{V_COLOR[vd]}">{e:+.1%}</b> ανά ευρώ.</div>'
            f'<div class="m">{esc(V_MEANS[vd] if not (vd == "suspect" and B.is_tail(line, mu, side)) else "Η γραμμή είναι πολύ πάνω από την πρόβλεψη. Σε τέτοιες γραμμές το μοντέλο υπερεκτιμά το Over, και στις μεγάλες αποδόσεις αυτό φαίνεται σαν μεγάλο κέρδος.")}</div></div>', unsafe_allow_html=True)
        bc = st.columns([1, 1, 4])
        stake = bc[0].number_input("Ποσό (€)", min_value=0.5, value=float(O.load_config().get("stake", 5.0)), step=0.5,
                                   key=f"{key}_stake", help="Πόσα έπαιξες. Το θυμάται για την επόμενη φορά.")
        bc[1].markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
        if bc[1].button("Το έπαιξα", key=f"{key}_bet", help="Καταγράφει το στοίχημα στο Ημερολόγιο. Δεν ποντάρει πουθενά."):
            g_ = nxt.loc[team] if team in nxt.index else None
            BT.place(con, player_id=pid, player=current.at[pid, "player"], team=team, opp=opp or "—", stat=stat, line=line,
                     side=side, price=price, stake=stake, p_model=p_side, p_market=pk_side, p_final=p_win, ev=e, verdict=vd,
                     season=None if g_ is None else g_["season"], game_code=None if g_ is None else g_["game_code"],
                     game_date=None if g_ is None else g_["date"])
            O.save_config({**O.load_config(), "stake": stake})
            st.toast("Καταγράφηκε στο Ημερολόγιο.")
        bc[2].markdown('<div class="el-leg" style="margin-top:1.9rem">Η καταγραφή είναι μόνο για το δικό σου αρχείο. '
                       'Το αποτέλεσμα θα συμπληρωθεί μόνο του μετά τον αγώνα.</div>', unsafe_allow_html=True)
    else:
        e, vd = None, None
        st.markdown(f'<div class="el-sub">{side_lab} {line:g} {esc(STAT_LABEL[stat])}. '
                    'Γράψε απόδοση για να δεις αν αξίζει.</div>', unsafe_allow_html=True)

    h5, t5 = M.hit_rate(vals, line, side, 5)
    h10, t10 = M.hit_rate(vals, line, side, 10)
    lo10, hi10 = M.beta_interval(h10, t10) if t10 else (0, 1)
    items = [
        ("Πρόβλεψη μοντέλου", f"{mu:.1f}", f"Μέσος όρος που περιμένει. Στους 8 από τους 10 αγώνες: {q10:.0f} έως {q90:.0f}."),
        ("Αναμενόμενα λεπτά", f"{min_hat:.0f}′", "Η πιο αβέβαιη εκτίμηση: σφάλμα περίπου 4–5 λεπτά."),
        (f"Πιθανότητα {side_lab}", pct(p_side), "Μόνο από το μοντέλο, πριν ληφθεί υπόψη η αγορά."),
        (f"{side_lab} στους 5 τελευταίους", f"{h5}/{t5}", "Πόσες φορές θα κέρδιζε. Πολύ μικρό δείγμα, μόνο για εικόνα."),
        (f"{side_lab} στους 10 τελευταίους", f"{h10}/{t10}", f"Το πραγματικό ποσοστό είναι κάπου στο {pct(lo10)}–{pct(hi10)}."),
    ]
    if opp:
        good = (f_opp > 0) == (side == "over")
        tone = V_COLOR["strong"] if good and abs(f_opp) >= 0.02 else V_COLOR["suspect"] if abs(f_opp) >= 0.02 else None
        items.append((f"Τι δίνει η {opp}", f"{f_opp:+.0%}",
                      f"{STAT_LABEL[stat]} που δέχεται σε σχέση με τη λίγκα. "
                      + ("Ουδέτερο." if abs(f_opp) < 0.02 else f"{'Υπέρ' if good else 'Κατά'} του {side_lab}."), tone))
    tiles(st, items)

    left, right = st.columns([3, 2], gap="large")
    W, L, T = V_COLOR["strong"], GREY, WARN      # κέρδιζε / έχανε / ισοπαλία

    # --- 1. τελευταίοι αγώνες
    mob = is_mobile()
    n_max = min(15 if mob else 30, len(h))
    n_show = left.slider("Πόσους αγώνες να δείξω", 5, n_max, min(8 if mob else 12, n_max), key=f"{key}_n") if n_max > 5 else n_max
    g = h.tail(n_show).copy()
    g["label"] = g["date"].dt.strftime("%d/%m") + np.where(g["is_home"], " vs ", " @ ") + g["opp"]
    win = (g[stat] > line) if side == "over" else (g[stat] < line)
    g["Αποτέλεσμα"] = np.where(g[stat] == line, "Ισοπαλία", np.where(win, "Κέρδιζε", "Έχανε"))
    g["Λεπτά"] = g["minutes"].round(0)
    g["Ρόλος"] = np.where(g["starter"] == 1, "Βασικός", "Αλλαγή")
    order = list(g["label"])
    bw = max(8, min(38, (250 if mob else 560) // max(len(g), 1)))
    hit_scale = alt.Scale(domain=["Κέρδιζε", "Έχανε", "Ισοπαλία"], range=[W, L, T])
    base = alt.Chart(g).encode(x=alt.X("label:N", sort=order, title=None, axis=alt.Axis(labelAngle=-40)))
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=bw).encode(
        y=alt.Y(f"{stat}:Q", title=STAT_LABEL[stat]),
        color=alt.Color("Αποτέλεσμα:N", scale=hit_scale, legend=None),
        tooltip=[alt.Tooltip("label:N", title="Αγώνας"), alt.Tooltip(f"{stat}:Q", title=STAT_LABEL[stat]),
                 "Λεπτά", "Ρόλος", "Αποτέλεσμα"])
    labels = base.mark_text(dy=-8, color=TEXT, fontSize=12, fontWeight=600).encode(
        y=f"{stat}:Q", text=alt.Text(f"{stat}:Q", format=".0f"))
    r_line = alt.Chart(pd.DataFrame({"y": [line]})).mark_rule(strokeWidth=2, color=TEXT).encode(y="y:Q")
    r_mu = alt.Chart(pd.DataFrame({"y": [mu]})).mark_rule(strokeWidth=2, strokeDash=[5, 4], color=BLUE).encode(y="y:Q")
    hits_n, tot_n = M.hit_rate(g[stat].to_numpy(float), line, side)
    heading(left, f"Τι έκανε στους τελευταίους {len(g)} αγώνες",
            sw(W, f"{side_lab} {line:g} κέρδιζε ({hits_n} από {tot_n})") + sw(L, "έχανε")
            + sw(TEXT, f"γραμμή {line:g}", "ln") + sw(BLUE, f"πρόβλεψη μοντέλου {mu:.1f}", "ln dash")
            + "<br>Κάθε μπάρα είναι ένας αγώνας, ο πιο πρόσφατος δεξιά. «vs» εντός έδρας, «@» εκτός.")
    show_chart(left, alt.layer(bars, labels, r_line, r_mu).properties(height=320))

    # --- 2. λεπτά
    mins = base.mark_bar(size=bw, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        y=alt.Y("Λεπτά:Q", title="Λεπτά"),
        color=alt.Color("Ρόλος:N", scale=alt.Scale(domain=["Βασικός", "Αλλαγή"], range=[BLUE, GREY]), legend=None),
        tooltip=[alt.Tooltip("label:N", title="Αγώνας"), "Λεπτά", "Ρόλος"])
    mrule = alt.Chart(pd.DataFrame({"y": [min_hat]})).mark_rule(strokeDash=[5, 4], strokeWidth=2, color=TEXT).encode(y="y:Q")
    heading(left, "Χρόνος συμμετοχής",
            sw(BLUE, "ξεκίνησε βασικός") + sw(GREY, "ήρθε από τον πάγκο")
            + sw(TEXT, f"το μοντέλο περιμένει {min_hat:.0f}′", "ln dash")
            + "<br>Τα λεπτά καθορίζουν σχεδόν όλα τα υπόλοιπα. Αν έχουν αλλάξει πρόσφατα, η πρόβλεψη είναι πιο αβέβαιη.")
    show_chart(left, (mins + mrule).properties(height=220))

    # --- 3. πιθανότητα ανά γραμμή
    lo_l, hi_l = max(0, int(M.quantiles(mu, kk, (0.03,))[0])), int(M.quantiles(mu, kk, (0.97,))[0]) + 1
    ls = np.arange(lo_l, hi_l + 1) + 0.5
    pov, _, pun = M.probs(mu, kk, ls)
    ps = pov if side == "over" else pun
    curve = pd.DataFrame({"Γραμμή": ls, "p": ps, "Δίκαιη απόδοση": np.round(1 / np.clip(ps, 1e-6, 1), 2)})
    cl = alt.Chart(curve).mark_line(strokeWidth=2, color=BLUE, point=alt.OverlayMarkDef(size=45, color=BLUE)).encode(
        x=alt.X("Γραμμή:Q", title=f"Γραμμή ({STAT_LABEL[stat]})", scale=alt.Scale(zero=False)),
        y=alt.Y("p:Q", title=f"Πιθανότητα {side_lab}", axis=alt.Axis(format="%", minExtent=44), scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("Γραμμή:Q", format=".1f"), alt.Tooltip("p:Q", format=".0%", title=f"Πιθανότητα {side_lab}"),
                 alt.Tooltip("Δίκαιη απόδοση:Q", format=".2f")])
    layers = [cl, alt.Chart(pd.DataFrame({"x": [line]})).mark_rule(strokeWidth=1.5, color=MUTED).encode(x="x:Q")]
    leg = sw(BLUE, "πιθανότητα μοντέλου σε κάθε γραμμή", "ln") + sw(MUTED, f"η γραμμή {line:g}", "ln")
    tail = "Πέρνα το ποντίκι πάνω από μια κουκκίδα για να δεις τη δίκαιη απόδοση εκείνης της γραμμής."
    if price and price > 1:
        mk = pd.DataFrame({"x": [line], "p": [1 / price], "Τι": [f"Νεκρό σημείο {1 / price:.0%} για απόδοση {price:.2f}"]})
        layers.append(alt.Chart(mk).mark_point(size=190, filled=True, shape="diamond", color=V_COLOR["marginal"],
                                               stroke=BG, strokeWidth=2).encode(x="x:Q", y="p:Q", tooltip=["Τι"]))
        mm = pd.DataFrame({"x": [line], "p": [p_side], "Τι": [f"Μοντέλο {p_side:.0%}"]})
        layers.append(alt.Chart(mm).mark_point(size=170, filled=True, color=BLUE, stroke=TEXT, strokeWidth=2).encode(
            x="x:Q", y="p:Q", tooltip=["Τι"]))
        leg = (sw(BLUE, f"μοντέλο {p_side:.0%}") + sw(V_COLOR["marginal"], f"νεκρό σημείο {1 / price:.0%}")
               + sw(MUTED, f"η γραμμή {line:g}", "ln"))
        tail = ("Νεκρό σημείο: η πιθανότητα που χρειάζεται για να μη χάνεις στην απόδοση που σου δίνουν. "
                "Αξία υπάρχει όταν το μπλε είναι πάνω από το κίτρινο.")
    heading(right, f"Πιθανότητα {side_lab} σε κάθε γραμμή", leg + "<br>" + tail)
    show_chart(right, alt.layer(*layers).properties(height=250))

    # --- 4. κατανομή
    pm = M.pmf_table(mu, kk)
    w = (pm["x"] > line) if side == "over" else (pm["x"] < line)
    pm["Αποτέλεσμα"] = np.where(pm["x"] == line, "Ισοπαλία", np.where(w, "Κέρδιζε", "Έχανε"))
    dist = alt.Chart(pm).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("x:O", title=STAT_LABEL[stat], axis=alt.Axis(labelAngle=0, labelOverlap=True)),
        y=alt.Y("p:Q", title="Πιθανότητα", axis=alt.Axis(format="%", minExtent=44)),
        color=alt.Color("Αποτέλεσμα:N", scale=hit_scale, legend=None),
        tooltip=[alt.Tooltip("x:O", title=STAT_LABEL[stat]), alt.Tooltip("p:Q", format=".1%", title="Πιθανότητα")])
    heading(right, "Τι περιμένει το μοντέλο στον επόμενο αγώνα",
            sw(W, f"{side_lab} κερδίζει ({pct(p_side)})") + sw(L, "χάνει")
            + "<br>Κάθε μπάρα είναι η πιθανότητα να κάνει ακριβώς τόσα. Το άθροισμα των πράσινων είναι η πιθανότητα του στοιχήματος.")
    show_chart(right, dist.properties(height=200))

    # --- 5. αντίπαλος, γενικά και για τη θέση του παίκτη
    if opp in D["ctx"]["opp"]:
        pos = pos_of(D, pid)
        show_k = list(dict.fromkeys([k for k in [stat] if k in M.BASE_KEYS] + ["pts", "reb", "ast", "fg3m", "fg2m", "stl", "blk", "tov"]))[:6]
        dp = D["ctx"].get("def_pos", {}).get(opp, {}).get(pos, {})
        lab_all, lab_pos = "Σε όλους", f"Σε {M.ROLE_PLURAL[pos]}"
        od = pd.DataFrame([{"Αγορά": STAT_LABEL[k], "Σειρά": lab_all, "d": float(D["ctx"]["opp"][opp][M.KEYS.index(k)] - 1)} for k in show_k]
                          + [{"Αγορά": STAT_LABEL[k], "Σειρά": lab_pos, "d": float(dp.get(k, 1.0) - 1)} for k in show_k])
        lim = max(0.12, float(od["d"].abs().max()) * 1.15)
        oc = alt.Chart(od).mark_bar(cornerRadius=3, size=11).encode(
            y=alt.Y("Αγορά:N", sort=[STAT_LABEL[k] for k in show_k], title=None, axis=alt.Axis(labelLimit=160, labelOverlap=False)),
            yOffset=alt.YOffset("Σειρά:N", sort=[lab_all, lab_pos]),
            x=alt.X("d:Q", title=None, axis=alt.Axis(format="+%"), scale=alt.Scale(domain=[-lim, lim])),
            color=alt.Color("Σειρά:N", scale=alt.Scale(domain=[lab_all, lab_pos], range=[GREY, POS_COLORS[pos]]), legend=None),
            tooltip=["Αγορά", "Σειρά", alt.Tooltip("d:Q", format="+.1%", title="Έναντι λίγκας")])
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED).encode(x="x:Q")
        heading(right, f"Τι δέχεται η άμυνα της {opp}",
                sw(GREY, "από όλους τους αντιπάλους") + sw(POS_COLORS[pos], f"από {M.ROLE_PLURAL[pos]}")
                + "<br>Δεξιά του μηδενός σημαίνει ότι δίνει περισσότερα από τον μέσο όρο. Το «από όλους» το έχει ήδη λάβει "
                  "υπόψη το μοντέλο· το «ανά θέση» είναι για εικόνα, γιατί στους ελέγχους δεν βελτίωσε τις προβλέψεις.")
        show_chart(right, (oc + zero).properties(height=48 * len(show_k) + 30))

    cur_season = int(df["season"].max())
    e1, e2, e3 = st.columns(3)
    with e1.expander("Δίκαιες αποδόσεις ανά γραμμή"):
        ft = pd.DataFrame({"Γραμμή": ls, "Over %": pov * 100, "Δίκαιο Over": 1 / np.clip(pov, 1e-6, 1),
                           "Under %": pun * 100, "Δίκαιο Under": 1 / np.clip(pun, 1e-6, 1)})
        st.caption(GLOSS["Δίκαιη απόδοση"] + " Αν η εταιρεία δίνει περισσότερο από τη δίκαιη, το μοντέλο βλέπει αξία.")
        st.dataframe(ft, hide_index=True, width="stretch", column_config={
            "Γραμμή": st.column_config.NumberColumn(format="%.1f"),
            "Over %": st.column_config.NumberColumn(format="%.0f%%"), "Under %": st.column_config.NumberColumn(format="%.0f%%"),
            "Δίκαιο Over": st.column_config.NumberColumn(format="%.2f"), "Δίκαιο Under": st.column_config.NumberColumn(format="%.2f")})
    with e2.expander("Εντός, εκτός, ανά σεζόν"):
        def split(mask, name):
            v = h.loc[mask, stat].to_numpy(float)
            hi, n = M.hit_rate(v, line, side)
            return {"Δείγμα": name, "Αγώνες": len(v), "Μέσος όρος": round(float(v.mean()), 1) if len(v) else None,
                    side_lab: f"{hi}/{n}", "Λεπτά": round(float(h.loc[mask, 'minutes'].mean()), 1) if len(v) else None}
        sp = [split(h["season"] == cur_season, "Φέτος"), split(h["season"] < cur_season, "Πέρσι"),
              split(h["is_home"], "Εντός"), split(~h["is_home"], "Εκτός"),
              split(h["starter"] == 1, "Ως βασικός"), split(h["starter"] != 1, "Ως αλλαγή")]
        if opp:
            sp.append(split(h["opp"] == opp, f"Εναντίον {opp}"))
        st.dataframe(pd.DataFrame(sp), hide_index=True, width="stretch")
    with e3.expander("Όλοι οι αγώνες"):
        log = h.sort_values("date", ascending=False).assign(
            Ημ=lambda d: d["date"].dt.strftime("%d/%m/%y"), Λεπτά=lambda d: d["minutes"].round(0))
        st.dataframe(log[["Ημ", "opp", "is_home", "Λεπτά", "pts", "reb", "ast", "fg2m", "fg3m", "stl", "blk", "tov"]]
                     .rename(columns={"opp": "Αντίπ.", "is_home": "Εντός", "pts": "Π", "reb": "Ρ", "ast": "Α",
                                      "fg2m": "2Π", "fg3m": "3Π", "stl": "Κλ", "blk": "Κοψ", "tov": "Λ"}),
                     hide_index=True, width="stretch")


# ------------------------------------------------------------------ σελίδες

def page_value(D, nxt, w_model):
    st.title("Στοιχήματα με αξία")
    st.markdown('<div class="el-sub">Για κάθε απόδοση που πέρασες, το μοντέλο εκτιμά πόσο συχνά θα κέρδιζε το '
                'στοίχημα και το συγκρίνει με αυτό που πληρώνει η εταιρεία.</div>', unsafe_allow_html=True)
    odds, b = board_for(D, nxt, w_model)
    if odds.empty:
        if not O.latest(con).empty:
            st.info("Οι αποδόσεις που έχεις περάσει αφορούν αγώνες που έχουν ήδη αρχίσει. Πέρασε αποδόσεις για "
                    "την επόμενη αγωνιστική στη σελίδα «Αποδόσεις».")
            return
        st.info("Δεν έχεις περάσει αποδόσεις ακόμα. Πήγαινε στη σελίδα «Αποδόσεις», διάλεξε αγώνα και γράψε "
                "τις γραμμές που δίνει η εταιρεία σου.")
        return
    if b.empty:
        st.info("Καμία απόδοση δεν αντιστοιχεί σε παίκτη με ιστορικό.")
        return
    # --- κρυμμένα στοιχήματα (επιλογή του χρήστη, για τον επόμενο αγώνα)
    hid = B.hidden_keys(con)
    b = b.assign(_key=[B.bet_key(r, nxt) for r in b.to_dict("records")])
    disc = B.discarded_keys(con)
    disc_rows = b[b["_key"].isin(disc)]
    b = b[~b["_key"].isin(disc)].reset_index(drop=True)
    b["_fav"] = b["_key"].isin(PK.keys_with_tag(con))
    fav_section(b[b["_fav"]], nxt)
    b["_hidden"] = b["_key"].isin(hid)
    hidden_rows = b[b["_hidden"]]
    show_hidden = st.session_state.get("show_hidden", False)
    if not show_hidden:
        b = b[~b["_hidden"]].reset_index(drop=True)
    if b.empty:
        st.info("Όλα τα στοιχήματα είναι κρυμμένα.")
    n_by = b["verdict"].value_counts()
    tiles(st, [(V_NAME[k], str(int(n_by.get(k, 0))), V_MEANS[k], V_COLOR[k]) for k in ("strong", "marginal", "suspect", "no")], "counts")

    # --- κορυφαίες επιλογές σε κάρτες
    # πρώτα τα «Αξίζει», και μέσα σε αυτά όσα έχουν κέρδος και στην τυπική απόδοση (αυτή που βρίσκεις παντού)
    top = (b[b["verdict"].isin(["strong", "marginal"]) & ~b["_hidden"]]
           .assign(_s=lambda x: (x["verdict"] == "strong").astype(int))
           .sort_values(["_s", "EV τυπικής"], ascending=False).head(3))
    if len(top):
        heading(st, "Οι καλύτερες επιλογές τώρα", "Πρώτα τα «Αξίζει», με σειρά το κέρδος στην τυπική απόδοση. "
                "Η πιθανότητα είναι ο μέσος όρος μοντέλου και αγοράς, επίτηδες συντηρητικός.")
        for col, (_, r) in zip(st.columns(3), top.iterrows()):
            col.markdown(
                f'<div class="el-card" style="--vc:{V_COLOR[r["verdict"]]}">'
                f'<div class="chip">{V_NAME[r["verdict"]]}</div>'
                f'<div class="who">{esc(r["Παίκτης"].title())}, {esc(r["Ομάδα"])} με {esc(r["Αντίπαλος"])}</div>'
                f'<div class="bet">{esc(r["Στοίχημα"])}</div>'
                f'<div class="row"><div><div class="l">Απόδοση</div><div class="v">{r["Απόδοση"]:.2f}</div></div>'
                f'<div><div class="l">Τυπική απόδοση</div><div class="v">{r["Τυπική απόδοση"]:.2f}'
                f'<small style="color:{V_COLOR["strong"] if r["EV τυπικής"] > 0 else V_COLOR["suspect"]}"> {r["EV τυπικής"]:+.0%}</small></div></div>'
                f'<div><div class="l">Πιθανότητα</div><div class="v">{pct(r["P τελική"])}</div></div></div>'
                f'<div class="why">{esc(r["Γιατί"])}</div></div>', unsafe_allow_html=True)
        st.write("")
    else:
        st.info("Κανένα από τα στοιχήματα που πέρασες δεν έχει αξία σύμφωνα με το μοντέλο. Αυτό είναι το συνηθισμένο.")

    with st.expander("Φίλτρα"):
        f = st.columns([3, 1, 1])
        stats = f[0].multiselect("Αγορές", M.KEYS, format_func=STAT_LABEL.get, placeholder="Όλες")
        side = f[1].selectbox("Πλευρά", ["Όλες", "Over", "Under"])
        min_g = f[2].number_input("Ελάχιστοι αγώνες ιστορικού", value=8, step=1, min_value=0)
    t1, t2, t3, t4 = st.columns(4)
    t4.toggle(f"Δείξε τα κρυμμένα ({len(hidden_rows)})", key="show_hidden",
              help="Στοιχήματα που έκρυψες εσύ. Μένουν κρυμμένα μόνο για τον επόμενο αγώνα του παίκτη.")
    hide_no = t1.toggle("Κρύψε όσα δεν αξίζουν", True)
    detail = t2.toggle("Αναλυτικές στήλες", False)
    only_fav = st.toggle("Μόνο τα ⭐ Θα έπαιζα", False, key="only_fav") if b["_fav"].any() else False
    no_role = t3.toggle("Κρύψε παίκτες με νέο ρόλο", False,
                        help="Άλλαξαν ομάδα ή παίζουν τουλάχιστον 6′ διαφορετικά από πέρσι. Εκεί η πρόβλεψη είναι πιο αβέβαιη.")
    v = b[b["Αγώνες"] >= min_g]
    if hide_no:
        v = v[v["verdict"] != "no"]
    if only_fav:
        v = v[v["_fav"]]
    if no_role and "Νέος ρόλος" in v:
        v = v[~v["Νέος ρόλος"]]
    if stats:
        v = v[v["stat"].isin(stats)]
    if side != "Όλες":
        v = v[v["Πλευρά"] == side]
    v = v.assign(_game=[game_label(nxt.at[t, "date"], t, o) if t in nxt.index else "—"
                        for t, o in zip(v["Ομάδα"], v["Αντίπαλος"])])
    v = study_filters(v, D, "vf", price="Τυπική απόδοση", prob="P τελική", ev="EV", team="Ομάδα", opp="Αντίπαλος",
                      game="_game", pid="player_id", minutes="Λεπτά", player="Παίκτης")
    v = v.reset_index(drop=True)
    if v.empty:
        st.caption("Τίποτα να δείξω με αυτά τα φίλτρα.")
        hidden_list(hidden_rows)
        return
    if is_mobile():
        heading(st, "Όλα τα στοιχήματα", f"{len(v)} στοιχήματα, τα καλύτερα πρώτα. Διάλεξε ένα από τη λίστα στο τέλος για την ανάλυσή του.")
        cards = ""
        for _, r in v.head(40).iterrows():
            cards += (f'<div class="el-bet" style="--vc:{V_COLOR[r["verdict"]]}"><div class="top"><span class="who">{"⭐ " if r.get("_fav") else ""}{esc(short(r["Παίκτης"]))}{" ⚠️" if r.get("Νέος ρόλος") else ""} '
                      f'<span style="color:var(--muted);font-weight:400">με {esc(r["Αντίπαλος"])}</span></span>'
                      f'<span class="chip">{V_NAME[r["verdict"]]}</span></div><div class="bet">{esc(r["Στοίχημα"])}</div>'
                      f'<div class="nums"><span><i>Απόδοση</i><b>{r["Απόδοση"]:.2f}</b></span>'
                      f'<span><i>Αναμ. κέρδος</i><b style="color:{V_COLOR[r["verdict"]]}">{r["EV"]:+.1%}</b></span>'
                      f'<span><i>Πιθανότητα</i><b>{pct(r["P τελική"])}</b></span><span><i>Φόρμα</i><b>{esc(r["Φόρμα"])}</b></span></div>'
                      f'<div class="why">{esc(r["Γιατί"])}</div></div>')
        st.markdown(cards, unsafe_allow_html=True)
        if len(v) > 40:
            st.caption(f"Δείχνω τα 40 καλύτερα από {len(v)}. Στένεψε με τα φίλτρα για τα υπόλοιπα.")
        with st.expander("⭐ Θα έπαιζα / 🗑️ Πέταγμα (πολλά μαζί)"):
            many = st.multiselect("Διάλεξε στοιχήματα", list(range(len(v))), key="m_disc", placeholder="Κανένα",
                                  format_func=lambda i: ("⭐ " if v.at[i, "_fav"] else "") +
                                  f"{short(v.at[i, 'Παίκτης'])} · {v.at[i, 'Στοίχημα']} @{v.at[i, 'Απόδοση']:.2f}")
            sel_ = v.iloc[many] if many else v.iloc[[]]
            m1, m2 = st.columns(2)
            fav_many(m1, sel_, nxt, "fav_m")
            discard_many(m2, sel_, "disc_m")
        pick = st.selectbox("Ανάλυση παίκτη", [None] + list(range(len(v))), key="m_pick",
                            format_func=lambda i: "Διάλεξε στοίχημα…" if i is None else f"{short(v.at[i, 'Παίκτης'])} · {v.at[i, 'Στοίχημα']}")
        if pick is not None:
            r = v.iloc[pick]
            hide_button(st, r, nxt, "hide_m")
            o = odds[(odds.player_id == r.player_id) & (odds.stat == r.stat) & (odds.line == r["Γραμμή"])]
            best = o.groupby("side")["price"].max()
            render_player(D, nxt, r.player_id, r.stat, float(r["Γραμμή"]), best.get("over"), best.get("under"),
                          key="v", side=r["side"], w_model=w_model)
        hidden_list(hidden_rows)
        discard_list(disc_rows)
        return
    show = v.assign(**{"Αναμ. κέρδος": v["EV"] * 100, "Μοντέλο": v["P μοντέλου"] * 100,
                       "Αγορά ": v["P αγοράς"].astype(float) * 100, "Πιθανότητα": v["P τελική"] * 100,
                       "Ποντάρισμα": v["Kelly/4"] * 100,
                       "Παίκτης": [("⭐ " if f_ else "") + ("🙈 " if h_ else "") + n.title() + (" ⚠️" if r_ else "")
                                   for n, r_, h_, f_ in zip(v["Παίκτης"], v.get("Νέος ρόλος", [False] * len(v)), v["_hidden"], v["_fav"])]})
    cols = ["Ετυμηγορία", "Παίκτης", "Στοίχημα", "Αντίπαλος", "Απόδοση", "Εταιρεία", "Τυπική απόδοση", "Αναμ. κέρδος", "Πιθανότητα",
            "Τελευταίοι 10", "Φόρμα", "Γιατί"]
    if detail:
        cols = cols[:-1] + ["Μοντέλο", "Αγορά ", "Πρόβλεψη", "Λεπτά", "Σεζόν", "Ποντάρισμα", "Εταιρείες αγοράς", "Γιατί"]
    heading(st, "Όλα τα στοιχήματα", "Κλικ στο κουτάκι αριστερά από μια γραμμή για να ανοίξει η ανάλυση του παίκτη από κάτω. "
            "Τσέκαρε ένα ή πολλά για να τα βάλεις στα «⭐ Θα έπαιζα» ή να τα πετάξεις. "
            "Άφησε το ποντίκι πάνω στον τίτλο μιας στήλης για να δεις τι σημαίνει.")
    ev = st.dataframe(
        show[cols], hide_index=True, width="stretch", on_select="rerun", selection_mode="multi-row", key="board",
        column_config={
            "Ετυμηγορία": st.column_config.TextColumn(width="small", help=GLOSS["Ετυμηγορία"]),
            "Στοίχημα": st.column_config.TextColumn(help=GLOSS["Over / Under"]),
            "Αναμ. κέρδος": st.column_config.NumberColumn(format="%+.1f%%", help=GLOSS["Αναμενόμενο κέρδος"]),
            "Πιθανότητα": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100,
                                                        help=GLOSS["Πιθανότητα"] + f" Εδώ: {w_model:.0%} μοντέλο και {1 - w_model:.0%} αγορά."),
            "Μοντέλο": st.column_config.NumberColumn(format="%.0f%%", help=GLOSS["Πιθανότητα μοντέλου"]),
            "Αγορά ": st.column_config.NumberColumn(format="%.0f%%", help=GLOSS["Πιθανότητα αγοράς"]),
            "Απόδοση": st.column_config.NumberColumn("Καλύτερη", format="%.2f", help=GLOSS["Απόδοση"] + " Η καλύτερη από όλες τις εταιρείες."),
            "Τυπική απόδοση": st.column_config.NumberColumn("Τυπική", format="%.2f",
                                                            help="Ο διάμεσος όλων των εταιρειών: περίπου όσο βρίσκεις στη Stoiximan. "
                                                                 "Αν η Stoiximan δίνει κάτω από αυτό, το αναμενόμενο κέρδος είναι μικρότερο."),
            "Εταιρείες αγοράς": st.column_config.NumberColumn(help="Από πόσες εταιρείες βγήκε η γνώμη της αγοράς."),
            "Πρόβλεψη": st.column_config.NumberColumn(format="%.1f", help=GLOSS["Πρόβλεψη"]),
            "Λεπτά": st.column_config.NumberColumn(format="%.0f", help=GLOSS["Αναμενόμενα λεπτά"]),
            "Αντίπαλος": st.column_config.TextColumn(help="Η ομάδα που αντιμετωπίζει στον επόμενο αγώνα."),
            "Παίκτης": st.column_config.TextColumn(help="Επώνυμο, όνομα."),
            "Εταιρεία": st.column_config.TextColumn(help="Η εταιρεία με την καλύτερη απόδοση για αυτό το στοίχημα."),
            "Αναμ. κέρδος ": st.column_config.NumberColumn(format="%+.1f%%"),
            "Τελευταίοι 10": st.column_config.BarChartColumn(y_min=0, help=GLOSS["Τελευταίοι 10"]),
            "Φόρμα": st.column_config.TextColumn(help=GLOSS["Φόρμα"]),
            "Σεζόν": st.column_config.TextColumn(help="Πόσες φορές θα κέρδιζε φέτος"),
            "Ποντάρισμα": st.column_config.NumberColumn(format="%.1f%%", help=GLOSS["Ποντάρισμα"]),
            "Γιατί": st.column_config.TextColumn(width="large", help="Οι κύριοι λόγοι πίσω από την ετυμηγορία."),
        })
    with st.expander("Υπόμνημα: τι σημαίνει κάθε στήλη"):
        g1, g2 = st.columns(2)
        g1.markdown(gloss_html(["Ετυμηγορία", "Over / Under", "Απόδοση", "Αναμενόμενο κέρδος", "Πιθανότητα"]), unsafe_allow_html=True)
        g2.markdown(gloss_html(["Τελευταίοι 10", "Φόρμα", "Πιθανότητα μοντέλου", "Πιθανότητα αγοράς", "Ποντάρισμα"]), unsafe_allow_html=True)
    hidden_list(hidden_rows)
    discard_list(disc_rows)
    sel = [i for i in (ev.selection.rows if ev and ev.selection else []) if i < len(v)]
    if sel:
        bc_ = st.columns([1, 1, 3])
        fav_many(bc_[0], v.iloc[sel], nxt, "fav_d")
        discard_many(bc_[1], v.iloc[sel], "disc_d")
    if len(sel) > 1:
        st.caption(f"Επέλεξες {len(sel)} στοιχήματα. Η ανάλυση δείχνει το τελευταίο.")
    if sel:
        r = v.iloc[sel[-1]]
        o = odds[(odds.player_id == r.player_id) & (odds.stat == r.stat) & (odds.line == r["Γραμμή"])]
        best = o.groupby("side")["price"].max()
        st.divider()
        hide_button(st, r, nxt, "hide_d")
        render_player(D, nxt, r.player_id, r.stat, float(r["Γραμμή"]), best.get("over"), best.get("under"),
                      key="v", side=r["side"], w_model=w_model)


def hide_button(target, row, nxt, key: str):
    """Κουμπί «Κρύψε / Ξαναδείξε» για ένα στοίχημα του πίνακα αξίας."""
    k = B.bet_key(row, nxt)
    if k is None:
        return
    hidden = k in B.hidden_keys(con)
    lab = "Ξαναδείξε αυτό το στοίχημα" if hidden else "🙈 Κρύψε αυτό το στοίχημα"
    if target.button(lab, key=key, help="Το κρύβει από τη σελίδα «Αξία» μέχρι τον επόμενο αγώνα του παίκτη. "
                                        "Στο Αρχείο μένει, με σήμανση «κρυμμένο», για να φαίνεται αν είχες δίκιο."):
        B.set_hidden(con, k, not hidden)
        st.session_state.pop("board", None)
        st.rerun()


def discard_list(rows: pd.DataFrame):
    if rows.empty:
        return
    with st.expander(f"Πεταμένα στοιχήματα ({len(rows)})"):
        st.caption("Δεν φαίνονται στην Αξία και δεν καταγράφονται στο Αρχείο. Η επαναφορά τα ξαναφέρνει και τα καταγράφει από τώρα.")
        if st.button("Επαναφορά όλων", key="undisc_all"):
            B.set_discarded(con, list(rows["_key"]), False)
            st.rerun()
        for i, r in enumerate(rows.to_dict("records")):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f'{esc(short(r["Παίκτης"]))} · {esc(r["Στοίχημα"])} · {r["Απόδοση"]:.2f}')
            if c2.button("Επαναφορά", key=f"undisc_{i}"):
                B.set_discarded(con, [r["_key"]], False)
                st.rerun()


def fav_section(rows: pd.DataFrame, nxt):
    """Τα στοιχήματα που σημείωσες «Θα το έπαιζα», ξεχωριστά στην κορυφή."""
    if rows.empty:
        return
    heading(st, f"⭐ Θα έπαιζα ({len(rows)})", "Όσα διάλεξες. Μετράνε στο Αρχείο με την ετικέτα «Θα τα έπαιζα», "
            "σημειωμένα πριν τον αγώνα, άρα με τίμιο ποσοστό.")
    rows = rows.reset_index(drop=True)
    t = pd.DataFrame({"Παίκτης": [short(n) if is_mobile() else n.title() for n in rows["Παίκτης"]],
                      "Στοίχημα": rows["Στοίχημα"], "Αντίπαλος": rows["Αντίπαλος"],
                      "Τυπική": rows["Τυπική απόδοση"], "Καλύτερη": rows["Απόδοση"],
                      "Πιθανότητα": rows["P τελική"] * 100, "Αναμ. κέρδος": rows["EV τυπικής"] * 100,
                      "Ετυμηγορία": rows["verdict"].map(V_NAME), "Βγάλε": False})
    cols = ["Βγάλε", "Παίκτης", "Στοίχημα", "Τυπική", "Αναμ. κέρδος"] if is_mobile() else list(t.columns[-1:]) + list(t.columns[:-1])
    ed = st.data_editor(t[cols], hide_index=True, width="stretch", key="fav_tbl",
                        disabled=[c for c in cols if c != "Βγάλε"], column_config={
                            "Βγάλε": st.column_config.CheckboxColumn("✕", width=34, help="Τσέκαρε για να το βγάλεις από τα «Θα έπαιζα»."),
                            "Τυπική": st.column_config.NumberColumn(format="%.2f"), "Καλύτερη": st.column_config.NumberColumn(format="%.2f"),
                            "Πιθανότητα": st.column_config.NumberColumn(format="%.0f%%"),
                            "Αναμ. κέρδος": st.column_config.NumberColumn(format="%+.1f%%", help="Στην τυπική απόδοση.")})
    out = rows[ed["Βγάλε"].to_numpy()]
    if len(out) and st.button(f"Βγάλε {len(out)} από τα «Θα έπαιζα»", key="fav_rm"):
        PK.mark(con, out, nxt, False)
        st.rerun()
    st.write("")


def fav_many(target, rows: pd.DataFrame, nxt, key: str):
    """Κουμπί που βάζει τα επιλεγμένα στα «Θα έπαιζα» (ή τα βγάζει αν είναι όλα ήδη εκεί)."""
    if rows.empty:
        return
    all_in = bool(rows["_fav"].all())
    lab = f"Βγάλε από τα «Θα έπαιζα» ({len(rows)})" if all_in else f"⭐ Θα το έπαιζα ({len(rows)})"
    if target.button(lab, key=key, type="secondary" if all_in else "primary",
                     help="Τα ξεχωρίζει στην κορυφή της σελίδας και τα μετράει στο Αρχείο με την ετικέτα «Θα τα έπαιζα»."):
        PK.mark(con, rows, nxt, not all_in)
        st.session_state.pop("board", None)
        st.rerun()


def discard_many(target, rows: pd.DataFrame, key: str):
    """Κουμπί που πετάει όλα τα στοιχήματα του rows μαζί."""
    keys = [k for k in rows["_key"] if k is not None]
    if keys and target.button(f"🗑️ Πέτα τα επιλεγμένα ({len(keys)})", key=key, type="primary",
                              help="Φεύγουν από την Αξία και από το Αρχείο (και δεν ξανακαταγράφονται). Επαναφέρονται από τη λίστα «Πεταμένα»."):
        B.set_discarded(con, keys, True)
        st.session_state.pop("board", None)
        st.rerun()


def hidden_list(rows: pd.DataFrame):
    if rows.empty:
        return
    with st.expander(f"Κρυμμένα στοιχήματα ({len(rows)})"):
        for i, r in enumerate(rows.to_dict("records")):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f'{esc(short(r["Παίκτης"]))} · {esc(r["Στοίχημα"])} · {r["Απόδοση"]:.2f} '
                        f'<span style="color:{V_COLOR[r["verdict"]]}">{V_NAME[r["verdict"]]}</span>', unsafe_allow_html=True)
            if c2.button("Επαναφορά", key=f"unhide_{i}"):
                B.set_hidden(con, r["_key"], False)
                st.rerun()


def _fold(x) -> str:
    """Πεζά, χωρίς τόνους: για αναζήτηση ονομάτων."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", str(x)) if not unicodedata.combining(c)).lower()


def game_label(date, a, b) -> str:
    d = pd.Timestamp(date)
    d = d.tz_convert("Europe/Athens") if d.tzinfo else d
    x, y = sorted([str(a), str(b)])
    return f"{d:%d/%m} {x}–{y}"


def study_filters(df: pd.DataFrame, D, key: str, *, price=None, prob=None, ev=None, team=None, opp=None,
                  game=None, pid=None, minutes=None, player=None, expanded=False) -> pd.DataFrame:
    """Κοινά φίλτρα μελέτης. Κάθε όρισμα είναι το όνομα της στήλης στο df (ή None αν δεν υπάρχει).
    Τα εύρη που μένουν στην πλήρη έκτασή τους δεν φιλτράρουν τίποτα (ούτε τις κενές τιμές)."""
    if df.empty:
        return df
    out = df
    mob = is_mobile()
    n_on = 0
    with st.expander("Φίλτρα μελέτης", expanded=expanded):
        c = st.columns(1 if mob else 3)
        col = lambda i: c[0] if mob else c[i % 3]  # noqa: E731

        def rng(i, label, colname, lo, hi, step, fmt, scale=1.0, help_=None):
            nonlocal out, n_on
            vals = df[colname].astype(float) * scale
            if vals.notna().sum() == 0:
                return
            lo_, hi_ = float(np.floor(min(lo, vals.min()) / step) * step), float(np.ceil(max(hi, vals.max()) / step) * step)
            if hi_ <= lo_:
                return
            a, b = col(i).slider(label, lo_, hi_, (lo_, hi_), step, format=fmt, key=f"{key}_{colname}", help=help_)
            if a > lo_ or b < hi_:
                x = out[colname].astype(float) * scale
                out = out[(x >= a - 1e-9) & (x <= b + 1e-9)]
                n_on += 1
        i = 0
        if price:
            rng(i, "Απόδοση", price, 1.0, 3.0, 0.05, "%.2f",
                help_="Η τυπική απόδοση (διάμεσος των εταιρειών) όπου υπάρχει, αλλιώς η απόδοση της γραμμής."); i += 1
        if prob:
            rng(i, "Πιθανότητα (%)", prob, 0, 100, 1.0, "%.0f%%", 100.0, GLOSS["Πιθανότητα"]); i += 1
        if ev:
            rng(i, "Αναμενόμενο κέρδος (%)", ev, 0, 10, 1.0, "%+.0f%%", 100.0, GLOSS["Αναμενόμενο κέρδος"]); i += 1
        if minutes:
            rng(i, "Αναμενόμενα λεπτά", minutes, 0, 40, 1.0, "%.0f′", 1.0, GLOSS["Αναμενόμενα λεπτά"]); i += 1
        if pid:
            ps = col(i).multiselect("Θέση", [0, 1, 2], format_func=M.ROLE_NAMES.get, placeholder="Όλες", key=f"{key}_pos")
            if ps:
                out = out[[pos_of(D, p_) in ps for p_ in out[pid]]]
                n_on += 1
            i += 1
        for lab, colname in (("Ομάδα", team), ("Αντίπαλος", opp)):
            if colname:
                opts = sorted(df[colname].dropna().astype(str).unique(), key=team_short)
                sel = col(i).multiselect(lab, opts, format_func=lambda t: f"{team_short(t)} ({t})", placeholder="Όλες",
                                         key=f"{key}_{colname}")
                if sel:
                    out = out[out[colname].astype(str).isin(sel)]
                    n_on += 1
                i += 1
        if game:
            opts = sorted(df[game].dropna().unique())
            sel = col(i).multiselect("Αγώνας", opts, placeholder="Όλοι", key=f"{key}_game")
            if sel:
                out = out[out[game].isin(sel)]
                n_on += 1
            i += 1
        if player:
            q = col(i).text_input("Παίκτης", key=f"{key}_q", placeholder="π.χ. Sloukas")
            if q.strip():
                qq = _fold(q.strip())
                out = out[[qq in _fold(n) for n in out[player]]]
                n_on += 1
        if n_on:
            st.caption(f"{n_on} ενεργά φίλτρα · {len(out)} από {len(df)}.")
    return out



def page_player(D, nxt, w_model):
    st.title("Παίκτης")
    st.markdown('<div class="el-sub">Δες οποιονδήποτε παίκτη σε οποιαδήποτε αγορά και γραμμή, '
                'με ή χωρίς απόδοση.</div>', unsafe_allow_html=True)
    cur = D["current"]
    season = int(D["df"]["season"].max())
    c = st.columns([2, 2, 3, 3])
    pool = cur[cur["season"] == season]
    team = c[0].selectbox("Ομάδα", ["Όλες"] + sorted(pool["team"].unique()))
    pos_f = c[1].selectbox("Θέση", [None, 0, 1, 2], format_func=lambda p_: "Όλες" if p_ is None else M.ROLE_NAMES[p_])
    if team != "Όλες":
        pool = pool[pool["team"] == team]
    if pos_f is not None:
        pool = pool[[pos_of(D, i) == pos_f for i in pool.index]]
    pool = pool.sort_values("min_hat", ascending=False) if team != "Όλες" else pool.sort_values("player")
    if pool.empty:
        st.info("Κανένας παίκτης.")
        return
    pid = c[2].selectbox("Παίκτης", list(pool.index), format_func=lambda i: f"{pool.at[i, 'player'].title()} · {pool.at[i, 'team']}")
    stat = c[3].selectbox("Αγορά", M.KEYS, format_func=STAT_LABEL.get)
    mu, _, _ = B.projection(pid, stat, cur, D["ctx"], nxt)
    d = st.columns([2, 2, 2, 2])
    side = d[0].radio("Πλευρά", ["over", "under"], format_func=lambda s_: s_.title(), horizontal=True)
    line = d[1].number_input("Γραμμή", value=float(max(math.floor(mu), 0) + 0.5), step=0.5, min_value=0.0,
                             key=f"line_{pid}_{stat}")
    over = d[2].number_input("Απόδοση Over", value=None, min_value=1.01, step=0.05, placeholder="π.χ. 1.85")
    under = d[3].number_input("Απόδοση Under", value=None, min_value=1.01, step=0.05, placeholder="π.χ. 1.85")
    st.divider()
    render_player(D, nxt, pid, stat, line, over, under, side=side, w_model=w_model)


def upcoming_games(days: int = 10) -> pd.DataFrame:
    up = pd.read_sql_query("SELECT * FROM games WHERE played=0", con)
    up["date"] = pd.to_datetime(up["date"], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    up = up[(up["date"] >= now - pd.Timedelta(hours=3)) & (up["date"] <= now + pd.Timedelta(days=days))]
    up = up.sort_values("date")
    up["label"] = [f"{d.tz_convert('Europe/Athens'):%a %d/%m %H:%M} · {h} – {a}"
                   for d, h, a in zip(up["date"], up["home_name"], up["away_name"])]
    return up


def defense_chart(target, D, team: str, first: str | None = None):
    show_k = list(dict.fromkeys(([first] if first else []) + ["pts", "reb", "ast", "fg3m", "fg2m", "stl", "blk", "tov"]))[:8]
    od = pd.DataFrame({"Αγορά": [STAT_LABEL[k] for k in show_k],
                       "d": [float(D["ctx"]["opp"][team][M.KEYS.index(k)] - 1) for k in show_k]})
    od["Σε σχέση με τη λίγκα"] = np.where(od["d"] >= 0, "Δίνει περισσότερα", "Δίνει λιγότερα")
    lim = max(0.12, float(od["d"].abs().max()) * 1.15)
    oc = alt.Chart(od).mark_bar(cornerRadius=3, size=14).encode(
        y=alt.Y("Αγορά:N", sort=list(od["Αγορά"]), title=None, axis=alt.Axis(labelLimit=160, labelOverlap=False)),
        x=alt.X("d:Q", title=None, axis=alt.Axis(format="+%"), scale=alt.Scale(domain=[-lim, lim])),
        color=alt.Color("Σε σχέση με τη λίγκα:N", scale=alt.Scale(domain=["Δίνει περισσότερα", "Δίνει λιγότερα"],
                                                                 range=[BLUE, ORANGE]), legend=None),
        tooltip=["Αγορά", alt.Tooltip("d:Q", format="+.1%", title="Έναντι λίγκας")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED).encode(x="x:Q")
    show_chart(target, (oc + zero).properties(height=34 * len(od) + 40))


def defense_pos_chart(target, D, team: str):
    ks = ["pts", "reb", "ast", "fg3m"]
    dp = D["ctx"]["def_pos"][team]
    od = pd.DataFrame([{"Αγορά": STAT_LABEL[k], "Θέση": M.ROLE_NAMES[p], "d": float(dp[p][k] - 1)} for k in ks for p in (0, 1, 2)])
    lim = max(0.12, float(od["d"].abs().max()) * 1.15)
    names = [M.ROLE_NAMES[i] for i in (0, 1, 2)]
    oc = alt.Chart(od).mark_bar(cornerRadius=3, size=9).encode(
        y=alt.Y("Αγορά:N", sort=[STAT_LABEL[k] for k in ks], title=None, axis=alt.Axis(labelLimit=160)),
        yOffset=alt.YOffset("Θέση:N", sort=names),
        x=alt.X("d:Q", title=None, axis=alt.Axis(format="+%"), scale=alt.Scale(domain=[-lim, lim])),
        color=alt.Color("Θέση:N", scale=alt.Scale(domain=names, range=POS_COLORS), legend=None),
        tooltip=["Αγορά", "Θέση", alt.Tooltip("d:Q", format="+.1%", title="Έναντι λίγκας")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED).encode(x="x:Q")
    show_chart(target, (oc + zero).properties(height=60 * len(ks) + 30))


def page_game(D, nxt, w_model):
    st.title("Αγώνας")
    st.markdown('<div class="el-sub">Όλοι οι παίκτες ενός αγώνα μαζί: τι περιμένει το μοντέλο από τον καθένα, '
                'ποιος λείπει και τι δίνει κάθε άμυνα. Ξεκίνα από εδώ για να δεις πού αξίζει να ψάξεις γραμμές.</div>',
                unsafe_allow_html=True)
    up = upcoming_games()
    if up.empty:
        st.info("Δεν υπάρχουν αγώνες στις επόμενες 10 ημέρες. Κάνε ενημέρωση στατιστικών στα «Δεδομένα και αντίγραφα» (στήλη αριστερά).")
        return
    cur, ctx = D["current"], D["ctx"]
    season = int(D["df"]["season"].max())
    lab = st.selectbox("Αγώνας", list(up["label"]))
    g = up[up["label"] == lab].iloc[0]
    absent = ctx.get("absent", set())
    show = ["pts", "reb", "ast", "fg3m", "pra"]
    hist = {pid: h for pid, h in D["df"][D["df"]["team"].isin([g.home_code, g.away_code])].groupby("player_id")}
    cols = st.columns(2, gap="large")
    edited = {}
    for col, team, name, opp in ((cols[0], g.home_code, g.home_name, g.away_code),
                                 (cols[1], g.away_code, g.away_name, g.home_code)):
        pool = cur[(cur["team"] == team) & (cur["season"] == season)].sort_values("min_hat", ascending=False)
        pool = pool[(pool["min_hat"] >= 6) | pool.index.isin(absent)]
        pool = pool.assign(_p=[pos_of(D, i) for i in pool.index]).sort_values(["_p", "min_hat"], ascending=[True, False])
        rows = []
        for pid in pool.index:
            out = pid in absent
            eff = M.absence_effect(pid, cur, ctx, absent)
            r = {"Λείπει": out, "Παίκτης": pool.at[pid, "player"].title() + (" ⚠️" if M.role_note(pid, ctx, cur) else ""),
                 "Θέση": pos_name(D, pid),
                 "Λεπτά": np.nan if out else eff["min"]}
            for k in show:
                r[STAT_LABEL[k] if k != "pra" else "Π+Ρ+Α"] = np.nan if out else B.projection(pid, k, cur, ctx, nxt)[0]
            h = hist.get(pid)
            l5 = float(h.sort_values("date")["pra"].tail(5).mean()) if h is not None and len(h) else np.nan
            base = B.projection(pid, "pra", cur, ctx, nxt)[0]
            r["Φόρμα"] = "" if out or np.isnan(l5) else ("▲ ανεβασμένος" if l5 > base * 1.15 else "▼ πεσμένος" if l5 < base * 0.85 else "σταθερός")
            r["Λόγω απουσιών"] = "" if out or not eff["out"] else f"+{eff['min'] - pool.at[pid, 'min_hat']:.1f}′, πόντοι ×{eff['mult']['pts']:.2f}"
            rows.append(r)
        t = pd.DataFrame(rows)
        if is_mobile():
            t["Παίκτης"] = [short(pool.at[pid, "player"]) + (" ⚠️" if M.role_note(pid, ctx, cur) else "") for pid in pool.index]
            t["Θέση"] = [pos_name(D, pid, False)[:1] for pid in pool.index]
            t = t.rename(columns={"Πόντοι": "Π", "Ριμπάουντ": "Ρ", "Ασίστ": "Α", "Π+Ρ+Α": "ΠΡΑ", "Λεπτά": "Λεπ"})[
                ["Λείπει", "Θέση", "Παίκτης", "Λεπ", "Π", "Ρ", "Α", "ΠΡΑ"]]
        heading(col, f"{name}", f"{'Εντός' if team == g.home_code else 'Εκτός'} έδρας. Τσέκαρε «Λείπει» σε όποιον δεν θα παίξει "
                "και πάτα αποθήκευση: οι προβλέψεις των συμπαικτών του προσαρμόζονται.")
        num = {c: st.column_config.NumberColumn(format="%.1f") for c in t.columns if c not in ("Λείπει", "Παίκτης", "Θέση", "Φόρμα", "Λόγω απουσιών", "Λεπτά")}
        ed = col.data_editor(
            t, hide_index=True, width="stretch", key=f"game_{g.season}_{g.game_code}_{team}",
            disabled=[c for c in t.columns if c != "Λείπει"], height=38 * (len(t) + 1) + 2,
            column_config={**num,
                           "Λεπ": st.column_config.NumberColumn(format="%.0f", width=42, help=GLOSS["Αναμενόμενα λεπτά"]),
                           "Π": st.column_config.NumberColumn(format="%.1f", width=42, help="Πόντοι: " + GLOSS["Πρόβλεψη"]),
                           "Ρ": st.column_config.NumberColumn(format="%.1f", width=38, help="Ριμπάουντ"),
                           "Α": st.column_config.NumberColumn(format="%.1f", width=38, help="Ασίστ"),
                           "ΠΡΑ": st.column_config.NumberColumn(format="%.1f", width=44, help="Πόντοι + ριμπάουντ + ασίστ"),
                           **({"Παίκτης": st.column_config.TextColumn(width=104)} if is_mobile() else {}),
                           "Λείπει": st.column_config.CheckboxColumn("Εκτός" if is_mobile() else "Λείπει", width=48 if is_mobile() else "small",
                                                                    help=GLOSS["Απουσίες"]),
                           "Θέση": st.column_config.TextColumn(width=30 if is_mobile() else "small",
                                                               help="Γκαρντ, Φόργουορντ ή Σέντερ από το επίσημο ρόστερ. Με «~» όταν είναι εκτίμηση από τα στατιστικά του."),
                           "Λεπτά": st.column_config.NumberColumn(format="%.0f", help=GLOSS["Αναμενόμενα λεπτά"]),
                           "Πόντοι": st.column_config.NumberColumn(format="%.1f", help=GLOSS["Πρόβλεψη"]),
                           "Φόρμα": st.column_config.TextColumn(help="Οι 5 τελευταίοι αγώνες του σε σχέση με την πρόβλεψη (Π+Ρ+Α). Μικρό δείγμα, μόνο για εικόνα."),
                           "Λόγω απουσιών": st.column_config.TextColumn(help="Πόσο άλλαξε η πρόβλεψή του επειδή λείπουν συμπαίκτες.")})
        edited[team] = (list(pool.index), {pid for pid, out in zip(pool.index, ed["Λείπει"]) if out})
        heading(col, f"Τι δέχεται η άμυνα της {name} ανά θέση",
                "".join(sw(POS_COLORS[i], M.ROLE_NAMES[i]) for i in (0, 1, 2))
                + f"<br>Πόσο περισσότερο (δεξιά) ή λιγότερο (αριστερά) από τον μέσο όρο της λίγκας δέχεται από κάθε θέση, "
                  f"στους τελευταίους 20 αγώνες. Αφορά τους παίκτες της {esc(opp)}.")
        if team in ctx.get("def_pos", {}):
            defense_pos_chart(col, D, team)
    changed = any(sel != (set(ids) & absent) for ids, sel in edited.values())
    b1, b2 = st.columns([1, 4])
    if b1.button("Αποθήκευση απουσιών", type="primary", disabled=not changed):
        for ids, sel in edited.values():
            B.set_absent(con, ids, sel)
        st.rerun()
    b2.markdown('<div class="el-leg">Πώς υπολογίζεται: από 351 περσινούς αγώνες με απουσία παίκτη του ροτέισον. '
                'Όσοι έρχονται από τον πάγκο κερδίζουν τα περισσότερα λεπτά, ειδικά στον ίδιο ρόλο με τον απόντα. '
                'Οι βασικοί κερδίζουν κυρίως σε σκοράρισμα και δημιουργία. Το μέσο όφελος είναι μικρότερο από όσο '
                'φαντάζεται κανείς: συνήθως +5% έως +15% στους πόντους. Η απουσία σβήνει μόνη της όταν ο παίκτης ξαναπαίξει.</div>',
                unsafe_allow_html=True)

    st.divider()
    both = cur[(cur["team"].isin([g.home_code, g.away_code])) & (cur["season"] == season) & (cur["min_hat"] >= 6)
               & ~cur.index.isin(absent)].sort_values(["team", "min_hat"], ascending=[True, False])
    c = st.columns([3, 3, 2, 2])
    pid = c[0].selectbox("Δες παίκτη αυτού του αγώνα", list(both.index),
                         format_func=lambda i: f"{both.at[i, 'player'].title()} · {both.at[i, 'team']}", key="g_pid")
    stat = c[1].selectbox("Αγορά", M.KEYS, format_func=STAT_LABEL.get, key="g_stat")
    side = c[2].radio("Πλευρά", ["over", "under"], format_func=lambda s_: s_.title(), horizontal=True, key="g_side")
    mu, _, _ = B.projection(pid, stat, cur, ctx, nxt)
    line = c[3].number_input("Γραμμή", value=float(max(math.floor(mu), 0) + 0.5), step=0.5, min_value=0.0, key=f"g_line_{pid}_{stat}")
    render_player(D, nxt, pid, stat, line, None, None, key="g", side=side, w_model=w_model)


STATUS = {"open": "⏳ Ανοιχτό", "won": "✅ Κέρδισε", "lost": "❌ Έχασε", "push": "↔️ Ισοπαλία", "void": "🚫 Ακυρώθηκε"}


def page_bets(D, embedded: bool = False):
    if not embedded:
        st.title("Ημερολόγιο")
    st.markdown('<div class="el-sub">Ό,τι έπαιξες, με το αποτέλεσμα να συμπληρώνεται μόνο του μόλις περαστούν τα '
                'στατιστικά του αγώνα. Εδώ φαίνεται αν το εργαλείο σε βοηθάει στην πράξη.</div>', unsafe_allow_html=True)
    d = BT.load(con)
    if d.empty:
        st.info("Δεν έχεις καταγράψει στοιχήματα. Όταν δεις κάτι που θα παίξεις, στη σελίδα «Αξία» ή «Παίκτης», "
                "γράψε το ποσό κάτω από την ετυμηγορία και πάτα «Το έπαιξα».")
        return
    sm = BT.summary(d)
    pnl_col = V_COLOR["strong"] if sm["pnl"] > 0 else V_COLOR["suspect"] if sm["pnl"] < 0 else None
    tiles(st, [
        ("Ανοιχτά", str(sm["n_open"]), "Περιμένουν να παιχτεί ο αγώνας ή να περαστούν τα στατιστικά του."),
        ("Κλεισμένα", str(sm["n_settled"]), f"Κέρδισαν ή έχασαν. Άλλα {sm['n_void']} ακυρώθηκαν ή ήρθαν ισοπαλία."),
        ("Κέρδος / ζημιά", f"{sm['pnl']:+.2f} €", f"Από {sm['staked']:.2f} € που ποντάρισες στα κλεισμένα.", pnl_col),
        ("Απόδοση επένδυσης", "—" if sm["roi"] is None else f"{sm['roi']:+.1%}", GLOSS["Απόδοση επένδυσης"], pnl_col),
        ("Ποσοστό επιτυχίας", "—" if sm["hit"] is None else pct(sm["hit"]),
         "Το μοντέλο περίμενε " + (pct(sm["expected_hit"]) if sm["expected_hit"] is not None else "—") + "."),
        ("Αναμενόμενο κέρδος", "—" if sm["expected_pnl"] is None else f"{sm['expected_pnl']:+.2f} €",
         "Τι θα έπρεπε να έχεις βγάλει αν οι πιθανότητες ήταν σωστές. Σύγκρινέ το με το πραγματικό."),
    ])
    s = d[d["status"].isin(["won", "lost"])].sort_values("settled_at").reset_index(drop=True)
    if sm["n_settled"] < 50:
        st.markdown(f'<div class="el-leg">Με {sm["n_settled"]} κλεισμένα στοιχήματα το αποτέλεσμα είναι κυρίως τύχη. '
                    'Χρειάζονται τουλάχιστον 100–200 για να φανεί αν υπάρχει πραγματικό πλεονέκτημα.</div>', unsafe_allow_html=True)
    if len(s) >= 2:
        s["n"] = np.arange(1, len(s) + 1)
        long = pd.concat([pd.DataFrame({"n": s["n"], "€": s["pnl"].cumsum(), "Σειρά": "Πραγματικό"}),
                          pd.DataFrame({"n": s["n"], "€": (s["stake"] * s["ev"].fillna(0)).cumsum(), "Σειρά": "Αναμενόμενο"})])
        ch = alt.Chart(long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40)).encode(
            x=alt.X("n:Q", title="Κλεισμένα στοιχήματα με τη σειρά", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("€:Q", title="Συνολικό κέρδος / ζημιά (€)"),
            color=alt.Color("Σειρά:N", scale=alt.Scale(domain=["Πραγματικό", "Αναμενόμενο"], range=[BLUE, MUTED]), legend=None),
            strokeDash=alt.StrokeDash("Σειρά:N", scale=alt.Scale(domain=["Πραγματικό", "Αναμενόμενο"], range=[[1, 0], [5, 4]]), legend=None),
            tooltip=["Σειρά", alt.Tooltip("n:Q", title="Στοίχημα"), alt.Tooltip("€:Q", format="+.2f")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=HAIR).encode(y="y:Q")
        a, b = st.columns([3, 2], gap="large")
        heading(a, "Πορεία κεφαλαίου", sw(BLUE, "τι έβγαλες πραγματικά", "ln") + sw(MUTED, "τι περίμενε το μοντέλο", "ln dash")
                + "<br>Αν η μπλε μένει σταθερά κάτω από τη διακεκομμένη, το μοντέλο είναι υπερβολικά αισιόδοξο.")
        show_chart(a, (zero + ch).properties(height=300))
        by = s.assign(Ετυμηγορία=s["verdict"].map(V_NAME).fillna("χωρίς")).groupby("Ετυμηγορία").agg(
            Στοιχήματα=("id", "size"), Κέρδισαν=("status", lambda x: int((x == "won").sum())),
            Ποντάρισμα=("stake", "sum"), Κέρδος=("pnl", "sum")).reset_index()
        by["Απόδοση %"] = by["Κέρδος"] / by["Ποντάρισμα"] * 100
        heading(b, "Ανά ετυμηγορία", "Αν τα «Αξίζει» δεν πάνε καλύτερα από τα υπόλοιπα, η ετυμηγορία δεν ξεχωρίζει τίποτα.")
        b.dataframe(by, hide_index=True, width="stretch", column_config={
            "Ποντάρισμα": st.column_config.NumberColumn(format="%.2f €"), "Κέρδος": st.column_config.NumberColumn(format="%+.2f €"),
            "Απόδοση %": st.column_config.NumberColumn(format="%+.1f%%", help=GLOSS["Απόδοση επένδυσης"])})

    heading(st, "Όλα τα στοιχήματα", "Το αποτέλεσμα συμπληρώνεται μόνο του. «Ακυρώθηκε» σημαίνει ότι ο παίκτης δεν αγωνίστηκε, "
            "οπότε οι εταιρείες επιστρέφουν το ποντάρισμα.")
    v = d.assign(Ημερομηνία=d["placed_at"].dt.tz_convert("Europe/Athens").dt.strftime("%d/%m %H:%M"),
                 Παίκτης=d["player"].str.title(),
                 Στοίχημα=[f"{'Over' if sd == 'over' else 'Under'} {ln:g} {STAT_LABEL.get(stt, stt)}" for sd, ln, stt in zip(d["side"], d["line"], d["stat"])],
                 Κατάσταση=d["status"].map(STATUS), Ετυμηγορία=d["verdict"].map(V_NAME),
                 **{"Αναμ. κέρδος": d["ev"] * 100, "Σβήσιμο": False})
    bet_cols = ["Σβήσιμο", "Ημερομηνία", "Παίκτης", "opp", "Στοίχημα", "price", "stake", "Ετυμηγορία", "Αναμ. κέρδος", "Κατάσταση", "result", "pnl"]
    if is_mobile():
        v["Παίκτης"] = [short(n) for n in d["player"]]
        v["Κατάσταση"] = v["Κατάσταση"].str.split(" ").str[0]
        v["Στοίχημα"] = [f"{'O' if sd == 'over' else 'U'} {ln:g} {STAT_LABEL.get(stt, stt)} @{pr:.2f}"
                         for sd, ln, stt, pr in zip(d["side"], d["line"], d["stat"], d["price"])]
        bet_cols = ["Σβήσιμο", "Παίκτης", "Στοίχημα", "Κατάσταση", "pnl"]
    ed = st.data_editor(
        v[bet_cols]
        .rename(columns={"opp": "Αντίπαλος", "price": "Απόδοση", "stake": "Ποσό", "result": "Έκανε", "pnl": "Κέρδος"}),
        hide_index=True, width="stretch", key="bets_tbl",
        disabled=["Ημερομηνία", "Παίκτης", "Αντίπαλος", "Στοίχημα", "Απόδοση", "Ποσό", "Ετυμηγορία", "Αναμ. κέρδος", "Κατάσταση", "Έκανε", "Κέρδος"],
        height=min(38 * (len(v) + 1) + 2, 600),
        column_config={"Απόδοση": st.column_config.NumberColumn(format="%.2f", help=GLOSS["Απόδοση"]),
                       "Ποσό": st.column_config.NumberColumn(format="%.2f €"),
                       "Κέρδος": st.column_config.NumberColumn(format="%+.2f €"),
                       "Έκανε": st.column_config.NumberColumn(format="%.0f", help="Τι έκανε τελικά ο παίκτης στο στατιστικό."),
                       "Αναμ. κέρδος": st.column_config.NumberColumn(format="%+.1f%%", help=GLOSS["Αναμενόμενο κέρδος"]),
                       "Ετυμηγορία": st.column_config.TextColumn(help="Η ετυμηγορία τη στιγμή που το έπαιξες."),
                       **({"Παίκτης": st.column_config.TextColumn(width=88),
                           "Στοίχημα": st.column_config.TextColumn(width=132, help="O = Over, U = Under, @ απόδοση"),
                           "Κατάσταση": st.column_config.TextColumn("", width=30, help="⏳ ανοιχτό · ✅ κέρδισε · ❌ έχασε · ↔️ ισοπαλία · 🚫 ακυρώθηκε"),
                           "Κέρδος": st.column_config.NumberColumn(format="%+.2f", width=58)} if is_mobile() else {}),
                       "Σβήσιμο": st.column_config.CheckboxColumn("✕" if is_mobile() else "Σβήσιμο", width=40 if is_mobile() else "small",
                                                                 help="Τσέκαρε για να σβήσεις λάθος καταχώριση.")})
    kill = [int(i) for i, x in zip(d["id"], ed["Σβήσιμο"]) if x]
    if kill and st.button(f"Σβήσε {len(kill)} στοιχήματα"):
        con.executemany("DELETE FROM bets WHERE id=?", [(i,) for i in kill])
        con.commit()
        st.rerun()


def refresh_odds(D) -> str:
    """Τραβάει αποδόσεις από το OddsPapi με το αποθηκευμένο κλειδί. Επιστρέφει μήνυμα."""
    cfg = O.load_config()
    try:
        res = O.fetch_oddspapi(cfg["oddspapi_key"], cfg.get("books", ""), 7)
        n, unmatched = O.store(con, res["rows"], O.PlayerIndex(D["current"]), "oddspapi")
    except Exception as e:  # noqa: BLE001
        return f"Οι αποδόσεις δεν ανανεώθηκαν: {e}"
    O.save_config({**cfg, "odds_at": pd.Timestamp.now(tz="UTC").isoformat()})
    if not n:
        return "Το OddsPapi δεν έδωσε ειδικά παικτών για τους επόμενους αγώνες."
    return f"Πέρασαν {n} αποδόσεις από {res['n_upcoming']} αγώνες." + (f" {len(unmatched)} παίκτες δεν αναγνωρίστηκαν." if unmatched else "")


def page_odds(D):
    st.title("Αποδόσεις")
    st.markdown('<div class="el-sub">Πέρνα εδώ τις γραμμές και τις αποδόσεις της εταιρείας σου. '
                'Χωρίς αυτές η σελίδα «Αξία» δεν έχει τι να συγκρίνει.</div>', unsafe_allow_html=True)
    cur = D["current"]
    index = O.PlayerIndex(cur)
    cfg = O.load_config()
    t_shot, t_fast, t_man, t_csv, t_api = st.tabs(["Από screenshot", "Γρήγορη εισαγωγή ανά αγώνα", "Μεμονωμένα",
                                                   "Επικόλληση CSV", "Αυτόματα (OddsPapi API)"])

    with t_shot:
        up_s = upcoming_games()
        if up_s.empty:
            st.info("Δεν υπάρχουν αγώνες στις επόμενες 10 ημέρες. Κάνε ενημέρωση στατιστικών στα «Δεδομένα και αντίγραφα» (στήλη αριστερά).")
        else:
            st.markdown('<div class="el-leg">Τράβηξε screenshot μία αγορά της Stoiximan, π.χ. «Πόντοι», με τα ονόματα και τις '
                        'αποδόσεις να φαίνονται καθαρά, και ανέβασέ το εδώ. Διαβάζει και τις δύο μορφές: τον πίνακα '
                        '«Line / Over / Under» του κινητού και τα κελιά «10+ / 1.47» του υπολογιστή (το «10+» περνιέται ως Over 9.5). '
                        'Ένα screenshot ανά αγορά· αν η αγορά δεν χωράει σε ένα, ανέβασε πολλά μαζί.</div>', unsafe_allow_html=True)
            s1, s2, s3 = st.columns([3, 2, 1])
            g_lab = s1.selectbox("Αγώνας", list(up_s["label"]), key="shot_game")
            gs = up_s[up_s["label"] == g_lab].iloc[0]
            s_stat = s2.selectbox("Αγορά του screenshot", M.KEYS, format_func=STAT_LABEL.get, key="shot_stat",
                                  help="Διάλεξε την αγορά που δείχνει το screenshot. Ένα screenshot ανά αγορά.")
            s_book = s3.text_input("Εταιρεία", value=cfg.get("fast_book", "stoiximan"), key="shot_book")
            files = st.file_uploader("Screenshot", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="shot_files",
                                     help="Μπορείς να ρίξεις πολλά μαζί αν η αγορά δεν χωράει σε ένα.")
            season_s = int(D["df"]["season"].max())
            pool_s = cur[cur["team"].isin([gs.home_code, gs.away_code]) & (cur["season"] == season_s)]
            idx_s = O.PlayerIndex(pool_s) if len(pool_s) else index
            lab_s = {f"{r.player.title()} · {r.team}": pid for pid, r in pool_s.sort_values(["team", "player"]).iterrows()}
            inv_s = {v: k for k, v in lab_s.items()}
            parsed, problems = [], []
            for f in files or []:
                raw = f.getvalue()
                ck = f"ocr_{hash(raw)}"
                if ck not in st.session_state:
                    tmp = db.DATA_DIR / "shot_tmp.png"
                    tmp.parent.mkdir(parents=True, exist_ok=True)
                    tmp.write_bytes(raw)
                    try:
                        with st.spinner("Διαβάζω το screenshot, περίπου 10–15 δευτερόλεπτα…"):
                            st.session_state[ck] = SH.read_image(str(tmp))
                    except SH.OcrUnavailable as e:
                        problems.append(f"Η αναγνώριση κειμένου δεν λειτούργησε: {e}")
                        continue
                rows_f = st.session_state[ck]
                if not rows_f:
                    problems.append(f"Στο «{f.name}» δεν βρήκα κελιά με αποδόσεις. Δοκίμασε πιο καθαρό ή μεγαλύτερο screenshot.")
                parsed += rows_f
            for p_ in problems:
                st.error(p_)
            if parsed:
                t = pd.DataFrame(parsed).drop_duplicates(["name", "line", "side"], keep="last")
                t["pid"] = [idx_s.match(n) for n in t["name"]]
                t = t.assign(_u=t["pid"].isna()).sort_values(["_u", "name", "line", "side"]).drop(columns="_u").reset_index(drop=True)
                tbl_s = pd.DataFrame({
                    "Παίκτης": [inv_s.get(p_, None) for p_ in t["pid"]], "Διάβασα": t["name"].to_numpy(),
                    "Πλευρά": t["side"].map({"over": "Over", "under": "Under"}).to_numpy(),
                    "Γραμμή": t["line"].to_numpy(float), "Απόδοση": t["price"].to_numpy(float),
                    "Έλεγχος": ["" if ok and pd.notna(p_) else "⚠ " + (why or "δεν βρήκα τον παίκτη σε αυτόν τον αγώνα")
                                for ok, p_, why in zip(t["ok"], t["pid"], t["why"])]})
                if is_mobile():
                    tbl_s = tbl_s.drop(columns=["Διάβασα"])
                n_bad = int((tbl_s["Έλεγχος"] != "").sum())
                heading(st, f"Διάβασα {len(tbl_s)} αποδόσεις για {t['name'].nunique()} παίκτες",
                        ("Όλα φαίνονται συνεπή. " if not n_bad else f"{n_bad} γραμμές θέλουν έλεγχο· η στήλη «Έλεγχος» γράφει τον λόγο. ")
                        + "Η εικόνα διαβάζεται δύο φορές και οι αποδόσεις ελέγχονται μεταξύ τους, οπότε τα λάθη συνήθως σημαδεύονται. "
                          "Διόρθωσε ό,τι χρειάζεται απευθείας στον πίνακα.")
                ed_s = st.data_editor(tbl_s, hide_index=True, width="stretch", key=f"shot_ed_{hash(tuple(t['name']))}_{s_stat}",
                                      disabled=["Διάβασα", "Έλεγχος"], num_rows="dynamic", column_config={
                    "Παίκτης": st.column_config.SelectboxColumn(options=list(lab_s), width=128 if is_mobile() else "medium",
                                                                help="Ο παίκτης στη βάση. Άλλαξέ τον αν η αντιστοίχιση είναι λάθος."),
                    "Διάβασα": st.column_config.TextColumn(help="Το όνομα όπως διαβάστηκε από την εικόνα."),
                    "Πλευρά": st.column_config.SelectboxColumn(options=["Over", "Under"], width=66 if is_mobile() else "small", help=GLOSS["Over / Under"]),
                    "Γραμμή": st.column_config.NumberColumn("Γραμ." if is_mobile() else "Γραμμή", format="%.1f", step=0.5, width=52 if is_mobile() else None,
                                                           help=GLOSS["Γραμμή"] + " Το «10+» της σκάλας είναι Over 9.5."),
                    "Απόδοση": st.column_config.NumberColumn("Απόδ." if is_mobile() else "Απόδοση", format="%.2f", min_value=1.01,
                                                            width=54 if is_mobile() else None, help=GLOSS["Απόδοση"]),
                    "Έλεγχος": st.column_config.TextColumn(width=150 if is_mobile() else None, help="Κενό σημαίνει ότι δεν βρέθηκε πρόβλημα.")})
                good = ed_s[ed_s["Παίκτης"].notna() & ed_s["Γραμμή"].notna() & ed_s["Απόδοση"].notna() & ed_s["Πλευρά"].notna()]
                if st.button(f"Αποθήκευση ({len(good)})", type="primary", key="shot_save", disabled=good.empty):
                    rows_s = [{"player_raw": pl, "player_id": lab_s[pl], "stat": s_stat, "line": float(ln),
                               "side": sd.lower(), "price": float(pr), "bookmaker": s_book or "stoiximan"}
                              for pl, ln, sd, pr in zip(good["Παίκτης"], good["Γραμμή"], good["Πλευρά"], good["Απόδοση"])]
                    n, _ = O.store(con, rows_s, index, "screenshot")
                    st.success(f"Αποθηκεύτηκαν {n} αποδόσεις. Άλλαξε αγορά και ρίξε το επόμενο screenshot.")

    with t_fast:
        up = pd.read_sql_query("SELECT * FROM games WHERE played=0", con)
        up["date"] = pd.to_datetime(up["date"], utc=True, errors="coerce")
        now = pd.Timestamp.now(tz="UTC")
        up = up[(up["date"] >= now - pd.Timedelta(hours=3)) & (up["date"] <= now + pd.Timedelta(days=10))].sort_values("date")
        if up.empty:
            st.info("Δεν υπάρχουν αγώνες στις επόμενες 10 ημέρες. Κάνε «Ενημέρωση στατιστικών» στα «Δεδομένα και αντίγραφα» (στήλη αριστερά).")
        else:
            c1, c2, c3 = st.columns([3, 2, 1])
            labels_g = {f"{r.date.tz_convert('Europe/Athens'):%a %d/%m %H:%M} · {r.home_name} – {r.away_name}":
                        (r.home_code, r.away_code) for r in up.itertuples()}
            g_lab = c1.selectbox("Αγώνας", list(labels_g))
            f_stat = c2.selectbox("Αγορά", M.KEYS, format_func=STAT_LABEL.get, key="fast_stat")
            book = c3.text_input("Εταιρεία", value=cfg.get("fast_book", "stoiximan"))
            teams = labels_g[g_lab]
            season_f = int(D["df"]["season"].max())
            pool_f = cur[(cur["team"].isin(teams)) & (cur["season"] == season_f) & (cur["min_hat"] >= 8)]
            pool_f = pool_f.assign(_t=pool_f["team"].map({teams[0]: 0, teams[1]: 1})).sort_values(
                ["_t", "min_hat"], ascending=[True, False])
            nxt_f = B.next_games(con)
            tbl = pd.DataFrame({
                "Παίκτης": pool_f["player"].to_numpy(), "Ομάδα": pool_f["team"].to_numpy(),
                "Πρόβλεψη": [B.projection(i, f_stat, cur, D["ctx"], nxt_f)[0] for i in pool_f.index],
                "Γραμμή": np.nan, "Over": np.nan, "Under": np.nan})
            if is_mobile():
                tbl["Παίκτης"] = [short(n) for n in pool_f["player"]]
                tbl = tbl.drop(columns=["Ομάδα"]).rename(columns={"Πρόβλεψη": "Πρόβλ."})
            st.caption("Γράψε γραμμή και αποδόσεις όπως τις δίνει η εταιρεία, μόνο για όσους παίκτες έχει. "
                       "Με Tab πας στο επόμενο κελί. Η «Πρόβλεψη» είναι του μοντέλου, για να βρίσκεις γρήγορα τον παίκτη.")
            ed_f = st.data_editor(tbl, hide_index=True, width="stretch", key=f"fast_{teams}_{f_stat}",
                                  disabled=["Παίκτης", "Ομάδα", "Πρόβλεψη", "Πρόβλ."], column_config={
                "Πρόβλεψη": st.column_config.NumberColumn(format="%.1f"),
                "Πρόβλ.": st.column_config.NumberColumn(format="%.1f", width=52, help=GLOSS["Πρόβλεψη"]),
                **({"Παίκτης": st.column_config.TextColumn(width=104)} if is_mobile() else {}),
                "Γραμμή": st.column_config.NumberColumn(step=0.5, min_value=0.0, format="%.1f"),
                "Over": st.column_config.NumberColumn(min_value=1.01, format="%.2f"),
                "Under": st.column_config.NumberColumn(min_value=1.01, format="%.2f")})
            filled = ed_f[ed_f["Γραμμή"].notna() & (ed_f["Over"].notna() | ed_f["Under"].notna())]
            if st.button(f"Αποθήκευση ({len(filled)})", key="save_fast", type="primary", disabled=filled.empty):
                w = pd.DataFrame({"player": filled["Παίκτης"], "player_id": pool_f.index[filled.index],
                                  "stat": f_stat, "line": filled["Γραμμή"], "over": filled["Over"],
                                  "under": filled["Under"], "bookmaker": book or "χειροκίνητα"})
                n, _ = O.store(con, O.wide_to_rows(w), index, "manual")
                O.save_config({**cfg, "fast_book": book})
                st.success(f"Αποθηκεύτηκαν {n} τιμές. Άλλαξε αγορά ή αγώνα για να συνεχίσεις.")

    with t_api:
        st.caption("Χρειάζεται δωρεάν κλειδί από το oddspapi.io. Το κλειδί αποθηκεύεται μαζί με τα δεδομένα σου "
                   "και από εκεί και πέρα οι αποδόσεις ανανεώνονται μόνες τους (κάθε 6 ώρες, όταν ανοίγεις τη σελίδα «Αξία»).")
        key = st.text_input("API key", value=cfg.get("oddspapi_key", ""), type="password")
        seen = sorted({b for b in O.latest(con).query("source == 'oddspapi'")["bookmaker"].unique()} if not O.latest(con).empty else set())
        mine = O.my_books(cfg)
        opts = list(dict.fromkeys(mine + O.SUGGESTED_BOOKS + seen))
        new_mine = st.multiselect(
            "Εταιρείες που παίζω (κενό = όλες)", opts, default=mine, key="my_books_sel", placeholder="Όλες",
            format_func=lambda b: b + ("" if not seen or b in seen else " (δεν έρχεται από το API)"),
            help="Μόνο οι αποδόσεις αυτών μετράνε για «Αξίζει» και για το Αρχείο. Οι υπόλοιπες εταιρείες χρησιμοποιούνται "
                 "μόνο για να βγει η γνώμη της αγοράς. Ό,τι περνάς εσύ (screenshot, χειροκίνητα) μετράει πάντα.")
        if new_mine != mine and st.button("Αποθήκευση εταιρειών"):
            O.save_config({**cfg, "my_books": new_mine})
            st.session_state.pop("_board", None)
            st.rerun()
        books = st.text_input("Από ποιες εταιρείες να φέρνει (slugs, με κόμμα· κενό = όλες)", value=cfg.get("books", ""),
                              help="Καλύτερα κενό: όσο περισσότερες εταιρείες, τόσο πιο αξιόπιστη η γνώμη της αγοράς.")
        days = st.slider("Ημέρες μπροστά", 1, 10, 7)
        if st.button("Λήψη αποδόσεων", type="primary", disabled=not key):
            O.save_config({**cfg, "oddspapi_key": key, "books": books})
            bar = st.progress(0.0, "Αγορές…")
            try:
                res = O.fetch_oddspapi(key, books, days, lambda p, t: bar.progress(p, t))
            except Exception as e:  # noqa: BLE001
                bar.empty()
                st.error(f"Αποτυχία: {e}")
            else:
                bar.empty()
                n, unmatched = O.store(con, res["rows"], index, "oddspapi")
                st.success(f"Αποθηκεύτηκαν {n} τιμές από {res['n_upcoming']} επερχόμενους αγώνες.")
                st.write(f"Αγορές ειδικών παικτών μπάσκετ στο API: {res['n_prop_markets']} "
                         f"(αναγνωρίστηκαν {res['n_mapped_markets']}).")
                if res["n_prop_markets"] == 0 or not res["rows"]:
                    st.warning("Το API δεν έδωσε ειδικά παικτών για Euroleague με αυτές τις εταιρείες. "
                               "Δοκίμασε κενό πεδίο εταιρειών ή πέρασε τις αποδόσεις χειροκίνητα.")
                if res["unknown"]:
                    st.write("Αγορές που αγνοήθηκαν:", ", ".join(res["unknown"][:30]))
                if unmatched:
                    st.write("Παίκτες που δεν αντιστοιχίστηκαν:", ", ".join(unmatched[:40]))

    season = int(D["df"]["season"].max())
    pool = cur[cur["season"] == season].sort_values(["team", "player"])
    labels = {f"{r.player} · {r.team}": pid for pid, r in pool.iterrows()}
    with t_man:
        st.caption("Μία γραμμή ανά στοίχημα. Συμπλήρωσε Over, Under ή και τα δύο (με τα δύο αφαιρείται σωστά η γκανιότα).")
        empty = pd.DataFrame({"Παίκτης": pd.Series(dtype="str"), "Αγορά": pd.Series(dtype="str"),
                              "Γραμμή": pd.Series(dtype="float"), "Over": pd.Series(dtype="float"),
                              "Under": pd.Series(dtype="float"), "Εταιρεία": pd.Series(dtype="str")})
        ed = st.data_editor(empty, num_rows="dynamic", width="stretch", key="manual", column_config={
            "Παίκτης": st.column_config.SelectboxColumn(options=list(labels), required=True, width="large"),
            "Αγορά": st.column_config.SelectboxColumn(options=list(STAT_LABEL.values()), required=True),
            "Γραμμή": st.column_config.NumberColumn(step=0.5, min_value=0.0, required=True),
            "Over": st.column_config.NumberColumn(min_value=1.01, format="%.2f"),
            "Under": st.column_config.NumberColumn(min_value=1.01, format="%.2f")})
        if st.button("Αποθήκευση", key="save_manual", disabled=ed.dropna(how="all").empty):
            inv = {v: k for k, v in STAT_LABEL.items()}
            w = pd.DataFrame({"player": ed["Παίκτης"], "player_id": ed["Παίκτης"].map(labels),
                              "stat": ed["Αγορά"].map(inv), "line": ed["Γραμμή"], "over": ed["Over"],
                              "under": ed["Under"], "bookmaker": ed["Εταιρεία"]})
            n, _ = O.store(con, O.wide_to_rows(w.dropna(subset=["player", "stat", "line"])), index, "manual")
            st.success(f"Αποθηκεύτηκαν {n} τιμές.")

    with t_csv:
        st.caption("Στήλες: player, stat, line, over, under, bookmaker. "
                   f"Τιμές stat: {', '.join(M.KEYS)}. Τα ονόματα ταιριάζονται αυτόματα (π.χ. «Sasha Vezenkov»).")
        txt = st.text_area("CSV", height=160, placeholder="player,stat,line,over,under,bookmaker\nSasha Vezenkov,pts,17.5,1.85,1.90,bet365")
        if st.button("Εισαγωγή", disabled=not txt.strip()):
            try:
                w = pd.read_csv(io.StringIO(txt), sep=None, engine="python")
                w.columns = [c.strip().lower() for c in w.columns]
                w["stat"] = w["stat"].astype(str).str.strip().str.lower()
                bad = sorted(set(w["stat"]) - set(M.KEYS))
                n, unmatched = O.store(con, O.wide_to_rows(w[w["stat"].isin(M.KEYS)]), index, "csv")
                st.success(f"Αποθηκεύτηκαν {n} τιμές.")
                if bad:
                    st.warning("Άγνωστες αγορές: " + ", ".join(bad))
                if unmatched:
                    st.warning("Δεν βρέθηκαν οι παίκτες: " + ", ".join(unmatched))
            except Exception as e:  # noqa: BLE001
                st.error(f"Δεν διαβάστηκε το CSV: {e}")

    st.divider()
    lat = O.latest(con)
    st.subheader(f"Ενεργές αποδόσεις ({len(lat)})")
    st.caption("Μετράνε όσες πέρασαν τις τελευταίες 72 ώρες· κρατιέται η πιο πρόσφατη τιμή ανά εταιρεία.")
    if len(lat):
        lat = lat.assign(Παίκτης=lat["player_id"].map(cur["player"]), Αγορά=lat["stat"].map(STAT_LABEL),
                         Ώρα=lat["fetched_at"].dt.tz_convert("Europe/Athens").dt.strftime("%d/%m %H:%M"))
        nxt_ = B.next_games(con)
        lat["Ομάδα"] = lat["player_id"].map(cur["team"])
        lat["Αντίπαλος"] = lat["Ομάδα"].map(nxt_["opp"]) if len(nxt_) else None
        lat["_game"] = [game_label(nxt_.at[t, "date"], t, o) if isinstance(t, str) and t in nxt_.index else "—"
                        for t, o in zip(lat["Ομάδα"], lat["Αντίπαλος"])]
        lat["_min"] = lat["player_id"].map(cur["min_hat"])
        # πιθανότητα και κέρδος του μοντέλου στη συγκεκριμένη τιμή (από τον πίνακα αξίας, όπου υπάρχει)
        _, bb_ = board_for(D, nxt_, 0.5)
        if len(bb_):
            pw = bb_.set_index(["player_id", "stat", "Γραμμή", "side"])["P τελική"]
            pw = pw[~pw.index.duplicated()]
            lat["_p"] = [pw.get((p_, s_, float(l_), sd_), np.nan) for p_, s_, l_, sd_ in
                         zip(lat["player_id"], lat["stat"], lat["line"], lat["side"])]
            lat["_ev"] = lat["_p"] * lat["price"] - 1
        else:
            lat["_p"] = lat["_ev"] = np.nan
        lat = study_filters(lat, D, "of", price="price", prob="_p", ev="_ev", team="Ομάδα", opp="Αντίπαλος",
                            game="_game", pid="player_id", minutes="_min", player="Παίκτης")
        if is_mobile():
            m_ = pd.DataFrame({"Παίκτης": [short(n) for n in lat["Παίκτης"].fillna("—")],
                               "Στοίχημα": [f"{'O' if sd == 'over' else 'U'} {ln:g} {STAT_LABEL.get(stt, stt)}"
                                            for sd, ln, stt in zip(lat["side"], lat["line"], lat["stat"])],
                               "Απόδ.": lat["price"].to_numpy(float)})
            st.dataframe(m_, hide_index=True, width="stretch", column_config={
                "Παίκτης": st.column_config.TextColumn(width=110), "Απόδ.": st.column_config.NumberColumn(format="%.2f", width=56),
                "Στοίχημα": st.column_config.TextColumn(help="O = Over, U = Under")})
        else:
            st.dataframe(lat[["Παίκτης", "Ομάδα", "Αντίπαλος", "Αγορά", "line", "side", "price", "_p", "_ev", "bookmaker", "source", "Ώρα"]].rename(
                columns={"line": "Γραμμή", "side": "Πλευρά", "price": "Απόδοση", "bookmaker": "Εταιρεία",
                         "source": "Πηγή"}).assign(_p=lambda x: x["_p"] * 100, _ev=lambda x: x["_ev"] * 100),
                hide_index=True, width="stretch", column_config={
                    "_p": st.column_config.NumberColumn("Πιθανότητα", format="%.0f%%", help=GLOSS["Πιθανότητα"]),
                    "_ev": st.column_config.NumberColumn("Αναμ. κέρδος", format="%+.1f%%",
                                                         help=GLOSS["Αναμενόμενο κέρδος"] + " Σε αυτή την τιμή.")})
        if st.button("Διαγραφή όλων των αποδόσεων"):
            con.execute("DELETE FROM odds")
            con.commit()
            st.rerun()


def page_data(D):
    st.title("Δεδομένα")
    if D is not None and st.button("← Πίσω", key="data_back"):
        _leave_data()
        st.rerun()
    n_games, n_played = con.execute("SELECT COUNT(*), COALESCE(SUM(played),0) FROM games").fetchone()
    n_box = con.execute("SELECT COUNT(DISTINCT season||'-'||game_code) FROM box").fetchone()[0]
    last = con.execute("SELECT MAX(g.date) FROM box b JOIN games g USING(season, game_code)").fetchone()[0]
    st.markdown('<div class="el-sub">Τα στατιστικά κάθε αγώνα, από τα οποία μαθαίνει το μοντέλο.</div>', unsafe_allow_html=True)
    tiles(st, [
        ("Αγώνες στο πρόγραμμα", str(n_games), "Όλοι οι αγώνες των σεζόν που έχεις κατεβάσει, παιγμένοι και μη."),
        ("Παιγμένοι", str(n_played), "Όσοι έχουν ολοκληρωθεί σύμφωνα με την τελευταία ενημέρωση."),
        ("Με στατιστικά", str(n_box), "Όσοι έχουν περαστεί στη βάση. Πρέπει να είναι ίσοι με τους παιγμένους.",
         V_COLOR["strong"] if n_box == n_played else V_COLOR["marginal"]),
        ("Τελευταίος αγώνας", pd.to_datetime(last).strftime("%d/%m/%Y") if last else "—", "Ο πιο πρόσφατος αγώνας που έχει το μοντέλο."),
        ("Θέσεις από ρόστερ", str(con.execute("SELECT COUNT(DISTINCT player_id) FROM players WHERE position IS NOT NULL").fetchone()[0]),
         "Παίκτες με επίσημη θέση (γκαρντ, φόργουορντ, σέντερ). Για τους υπόλοιπους η θέση εκτιμάται από τα στατιστικά."),
    ])
    today = pd.Timestamp.now()
    cur_season = today.year if today.month >= 9 else today.year - 1
    seasons = st.multiselect("Σεζόν (έτος έναρξης)", list(range(cur_season - 4, cur_season + 1)),
                             default=[cur_season - 1, cur_season])
    st.caption("Πηγή: τα δημόσια feeds στατιστικών της Euroleague. Κατεβαίνουν μόνο όσα boxscores λείπουν· "
               "η πρώτη φορά για μια ολόκληρη σεζόν παίρνει 5–6 λεπτά. Αν το API κόψει τα αιτήματα (429), περίμενε λίγο και ξαναπάτα.")
    if st.button("Ενημέρωση στατιστικών", type="primary", disabled=not seasons):
        bar = st.progress(0.0, "Πρόγραμμα αγώνων…")
        res = E.update(con, sorted(seasons), lambda p, t: bar.progress(p, t))
        bar.empty()
        msg = f"Νέα boxscores: {res['new_box']} · λείπουν ακόμα: {res['missing']}"
        (st.success if not res["missing"] else st.info)(msg)
        for w in res["warnings"]:
            st.warning(w)
        if res["errors"]:
            st.error("Σφάλματα:\n\n" + "\n\n".join(res["errors"][:10]))
        if SYNC:
            SYNC.maybe_push(force=True)
        st.button("Συνέχεια")

    heading(st, "Αντίγραφο ασφαλείας", "Όλα τα δεδομένα σε ένα αρχείο: στατιστικά, αποδόσεις, Ημερολόγιο, Αρχείο. "
            "Κατέβασέ το για να το κρατήσεις ή για να μεταφέρεις τα δεδομένα από τη μία εγκατάσταση στην άλλη."
            + (" Στο cloud αποθηκεύονται έτσι κι αλλιώς μόνα τους, με αντίγραφα των τελευταίων 7 ημερών." if CLOUD else ""))
    b1, b2 = st.columns(2)
    if b1.button("Ετοίμασε αντίγραφο"):
        st.session_state["db_dump"] = CL.snapshot(con)
    if st.session_state.get("db_dump"):
        b1.download_button("Κατέβασμα αρχείου", st.session_state["db_dump"], type="primary",
                           file_name=f"euroleague-props-{pd.Timestamp.now():%Y%m%d-%H%M}.sqlite.gz",
                           mime="application/gzip")
    up_ = b2.file_uploader("Επαναφορά από αρχείο", type=["gz", "sqlite"],
                           help="ΑΝΤΙΚΑΘΙΣΤΑ όλα τα τωρινά δεδομένα με αυτά του αρχείου.")
    if up_ is not None and b2.button("Αντικατάσταση δεδομένων με το αρχείο", type="primary"):
        try:
            CL.restore_into(con, up_.getvalue())
            _schema(con)
            O.bind(con)
            con.commit()
            st.session_state.pop("_board", None)
            if SYNC:
                SYNC.saved_changes = -1          # η επαναφορά δεν μετράει στο total_changes
                SYNC.maybe_push(force=True)
            st.success("Τα δεδομένα αντικαταστάθηκαν.")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Η επαναφορά απέτυχε: {e}")


def page_model(D, sig, embedded: bool = False):
    if not embedded:
        st.title("Έλεγχος μοντέλου")
    st.markdown('<div class="el-sub">Μπορώ να εμπιστευτώ τις πιθανότητες; Για κάθε παλιό αγώνα το μοντέλο προβλέπει '
                'χρησιμοποιώντας μόνο ό,τι είχε προηγηθεί, και μετά βλέπουμε τι έγινε στην πραγματικότητα.</div>',
                unsafe_allow_html=True)
    if D["preds"].empty:
        st.info("Χρειάζονται περισσότεροι αγώνες.")
        return
    bt = compute_bt(sig)
    f1, f2 = st.columns(2)
    stat = f1.selectbox("Αγορά", ["ΣΥΝΟΛΟ"] + M.KEYS, format_func=lambda k: STAT_LABEL.get(k, "Όλες οι αγορές"))
    grp = f2.selectbox("Παίκτες", ["Όλοι", "Βασικός ρόλος (18+ λεπτά)", "Ρόλος αλλαγής (κάτω από 18)"],
                       help="Οι εταιρείες δίνουν γραμμές κυρίως για όσους παίζουν πολύ.")
    bt = bt if grp == "Όλοι" else bt[bt["min_hat"] >= 18] if grp.startswith("Βασικός") else bt[bt["min_hat"] < 18]
    d = bt if stat == "ΣΥΝΟΛΟ" else bt[bt["stat"] == stat]
    cal = pd.concat([M.calibration(d, "p_model").assign(Μέθοδος="Μοντέλο"),
                     M.calibration(d, "p_emp").assign(Μέθοδος="Ποσοστό τελευταίων 10")])
    cal = cal[cal["n"] >= 30]
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color=MUTED).encode(x="x", y="y")
    pts = alt.Chart(cal).mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=2).encode(
        x=alt.X("pred:Q", title="Προβλεπόμενη πιθανότητα", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("obs:Q", title="Πραγματική συχνότητα", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Μέθοδος:N", scale=alt.Scale(domain=["Μοντέλο", "Ποσοστό τελευταίων 10"],
                                                     range=[BLUE, ORANGE]), legend=None),
        shape=alt.Shape("Μέθοδος:N", legend=None),
        tooltip=["Μέθοδος", alt.Tooltip("pred:Q", format=".1%", title="Πρόβλεψη"),
                 alt.Tooltip("obs:Q", format=".1%", title="Πραγματικό"), alt.Tooltip("n:Q", title="Δείγμα")])
    a, b = st.columns([3, 2], gap="large")
    heading(a, "Όταν λέει 60%, βγαίνει 60%;",
            sw(BLUE, "το μοντέλο") + sw(ORANGE, "το σκέτο ποσοστό των 10 τελευταίων αγώνων") + sw(MUTED, "το ιδανικό", "ln dash")
            + "<br>Οριζόντια: η πιθανότητα που προέβλεψε. Κάθετα: πόσο συχνά συνέβη. Όσο πιο κοντά στη διαγώνιο, "
              "τόσο πιο αξιόπιστες οι πιθανότητες. Κάτω από τη διαγώνιο σημαίνει υπερβολική αισιοδοξία.")
    show_chart(a, (diag + pts).properties(height=380))
    sc = M.score_backtest(bt)
    sc["stat"] = sc["stat"].map(lambda k: STAT_LABEL.get(k, k))
    heading(b, "Ποια μέθοδος προβλέπει καλύτερα",
            "Δείκτης Brier: το μέσο τετραγωνικό σφάλμα της πιθανότητας. Μικρότερο είναι καλύτερο. "
            "Το 0.25 αντιστοιχεί σε κορόνα-γράμματα.")
    b.dataframe(sc.rename(columns={"stat": "Αγορά", "n": "Δείγμα", "brier_model": "Μοντέλο",
                                   "brier_hitrate": "Ποσοστό L10", "brier_const": "Σταθερό"}),
                hide_index=True, width="stretch",
                column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ("Μοντέλο", "Ποσοστό L10", "Σταθερό")})
    b.markdown('<div class="el-leg"><b>Μοντέλο</b>: οι πιθανότητες της εφαρμογής. <b>Ποσοστό L10</b>: πόσες φορές '
               'το έκανε στους 10 τελευταίους. <b>Σταθερό</b>: πάντα το ίδιο μέσο ποσοστό, χωρίς καμία γνώση. '
               'Αν το L10 δεν κερδίζει το Σταθερό, το «τι έκανε τελευταία» δεν προβλέπει τίποτα από μόνο του.</div>',
               unsafe_allow_html=True)
    with st.expander("Τεχνικές παράμετροι του μοντέλου"):
        a_, b_ = D["ctx"]["min_cal"]
        st.caption(f"Var = μ + μ²/k, με 1/k = a + c/λεπτά². Λεπτά: {a_:.2f} + {b_:.3f} × πρόσφατος μέσος όρος.")
        st.dataframe(pd.DataFrame({
            "Αγορά": [STAT_LABEL[k] for k in M.KEYS],
            "k στα 15′": [float(M.k_of(D["disp"][k], 15)) for k in M.KEYS],
            "k στα 25′": [float(M.k_of(D["disp"][k], 25)) for k in M.KEYS],
            "Εντός/εκτός": [f"{D['ctx']['home'][0][i]:.3f} / {D['ctx']['home'][1][i]:.3f}" for i in range(len(M.KEYS))],
        }), hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.1f") for c in ("k στα 15′", "k στα 25′")})
    st.warning("Η βαθμονόμηση δείχνει ότι οι πιθανότητες είναι ειλικρινείς, όχι ότι κερδίζουν την αγορά. "
               "Αυτό ελέγχεται μόνο κρατώντας αρχείο με τις αποδόσεις που πήρες και τα αποτελέσματα.")


# ------------------------------------------------------------------ άμυνες ανά θέση

TEAM_SHORT = {"RED": "Crvena Zvezda", "DUB": "Dubai", "HTA": "Hapoel TA", "BAR": "Barcelona", "BAS": "Baskonia",
              "ASV": "ASVEL", "PAN": "Panathinaikos", "BES": "Besiktas", "ULK": "Fenerbahce", "PAR": "Partizan",
              "IST": "Efes", "ZAL": "Zalgiris", "PAM": "Valencia", "MIL": "Olimpia Milano", "PRS": "Paris",
              "TEL": "Maccabi TA", "VIR": "Virtus", "MUN": "Bayern", "MAD": "Real Madrid", "OLY": "Olympiacos",
              "MCO": "Monaco", "BER": "Alba Berlin"}
DEF_STAT_SHORT = {"pts": "Πόντοι", "reb": "Ριμπ.", "ast": "Ασίστ", "fg3m": "Τρίπ.", "fg2m": "Δίπ.",
                  "stl": "Κλεψ.", "blk": "Κοψ.", "tov": "Λάθη", "pra": "Π+Ρ+Α"}
POS4 = {0: "Γκαρντ", 1: "Φόργουορντ", 2: "Σέντερ", 3: "Σύνολο"}
DIV_MID = "#283247"     # ουδέτερο κέντρο της αποκλίνουσας κλίμακας, λίγο πιο ανοιχτό από την επιφάνεια
GLOSS.update({
    "Διορθωμένο για τύχη": "Με λίγους αγώνες ο μέσος όρος μιας ομάδας είναι κυρίως τύχη. Η διορθωμένη τιμή τον τραβάει "
                           "προς τον μέσο όρο της λίγκας, τόσο περισσότερο όσο λιγότερα και πιο θορυβώδη είναι τα δεδομένα.",
    "Ανά 100 κατοχές": "Διορθώνει για τον ρυθμό: μια γρήγορη ομάδα δίνει περισσότερα σε όλους επειδή γίνονται περισσότερες "
                       "επιθέσεις, όχι επειδή αμύνεται χειρότερα.",
    "Πέρα από τη συνολική άμυνα": "Πόσο περισσότερο δίνει στη θέση από όσο εξηγεί η συνολική της άμυνα. Το μοντέλο ήδη "
                                  "ξέρει τη συνολική άμυνα· αυτό το κομμάτι δεν το ξέρει.",
})


def pctf(x: float) -> str:
    """+12% / −3% / 0% (χωρίς «-0%»)."""
    return "0%" if abs(x) < 0.005 else f"{x:+.0%}".replace("-", "−")


def team_short(code: str) -> str:
    return TEAM_SHORT.get(code, code)


@st.cache_resource(show_spinner="Υπολογισμός αμυνών…")
def compute_def(sig):
    from elprops import defense as DF
    D_ = compute(sig)
    G = DF.game_table(D_["df"], D_["ctx"]["roles"])
    units = {u: DF.per_unit(G, u) for u in ("game", "poss100")}
    return {"G": units, "n0": {u: DF.estimate_n0(units[u]) for u in units},
            "pers": DF.persistence(units["game"])}


def _div_color(field: str, lim: float = 0.2):
    return alt.Color(f"{field}:Q", scale=alt.Scale(domain=[-lim, 0, lim], range=[ORANGE, DIV_MID, BLUE], clamp=True,
                                                         interpolate="lab"),
                     legend=None)


def _def_tooltip(unit_lbl):
    return [alt.Tooltip("Ομάδα:N"), alt.Tooltip("Θέση:N"), alt.Tooltip("Στατιστικό:N"),
            alt.Tooltip("raw:Q", format=".1f", title=f"Δέχεται {unit_lbl}"),
            alt.Tooltip("league:Q", format=".1f", title="Μέσος όρος λίγκας"),
            alt.Tooltip("pct_raw:Q", format="+.0%", title="Όπως έγινε"),
            alt.Tooltip("pct:Q", format="+.0%", title="Διορθωμένο για τύχη"),
            alt.Tooltip("rank_txt:N", title="Κατάταξη"), alt.Tooltip("n:Q", title="Αγώνες"),
            alt.Tooltip("rel:Q", format=".0%", title="Πόσο μετράνε τα δεδομένα της")]


def _def_frame(T, col):
    d = T.copy()
    d["Ομάδα"] = d["team"].map(team_short)
    d["Θέση"] = d["pos"].map(POS4)
    d["Στατιστικό"] = d["stat"].map(DEF_STAT_SHORT)
    d["c"] = d[col]
    d["rank_txt"] = [f"{r}η από {n} σε όσα δίνει" for r, n in zip(d["rank"], d["n_teams"])]
    return d


def def_heatmap(target, T, rows_field, cols_field, row_order, col_order, col, unit_lbl, text="raw", height=None,
                label_limit=120):
    d = _def_frame(T, col)
    d["txt"] = [pctf(v) if text == "pct" else (f"{v:.1f}" if v < 100 else f"{v:.0f}")
                for v in (d[col] if text == "pct" else d["raw"])]
    base = alt.Chart(d).encode(
        x=alt.X(f"{cols_field}:N", sort=col_order, title=None, axis=alt.Axis(orient="top", labelAngle=0, labelLimit=90)),
        y=alt.Y(f"{rows_field}:N", sort=row_order, title=None, axis=alt.Axis(labelLimit=label_limit)))
    rect = base.mark_rect(cornerRadius=3, stroke=BG, strokeWidth=2).encode(color=_div_color("c"),
                                                                           tooltip=_def_tooltip(unit_lbl))
    txt = base.mark_text(fontSize=11, color=TEXT, font="Commissioner").encode(text="txt:N", tooltip=_def_tooltip(unit_lbl))
    show_chart(target, (rect + txt).properties(height=height or 26 * len(row_order) + 30))


def def_legend(col_mode):
    return (sw(BLUE, "δίνει περισσότερα από τον μέσο όρο (καλό για Over)") + sw(ORANGE, "δίνει λιγότερα (καλό για Under)")
            + sw(DIV_MID, "κοντά στον μέσο όρο") + "<br>Ο αριθμός στο κελί είναι τι δέχεται πραγματικά. Το χρώμα είναι το "
            + ("ποσοστό έναντι της λίγκας <b>διορθωμένο για τύχη</b>, ώστε 3–4 καλά παιχνίδια να μη φαίνονται τάση."
               if col_mode == "pct" else "ποσοστό έναντι της λίγκας <b>όπως έγινε</b>, χωρίς διόρθωση: με λίγους αγώνες είναι κυρίως θόρυβος."))


def page_defense(D, nxt, w_model):
    from elprops import defense as DF
    st.title("Άμυνες ανά θέση")
    st.markdown('<div class="el-sub">Τι δίνει κάθε ομάδα στους γκαρντ, στους φόργουορντ και στους σέντερ των αντιπάλων. '
                'Για κάθε αγώνα αθροίζεται τι έκαναν οι αντίπαλοι κάθε θέσης και συγκρίνεται με τον μέσο όρο της λίγκας.</div>',
                unsafe_allow_html=True)
    X = compute_def(sig)
    df, cur, ctx = D["df"], D["current"], D["ctx"]
    seasons = sorted(int(s) for s in df["season"].unique())
    s_now = seasons[-1]
    lbl = lambda s_: f"{s_}-{str(s_ + 1)[-2:]}"  # noqa: E731
    periods = {"now": ([s_now], f"Φέτος ({lbl(s_now)})")}
    if len(seasons) > 1:
        periods["both"] = ([seasons[-2], s_now], f"Φέτος + πέρσι")
        periods["prev"] = ([seasons[-2]], f"Πέρσι ({lbl(seasons[-2])})")
    mob = is_mobile()
    c = st.columns([2, 2, 2, 2]) if not mob else st.columns(2)
    per = c[0].selectbox("Περίοδος", list(periods), format_func=lambda k: periods[k][1], key="def_per")
    stat = c[1].selectbox("Στατιστικό", DF.DSTATS, format_func=STAT_LABEL.get, key="def_stat")
    c2 = c if not mob else st.expander("Μονάδα και χρώμα").columns(2)
    unit = c2[2 if not mob else 0].selectbox("Μονάδα", ["game", "poss100"], key="def_unit",
                                             format_func={"game": "Ανά αγώνα", "poss100": "Ανά 100 κατοχές"}.get,
                                             help=GLOSS["Ανά 100 κατοχές"])
    col_mode = c2[3 if not mob else 1].selectbox("Χρώμα", ["pct", "pct_raw"], key="def_col",
                                                 format_func={"pct": "Διορθωμένο για τύχη", "pct_raw": "Όπως έγινε"}.get,
                                                 help=GLOSS["Διορθωμένο για τύχη"])
    unit_lbl = "ανά αγώνα" if unit == "game" else "ανά 100 κατοχές"
    G, n0 = X["G"][unit], X["n0"][unit]
    T = DF.table(G, periods[per][0], n0)
    if T.empty:
        st.info("Δεν υπάρχουν αγώνες για αυτή την περίοδο.")
        return
    n_med = int(T[T["pos"] == DF.ALL]["n"].median())
    official = len(ctx.get("official", set()))
    note = []
    if n_med < 12:
        note.append(f"Κάθε ομάδα έχει μόνο ~{n_med} αγώνες σε αυτή την περίοδο. Με τόσο λίγους, τα χρώματα «διορθωμένο για τύχη» "
                    "μένουν σχεδόν ουδέτερα: δεν είναι λάθος, απλώς {n} αγώνες δεν λένε σχεδόν τίποτα για άμυνα ανά θέση. "
                    "Δοκίμασε «Φέτος + πέρσι».".replace("{n}", str(n_med)))
    if not official:
        note.append("Οι θέσεις είναι εκτίμηση από τα στατιστικά, γιατί δεν έχουν κατέβει ακόμα τα ρόστερ. "
                    "Κάνε «Ενημέρωση» στα «Δεδομένα και αντίγραφα» (στήλη αριστερά) για τις επίσημες.")
    for n_ in note:
        st.caption("⚠️ " + n_)

    t_map, t_match, t_team, t_real = st.tabs(["Χάρτης", "Ματσάπ αγωνιστικής", "Προφίλ ομάδας", "Έχει ουσία;"])

    # ---------------- χάρτης
    with t_map:
        pos = st.segmented_control("Θέση", [0, 1, 2], format_func=M.ROLE_NAMES.get, default=0, key="def_pos_map") or 0
        sub = T[T["pos"] == pos]
        order = list(sub[sub["stat"] == stat].sort_values(col_mode, ascending=False)["team"].map(team_short))
        top3 = ", ".join(order[:3])
        bot3 = ", ".join(order[-3:][::-1])
        spread = float(sub[sub["stat"] == stat][col_mode].abs().max())
        tip = (f"Η μεγαλύτερη απόκλιση είναι {pctf(spread)}: πρακτικά όλες ίδιες." if spread < 0.03
               else f"{STAT_LABEL[stat]} {unit_lbl}")
        lg_v = float(sub[sub["stat"] == stat]["league"].iloc[0])
        tiles(st, [
            (f"Δίνουν τα περισσότερα στους {M.ROLE_PLURAL[pos]}", top3, tip, BLUE),
            ("Δίνουν τα λιγότερα", bot3, tip, ORANGE),
        ] + ([] if mob else [("Μέσος όρος λίγκας", f"{lg_v:.1f}",
                              f"{STAT_LABEL[stat]} που δίνει μια ομάδα στους {M.ROLE_PLURAL[pos]} {unit_lbl}")]))
        if mob:
            heading(st, f"{STAT_LABEL[stat]}: τι δίνει κάθε ομάδα ανά θέση",
                    f"Ταξινόμηση με βάση τους {M.ROLE_PLURAL[pos]}. Μέσος όρος λίγκας: {lg_v:.1f}. " + def_legend(col_mode))
            def_heatmap(st, T[T["stat"] == stat], "Ομάδα", "Θέση", order, list(POS4.values()), col_mode, unit_lbl,
                        label_limit=95)
        else:
            a, b = st.columns([5, 3], gap="large")
            heading(a, f"{M.ROLE_NAMES[pos]}: όλα τα στατιστικά",
                    f"Ταξινόμηση: όποια δίνει τα περισσότερα «{STAT_LABEL[stat]}» στους {M.ROLE_PLURAL[pos]} πάνω. "
                    + def_legend(col_mode))
            def_heatmap(a, sub, "Ομάδα", "Στατιστικό", order, list(DEF_STAT_SHORT.values()), col_mode, unit_lbl)
            heading(b, f"{STAT_LABEL[stat]}: σύγκριση θέσεων",
                    "Ίδια σειρά ομάδων. Το «Σύνολο» είναι η συνολική άμυνα, που ήδη ξέρει το μοντέλο. "
                    "Ενδιαφέρον έχει όταν μια θέση ξεφεύγει από το σύνολο.")
            def_heatmap(b, T[T["stat"] == stat], "Ομάδα", "Θέση", order, list(POS4.values()), col_mode, unit_lbl)
        with st.expander("Υπόμνημα"):
            st.markdown(gloss_html(["Διορθωμένο για τύχη", "Ανά 100 κατοχές", "Πέρα από τη συνολική άμυνα"]),
                        unsafe_allow_html=True)

    # ---------------- ματσάπ
    with t_match:
        up = upcoming_games(days=14)
        if up.empty:
            st.info("Δεν υπάρχουν προγραμματισμένοι αγώνες στις επόμενες δύο εβδομάδες.")
        else:
            rnd = int(up["round"].min())
            up = up[up["round"] == rnd]
            f1, f2 = st.columns(2)
            poss_f = f1.segmented_control("Θέσεις", [0, 1, 2], format_func=M.ROLE_NAMES.get, selection_mode="multi",
                                          default=[0, 1, 2], key="def_pos_match") or [0, 1, 2]
            min_m = f2.slider("Ελάχιστα αναμενόμενα λεπτά", 0, 30, 15, key="def_minm")
            absent = ctx.get("absent", set())
            _, bb = board_for(D, nxt, w_model)
            best_over = {}
            if not bb.empty:
                ov = bb[(bb["stat"] == stat) & (bb["side"] == "over")].sort_values("EV", ascending=False)
                best_over = {r["player_id"]: r for r in ov.drop_duplicates("player_id").to_dict("records")}
            Ti = T.set_index(["team", "pos", "stat"])
            rows = []
            for g_ in up.itertuples(index=False):
                for team, opp in ((g_.home_code, g_.away_code), (g_.away_code, g_.home_code)):
                    pool = cur[(cur["team"] == team) & (cur["season"] == s_now) & (cur["min_hat"] >= min_m)]
                    for pid in pool.index:
                        p_ = pos_of(D, pid)
                        if pid in absent or p_ not in poss_f or (opp, p_, stat) not in Ti.index:
                            continue
                        t_ = Ti.loc[(opp, p_, stat)]
                        mu = B.projection(pid, stat, cur, ctx, nxt)[0]
                        o_ = best_over.get(pid)
                        rows.append({"player_id": pid, "Παίκτης": pool.at[pid, "player"].title(), "short": short(pool.at[pid, "player"]),
                                     "Θέση": pos_name(D, pid), "Ομάδα": team_short(team), "Αντίπαλος": team_short(opp),
                                     "Δίνει": float(t_[col_mode]) * 100, "Κατάταξη": f"{int(t_['rank'])}/{int(t_['n_teams'])}",
                                     "rank": int(t_["rank"]), "Πέρα": float(t_["extra"]) * 100,
                                     "Πρόβλεψη": mu, "Λεπτά": float(pool.at[pid, "min_hat"]),
                                     "Γραμμή": float(o_["Γραμμή"]) if o_ else np.nan,
                                     "Απόδοση Over": float(o_["Απόδοση"]) if o_ else np.nan,
                                     "EV Over": float(o_["EV"]) * 100 if o_ else np.nan,
                                     "verdict": o_["verdict"] if o_ else None})
            R = pd.DataFrame(rows)
            heading(st, f"Αγωνιστική {rnd}: ποιοι παίζουν απέναντι σε άμυνα που δίνει «{STAT_LABEL[stat]}» στη θέση τους",
                    "Ταξινόμηση από την πιο «φιλική» άμυνα. Η <b>Πρόβλεψη</b> του μοντέλου ήδη περιέχει τη συνολική άμυνα του αντιπάλου, "
                    "όχι όμως τη θέση: το «Πέρα από τη συνολική» δείχνει τι μένει απ' έξω.")
            if R.empty:
                st.caption("Κανένας παίκτης με αυτά τα φίλτρα.")
            else:
                R = R.sort_values(["Δίνει", "Πρόβλεψη"], ascending=[False, False]).reset_index(drop=True)
                if mob:
                    cards = ""
                    for r in R.head(30).to_dict("records"):
                        vc = BLUE if r["Δίνει"] > 0 else ORANGE
                        ev_ = (f'<span><i>Over {r["Γραμμή"]:g} @ {r["Απόδοση Over"]:.2f}</i><b style="color:{V_COLOR.get(r["verdict"], MUTED)}">'
                               f'{r["EV Over"]:+.1f}%</b></span>') if not np.isnan(r["Γραμμή"]) else ""
                        cards += (f'<div class="el-bet" style="--vc:{vc}"><div class="top"><span class="who">{esc(r["short"])} '
                                  f'<span style="color:var(--muted);font-weight:400">{esc(r["Θέση"])} · με {esc(r["Αντίπαλος"])}</span></span>'
                                  f'<span class="chip">{pctf(r["Δίνει"] / 100)}</span></div>'
                                  f'<div class="nums"><span><i>Κατάταξη</i><b>{esc(r["Κατάταξη"])}</b></span>'
                                  f'<span><i>Πρόβλεψη</i><b>{r["Πρόβλεψη"]:.1f}</b></span>'
                                  f'<span><i>Πέρα από συνολ.</i><b>{pctf(r["Πέρα"] / 100)}</b></span>{ev_}</div></div>')
                    st.markdown(cards, unsafe_allow_html=True)
                else:
                    has_odds = R["Γραμμή"].notna().any()
                    show = R[["Παίκτης", "Θέση", "Ομάδα", "Αντίπαλος", "Δίνει", "Κατάταξη", "Πέρα", "Λεπτά", "Πρόβλεψη"]
                             + (["Γραμμή", "Απόδοση Over", "EV Over"] if has_odds else [])]
                    st.dataframe(show, hide_index=True, width="stretch", height=min(38 * (len(show) + 1) + 2, 640),
                                 column_config={
                                     "Δίνει": st.column_config.NumberColumn("Ο αντίπαλος δίνει", format="%+.0f%%",
                                                                           help="Πόσο πάνω ή κάτω από τον μέσο όρο της λίγκας δέχεται ο αντίπαλος "
                                                                                "σε αυτό το στατιστικό από παίκτες αυτής της θέσης."),
                                     "Κατάταξη": st.column_config.TextColumn(help="1 = η άμυνα που δίνει τα περισσότερα στη θέση."),
                                     "Πέρα": st.column_config.NumberColumn("Πέρα από τη συνολική", format="%+.0f%%",
                                                                          help=GLOSS["Πέρα από τη συνολική άμυνα"]),
                                     "Λεπτά": st.column_config.NumberColumn(format="%.0f", help=GLOSS["Αναμενόμενα λεπτά"]),
                                     "Πρόβλεψη": st.column_config.NumberColumn(format="%.1f", help=GLOSS["Πρόβλεψη"]),
                                     "Γραμμή": st.column_config.NumberColumn(format="%.1f", help="Η γραμμή Over με το καλύτερο αναμενόμενο κέρδος από όσες πέρασες."),
                                     "Απόδοση Over": st.column_config.NumberColumn(format="%.2f"),
                                     "EV Over": st.column_config.NumberColumn("Αναμ. κέρδος", format="%+.1f%%", help=GLOSS["Αναμενόμενο κέρδος"]),
                                 })
                st.caption("Οι στήλες γραμμής και απόδοσης εμφανίζονται όταν έχεις περάσει αποδόσεις για την αγωνιστική. "
                           "Για ανάλυση κάποιου, άνοιξέ τον στη σελίδα «Παίκτης».")

    # ---------------- προφίλ ομάδας
    with t_team:
        teams = sorted(T["team"].unique(), key=team_short)
        a1, a2 = st.columns(2)
        team = a1.selectbox("Ομάδα", teams, format_func=team_short, key="def_team")
        pos_t = a2.segmented_control("Θέση αντιπάλων", [0, 1, 2, 3], format_func=POS4.get, default=0, key="def_pos_team")
        pos_t = 0 if pos_t is None else pos_t
        r_ = T[(T["team"] == team) & (T["pos"] == pos_t) & (T["stat"] == stat)].iloc[0]
        who = f"στους {M.ROLE_PLURAL[pos_t]}" if pos_t != 3 else "συνολικά"
        col_v = BLUE if r_[col_mode] > 0.005 else ORANGE if r_[col_mode] < -0.005 else None
        tiles(st, [
            (f"Δίνει {who}", f"{r_['raw']:.1f}", f"{STAT_LABEL[stat]} {unit_lbl}, όπως έγινε"),
            ("Μέσος όρος λίγκας", f"{r_['league']:.1f}", f"{STAT_LABEL[stat]} {who} {unit_lbl}"),
            ("Έναντι λίγκας", pctf(r_[col_mode]), "διορθωμένο για τύχη" if col_mode == "pct" else "όπως έγινε, χωρίς διόρθωση", col_v),
            ("Κατάταξη", f"{int(r_['rank'])}<small>/ {int(r_['n_teams'])}</small>", "1 = δίνει τα περισσότερα στη λίγκα"),
            ("Αγώνες", f"{int(r_['n'])}", f"Τα δεδομένα της μετράνε {r_['rel']:.0%}, ο μέσος όρος της λίγκας τα υπόλοιπα."),
        ])
        heading(st, f"{team_short(team)}: τι δίνει σε κάθε θέση", def_legend(col_mode))
        def_heatmap(st, T[T["team"] == team], "Θέση", "Στατιστικό", list(POS4.values()), list(DEF_STAT_SHORT.values()),
                    col_mode, unit_lbl, height=4 * 34 + 30, label_limit=90)
        Gs = G[(G["team"] == team) & (G["pos"] == pos_t) & (G["season"].isin(periods[per][0]))].sort_values("date").copy()
        if len(Gs):
            Gs["Αγώνας"] = [f"{d:%d/%m} {team_short(v)}" for d, v in zip(Gs["date"].dt.tz_convert("Europe/Athens"), Gs["vs"])]
            Gs["Έδρα"] = np.where(Gs["is_home"], "Εντός", "Εκτός")
            Gs["v"] = Gs[stat]
            lg = float(r_["league"])
            heading(st, f"Αγώνα με τον αγώνα: {STAT_LABEL[stat]} που έδωσε {who}",
                    sw(POS_COLORS[pos_t] if pos_t < 3 else GREY, "κάθε αγώνας, ο πιο πρόσφατος δεξιά")
                    + sw(MUTED, f"μέσος όρος λίγκας ({lg:.1f})", "ln dash")
                    + "Δες πόσο πηδάει από αγώνα σε αγώνα: γι' αυτό λίγοι αγώνες δεν αρκούν.")
            bars = alt.Chart(Gs).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3,
                                          color=POS_COLORS[pos_t] if pos_t < 3 else GREY).encode(
                x=alt.X("Αγώνας:N", sort=list(Gs["Αγώνας"]), title=None,
                        axis=alt.Axis(labelAngle=-50, labelLimit=110, labels=len(Gs) <= 45)),
                y=alt.Y("v:Q", title=f"{STAT_LABEL[stat]} {unit_lbl}"),
                tooltip=["Αγώνας", "Έδρα", alt.Tooltip("round:Q", title="Αγωνιστική"),
                         alt.Tooltip("v:Q", format=".1f", title=STAT_LABEL[stat])])
            rule = alt.Chart(pd.DataFrame({"y": [lg]})).mark_rule(color=MUTED, strokeDash=[5, 4], strokeWidth=1.5).encode(y="y:Q")
            show_chart(st, (bars + rule).properties(height=260))
        H = DF.hurt_list(df, ctx["roles"], team, periods[per][0], pos_t, stat)
        if len(H):
            heading(st, "Ποιοι την «πόνεσαν»: οι καλύτερες εμφανίσεις "
                    + (f"αντίπαλων {M.ROLE_PLURAL[pos_t]}" if pos_t != 3 else "αντιπάλων") + " απέναντί της",
                    "Σύγκριση με τον μέσο όρο του ίδιου παίκτη στους υπόλοιπους αγώνες του. "
                    "Αν η διαφορά είναι μεγάλη σε πολλούς, υπάρχει μοτίβο· αν είναι ένας ή δύο, μάλλον ήταν η βραδιά τους.")
            H = H.assign(Παίκτης=[(short(n) if mob else n.title()) for n in H["player"]],
                         Ομάδα=H["team"].map(team_short),
                         Ημερομηνία=H["date"].dt.tz_convert("Europe/Athens").dt.strftime("%d/%m/%y"))
            H = H.rename(columns={stat: "Έκανε", "avg": "Μ.Ο. του", "diff": "Διαφορά", "minutes": "Λεπτά"})
            cols_h = ["Παίκτης", "Έκανε", "Μ.Ο. του", "Διαφορά"] if mob else ["Παίκτης", "Ομάδα", "Ημερομηνία", "Λεπτά", "Έκανε", "Μ.Ο. του", "Διαφορά"]
            st.dataframe(H[cols_h], hide_index=True, width="stretch",
                         column_config={"Λεπτά": st.column_config.NumberColumn(format="%.0f"),
                                        "Μ.Ο. του": st.column_config.NumberColumn(format="%.1f", help="Μέσος όρος του παίκτη στους άλλους αγώνες της ίδιας περιόδου."),
                                        "Διαφορά": st.column_config.NumberColumn(format="%+.1f")})

    # ---------------- έχει ουσία;
    with t_real:
        P = X["pers"]
        st.markdown(
            '<div class="el-sub" style="margin-top:.4rem">Ο έλεγχος: για κάθε ομάδα χωρίζουμε τη σεζόν στα δύο. Αν μια ομάδα που έδωσε πολλά '
            'στους γκαρντ στο πρώτο μισό δίνει πολλά και στο δεύτερο, το μοτίβο είναι πραγματικό. Αν όχι, ήταν τύχη. '
            'Μετράει η <b>συσχέτιση</b> r: 1 σημαίνει ότι επαναλαμβάνεται τέλεια, 0 ότι δεν επαναλαμβάνεται καθόλου. '
            'Με 20 ομάδες, κάτω από ~0.45 δεν ξεχωρίζει με σιγουριά από την τύχη.</div>', unsafe_allow_html=True)
        if P.empty or P["r_total"].isna().all():
            st.info("Χρειάζεται τουλάχιστον μία σεζόν με 16+ αγώνες ανά ομάδα.")
        else:
            def cls(r):
                return "Επαναλαμβάνεται" if r >= 0.45 else "Ασθενές" if r >= 0.25 else "Θόρυβος"
            CL = {"Επαναλαμβάνεται": GOOD, "Ασθενές": WARN, "Θόρυβος": HAIR}
            Pm = P.melt(id_vars=["pos", "stat", "n_teams"], value_vars=["r_total", "r_specific"], var_name="τι", value_name="r")
            Pm["Θέση"] = Pm["pos"].map(POS4)
            Pm["Στατιστικό"] = Pm["stat"].map(DEF_STAT_SHORT)
            Pm["Κρίση"] = Pm["r"].map(lambda r: "Θόρυβος" if np.isnan(r) else cls(r))
            Pm["txt"] = Pm["r"].map(lambda r: "—" if np.isnan(r) else f"{r:.2f}")
            leg = "".join(sw(CL[k], k + t) for k, t in (("Επαναλαμβάνεται", " (r ≥ 0.45)"), ("Ασθενές", " (0.25–0.45)"), ("Θόρυβος", " (κάτω από 0.25)")))
            panels = (("r_total", "Συνολικά", "Δίνει πολλά στη θέση στο 1ο μισό → και στο 2ο;"),
                      ("r_specific", "Πέρα από τη συνολική άμυνα",
                       "Το ίδιο, αφού αφαιρεθεί η συνολική άμυνα. Αυτό είναι το κομμάτι που δεν ξέρει ήδη το μοντέλο."))
            cols_ = [st, st] if mob else st.columns(2, gap="large")
            for colx, (key_, title_, sub_) in zip(cols_, panels):
                heading(colx, title_, sub_ + "<br>" + leg)
                d_ = Pm[Pm["τι"] == key_]
                base = alt.Chart(d_).encode(
                    x=alt.X("Θέση:N", sort=[POS4[i] for i in DF.POS], title=None, axis=alt.Axis(orient="top", labelAngle=0)),
                    y=alt.Y("Στατιστικό:N", sort=list(DEF_STAT_SHORT.values()), title=None))
                ch = base.mark_rect(cornerRadius=3, stroke=BG, strokeWidth=2).encode(
                    color=alt.Color("Κρίση:N", scale=alt.Scale(domain=list(CL), range=list(CL.values())), legend=None),
                    tooltip=["Θέση", "Στατιστικό", alt.Tooltip("r:Q", format=".2f"), "Κρίση", alt.Tooltip("n_teams:Q", title="Ομάδες")])
                show_chart(colx, (ch + base.mark_text(fontSize=12, color=TEXT, font="Commissioner").encode(text="txt:N")).properties(height=9 * 30 + 30))
            # πόσοι αγώνες χρειάζονται
            n0g = X["n0"]["game"]
            N = pd.DataFrame([{"Θέση": POS4[p], "Στατιστικό": DEF_STAT_SHORT[k], "n0": n0g.get((p, k), np.nan)}
                              for p in (0, 1, 2, 3) for k in DF.DSTATS])
            N["txt"] = N["n0"].map(lambda v: "400+" if v >= DF.N0_MAX else f"{v:.0f}")
            N["Κρίση"] = pd.cut(N["n0"], [0, 20, 60, 1e9], labels=["Μέσα στη σεζόν", "Πάνω από μισή σεζόν", "Πάνω από σεζόν"]).astype(str)
            CL2 = {"Μέσα στη σεζόν": GOOD, "Πάνω από μισή σεζόν": WARN, "Πάνω από σεζόν": HAIR}
            heading(st, "Πόσοι αγώνες χρειάζονται για να πιστέψεις τα δεδομένα μιας ομάδας κατά 50%",
                    "Όσο μεγαλύτερος ο αριθμός, τόσο περισσότερος θόρυβος από αγώνα σε αγώνα σε σχέση με τις πραγματικές διαφορές "
                    "μεταξύ ομάδων. Μια σεζόν έχει 38 αγώνες. Από αυτόν τον αριθμό βγαίνει η διόρθωση για τύχη.<br>"
                    + "".join(sw(v, k) for k, v in CL2.items()))
            base = alt.Chart(N).encode(x=alt.X("Στατιστικό:N", sort=list(DEF_STAT_SHORT.values()), title=None,
                                               axis=alt.Axis(orient="top", labelAngle=0)),
                                       y=alt.Y("Θέση:N", sort=list(POS4.values()), title=None))
            ch = base.mark_rect(cornerRadius=3, stroke=BG, strokeWidth=2).encode(
                color=alt.Color("Κρίση:N", scale=alt.Scale(domain=list(CL2), range=list(CL2.values())), legend=None),
                tooltip=["Θέση", "Στατιστικό", alt.Tooltip("n0:Q", format=".0f", title="Αγώνες")])
            show_chart(st, (ch + base.mark_text(fontSize=12, color=TEXT, font="Commissioner").encode(text="txt:N")).properties(height=4 * 34 + 30))
            strong = P[P["r_specific"] >= 0.45]
            pairs = ", ".join(f"{DEF_STAT_SHORT[k].lower()} στους {M.ROLE_PLURAL[p]}" for p, k in zip(strong["pos"], strong["stat"]))
            st.markdown(
                f'<div class="el-verdict" style="--vc:{WARN}"><div class="t">Συμπέρασμα</div><div class="s">'
                f'Η <b>συνολική</b> άμυνα μιας ομάδας είναι πραγματική και ήδη μπαίνει στις προβλέψεις. Το «τι δίνει σε κάθε θέση» '
                f'είναι κυρίως θόρυβος: για πόντους στους γκαρντ χρειάζονται περίπου {n0g.get((0, "pts"), 0):.0f} αγώνες για να '
                f'μετρήσουν τα δεδομένα της ομάδας μισά-μισά με τον μέσο όρο. '
                + (f"Ξεχωρίζουν μόνο: {esc(pairs)}. " if pairs else "Κανένας συνδυασμός δεν επαναλαμβάνεται καθαρά. ")
                + 'Στον έλεγχο του μοντέλου, η προσθήκη της θέσης δεν βελτίωσε τις προβλέψεις. Χρησιμοποίησέ το ως '
                  'δεύτερη ματιά, όχι ως λόγο να παίξεις.</div>'
                  f'<div class="m">Έλεγχος στις σεζόν με 16+ αγώνες ανά ομάδα · θέσεις: '
                  f'{"επίσημες από τα ρόστερ" if official else "εκτίμηση από τα στατιστικά"}.</div></div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ αυτόματο αρχείο

TAG_ICON = {"easy_win": "✅", "close_win": "✅", "close_loss": "❌", "clear_loss": "❌", "low_min": "⏱️",
            "void": "🚫", "push": "↔️"}
TAG_COLOR = {"easy_win": "#3CCB7F", "close_win": "#2E8F5E", "close_loss": "#C9623F", "clear_loss": "#F2785C",
             "low_min": "#C98500", "void": GREY, "push": GREY}
GLOSS.update({
    "Μονάδα": "Ένα ίδιο ποντάρισμα σε κάθε στοίχημα, π.χ. 1 €. +3 μονάδες σημαίνει 3 € κέρδος αν έπαιζες 1 € στο καθένα.",
    "Κίνηση αγοράς": "Πόσο άλλαξε η απόδοση από την καταγραφή μέχρι την τελευταία φορά που την είδαμε πριν τον αγώνα. "
                     "Θετικό σημαίνει ότι έπεσε, δηλαδή η αγορά συμφώνησε μετά με το μοντέλο. Είναι το πιο γρήγορο σημάδι πλεονεκτήματος.",
    "Περιθώριο τύχης": "Πόσο μπορεί να ανεβοκατέβει η απόδοση επένδυσης μόνο από τύχη με τόσα στοιχήματα (90%).",
})


def outcome_text(r) -> str:
    if r["status"] == "open":
        return "⏳ Ανοιχτό"
    t = r["tag"] or r["status"]
    txt = f'{TAG_ICON.get(t, "")} {PK.TAGS.get(t, t)}'
    if r["status"] in ("won", "lost", "push") and r["result"] is not None and not pd.isna(r["result"]):
        txt += f' · {r["result"]:.0f}'
        if t == "low_min" and not pd.isna(r["minutes"]):
            txt += f' σε {r["minutes"]:.0f}′ αντί {r["min_hat"]:.0f}′'
    return txt


def page_archive(D, embedded: bool = False):
    if not embedded:
        st.title("Αρχείο «Αξίζει»")
    st.markdown('<div class="el-sub">Κάθε στοίχημα που βγαίνει «Αξίζει» καταγράφεται μόνο του τη στιγμή που εμφανίζεται, '
                'με την απόδοση εκείνης της στιγμής, και κλείνει μόνο του όταν περαστούν τα στατιστικά. Μετράει αν '
                'το μοντέλο κερδίζει, ανεξάρτητα από το τι έπαιξες εσύ. Καταγράφονται και τα «Οριακά», για σύγκριση.</div>',
                unsafe_allow_html=True)
    d = PK.load(con)
    if d.empty:
        st.info("Δεν έχει καταγραφεί τίποτα ακόμα. Μόλις περάσεις αποδόσεις και κάποιο στοίχημα βγει «Αξίζει» "
                "(ή «Οριακό»), θα εμφανιστεί εδώ μόνο του.")
        return
    mob = is_mobile()
    f = st.columns(3)
    vs = f[0].segmented_control("Ετυμηγορία", ["strong", "marginal"], format_func=V_NAME.get, selection_mode="multi",
                                default=["strong"], key="arc_v") or ["strong"]
    stats = f[1].multiselect("Αγορές", sorted(d["stat"].unique(), key=M.KEYS.index), format_func=STAT_LABEL.get,
                             placeholder="Όλες", key="arc_s")
    side = f[2].segmented_control("Πλευρά", ["over", "under"], format_func=str.title, key="arc_side")
    n_hid = int(d["hidden"].sum()) if "hidden" in d else 0
    hid_mode = "all"
    if n_hid:
        hid_mode = st.segmented_control(
            f"Κρυμμένα από σένα ({n_hid})", ["all", "visible", "hidden"], default="all", key="arc_hid",
            format_func={"all": "Όλα", "visible": "Χωρίς τα κρυμμένα", "hidden": "Μόνο τα κρυμμένα"}.get,
            help="Τα στοιχήματα που έκρυψες στη σελίδα «Αξία». Αν τα κρυμμένα βγαίνουν λιγότερο από τα υπόλοιπα, "
                 "οι απορρίψεις σου βοηθούν.") or "all"
    tags_all = PK.all_tags(con)
    tg = PK.load_tags(con, d)
    n_exc = int(d["excluded"].sum())
    g1, g2, g3 = st.columns([3, 2, 2])
    tag_f = g1.multiselect("Ετικέτες", tags_all + ["— χωρίς ετικέτα"], key="arc_tags", placeholder="Όλα τα στοιχήματα",
                           help="Δείχνει όσα έχουν τουλάχιστον μία από τις ετικέτες. Τις βάζεις στον πίνακα στο τέλος της σελίδας.")
    only_before = g2.toggle("Μόνο σημειωμένα πριν τον αγώνα", True, key="arc_before",
                            help="Μετράει μια ετικέτα μόνο αν τη έβαλες πριν το τζάμπολ, δηλαδή χωρίς να ξέρεις το αποτέλεσμα. "
                                 "Αυτό είναι το τίμιο ποσοστό.")
    view = g3.segmented_control(f"Εξαιρέσεις ({n_exc})", ["all", "exc"], default="all", key="arc_view",
                                format_func={"all": "Χωρίς αυτές", "exc": "Μόνο αυτές"}.get,
                                help="Όσα εξαίρεσες (στήλη ✕ του πίνακα) δεν μετράνε πουθενά, εκτός αν δεις «Μόνο αυτές».") or "all"
    v = d[d["verdict"].isin(vs)]
    v = v[v["excluded"] == (1 if view == "exc" else 0)]
    mx = max_odds()
    n_over = int((v["price_typ"] > mx).sum())
    if n_over and st.toggle(f"Με τους τωρινούς κανόνες: χωρίς τυπική απόδοση πάνω από {mx:.2f} ({n_over})", True, key="arc_mx",
                            help="Οι παλιές καταγραφές έγιναν πριν μπει το όριο. Με αυτό ανοιχτό, τα ποσοστά βγαίνουν σαν να "
                                 "ίσχυε από την αρχή. Το όριο αλλάζει στις Ρυθμίσεις."):
        v = v[v["price_typ"] <= mx]
    tg_ok = tg[tg["before"]] if only_before else tg

    def with_tags(x: pd.DataFrame, chosen) -> pd.DataFrame:
        if not chosen:
            return x
        ids = set(tg_ok.loc[tg_ok["tag"].isin(chosen), "pick_id"])
        if "— χωρίς ετικέτα" in chosen:
            ids |= set(x["id"]) - set(tg["pick_id"])
        return x[x["id"].isin(ids)]
    if hid_mode == "visible":
        v = v[v["hidden"] == 0]
    elif hid_mode == "hidden":
        v = v[v["hidden"] == 1]
    if stats:
        v = v[v["stat"].isin(stats)]
    if side:
        v = v[v["side"] == side]
    v = v.assign(_game=[game_label(gd, t, o) for gd, t, o in zip(v["game_date"], v["team"], v["opp"])])
    v = study_filters(v, D, "af", price="price_typ", prob="p_final", ev="ev", team="team", opp="opp",
                      game="_game", pid="player_id", minutes="min_hat", player="player")
    v_all = v                                  # χωρίς το φίλτρο ετικετών, για τη σύγκριση ανά ετικέτα
    v = with_tags(v, tag_f)
    if v.empty:
        st.caption("Τίποτα με αυτά τα φίλτρα.")
        return
    sm = PK.summary(v)
    if sm["n"]:
        GOODC, BADC = V_COLOR["strong"], V_COLOR["suspect"]
        edge = sm["hit"] - sm["exp_market"]
        col_h = GOODC if edge > 0.005 else BADC if edge < -0.005 else None
        col_u = GOODC if sm["units"] > 0 else BADC if sm["units"] < 0 else None
        heading(st, "Έκβαση", f'{sm["n"]} κλεισμένα σε {sm["games"]} αγώνες · ανοιχτά {sm["n_open"]} · ακυρωμένα {sm["n_void"]}. '
                "Η έκβαση δεν εξαρτάται από την εταιρεία: μετράει μόνο αν βγήκε το στοίχημα.")
        tiles(st, [
            ("Βγήκαν", pct(sm["hit"]), f'{int(round(sm["hit"] * sm["n"]))} από {sm["n"]}.', col_h),
            ("Περίμενε η αγορά", pct(sm["exp_market"]), "Η πιθανότητα από τις αποδόσεις, χωρίς γκανιότα. Αυτό πρέπει να ξεπερνάμε."),
            ("Περίμενε το μοντέλο", pct(sm["exp_model"]), "Αν τα «Βγήκαν» είναι κοντά εδώ, οι πιθανότητες του μοντέλου είναι σωστές."),
            ("Πάνω από την αγορά", f'{edge * 100:+.1f}<small>μον.</small>',
             f'Περιθώριο τύχης ±{sm["hit_band"] * 100:.0f} μονάδες (λογαριάζει ότι στοιχήματα του ίδιου αγώνα κινούνται μαζί).', col_h),
        ])
        heading(st, "Κέρδος με 1 μονάδα σε κάθε στοίχημα", "Στην <b>τυπική απόδοση</b>: ο διάμεσος όλων των εταιρειών, "
                "περίπου όσο βρίσκεις σε μια μεγάλη εταιρεία όπως η Stoiximan. Η καλύτερη απόδοση της αγοράς δίνεται για σύγκριση."
                + (f' Για {sm["approx"]:.0%} των στοιχημάτων (παλιές καταγραφές) η τυπική απόδοση είναι εκτίμηση από την αγορά.'
                   if sm["approx"] > 0 else ""))
        tiles(st, [
            ("Κέρδος (τυπική απόδοση)", f'{sm["units"]:+.1f}<small>μον.</small>', f'Μέση τυπική απόδοση {sm["avg_price"]:.2f}.', col_u),
            ("Απόδοση επένδυσης", f'{sm["roi"]:+.1%}', f'Περιθώριο τύχης ±{sm["roi_band"]:.0%}.', col_u),
            ("Στην καλύτερη απόδοση", f'{sm["units_best"]:+.1f}<small>μον.</small>',
             f'{sm["roi_best"]:+.1%} με μέση απόδοση {sm["avg_best"]:.2f}. Μόνο αν έπαιζες κάθε φορά στην πιο γενναιόδωρη εταιρεία.'),
            ("Κίνηση αγοράς", "—" if sm["clv"] is None else f'{sm["clv"]:+.1%}',
             "Καμία αλλαγή απόδοσης ακόμα." if sm["clv"] is None else f'Σε {sm["n_moved"]} που άλλαξαν, το {pct(sm["clv_pos"])} κινήθηκε υπέρ μας.',
             None if sm["clv"] is None else (GOODC if sm["clv"] > 0 else BADC)),
        ])
        lo, hi = edge - sm["hit_band"], edge + sm["hit_band"]
        if sm["games"] < 15:
            vc, t, msg = GREY, "Πολύ νωρίς", (f'Μόνο {sm["games"]} αγώνες. Τα στοιχήματα του ίδιου αγώνα κερδίζουν και χάνουν μαζί, '
                                              f'οπότε το πραγματικό δείγμα είναι μικρό: η έκβαση μπορεί να απέχει ±{sm["hit_band"] * 100:.0f} '
                                              'μονάδες από την αγορά μόνο από τύχη. Χρειάζονται τουλάχιστον 3–4 αγωνιστικές.')
        elif lo > 0:
            vc, t, msg = GOODC, "Το μοντέλο βλέπει κάτι που η αγορά δεν βλέπει", (
                f'Τα στοιχήματα βγαίνουν {edge * 100:+.1f} μονάδες πιο συχνά από όσο λένε οι αποδόσεις, πέρα από το περιθώριο τύχης. '
                'Συνέχισε να μετράς: μια κακή εβδομάδα μπορεί να το αλλάξει.')
        elif hi < 0:
            vc, t, msg = BADC, "Η αγορά ξέρει καλύτερα", (
                f'Τα στοιχήματα βγαίνουν {edge * 100:+.1f} μονάδες λιγότερο από όσο λένε οι αποδόσεις. Το μοντέλο υπερεκτιμά κάπου: '
                'δες τον πίνακα ανά αγορά παρακάτω.')
        else:
            vc, t, msg = V_COLOR["marginal"], "Ακόμα μέσα στην τύχη", (
                f'Η διαφορά από την αγορά ({edge * 100:+.1f} μονάδες) είναι μέσα στο περιθώριο ±{sm["hit_band"] * 100:.0f}.')
        st.markdown(f'<div class="el-verdict" style="--vc:{vc}"><div class="t"><b>{t}</b></div><div class="s">{esc(msg)}</div></div>',
                    unsafe_allow_html=True)
    else:
        st.caption(f'{sm["n_open"]} ανοιχτά, κανένα κλεισμένο ακόμα. Κλείνουν μόνα τους μετά τον αγώνα, με την ενημέρωση στατιστικών.')

    after = v[(v["tag_after"] & (not only_before)) | ((v["excluded"] == 1) & ~v["excluded_before"])]
    after = after[after["status"] != "open"]
    if len(after):
        st.markdown(f'<div class="el-leg" style="color:{WARN}">⚠️ {len(after)} από αυτά τα σημείωσες ή τα εξαίρεσες <b>μετά</b> τον αγώνα. '
                    'Εκ των υστέρων επιλογές κάνουν το ποσοστό να φαίνεται καλύτερο απ\' ό,τι είναι.</div>', unsafe_allow_html=True)
    cl_all = v_all[v_all["status"].isin(["won", "lost"])]
    if len(tg) and len(cl_all):
        def line_(x, lab):
            r_ = {"Ομάδα": lab, "Κλεισμένα": len(x)}
            if len(x):
                r_.update({"Βγήκαν": 100 * (x["status"] == "won").mean(), "Περίμενε η αγορά": 100 * x["exp_market"].mean(),
                           "Μονάδες": x["units"].sum(), "Απόδ. επένδυσης": 100 * x["units"].mean()})
            return r_
        rows_ = [line_(cl_all, "Όλα (με τα τωρινά φίλτρα)")]
        for t_ in tags_all:
            pre = cl_all[cl_all["id"].isin(tg.loc[(tg["tag"] == t_) & tg["before"], "pick_id"])]
            post = cl_all[cl_all["id"].isin(tg.loc[(tg["tag"] == t_) & ~tg["before"], "pick_id"])]
            if len(pre) or len(post):
                rows_.append(line_(pre, f"🏷️ {t_} · πριν τον αγώνα"))
            if len(post):
                rows_.append(line_(post, f"🏷️ {t_} · μετά (εκ των υστέρων)"))
        rows_.append(line_(cl_all[~cl_all["id"].isin(tg["pick_id"])], "Χωρίς ετικέτα"))
        heading(st, "Ανά ετικέτα", "Με τα ίδια φίλτρα (ετυμηγορία, αγορά, πλευρά). Μετράει η γραμμή «πριν τον αγώνα»: τότε δεν ήξερες "
                "το αποτέλεσμα. Αν μια ετικέτα βγαίνει σταθερά πάνω από τα «Όλα», η κρίση σου προσθέτει αξία.")
        st.dataframe(pd.DataFrame(rows_), hide_index=True, width="stretch", column_config={
            "Βγήκαν": st.column_config.NumberColumn(format="%.0f%%"), "Περίμενε η αγορά": st.column_config.NumberColumn(format="%.0f%%"),
            "Μονάδες": st.column_config.NumberColumn(format="%+.1f"), "Απόδ. επένδυσης": st.column_config.NumberColumn(format="%+.1f%%")})
    if n_hid and hid_mode == "all":
        cl = v[v["status"].isin(["won", "lost"])]
        hh, vv = cl[cl["hidden"] == 1], cl[cl["hidden"] == 0]
        if len(hh):
            st.markdown(
                f'<div class="el-leg">🙈 <b>Οι απορρίψεις σου:</b> από τα {len(hh)} κλεισμένα που έκρυψες βγήκε το '
                f'{pct((hh["status"] == "won").mean())} (η αγορά περίμενε {pct(hh["exp_market"].mean())}), '
                f'{hh["units"].sum():+.1f} μον. Τα υπόλοιπα: {pct((vv["status"] == "won").mean()) if len(vv) else "—"}, '
                f'{vv["units"].sum():+.1f} μον. Αν τα κρυμμένα βγαίνουν συστηματικά λιγότερο, η κρίση σου προσθέτει αξία.</div>',
                unsafe_allow_html=True)
    s = v[v["status"].isin(["won", "lost"])].sort_values(["game_date", "id"]).reset_index(drop=True)
    closed = v[v["status"] != "open"]
    if len(s) >= 2 or len(closed):
        a, b = (st, st) if mob else st.columns([3, 2], gap="large")
        if len(s) >= 2:
            s["n"] = np.arange(1, len(s) + 1)
            long = pd.concat([pd.DataFrame({"n": s["n"], "u": s["units"].cumsum(), "Σειρά": "Πραγματικό"}),
                              pd.DataFrame({"n": s["n"], "u": (s["p_final"].fillna(s["exp_market"]) * s["price_typ"] - 1).cumsum(),
                                            "Σειρά": "Αναμενόμενο"})])
            ch = alt.Chart(long).mark_line(strokeWidth=2).encode(
                x=alt.X("n:Q", title="Κλεισμένα με τη σειρά του αγώνα", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("u:Q", title="Μονάδες"),
                color=alt.Color("Σειρά:N", scale=alt.Scale(domain=["Πραγματικό", "Αναμενόμενο"], range=[BLUE, MUTED]), legend=None),
                strokeDash=alt.StrokeDash("Σειρά:N", scale=alt.Scale(domain=["Πραγματικό", "Αναμενόμενο"], range=[[1, 0], [5, 4]]), legend=None),
                tooltip=["Σειρά", alt.Tooltip("n:Q", title="Στοίχημα"), alt.Tooltip("u:Q", format="+.2f", title="Μονάδες")])
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=HAIR).encode(y="y:Q")
            heading(a, "Πορεία σε μονάδες (τυπική απόδοση)", sw(BLUE, "πραγματικό", "ln") + sw(MUTED, "αναμενόμενο από το μοντέλο", "ln dash")
                    + "<br>Η μπλε θα ανεβοκατεβαίνει γύρω από τη διακεκομμένη. Αν μένει σταθερά από κάτω, το μοντέλο είναι αισιόδοξο.")
            show_chart(a, (zero + ch).properties(height=280))
        if len(closed):
            order = [k for k in PK.TAGS if k in set(closed["tag"])]
            cnt = closed["tag"].value_counts().reindex(order).fillna(0).astype(int)
            od = pd.DataFrame({"Έκβαση": [PK.TAGS[k] for k in order], "n": cnt.to_numpy(), "k": order,
                               "Τι σημαίνει": [PK.TAG_MEANS[k] for k in order]})
            heading(b, "Πώς κρίθηκαν", "Μια ήττα «παρά τρίχα» ή με λίγα λεπτά είναι κυρίως ατυχία. Οι «καθαρές» ήττες "
                    "δείχνουν πού κάνει λάθος η εκτίμηση. Άφησε το ποντίκι πάνω σε μια μπάρα για εξήγηση.")
            bars = alt.Chart(od).mark_bar(cornerRadius=3, size=16).encode(
                y=alt.Y("Έκβαση:N", sort=list(od["Έκβαση"]), title=None, axis=alt.Axis(labelLimit=200)),
                x=alt.X("n:Q", title="Στοιχήματα", axis=alt.Axis(tickMinStep=1)),
                color=alt.Color("k:N", scale=alt.Scale(domain=order, range=[TAG_COLOR[k] for k in order]), legend=None),
                tooltip=["Έκβαση", alt.Tooltip("n:Q", title="Πόσα"), "Τι σημαίνει"])
            show_chart(b, (bars + bars.mark_text(align="left", dx=4, color=TEXT, font="Commissioner").encode(text="n:Q"))
                       .properties(height=34 * len(od) + 30))

    if len(s) >= 5:
        by = s.groupby("stat").agg(n=("id", "size"), hit=("status", lambda x: (x == "won").mean()),
                                   exp=("exp_market", "mean"), u=("units", "sum"), roi=("units", "mean"),
                                   eroi=("ev", "mean")).reset_index()
        by["Αγορά"] = by["stat"].map(STAT_LABEL)
        sd = s.groupby("side").agg(n=("id", "size"), hit=("status", lambda x: (x == "won").mean()),
                                   exp=("exp_market", "mean"), u=("units", "sum"), roi=("units", "mean"),
                                   eroi=("ev", "mean")).reset_index()
        sd["Αγορά"] = sd["side"].str.title() + " (όλες)"
        T = pd.concat([by, sd], ignore_index=True)
        T = T.assign(hit=T["hit"] * 100, exp=T["exp"] * 100, roi=T["roi"] * 100, eroi=T["eroi"] * 100)
        heading(st, "Ανά αγορά και πλευρά", "Με λίγα στοιχήματα ανά γραμμή οι διαφορές είναι θόρυβος. "
                "Ψάξε για αγορές όπου η επιτυχία μένει σταθερά κάτω από την αναμενόμενη.")
        st.dataframe(T[["Αγορά", "n", "hit", "u"] if mob else ["Αγορά", "n", "hit", "exp", "u", "roi", "eroi"]], hide_index=True, width="stretch", column_config={
            "n": st.column_config.NumberColumn("Κλεισμένα"),
            "hit": st.column_config.NumberColumn("Επιτυχία", format="%.0f%%"),
            "exp": st.column_config.NumberColumn("Περίμενε η αγορά", format="%.0f%%"),
            "u": st.column_config.NumberColumn("Μονάδες", format="%+.1f", help=GLOSS["Μονάδα"]),
            "roi": st.column_config.NumberColumn("Απόδ. επένδυσης", format="%+.1f%%", help=GLOSS["Απόδοση επένδυσης"]),
            "eroi": st.column_config.NumberColumn("Αναμ. κέρδος (μοντέλο)", format="%+.1f%%")})

    heading(st, "Όλες οι καταγραφές", "Η απόδοση είναι αυτή τη στιγμή της καταγραφής. Η «τελευταία» είναι η πιο πρόσφατη "
            "που είδαμε πριν τον αγώνα. Το αποτέλεσμα και ο χαρακτηρισμός συμπληρώνονται μόνα τους.")
    w = v.assign(
        Αγώνας=v["game_date"].dt.tz_convert("Europe/Athens").dt.strftime("%d/%m"),
        Παίκτης=[short(n) if mob else n.title() for n in v["player"]],
        Στοίχημα=[f"{'Over' if sd_ == 'over' else 'Under'} {ln:g} {STAT_LABEL.get(stt, stt)}" for sd_, ln, stt in zip(v["side"], v["line"], v["stat"])],
        Ετυμηγορία=v["verdict"].map(V_NAME),
        Έκβαση=[("🙈 " if r.get("hidden") else "") + outcome_text(r) for r in v.to_dict("records")],
        ev_pct=v["ev"] * 100, clv_pct=v["clv"] * 100)
    w = w.assign(Ετικέτες=w["tags"], Εξαίρεση=w["excluded"].astype(bool)).reset_index(drop=True)
    flag_cfg = {"Ετικέτες": st.column_config.MultiselectColumn(
                    "🏷️" if mob else "Ετικέτες", options=tags_all, accept_new_options=True, width=110 if mob else "medium",
                    help="Διάλεξε ή γράψε νέα ετικέτα (π.χ. «Σίγουρα», «Ρίσκο»). Ένα στοίχημα μπορεί να έχει πολλές. "
                         "Καλύτερα πριν τον αγώνα: μετά μετράει ως «εκ των υστέρων»."),
                "Εξαίρεση": st.column_config.CheckboxColumn("✕" if mob else "Εξαίρεση", width=34 if mob else "small",
                                                            help="Βγαίνει από τα ποσοστά (π.χ. ο παίκτης τραυματίστηκε στο 2ο λεπτό). "
                                                                 "Φαίνεται μόνο στο «Μόνο οι εξαιρέσεις».")}
    if mob:
        w["Στ."] = [f"{'O' if sd_ == 'over' else 'U'} {ln:g} {STAT_LABEL.get(stt, stt)} @{pt:.2f}"
                    for sd_, ln, stt, pt in zip(w["side"], w["line"], w["stat"], w["price_typ"])]
        w["Αποτ."] = [outcome_text(r).split(" ")[0] + (f' {r["result"]:.0f}' if r["status"] in ("won", "lost", "push") else "")
                      for r in w.to_dict("records")]
        cols_w = ["Ετικέτες", "Εξαίρεση", "Παίκτης", "Στ.", "Αποτ."]
        cfg = {**flag_cfg, "Παίκτης": st.column_config.TextColumn(width=86), "Στ.": st.column_config.TextColumn(width=140),
               "Αποτ.": st.column_config.TextColumn(width=52)}
    else:
        cols_w = ["Ετικέτες", "Εξαίρεση", "Αγώνας", "Παίκτης", "opp", "Στοίχημα", "Ετυμηγορία", "price_typ", "price", "bookmaker",
                  "price_last", "clv_pct", "ev_pct", "mu", "Έκβαση", "units"]
        cfg = {**flag_cfg,
               "opp": st.column_config.TextColumn("Αντίπαλος"),
               "price_typ": st.column_config.NumberColumn("Τυπική", format="%.2f", help="Ο διάμεσος όλων των εταιρειών: περίπου όσο βρίσκεις στη Stoiximan."),
               "price": st.column_config.NumberColumn("Καλύτερη", format="%.2f", help="Η καλύτερη απόδοση της αγοράς τη στιγμή της καταγραφής."),
               "bookmaker": st.column_config.TextColumn("Εταιρεία", help="Πού ήταν η καλύτερη απόδοση."),
               "price_last": st.column_config.NumberColumn("Τελευταία", format="%.2f", help="Η τελευταία απόδοση πριν τον αγώνα."),
               "clv_pct": st.column_config.NumberColumn("Κίνηση αγοράς", format="%+.1f%%", help=GLOSS["Κίνηση αγοράς"]),
               "ev_pct": st.column_config.NumberColumn("Αναμ. κέρδος", format="%+.1f%%", help=GLOSS["Αναμενόμενο κέρδος"]),
               "mu": st.column_config.NumberColumn("Πρόβλεψη", format="%.1f", help=GLOSS["Πρόβλεψη"]),
               "Έκβαση": st.column_config.TextColumn(width="large", help="Αποτέλεσμα, τι έκανε ο παίκτης, και χαρακτηρισμός."),
               "units": st.column_config.NumberColumn("Μονάδες", format="%+.2f", help=GLOSS["Μονάδα"] + " Στην τυπική απόδοση."),
               }
    ed = st.data_editor(w[cols_w], hide_index=True, width="stretch", key=f"arc_ed_{view}_{hid_mode}",
                        height=min(38 * (len(w) + 1) + 2, 640), column_config=cfg,
                        disabled=[c for c in cols_w if c not in ("Ετικέτες", "Εξαίρεση")])
    norm = lambda x: sorted(set(x or []))  # noqa: E731
    ch_fav = w.loc[[norm(a_) != norm(b_) for a_, b_ in zip(ed["Ετικέτες"], w["Ετικέτες"])]]
    ch_exc = w.loc[ed["Εξαίρεση"].to_numpy() != w["Εξαίρεση"].to_numpy()]
    if len(ch_fav) or len(ch_exc):
        n_after = int(((ch_fav["status"] != "open").sum()) + ((ch_exc["status"] != "open").sum()))
        if st.button(f"Αποθήκευση ({len(ch_fav) + len(ch_exc)} αλλαγές)", type="primary", key="arc_save"):
            for i, new_ in zip(ch_fav["id"], ed.loc[ch_fav.index, "Ετικέτες"]):
                PK.set_tags(con, i, list(new_ or []))
            for i, new_ in zip(ch_exc["id"], ed.loc[ch_exc.index, "Εξαίρεση"]):
                PK.set_flag(con, [i], "excluded", bool(new_))
            st.rerun()
        if n_after:
            st.caption(f"⚠️ {n_after} από τις αλλαγές αφορούν αγώνες που έχουν ήδη παιχτεί: θα μετρήσουν ως «εκ των υστέρων».")
    with st.expander("Υπόμνημα"):
        st.markdown('<dl class="el-gloss">' + "".join(
            f'<dt style="color:{TAG_COLOR[k]}">{TAG_ICON[k]} {esc(PK.TAGS[k])}</dt><dd>{esc(PK.TAG_MEANS[k])}</dd>' for k in PK.TAGS)
            + "</dl>" + gloss_html(["Μονάδα", "Κίνηση αγοράς", "Περιθώριο τύχης"]), unsafe_allow_html=True)


# ------------------------------------------------------------------ main

def pending_games() -> int:
    """Αγώνες που έχουν ήδη παιχτεί αλλά δεν έχουν περαστεί στη βάση."""
    cut = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return con.execute("SELECT COUNT(*) FROM games WHERE played=0 AND date < ?", (cut,)).fetchone()[0]


def quick_update():
    season = int(con.execute("SELECT MAX(season) FROM games").fetchone()[0])
    bar = st.sidebar.progress(0.0, "Λήψη…")
    res = E.update(con, [season], lambda p, t: bar.progress(p, t))
    bar.empty()
    st.session_state["upd_msg"] = (f"Πέρασαν {res['new_box']} νέοι αγώνες." if not res["errors"] else
                                   "Η ενημέρωση δεν ολοκληρώθηκε: " + res["errors"][-1])
    st.rerun()


PAGES = {"Αξία": ":material/target: Αξία", "Αγώνας": ":material/sports_basketball: Αγώνας",
         "Παίκτης": ":material/person: Παίκτης", "Άμυνες": ":material/shield: Άμυνες",
         "Αποδόσεις": ":material/edit_note: Αποδόσεις", "Αποτελέσματα": ":material/insights: Αποτελέσματα"}


def _leave_data():
    st.session_state.pop("page_override", None)


def page_results(D, sig):
    st.title("Αποτελέσματα")
    st.markdown('<div class="el-sub">Δουλεύει; Το Αρχείο μετράει όλα τα «Αξίζει» αυτόματα, το Ημερολόγιο όσα έπαιξες εσύ, '
                'και ο Έλεγχος μοντέλου αν οι πιθανότητες είναι ειλικρινείς σε όλους τους παλιούς αγώνες.</div>',
                unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["Αρχείο «Αξίζει»", "Ημερολόγιο", "Έλεγχος μοντέλου"])
    with t1:
        page_archive(D, embedded=True)
    with t2:
        page_bets(D, embedded=True)
    with t3:
        page_model(D, sig, embedded=True)

sig = data_sig()
D = compute(sig)
if D:
    B.clear_returned(con)
    n_settled = BT.settle(con)
    if n_settled:
        st.toast(f"Έκλεισαν {n_settled} στοιχήματα στο Ημερολόγιο.")
    n_picks = PK.settle(con)
    if n_picks:
        st.toast(f"Έκλεισαν {n_picks} καταγραφές στο Αρχείο.")
    D["ctx"]["absent"] = B.absent_ids(con)
with st.sidebar:
    st.markdown('<div class="el-brand">🏀 Euroleague Props <span>ειδικά παικτών</span></div>', unsafe_allow_html=True)
    MOB = is_mobile()
    page = "Αξία" if MOB else st.radio("Σελίδα", list(PAGES), format_func=PAGES.get, label_visibility="collapsed",
                                        on_change=_leave_data)
    w_model = 0.5
    if D:
        last = D["df"]["date"].max()
        st.caption(f"Στατιστικά έως {last.tz_convert('Europe/Athens'):%d/%m/%Y} · "
                   f"{D['df'].groupby(['season', 'game_code']).ngroups} αγώνες")
        n_pend = pending_games()
        if n_pend:
            st.warning(f"Έχουν παιχτεί {n_pend} αγώνες από την τελευταία ενημέρωση. Το μοντέλο δεν τους ξέρει ακόμα.")
            if st.button("Ενημέρωση τώρα", type="primary"):
                quick_update()
        if st.session_state.get("upd_msg"):
            st.caption(st.session_state.pop("upd_msg"))
        n_abs = len(D["ctx"]["absent"])
        if n_abs:
            st.caption(f"Δηλωμένες απουσίες: {n_abs}. Αλλάζουν στη σελίδα «Αγώνας».")
        if O.load_config().get("oddspapi_key"):
            if st.button("Ανανέωση αποδόσεων", help="Τραβάει τις τρέχουσες αποδόσεις από το OddsPapi με το κλειδί που έχεις αποθηκεύσει."):
                with st.spinner("Λήψη αποδόσεων…"):
                    st.session_state["odds_msg"] = refresh_odds(D)
            if st.session_state.get("odds_msg"):
                st.caption(st.session_state["odds_msg"])
        if st.button("Δεδομένα και αντίγραφα", icon=":material/database:", width="stretch",
                     help="Ενημέρωση στατιστικών, κατέβασμα αντιγράφου ασφαλείας, επαναφορά."):
            st.session_state["page_override"] = "Δεδομένα"
        with st.expander("Ρυθμίσεις"):
            w_model = st.slider("Πόσο εμπιστεύομαι το μοντέλο έναντι της αγοράς", 0.0, 1.0, 0.5, 0.05,
                                help="Η αγορά ξέρει πράγματα που το μοντέλο αγνοεί (τραυματισμούς, ρόλους). "
                                     "Στο 0.5 το αναμενόμενο κέρδος μετριέται συντηρητικά. Στο 1.0 μετράει μόνο το μοντέλο.")
            mx_ = st.number_input("Μέγιστη απόδοση για «Αξίζει»", 1.2, 20.0, max_odds(), 0.1, format="%.2f",
                                  help="Στην τυπική απόδοση (διάμεσος των εταιρειών). Από πάνω, τίποτα δεν βγαίνει «Αξίζει» ή "
                                       "«Οριακό» ούτε καταγράφεται στο Αρχείο. Κόβει τις «σκάλες» με τις μεγάλες αποδόσεις, όπου το "
                                       "μοντέλο υπερεκτιμά το κέρδος.")
            if abs(mx_ - max_odds()) > 1e-9:
                O.save_config({**O.load_config(), "max_odds": float(mx_)})
                st.session_state.pop("_board", None)
                st.rerun()
            st.toggle("Διάταξη κινητού", value=_auto_mobile(), key="mobile_view",
                      help="Μενού στο πάνω μέρος, κάρτες αντί για φαρδιούς πίνακες. Ανοίγει μόνη της όταν μπαίνεις από κινητό.")
            url_, rem_ = (None, None) if CLOUD else (lan_url(), remote_url())
            if CLOUD:
                pass
            elif rem_:
                st.caption(f"Από το κινητό, οπουδήποτε (με ανοιχτό το Tailscale): **{rem_}**")
            elif not MOB:
                st.caption("Για το κινητό εκτός σπιτιού: εγκατάστησε το Tailscale σε Mac και κινητό, με τον ίδιο λογαριασμό.")
            if url_ and not MOB:
                st.caption(f"Στο ίδιο Wi-Fi: {url_}")
            if SERVICE:
                st.caption("Τρέχει μόνιμα στο παρασκήνιο του Mac και ξεκινά μόνη της, για να την ανοίγεις από το κινητό.")
            if not CLOUD and st.button("Επανεκκίνηση εφαρμογής" if SERVICE else "Κλείσιμο εφαρμογής",
                         help="Ξαναξεκινά σε ~15 δευτερόλεπτα. Για μόνιμο σταμάτημα: bash ~/Documents/euroleague-props/install.command --stop"
                         if SERVICE else "Σταματά την εφαρμογή στον υπολογιστή. Την ξανανοίγεις από το εικονίδιο."):
                import signal
                import threading
                st.warning("Ξαναξεκινά. Ανανέωσε τη σελίδα σε λίγα δευτερόλεπτα." if SERVICE
                           else "Η εφαρμογή έκλεισε. Μπορείς να κλείσεις την καρτέλα.")
                threading.Timer(0.8, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
        if SYNC:
            if SYNC.last_error:
                st.error(f"Η αποθήκευση απέτυχε: {SYNC.last_error}")
            elif SYNC.dirty():
                st.caption("☁️ Υπάρχουν αλλαγές που θα αποθηκευτούν μέσα σε 3 λεπτά.")
            elif SYNC.last_ok:
                st.caption(f"☁️ Όλα αποθηκευμένα · τελευταία φορά {pd.Timestamp(SYNC.last_ok, unit='s', tz='UTC').tz_convert('Europe/Athens'):%H:%M}")
            else:
                st.caption("☁️ Όλα αποθηκευμένα.")
            if SYNC.dirty() and st.button("Αποθήκευση τώρα", help="Αποθηκεύει αμέσως στο GitHub. Γίνεται και μόνο του."):
                with st.spinner("Αποθήκευση…"):
                    ok_ = SYNC.push("χειροκίνητα")
                st.toast("Αποθηκεύτηκε." if ok_ else "Η αποθήκευση απέτυχε.")
        elif CLOUD:
            st.error("Τα δεδομένα δεν αποθηκεύονται σε αυτή τη συνεδρία.")
        with st.expander("Υπόμνημα όρων"):
            st.markdown(gloss_html(list(GLOSS)), unsafe_allow_html=True)
            st.markdown('<dl class="el-gloss">' + "".join(
                f'<dt style="color:{V_COLOR[k]}">{V_NAME[k]}</dt><dd>{esc(V_MEANS[k])}</dd>' for k in V_NAME) + "</dl>",
                unsafe_allow_html=True)

if MOB and D is not None:
    if "nav_m" not in st.session_state:
        st.session_state["nav_m"] = "Αξία"
    if st.session_state.get("nav_m") not in PAGES:
        st.session_state["nav_m"] = "Αξία"
    pick_ = st.pills("Σελίδα", list(PAGES), key="nav_m", label_visibility="collapsed", on_change=_leave_data)
    page = pick_ or st.session_state.get("last_page", "Αξία")
    st.session_state["last_page"] = page
page = st.session_state.get("page_override", page)

if BOOT and not st.session_state.get("boot_seen"):
    (st.error if BOOT[0] == "error" else st.info)(BOOT[1])
    st.session_state["boot_seen"] = BOOT[0] != "error"

if D is None:
    st.title("Euroleague Props")
    st.info("Η βάση είναι άδεια. Πάτα «Ενημέρωση στατιστικών» για να κατέβουν οι αγώνες.")
    page_data(None)
else:
    nxt = B.next_games(con)
    # αυτόματη καταγραφή: σε κάθε άνοιγμα/αλλαγή, όποια σελίδα κι αν είναι ανοιχτή
    try:
        # πάντα με τη σταθερή ρύθμιση 0.5, ό,τι κι αν δείχνει η οθόνη, για να μετράει το Αρχείο έναν κανόνα
        _, b_auto = board_for(D, nxt, 0.5)
        n_new = PK.record(con, b_auto, nxt, D["current"], 0.5, skip=B.discarded_keys(con))
        if n_new:
            st.toast(f"Καταγράφηκαν {n_new} νέα στοιχήματα στο Αρχείο.")
    except Exception as e:  # noqa: BLE001  — η καταγραφή δεν πρέπει ποτέ να ρίξει την εφαρμογή
        st.session_state["rec_err"] = str(e)
    if page == "Αξία":
        cfg_ = O.load_config()
        stale = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(cfg_.get("odds_at", "2000-01-01T00:00:00+00:00"))) > pd.Timedelta(hours=6)
        if cfg_.get("oddspapi_key") and stale and not st.session_state.get("auto_odds_done"):
            st.session_state["auto_odds_done"] = True       # μία προσπάθεια ανά άνοιγμα της εφαρμογής
            with st.spinner("Ανανέωση αποδόσεων…"):
                st.session_state["odds_msg"] = refresh_odds(D)
        page_value(D, nxt, w_model)
    elif page == "Αγώνας":
        page_game(D, nxt, w_model)
    elif page == "Άμυνες":
        page_defense(D, nxt, w_model)
    elif page == "Αποτελέσματα":
        page_results(D, sig)
    elif page == "Παίκτης":
        page_player(D, nxt, w_model)
    elif page == "Αποδόσεις":
        page_odds(D)
    else:
        page_data(D)

if SYNC:
    SYNC.maybe_push()
