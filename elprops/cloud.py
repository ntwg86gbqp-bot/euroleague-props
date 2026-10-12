"""Μόνιμη αποθήκευση της βάσης όταν η εφαρμογή τρέχει στο cloud (Streamlit Community Cloud).

Ο δίσκος του server σβήνεται σε κάθε «ύπνο»/επανεκκίνηση. Γι' αυτό η βάση SQLite αποθηκεύεται
συμπιεσμένη σε έναν κλάδο («data») του ιδιωτικού αποθετηρίου GitHub της εφαρμογής, και κατεβαίνει
από εκεί όταν ξεκινά ο server.

- Κάθε αποθήκευση φτιάχνει ένα commit ΧΩΡΙΣ ιστορικό (orphan) και μετακινεί τον κλάδο εκεί (force),
  ώστε το αποθετήριο να μη μεγαλώνει με τους μήνες.
- Μέσα στο ίδιο commit κρατιούνται 7 ημερήσια αντίγραφα (backups/<ημέρα>.sqlite.gz), για επαναφορά
  αν κάτι πάει στραβά.
- Ο κλάδος του κώδικα (main) δεν αγγίζεται, άρα η αποθήκευση δεν προκαλεί νέο deploy.

Ρυθμίσεις (Streamlit secrets):  GITHUB_TOKEN, DATA_REPO = "χρήστης/αποθετήριο", προαιρετικά DATA_BRANCH.
"""
from __future__ import annotations
import base64
import datetime as dt
import gzip
import os
import sqlite3
import tempfile
import threading
import time

import requests

API = os.environ.get("ELPROPS_GH_API", "https://api.github.com")   # αλλάζει μόνο στις δοκιμές
MAIN = "elprops.sqlite.gz"
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


class CloudError(RuntimeError):
    pass


def snapshot(con: sqlite3.Connection) -> bytes:
    """Συνεπές αντίγραφο της βάσης (με το backup API της SQLite), συμπιεσμένο."""
    fd, tmp = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        dst = sqlite3.connect(tmp)
        con.backup(dst)
        dst.close()
        with open(tmp, "rb") as f:
            return gzip.compress(f.read(), compresslevel=6)
    finally:
        os.remove(tmp)


def restore_into(con: sqlite3.Connection, data: bytes) -> None:
    """Αντικαθιστά το περιεχόμενο της ανοιχτής βάσης με το `data` (gzip ή σκέτο SQLite)."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    if not data.startswith(b"SQLite format 3"):
        raise CloudError("Το αρχείο δεν είναι βάση της εφαρμογής.")
    fd, tmp = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        src = sqlite3.connect(tmp)
        tables = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"games", "box"} <= tables:
            src.close()
            raise CloudError("Το αρχείο δεν έχει τους πίνακες της εφαρμογής.")
        src.backup(con)
        src.close()
    finally:
        os.remove(tmp)


class Store:
    def __init__(self, token: str, repo: str, branch: str = "data"):
        self.repo, self.branch = repo.strip().strip("/"), branch
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token.strip()}", "Accept": "application/vnd.github+json",
                               "X-GitHub-Api-Version": "2022-11-28"})

    def _req(self, method: str, path: str, ok=(200, 201), **kw):
        for attempt in range(3):
            try:
                r = self.s.request(method, f"{API}/repos/{self.repo}{path}", timeout=30, **kw)
            except requests.RequestException as e:
                if attempt == 2:
                    raise CloudError(f"Δεν απάντησε το GitHub: {e}") from e
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code in ok:
                return r
            if r.status_code in (401, 403):
                raise CloudError("Το GitHub απέρριψε το κλειδί (GITHUB_TOKEN). Έλεγξε ότι έχει δικαίωμα "
                                 "«Contents: Read and write» σε αυτό το αποθετήριο και ότι δεν έληξε.")
            if r.status_code >= 500 and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            return r
        return r

    def _tree_sha(self) -> str | None:
        r = self._req("GET", f"/git/ref/heads/{self.branch}", ok=(200,))
        if r.status_code != 200:
            return None
        commit = self._req("GET", f"/git/commits/{r.json()['object']['sha']}").json()
        return commit["tree"]["sha"]

    def download(self, path: str = MAIN) -> bytes | None:
        """Το αρχείο από τον κλάδο δεδομένων, ή None αν δεν υπάρχει ακόμα."""
        r = self._req("GET", f"/contents/{path}", ok=(200,), params={"ref": self.branch},
                      headers={"Accept": "application/vnd.github.raw+json"})
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            raise CloudError(f"Αποτυχία λήψης ({r.status_code}): {r.text[:200]}")
        return r.content

    def upload(self, data: bytes, note: str = "") -> None:
        """Ανεβάζει το αρχείο ως νέο commit χωρίς ιστορικό, κρατώντας τα αντίγραφα των άλλων ημερών."""
        blob = self._req("POST", "/git/blobs", json={"content": base64.b64encode(data).decode(),
                                                     "encoding": "base64"})
        if blob.status_code not in (200, 201):
            raise CloudError(f"Αποτυχία αποθήκευσης ({blob.status_code}): {blob.text[:200]}")
        sha = blob.json()["sha"]
        day = DAYS[dt.datetime.now(dt.timezone.utc).weekday()]
        entries = [{"path": MAIN, "mode": "100644", "type": "blob", "sha": sha},
                   {"path": f"backups/{day}.sqlite.gz", "mode": "100644", "type": "blob", "sha": sha}]
        body = {"tree": entries}
        base = self._tree_sha()
        if base:
            body["base_tree"] = base
        tree = self._req("POST", "/git/trees", json=body)
        if tree.status_code not in (200, 201):
            raise CloudError(f"Αποτυχία αποθήκευσης ({tree.status_code}): {tree.text[:200]}")
        msg = "Αποθήκευση δεδομένων" + (f" · {note}" if note else "")
        commit = self._req("POST", "/git/commits", json={"message": msg, "tree": tree.json()["sha"], "parents": []})
        if commit.status_code not in (200, 201):
            raise CloudError(f"Αποτυχία αποθήκευσης ({commit.status_code}): {commit.text[:200]}")
        csha = commit.json()["sha"]
        r = self._req("PATCH", f"/git/refs/heads/{self.branch}", ok=(200,), json={"sha": csha, "force": True})
        if r.status_code != 200:
            r = self._req("POST", "/git/refs", json={"ref": f"refs/heads/{self.branch}", "sha": csha})
            if r.status_code not in (200, 201):
                raise CloudError(f"Αποτυχία αποθήκευσης ({r.status_code}): {r.text[:200]}")


class Syncer:
    """Αποθηκεύει τη βάση όταν αλλάζει, το πολύ μία φορά ανά `every` δευτερόλεπτα.

    Οι αλλαγές μετρώνται με το total_changes της σύνδεσης (όλες οι εγγραφές της εφαρμογής περνούν από αυτήν).
    Αν υπάρχει εκκρεμότητα, ένα χρονόμετρο εξασφαλίζει ότι θα αποθηκευτεί χωρίς να χρειαστεί νέο κλικ."""

    def __init__(self, store: Store, con: sqlite3.Connection, every: float = 180.0):
        self.store, self.con, self.every = store, con, every
        self.lock = threading.Lock()
        self.saved_changes = con.total_changes
        self.last_push = 0.0
        self.last_ok: float | None = None
        self.last_error: str | None = None
        self.timer: threading.Timer | None = None

    def dirty(self) -> bool:
        return self.con.total_changes != self.saved_changes

    def maybe_push(self, force: bool = False) -> None:
        if not self.dirty() and not force:
            return
        wait = self.every - (time.time() - self.last_push)
        if force or wait <= 0:
            threading.Thread(target=self.push, daemon=True).start()
        elif self.timer is None or not self.timer.is_alive():
            self.timer = threading.Timer(wait + 1, self.push)
            self.timer.daemon = True
            self.timer.start()

    def push(self, note: str = "") -> bool:
        if not self.lock.acquire(blocking=False):
            return False
        try:
            changes = self.con.total_changes
            data = snapshot(self.con)
            self.store.upload(data, note)
            self.saved_changes, self.last_push, self.last_ok, self.last_error = changes, time.time(), time.time(), None
            return True
        except Exception as e:  # noqa: BLE001
            self.last_push, self.last_error = time.time(), str(e)
            return False
        finally:
            self.lock.release()
