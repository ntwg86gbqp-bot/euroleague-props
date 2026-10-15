"""SQLite: σχήμα και βοηθητικά."""
from __future__ import annotations
import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "elprops.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS games(
  season INTEGER, game_code INTEGER, round INTEGER, date TEXT,
  home_code TEXT, home_name TEXT, away_code TEXT, away_name TEXT,
  played INTEGER, home_score INTEGER, away_score INTEGER,
  PRIMARY KEY(season, game_code));
CREATE TABLE IF NOT EXISTS box(
  season INTEGER, game_code INTEGER, player_id TEXT, player TEXT, team TEXT,
  starter INTEGER, minutes REAL,
  pts INTEGER, fg2m INTEGER, fg2a INTEGER, fg3m INTEGER, fg3a INTEGER,
  ftm INTEGER, fta INTEGER, oreb INTEGER, dreb INTEGER, reb INTEGER,
  ast INTEGER, stl INTEGER, tov INTEGER, blk INTEGER, pf INTEGER,
  PRIMARY KEY(season, game_code, player_id));
CREATE TABLE IF NOT EXISTS odds(
  id INTEGER PRIMARY KEY AUTOINCREMENT, fetched_at TEXT, source TEXT,
  bookmaker TEXT, player_raw TEXT, player_id TEXT,
  stat TEXT, line REAL, side TEXT, price REAL);
CREATE INDEX IF NOT EXISTS odds_k ON odds(player_id, stat, line, side, bookmaker);
CREATE TABLE IF NOT EXISTS players(
  player_id TEXT, season INTEGER, team TEXT, name TEXT, position TEXT, height REAL,
  PRIMARY KEY(player_id, season, team));
CREATE TABLE IF NOT EXISTS absences(player_id TEXT PRIMARY KEY, set_at TEXT);
CREATE TABLE IF NOT EXISTS only_current(player_id TEXT PRIMARY KEY, set_at TEXT);
CREATE TABLE IF NOT EXISTS discarded_bets(player_id TEXT, stat TEXT, line REAL, side TEXT, season INTEGER, game_code INTEGER,
  set_at TEXT, PRIMARY KEY(player_id, stat, line, side, season, game_code));
CREATE TABLE IF NOT EXISTS hidden_bets(player_id TEXT, stat TEXT, line REAL, side TEXT, season INTEGER, game_code INTEGER,
  set_at TEXT, PRIMARY KEY(player_id, stat, line, side, season, game_code));
CREATE TABLE IF NOT EXISTS bets(
  id INTEGER PRIMARY KEY AUTOINCREMENT, placed_at TEXT, player_id TEXT, player TEXT, team TEXT, opp TEXT,
  stat TEXT, line REAL, side TEXT, price REAL, stake REAL, bookmaker TEXT,
  p_model REAL, p_market REAL, p_final REAL, ev REAL, verdict TEXT,
  season INTEGER, game_code INTEGER, game_date TEXT,
  status TEXT DEFAULT 'open', result REAL, pnl REAL, settled_at TEXT);
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    p = Path(path) if path else DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.executescript(SCHEMA)
    from .picks import SCHEMA as PICKS
    con.executescript(PICKS)
    return con
