#!/usr/bin/env python3
"""
NHL Unified Prediction Pipeline - Adapted from V12 CBB Pipeline
================================================================
Machine learning pipeline for NHL game predictions

Pipeline Flow:
    1. Fetch DraftKings odds FIRST (betting universe)
    2. If no odds → Skip predictions, process historical only
    3. If odds exist → Fetch team stats and merge
    4. Fetch historical odds (training data)
    5. Fetch game results ONLY for betting-relevant games
    6. Process archived games and update training data
    7. Train/load ML model
    8. Generate predictions (only for games with odds)
    9. Archive today's data for future training

Usage:
    python nhl_unified_pipeline.py              # Full pipeline
    python nhl_unified_pipeline.py --retrain    # Force model retraining
    python nhl_unified_pipeline.py --quick-odds # Quick odds refresh
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import pytz
import time
import os
import argparse
import random
import re
import hashlib

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for NHL pipeline"""
    # API Keys (same as CBB)
    ODDS_API_KEY_TODAY = "704d21dd7f686383fffd15d45a6d05c8"
    ODDS_API_KEY_HISTORICAL = "89d92a0a17cbbd55aa8fd731388cd1b8"
    ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

    # Sport configuration
    SPORT = "icehockey_nhl"  # Odds API sport key

    # Paths
    TEAM_MAPPING_CSV = "nhl_team_name_mapping.csv"
    MODEL_FILE = "nhl_random_forest_model.pkl"
    PREDICTIONS_OUTPUT = "nhl_today_predictions.csv"

    # Generated file paths
    TODAY_ODDS_RAW = "nhl_today_odds_raw.csv"
    TODAY_ODDS_STANDARDIZED = "nhl_today_odds_standardized.csv"
    TODAY_ODDS_WITH_STATS = "nhl_today_odds_with_stats.csv"

    TEAM_STATS_RAW = "nhl_team_stats_raw.csv"
    TEAM_STATS_STANDARDIZED = "nhl_team_stats.csv"

    HISTORICAL_ODDS_RAW = "nhl_historical_odds_raw.csv"
    HISTORICAL_ODDS_STANDARDIZED = "nhl_historical_odds_standardized.csv"

    GAME_RESULTS_RAW = "nhl_game_results_raw.csv"
    GAME_RESULTS_STANDARDIZED = "nhl_game_results_standardized.csv"

    TRAINING_DATA = "nhl_training_data.csv"
    MERGED_GAMES_ARCHIVE_DIR = "nhl_merged_games_archive"

    # Settings
    SEASON = "2026"  # NHL season (2025-2026 uses end year)
    SEASON_START_DATE = "2025-10-04"  # NHL season start
    HISTORICAL_DAYS_BACK = 120
    WINNER_UPDATE_WINDOW = 2
    CENTRAL_TZ = pytz.timezone('US/Central')

    # Model parameters
    MIN_TRAINING_SAMPLES = 100  # NHL has fewer games than CBB
    MODEL_RETRAIN_DAYS = 5

    # Feature engineering control
    AUTO_DETECT_FEATURE_AVAILABILITY = True
    MIN_AVG_GAMES_FOR_3GAME_ROLLING = 2.0
    MIN_AVG_GAMES_FOR_5GAME_ROLLING = 3.5

    # Hockey Reference URL
    HOCKEY_REFERENCE_URL = f"https://www.hockey-reference.com/leagues/NHL_{SEASON}.html"

    # Request settings
    REQUEST_DELAY = 3
    MAX_RETRIES = 3

    @classmethod
    def get_historical_cutoff_date(cls):
        """Dynamic cutoff for fetching historical data"""
        season_start = datetime.strptime(cls.SEASON_START_DATE, "%Y-%m-%d").date()
        days_back_cutoff = (datetime.now().date() - timedelta(days=cls.HISTORICAL_DAYS_BACK))
        return max(season_start, days_back_cutoff)

    @classmethod
    def get_winner_update_cutoff_date(cls):
        """Dynamic cutoff for updating winners"""
        return datetime.now().date() - timedelta(days=cls.WINNER_UPDATE_WINDOW)


# ============================================================================
# TEAM NAME MAPPING SYSTEM
# ============================================================================

class TeamNameMapper:
    """Handles team name standardization for NHL teams"""

    def __init__(self, mapping_csv_path):
        self.mapping_csv_path = mapping_csv_path
        self.standard_to_odds = {}
        self.odds_to_standard = {}
        self.reference_to_standard = {}
        self.standard_to_reference = {}
        self.unmapped_teams = []
        self.load_mapping()

    def load_mapping(self):
        """Load mapping from CSV file"""
        print(f"📋 Loading NHL team name mapping from {self.mapping_csv_path}...")

        if not os.path.exists(self.mapping_csv_path):
            raise FileNotFoundError(f"Team mapping file not found: {self.mapping_csv_path}")

        try:
            df = pd.read_csv(self.mapping_csv_path)
            if df.empty or 'list1' not in df.columns or 'list2' not in df.columns:
                raise ValueError("Team mapping file missing required columns")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Team mapping file is empty: {self.mapping_csv_path}")

        # CSV has: list1 (standard), list2 (odds API), list3 (hockey reference)
        for _, row in df.iterrows():
            standard = str(row['list1']).strip()
            odds_variant = str(row['list2']).strip()
            reference_variant = str(row.get('list3', row['list2'])).strip()

            if not standard or standard == 'nan':
                continue

            self.standard_to_odds[standard] = odds_variant
            self.odds_to_standard[odds_variant] = standard
            self.standard_to_reference[standard] = reference_variant
            self.reference_to_standard[reference_variant] = standard

        print(f"✅ Loaded {len(self.standard_to_reference)} NHL team mappings")

    def standardize(self, team_name, source='odds_api'):
        """Convert any team name variant to standard name"""
        if pd.isna(team_name) or team_name == '':
            return None

        team_name = str(team_name).strip()

        # If already standard, return as-is
        if team_name in self.standard_to_reference:
            return team_name

        # Try exact match based on source
        if source == 'hockey_reference':
            if team_name in self.reference_to_standard:
                return self.reference_to_standard[team_name]
        elif source == 'odds_api':
            if team_name in self.odds_to_standard:
                return self.odds_to_standard[team_name]

        # Try all mappings as fallback
        if team_name in self.odds_to_standard:
            return self.odds_to_standard[team_name]
        if team_name in self.reference_to_standard:
            return self.reference_to_standard[team_name]

        # Track unmapped team
        self.unmapped_teams.append({'team_name': team_name, 'source': source})
        return team_name

    def get_reference_name(self, standard_name):
        """Convert standard name to hockey reference variant"""
        return self.standard_to_reference.get(standard_name, standard_name)

    def clean_team_name(self, name):
        """Remove rankings and extra characters"""
        if pd.isna(name):
            return None
        return re.sub(r"\s*\(\d+\)", "", str(name)).strip()

    def save_unmapped_teams_report(self, output_file="nhl_unmapped_teams.txt"):
        """Save report of unmapped teams"""
        if not self.unmapped_teams:
            return

        unique_teams = {}
        for entry in self.unmapped_teams:
            team = entry['team_name']
            source = entry['source']
            if team not in unique_teams:
                unique_teams[team] = []
            if source not in unique_teams[team]:
                unique_teams[team].append(source)

        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NHL UNMAPPED TEAMS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total unmapped teams: {len(unique_teams)}\n\n")

            for team, sources in sorted(unique_teams.items()):
                f.write(f"Team: {team}\n")
                f.write(f"   Source(s): {', '.join(sources)}\n\n")

        print(f"\n📝 Unmapped teams report saved to: {output_file}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_game_id(date, team1, team2):
    """Generate unique game_id from date and team names"""
    teams = sorted([str(team1), str(team2)])
    game_str = f"{date}_{teams[0]}_{teams[1]}"
    return hashlib.md5(game_str.encode()).hexdigest()


# ============================================================================
# DATA COLLECTION - TEAM STATS (HOCKEY REFERENCE)
# ============================================================================

def fetch_team_stats(season=Config.SEASON):
    """
    Fetch NHL team stats from Hockey Reference

    Scrapes the Team Statistics table from hockey-reference.com

    Returns:
        DataFrame with team stats
    """
    print(f"\n📊 Fetching NHL team stats for {season} season...")

    url = Config.HOCKEY_REFERENCE_URL
    print(f"   URL: {url}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    try:
        time.sleep(Config.REQUEST_DELAY)
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the team stats table
        # Hockey Reference has multiple tables, we want "stats" or similar
        tables = soup.find_all('table')

        stats_df = None

        # Try to find team statistics table
        for table in tables:
            table_id = table.get('id', '')
            if 'stats' in table_id.lower() or 'team' in table_id.lower():
                # Parse table
                try:
                    df = pd.read_html(str(table))[0]

                    # Check if it looks like a team stats table
                    if len(df) >= 30:  # Should have ~32 teams
                        stats_df = df
                        print(f"   Found table: {table_id} with {len(df)} rows")
                        break
                except:
                    continue

        # If no specific table found, try parsing all tables
        if stats_df is None:
            all_tables = pd.read_html(str(soup))
            for df in all_tables:
                # Look for table with team names
                if len(df) >= 30 and any(col for col in df.columns if 'team' in str(col).lower() or df.iloc[0].astype(str).str.contains('Bruins|Rangers|Maple Leafs', case=False).any()):
                    stats_df = df
                    print(f"   Found stats table with {len(df)} rows")
                    break

        if stats_df is None:
            print("   ⚠️  Could not find team stats table, using fallback")
            return create_fallback_stats()

        # Clean up the dataframe
        # Handle multi-level columns if present
        if isinstance(stats_df.columns, pd.MultiIndex):
            stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]

        # Identify team column
        team_col = None
        for col in stats_df.columns:
            if 'team' in str(col).lower() or stats_df[col].astype(str).str.contains('Bruins|Rangers', case=False).any():
                team_col = col
                break

        if team_col is None:
            # Use first column as team
            team_col = stats_df.columns[0]

        # Rename team column to 'team'
        stats_df = stats_df.rename(columns={team_col: 'team'})

        # Remove any rows that aren't actual teams (headers, totals, etc.)
        stats_df = stats_df[stats_df['team'].notna()]
        stats_df = stats_df[~stats_df['team'].astype(str).str.contains('League|Average|Total', case=False, na=False)]

        # Clean team names
        stats_df['team'] = stats_df['team'].astype(str).str.strip()
        stats_df['team'] = stats_df['team'].str.replace(r'\*', '', regex=True)  # Remove playoff markers

        print(f"✅ Loaded stats for {len(stats_df)} NHL teams")

        # Save raw data
        stats_df.to_csv(Config.TEAM_STATS_RAW, index=False)
        print(f"💾 Saved raw stats to {Config.TEAM_STATS_RAW}")

        return stats_df

    except Exception as e:
        print(f"   ❌ Error fetching stats: {e}")
        print("   Using fallback stats")
        return create_fallback_stats()


def create_fallback_stats():
    """Create fallback stats DataFrame with all 32 NHL teams"""
    teams = [
        "Anaheim Ducks", "Arizona Coyotes", "Boston Bruins", "Buffalo Sabres",
        "Calgary Flames", "Carolina Hurricanes", "Chicago Blackhawks", "Colorado Avalanche",
        "Columbus Blue Jackets", "Dallas Stars", "Detroit Red Wings", "Edmonton Oilers",
        "Florida Panthers", "Los Angeles Kings", "Minnesota Wild", "Montreal Canadiens",
        "Nashville Predators", "New Jersey Devils", "New York Islanders", "New York Rangers",
        "Ottawa Senators", "Philadelphia Flyers", "Pittsburgh Penguins", "San Jose Sharks",
        "Seattle Kraken", "St. Louis Blues", "Tampa Bay Lightning", "Toronto Maple Leafs",
        "Utah Hockey Club", "Vancouver Canucks", "Vegas Golden Knights", "Washington Capitals",
        "Winnipeg Jets"
    ]

    # Create minimal stats DataFrame
    data = []
    for team in teams:
        data.append({
            'team': team,
            'GP': 40,  # Default games played
            'W': 20,
            'L': 15,
            'OL': 5,
            'PTS': 45,
            'PTS%': 0.56,
            'GF': 120,
            'GA': 110,
            'GF/G': 3.0,
            'GA/G': 2.75,
            'PP%': 20.0,
            'PK%': 80.0,
            'S/G': 30.0,
            'SA/G': 28.0,
            'SV%': 0.910,
            'SO': 2
        })

    df = pd.DataFrame(data)
    print(f"   Created fallback stats for {len(df)} teams")
    return df


# ============================================================================
# DATA COLLECTION - TODAY'S ODDS
# ============================================================================

def fetch_todays_odds():
    """Fetch NHL odds from The Odds API"""
    print(f"\n💰 Fetching NHL odds for today and tomorrow...")

    games = []
    current_time = datetime.now(pytz.utc)
    current_time_central = current_time.astimezone(Config.CENTRAL_TZ)
    today_date = current_time_central.date()

    url = f"{Config.ODDS_API_BASE_URL}/sports/{Config.SPORT}/odds/?apiKey={Config.ODDS_API_KEY_TODAY}&regions=us&markets=h2h"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if not data:
            print("⚠️  No NHL odds available")
            return pd.DataFrame()

        for match in data:
            try:
                start_time = datetime.fromisoformat(match['commence_time'].replace("Z", "+00:00"))
                start_time_central = start_time.astimezone(Config.CENTRAL_TZ)
                match_date = start_time_central.date()

                days_ahead = (match_date - today_date).days
                if days_ahead < 0 or days_ahead > 1:
                    continue

                bookmaker = next((b for b in match.get('bookmakers', []) if b['title'] == 'DraftKings'), None)
                if not bookmaker:
                    continue

                markets = bookmaker.get('markets', [])
                if not markets:
                    continue

                outcomes = markets[0].get('outcomes', [])
                if len(outcomes) < 2:
                    continue

                games.append({
                    "Sport": Config.SPORT,
                    "Start Time (CT)": start_time_central.strftime("%Y-%m-%d %H:%M:%S"),
                    "Date": match_date.strftime("%Y-%m-%d"),
                    "Team 1": outcomes[0]['name'],
                    "Team 2": outcomes[1]['name'],
                    "Odds 1": outcomes[0]['price'],
                    "Odds 2": outcomes[1]['price'],
                    "Home Team": match['home_team'],
                    "Away Team": match['away_team']
                })

            except (IndexError, KeyError) as e:
                continue

        if not games:
            print("⚠️  No valid NHL games found")
            return pd.DataFrame()

        df = pd.DataFrame(games)
        print(f"✅ Fetched odds for {len(df)} NHL games")

        df.to_csv(Config.TODAY_ODDS_RAW, index=False)
        print(f"💾 Saved raw data to {Config.TODAY_ODDS_RAW}")

        return df

    except Exception as e:
        print(f"❌ Error fetching today's odds: {e}")
        raise


# ============================================================================
# DATA COLLECTION - HISTORICAL ODDS
# ============================================================================

def fetch_historical_odds(days_back=Config.HISTORICAL_DAYS_BACK):
    """Fetch historical NHL odds from The Odds API"""
    print(f"\n📜 Fetching historical NHL odds...")

    if os.path.exists(Config.HISTORICAL_ODDS_RAW):
        try:
            print(f"📂 Loading existing historical odds from {Config.HISTORICAL_ODDS_RAW}")
            df = pd.read_csv(Config.HISTORICAL_ODDS_RAW)
            if df.empty or 'game_id' not in df.columns:
                existing_game_ids = set()
            else:
                existing_game_ids = set(df['game_id'])
                print(f"   Found {len(existing_game_ids)} existing games")
        except:
            existing_game_ids = set()
    else:
        existing_game_ids = set()

    all_games = []
    central_tz = Config.CENTRAL_TZ
    utc_tz = pytz.utc

    cutoff_date_obj = Config.get_historical_cutoff_date()
    print(f"   Fetching games from {cutoff_date_obj} onward")

    now = datetime.now(central_tz)
    yesterday = now - timedelta(days=1)

    times = [f"{hour:02}:00:00Z" for hour in range(0, 24, 4)]

    new_games_count = 0

    for i in range(days_back):
        current_date = (yesterday - timedelta(days=i)).date()
        if current_date < cutoff_date_obj:
            break

        date_str = current_date.strftime("%Y-%m-%d")

        for time_str in times:
            query_timestamp = f"{date_str}T{time_str}"
            url = f"{Config.ODDS_API_BASE_URL}/historical/sports/{Config.SPORT}/odds"
            params = {
                "api_key": Config.ODDS_API_KEY_HISTORICAL,
                "regions": "us",
                "bookmakers": "draftkings",
                "oddsFormat": "decimal",
                "markets": "h2h",
                "dateFormat": "iso",
                "date": query_timestamp,
            }

            try:
                time.sleep(0.5)
                response = requests.get(url, params=params, timeout=15)

                if response.status_code == 200:
                    data = response.json().get('data', [])

                    for game in data:
                        if "bookmakers" in game:
                            for bookmaker in game["bookmakers"]:
                                if bookmaker["key"] == "draftkings":
                                    outcomes = bookmaker["markets"][0]["outcomes"]
                                    if len(outcomes) == 2:
                                        game_id = game["id"]

                                        if game_id in existing_game_ids:
                                            continue

                                        home_team = game.get('home_team', 'Unknown')
                                        away_team = game.get('away_team', 'Unknown')

                                        commence_utc = datetime.strptime(game["commence_time"], "%Y-%m-%dT%H:%M:%SZ")
                                        commence_utc = utc_tz.localize(commence_utc)
                                        commence_central = commence_utc.astimezone(central_tz)
                                        local_date = commence_central.strftime("%Y-%m-%d")

                                        all_games.append({
                                            "game_id": game_id,
                                            "date": local_date,
                                            "team_1": outcomes[0]["name"],
                                            "team_1_odds": outcomes[0]["price"],
                                            "team_2": outcomes[1]["name"],
                                            "team_2_odds": outcomes[1]["price"],
                                            "home_team": home_team,
                                            "away_team": away_team,
                                            "winner": ""
                                        })
                                        new_games_count += 1
                                        existing_game_ids.add(game_id)

                elif response.status_code == 429:
                    print(f"⏳ Rate limited, waiting...")
                    time.sleep(60)

            except Exception as e:
                continue

        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{days_back} days, {new_games_count} new games")

    # Combine with existing
    if os.path.exists(Config.HISTORICAL_ODDS_RAW):
        try:
            existing_df = pd.read_csv(Config.HISTORICAL_ODDS_RAW)
            if not existing_df.empty and all_games:
                new_df = pd.DataFrame(all_games)
                df = pd.concat([existing_df, new_df], ignore_index=True)
            elif not existing_df.empty:
                df = existing_df
            else:
                df = pd.DataFrame(all_games)
        except:
            df = pd.DataFrame(all_games)
    else:
        df = pd.DataFrame(all_games)

    print(f"✅ Total historical games: {len(df)} ({new_games_count} new)")

    if not df.empty:
        df.to_csv(Config.HISTORICAL_ODDS_RAW, index=False)
        print(f"💾 Saved to {Config.HISTORICAL_ODDS_RAW}")

    return df


# ============================================================================
# DATA COLLECTION - GAME RESULTS (ESPN API)
# ============================================================================

def fetch_game_results(days_back=None):
    """Fetch NHL game results from ESPN API"""
    if days_back is None:
        days_back = Config.WINNER_UPDATE_WINDOW + 5

    print(f"\n🏒 Fetching NHL game results (last {days_back} days)...")

    if os.path.exists(Config.GAME_RESULTS_RAW):
        try:
            existing_df = pd.read_csv(Config.GAME_RESULTS_RAW)
            if existing_df.empty or 'Date' not in existing_df.columns:
                existing_df = pd.DataFrame()
                existing_dates = set()
            else:
                existing_dates = set(existing_df['Date'].unique())
                print(f"📂 Found {len(existing_dates)} existing dates")
        except:
            existing_df = pd.DataFrame()
            existing_dates = set()
    else:
        existing_df = pd.DataFrame()
        existing_dates = set()

    cutoff_date = datetime.combine(Config.get_historical_cutoff_date(), datetime.min.time())
    yesterday = datetime.today() - timedelta(days=1)
    current_date = yesterday

    all_results = []

    for _ in range(days_back):
        if current_date < cutoff_date:
            break

        date_str = current_date.strftime("%Y-%m-%d")

        if date_str in existing_dates:
            current_date -= timedelta(days=1)
            continue

        try:
            espn_date_str = current_date.strftime("%Y%m%d")
            url = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
            params = {'dates': espn_date_str}

            time.sleep(0.5)
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            events = data.get('events', [])

            games_found = 0
            for event in events:
                try:
                    status = event.get('status', {}).get('type', {})
                    if not status.get('completed', False):
                        continue

                    competitions = event.get('competitions', [])
                    if not competitions:
                        continue

                    comp = competitions[0]
                    competitors = comp.get('competitors', [])

                    if len(competitors) < 2:
                        continue

                    home_team = None
                    away_team = None
                    winner = None

                    for competitor in competitors:
                        team_name = competitor.get('team', {}).get('displayName', '')
                        is_home = competitor.get('homeAway') == 'home'
                        is_winner = competitor.get('winner', False)

                        if is_home:
                            home_team = team_name
                            if is_winner:
                                winner = team_name
                        else:
                            away_team = team_name
                            if is_winner:
                                winner = team_name

                    if home_team and away_team and winner:
                        all_results.append({
                            'Date': date_str,
                            'Team1': home_team,
                            'Team2': away_team,
                            'Winner': winner
                        })
                        games_found += 1

                except Exception:
                    continue

            if games_found > 0:
                print(f"   {date_str}: {games_found} completed games")

        except Exception as e:
            print(f"⚠️  Error fetching {date_str}: {e}")

        current_date -= timedelta(days=1)

    # Combine with existing
    if not existing_df.empty:
        if all_results:
            new_df = pd.DataFrame(all_results)
            df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            df = existing_df
    else:
        df = pd.DataFrame(all_results)

    print(f"✅ Total game results: {len(df)} ({len(all_results)} new)")

    if not df.empty:
        df.to_csv(Config.GAME_RESULTS_RAW, index=False)
        print(f"💾 Saved to {Config.GAME_RESULTS_RAW}")

    return df


# ============================================================================
# DATA STANDARDIZATION
# ============================================================================

def standardize_team_stats(df_raw, mapper):
    """Standardize team names in stats DataFrame"""
    print("\n🔄 Standardizing team stats...")

    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()

    # Standardize team names
    df['team'] = df['team'].apply(lambda x: mapper.standardize(x, source='hockey_reference'))

    # Remove rows with None team names
    initial_count = len(df)
    df = df[df['team'].notna()]
    removed = initial_count - len(df)
    if removed > 0:
        print(f"   Removed {removed} rows with unmapped team names")

    # Remove duplicates
    df = df.drop_duplicates(subset=['team'], keep='first')

    print(f"✅ Standardized {len(df)} teams")

    df.to_csv(Config.TEAM_STATS_STANDARDIZED, index=False)
    print(f"💾 Saved to {Config.TEAM_STATS_STANDARDIZED}")

    return df


def standardize_todays_odds(df_raw, mapper):
    """Standardize team names in today's odds DataFrame"""
    print("\n🔄 Standardizing today's odds...")

    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()

    df['Team 1'] = df['Team 1'].apply(lambda x: mapper.standardize(x, source='odds_api'))
    df['Team 2'] = df['Team 2'].apply(lambda x: mapper.standardize(x, source='odds_api'))
    df['Home Team'] = df['Home Team'].apply(lambda x: mapper.standardize(x, source='odds_api'))
    df['Away Team'] = df['Away Team'].apply(lambda x: mapper.standardize(x, source='odds_api'))

    df['game_id'] = df.apply(
        lambda row: generate_game_id(row['Date'], row['Team 1'], row['Team 2']),
        axis=1
    )

    cols = ['game_id'] + [col for col in df.columns if col != 'game_id']
    df = df[cols]

    print(f"✅ Standardized {len(df)} games")

    df.to_csv(Config.TODAY_ODDS_STANDARDIZED, index=False)
    print(f"💾 Saved to {Config.TODAY_ODDS_STANDARDIZED}")

    return df


def standardize_historical_odds(df_raw, mapper):
    """Standardize team names in historical odds DataFrame"""
    print("\n🔄 Standardizing historical odds...")

    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()

    df['team_1'] = df['team_1'].apply(lambda x: mapper.standardize(x, source='odds_api'))
    df['team_2'] = df['team_2'].apply(lambda x: mapper.standardize(x, source='odds_api'))
    df['home_team'] = df['home_team'].apply(lambda x: mapper.standardize(x, source='odds_api'))
    df['away_team'] = df['away_team'].apply(lambda x: mapper.standardize(x, source='odds_api'))

    if 'winner' in df.columns:
        df['winner'] = df['winner'].apply(
            lambda x: mapper.standardize(x, source='odds_api') if pd.notna(x) and x != '' else x
        )

    print(f"✅ Standardized {len(df)} games")

    df.to_csv(Config.HISTORICAL_ODDS_STANDARDIZED, index=False)
    print(f"💾 Saved to {Config.HISTORICAL_ODDS_STANDARDIZED}")

    return df


def standardize_game_results(df_raw, mapper):
    """Standardize team names in game results DataFrame"""
    print("\n🔄 Standardizing game results...")

    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()

    df['Team1'] = df['Team1'].apply(lambda x: mapper.standardize(mapper.clean_team_name(x), source='odds_api'))
    df['Team2'] = df['Team2'].apply(lambda x: mapper.standardize(mapper.clean_team_name(x), source='odds_api'))
    df['Winner'] = df['Winner'].apply(lambda x: mapper.standardize(mapper.clean_team_name(x), source='odds_api'))

    # Remove rows with unmapped teams
    initial_count = len(df)
    df = df[df['Team1'].notna() & df['Team2'].notna() & df['Winner'].notna()]
    removed = initial_count - len(df)
    if removed > 0:
        print(f"   Removed {removed} games with unmapped team names")

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date

    print(f"✅ Standardized {len(df)} game results")

    df.to_csv(Config.GAME_RESULTS_STANDARDIZED, index=False)
    print(f"💾 Saved to {Config.GAME_RESULTS_STANDARDIZED}")

    return df


# ============================================================================
# DATA MERGING
# ============================================================================

def merge_today_with_stats(odds_df, stats_df):
    """Merge today's odds with team stats"""
    print("\n🔗 Merging today's odds with team stats...")

    # Merge Team 1 stats
    merged = odds_df.merge(
        stats_df,
        left_on='Team 1',
        right_on='team',
        how='left',
        suffixes=('_team1', '_stats_team1')
    )
    merged.drop(['team'], axis=1, inplace=True, errors='ignore')

    # Merge Team 2 stats
    merged = merged.merge(
        stats_df,
        left_on='Team 2',
        right_on='team',
        how='left',
        suffixes=('_team1', '_stats_team2')
    )
    merged.drop(['team'], axis=1, inplace=True, errors='ignore')

    # Check for missing stats
    # Identify key stat columns dynamically
    team1_cols = [c for c in merged.columns if c.endswith('_team1')]
    team2_cols = [c for c in merged.columns if c.endswith('_stats_team2')]

    if team1_cols and team2_cols:
        missing_team1 = merged[team1_cols[0]].isna().sum()
        missing_team2 = merged[team2_cols[0]].isna().sum()

        if missing_team1 > 0 or missing_team2 > 0:
            print(f"   ⚠️  {missing_team1} games missing Team 1 stats, {missing_team2} missing Team 2 stats")

    merged['data_quality'] = 'COMPLETE'

    print(f"✅ Merged {len(merged)} games")

    merged.to_csv(Config.TODAY_ODDS_WITH_STATS, index=False)
    print(f"💾 Saved to {Config.TODAY_ODDS_WITH_STATS}")

    return merged


def update_historical_winners(historical_df, results_df):
    """Update historical odds with actual game results"""
    print("\n🏆 Updating historical data with winners...")

    if historical_df.empty:
        return pd.DataFrame()

    if results_df.empty:
        return historical_df

    df = historical_df.copy()

    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    results_df['Date'] = pd.to_datetime(results_df['Date'], errors='coerce').dt.date

    winner_update_cutoff = Config.get_winner_update_cutoff_date()

    results_dict = {}
    for _, row in results_df.iterrows():
        key1 = (row['Date'], row['Team1'], row['Team2'])
        key2 = (row['Date'], row['Team2'], row['Team1'])
        results_dict[key1] = row['Winner']
        results_dict[key2] = row['Winner']

    updated_count = 0

    for idx, row in df.iterrows():
        game_date = row['date']
        existing_winner = row.get('winner', '')

        should_update = (
            (game_date < datetime.now().date())
            and
            (pd.isna(existing_winner) or existing_winner == '' or game_date >= winner_update_cutoff)
        )

        if not should_update:
            continue

        key1 = (game_date, row['team_1'], row['team_2'])
        key2 = (game_date, row['team_2'], row['team_1'])

        if key1 in results_dict:
            df.at[idx, 'winner'] = results_dict[key1]
            updated_count += 1
        elif key2 in results_dict:
            df.at[idx, 'winner'] = results_dict[key2]
            updated_count += 1

    print(f"✅ Winners updated: {updated_count}")

    return df


# ============================================================================
# INCREMENTAL TRAINING DATA SYSTEM
# ============================================================================

def archive_todays_merged_data(merged_df, game_date):
    """Save today's merged data to archive"""
    if merged_df.empty:
        return

    os.makedirs(Config.MERGED_GAMES_ARCHIVE_DIR, exist_ok=True)
    archive_file = os.path.join(Config.MERGED_GAMES_ARCHIVE_DIR, f"{game_date}.csv")

    if os.path.exists(archive_file):
        try:
            existing_df = pd.read_csv(archive_file)
            if 'game_id' in existing_df.columns and 'game_id' in merged_df.columns:
                existing_ids = set(existing_df['game_id'].dropna())
                new_games = merged_df[~merged_df['game_id'].isin(existing_ids)]
                if new_games.empty:
                    print(f"   ✓ Archive for {game_date} already has all games")
                    return
                combined = pd.concat([existing_df, new_games], ignore_index=True)
                combined.to_csv(archive_file, index=False)
                print(f"📦 Appended {len(new_games)} games to {archive_file}")
                return
        except:
            pass

    merged_df.to_csv(archive_file, index=False)
    print(f"📦 Archived {len(merged_df)} games to {archive_file}")


def process_archived_games(results_df):
    """Process archived games: add winners, append to training data"""
    print("\n🔄 Processing archived games...")

    if not os.path.exists(Config.MERGED_GAMES_ARCHIVE_DIR):
        print("   No archive directory found")
        return 0

    archive_files = sorted([
        f for f in os.listdir(Config.MERGED_GAMES_ARCHIVE_DIR)
        if f.endswith('.csv') and 'backup' not in f.lower()
    ])

    if not archive_files:
        print("   No archived games found")
        return 0

    print(f"   Found {len(archive_files)} archived date(s)")

    total_added = 0
    today = datetime.now().date()

    for archive_file in archive_files:
        archive_path = os.path.join(Config.MERGED_GAMES_ARCHIVE_DIR, archive_file)
        game_date_str = archive_file.replace('.csv', '')

        try:
            archived_df = pd.read_csv(archive_path)

            if archived_df.empty:
                os.remove(archive_path)
                continue

            game_date = pd.to_datetime(game_date_str).date()

            if game_date >= today:
                continue

            # Check if archive already has winners
            has_winners = 'winner' in archived_df.columns and archived_df['winner'].notna().any()

            if not has_winners and not results_df.empty:
                results_for_date = results_df[results_df['Date'] == game_date]

                if results_for_date.empty:
                    print(f"   {game_date_str}: Results not available yet")
                    continue

                archived_df['winner'] = ''

                team1_col = 'team_1' if 'team_1' in archived_df.columns else 'Team 1'
                team2_col = 'team_2' if 'team_2' in archived_df.columns else 'Team 2'

                results_dict = {}
                for _, row in results_for_date.iterrows():
                    key1 = (row['Team1'], row['Team2'])
                    key2 = (row['Team2'], row['Team1'])
                    results_dict[key1] = row['Winner']
                    results_dict[key2] = row['Winner']

                for idx, row in archived_df.iterrows():
                    key1 = (row[team1_col], row[team2_col])
                    key2 = (row[team2_col], row[team1_col])

                    if key1 in results_dict:
                        archived_df.at[idx, 'winner'] = results_dict[key1]
                    elif key2 in results_dict:
                        archived_df.at[idx, 'winner'] = results_dict[key2]

            # Only add games with winners
            games_with_winners = archived_df[archived_df['winner'].notna() & (archived_df['winner'] != '')].copy()

            if len(games_with_winners) == 0:
                continue

            # Standardize column names
            rename_map = {
                'Team 1': 'team_1',
                'Team 2': 'team_2',
                'Home Team': 'home_team',
                'Away Team': 'away_team',
                'Odds 1': 'team_1_odds',
                'Odds 2': 'team_2_odds',
                'Date': 'date'
            }
            games_with_winners = games_with_winners.rename(columns=rename_map)

            # Drop metadata columns
            cols_to_drop = ['Sport', 'Start Time (CT)']
            games_with_winners = games_with_winners.drop(columns=cols_to_drop, errors='ignore')

            # Generate game_id if not present
            if 'game_id' not in games_with_winners.columns:
                games_with_winners['game_id'] = games_with_winners.apply(
                    lambda row: generate_game_id(row['date'], row['team_1'], row['team_2']),
                    axis=1
                )

            # Apply feature engineering
            games_with_winners = feature_engineering(games_with_winners)

            # Append to training data
            if os.path.exists(Config.TRAINING_DATA):
                training_df = pd.read_csv(Config.TRAINING_DATA)
                if 'game_id' in games_with_winners.columns and 'game_id' in training_df.columns:
                    combined = pd.concat([training_df, games_with_winners], ignore_index=True)
                    combined = combined.drop_duplicates(subset=['game_id'], keep='last')
                else:
                    combined = pd.concat([training_df, games_with_winners], ignore_index=True)
                combined.to_csv(Config.TRAINING_DATA, index=False)
            else:
                games_with_winners.to_csv(Config.TRAINING_DATA, index=False)

            print(f"   {game_date_str}: Added {len(games_with_winners)} games to training data")

            # Remove processed archive
            os.remove(archive_path)

            total_added += len(games_with_winners)

        except Exception as e:
            print(f"   {game_date_str}: Error - {e}")
            continue

    if total_added > 0:
        print(f"✅ Added {total_added} games to training data")

    return total_added


# ============================================================================
# FEATURE ENGINEERING (NHL-SPECIFIC)
# ============================================================================

def assess_data_availability(df):
    """Analyze dataset to determine which features can be reliably calculated"""
    team1_counts = df['team_1'].value_counts()
    team2_counts = df['team_2'].value_counts()

    all_teams = set(df['team_1'].unique()) | set(df['team_2'].unique())
    team_games = {}
    for team in all_teams:
        count = team1_counts.get(team, 0) + team2_counts.get(team, 0)
        team_games[team] = count

    avg_games_per_team = np.mean(list(team_games.values()))

    return {
        'avg_games_per_team': avg_games_per_team,
        'total_teams': len(all_teams),
        'total_games': len(df),
        'enable_3game_rolling': avg_games_per_team >= Config.MIN_AVG_GAMES_FOR_3GAME_ROLLING,
        'enable_5game_rolling': avg_games_per_team >= Config.MIN_AVG_GAMES_FOR_5GAME_ROLLING,
    }


def feature_engineering(df):
    """Create derived features for NHL predictions"""
    print("\n⚙️  Engineering NHL features...")

    # Identify available stat columns
    stat_cols = df.columns.tolist()

    # Find columns for team 1 and team 2
    team1_suffix = '_team1'
    team2_suffix = '_stats_team2'

    # Helper function to safely get column value
    def safe_col(col_name):
        if col_name in df.columns:
            return df[col_name].fillna(0)
        return pd.Series([0] * len(df))

    # ========== BASIC STATS DIFFERENTIALS ==========

    # Points percentage differential (key NHL stat)
    if 'PTS%_team1' in df.columns and 'PTS%_stats_team2' in df.columns:
        df['pts_pct_diff'] = safe_col('PTS%_team1') - safe_col('PTS%_stats_team2')
    elif 'PTS_team1' in df.columns and 'GP_team1' in df.columns:
        # Calculate from PTS and GP
        df['pts_pct_team1'] = np.where(safe_col('GP_team1') > 0,
                                        safe_col('PTS_team1') / (safe_col('GP_team1') * 2), 0.5)
        df['pts_pct_team2'] = np.where(safe_col('GP_stats_team2') > 0,
                                        safe_col('PTS_stats_team2') / (safe_col('GP_stats_team2') * 2), 0.5)
        df['pts_pct_diff'] = df['pts_pct_team1'] - df['pts_pct_team2']
    else:
        df['pts_pct_diff'] = 0

    # Win percentage
    if 'W_team1' in df.columns and 'GP_team1' in df.columns:
        df['win_pct_team1'] = np.where(safe_col('GP_team1') > 0,
                                        safe_col('W_team1') / safe_col('GP_team1'), 0.5)
        df['win_pct_team2'] = np.where(safe_col('GP_stats_team2') > 0,
                                        safe_col('W_stats_team2') / safe_col('GP_stats_team2'), 0.5)
        df['win_pct_diff'] = df['win_pct_team1'] - df['win_pct_team2']
    else:
        df['win_pct_diff'] = 0

    # Goals per game differential
    if 'GF/G_team1' in df.columns:
        df['goals_per_game_diff'] = safe_col('GF/G_team1') - safe_col('GF/G_stats_team2')
    elif 'GF_team1' in df.columns:
        df['gf_per_game_team1'] = np.where(safe_col('GP_team1') > 0,
                                           safe_col('GF_team1') / safe_col('GP_team1'), 2.5)
        df['gf_per_game_team2'] = np.where(safe_col('GP_stats_team2') > 0,
                                           safe_col('GF_stats_team2') / safe_col('GP_stats_team2'), 2.5)
        df['goals_per_game_diff'] = df['gf_per_game_team1'] - df['gf_per_game_team2']
    else:
        df['goals_per_game_diff'] = 0

    # Goals against differential (defense)
    if 'GA/G_team1' in df.columns:
        df['goals_against_diff'] = safe_col('GA/G_stats_team2') - safe_col('GA/G_team1')  # Lower is better
    elif 'GA_team1' in df.columns:
        df['ga_per_game_team1'] = np.where(safe_col('GP_team1') > 0,
                                           safe_col('GA_team1') / safe_col('GP_team1'), 2.5)
        df['ga_per_game_team2'] = np.where(safe_col('GP_stats_team2') > 0,
                                           safe_col('GA_stats_team2') / safe_col('GP_stats_team2'), 2.5)
        df['goals_against_diff'] = df['ga_per_game_team2'] - df['ga_per_game_team1']
    else:
        df['goals_against_diff'] = 0

    # Goal differential
    df['goal_diff_team1'] = safe_col('GF_team1') - safe_col('GA_team1')
    df['goal_diff_team2'] = safe_col('GF_stats_team2') - safe_col('GA_stats_team2')
    df['goal_diff_diff'] = df['goal_diff_team1'] - df['goal_diff_team2']

    # ========== SPECIAL TEAMS ==========

    # Power play differential
    if 'PP%_team1' in df.columns:
        df['pp_diff'] = safe_col('PP%_team1') - safe_col('PP%_stats_team2')
    else:
        df['pp_diff'] = 0

    # Penalty kill differential
    if 'PK%_team1' in df.columns:
        df['pk_diff'] = safe_col('PK%_team1') - safe_col('PK%_stats_team2')
    else:
        df['pk_diff'] = 0

    # Special teams combined advantage
    df['special_teams_diff'] = df['pp_diff'] + df['pk_diff']

    # ========== GOALTENDING ==========

    # Save percentage differential
    if 'SV%_team1' in df.columns:
        df['sv_pct_diff'] = safe_col('SV%_team1') - safe_col('SV%_stats_team2')
    else:
        df['sv_pct_diff'] = 0

    # ========== SHOTS ==========

    # Shots per game differential
    if 'S/G_team1' in df.columns:
        df['shots_diff'] = safe_col('S/G_team1') - safe_col('S/G_stats_team2')
    else:
        df['shots_diff'] = 0

    # Shots against differential
    if 'SA/G_team1' in df.columns:
        df['shots_against_diff'] = safe_col('SA/G_stats_team2') - safe_col('SA/G_team1')
    else:
        df['shots_against_diff'] = 0

    # ========== HOME ICE ADVANTAGE ==========

    # Home/away indicator
    if 'home_team' in df.columns and 'team_1' in df.columns:
        df['home_advantage'] = np.where(df['team_1'] == df['home_team'], 1, -1)
    elif 'Home Team' in df.columns and 'Team 1' in df.columns:
        df['home_advantage'] = np.where(df['Team 1'] == df['Home Team'], 1, -1)
    else:
        df['home_advantage'] = 0

    # Home advantage interactions
    df['home_x_pts_diff'] = df['home_advantage'] * df['pts_pct_diff']
    df['home_x_goal_diff'] = df['home_advantage'] * df['goal_diff_diff']

    # ========== COMPOSITE METRICS ==========

    # Overall team strength (combined metric)
    df['team_strength_diff'] = (
        df['pts_pct_diff'] * 2 +
        df['goals_per_game_diff'] * 0.5 +
        df['goals_against_diff'] * 0.5 +
        df['sv_pct_diff'] * 10 +
        df['special_teams_diff'] * 0.02
    )

    # Offensive vs defensive matchup
    df['off_def_mismatch'] = df['goals_per_game_diff'] + df['goals_against_diff']

    # Initialize upset factors
    df['Team1UpsetFactor'] = 0.0
    df['Team2UpsetFactor'] = 0.0

    # ========== ROLLING FEATURES (if enough data) ==========

    if Config.AUTO_DETECT_FEATURE_AVAILABILITY and len(df) > 50:
        data_stats = assess_data_availability(df)
        enable_rolling = data_stats['enable_5game_rolling']

        if enable_rolling and 'winner' in df.columns and df['winner'].notna().any():
            # Calculate recent win percentage
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            df['team1_won'] = (df['winner'] == df['team_1']).astype(int)
            df['team2_won'] = (df['winner'] == df['team_2']).astype(int)

            df['recent_win_pct_team1'] = (
                df.groupby('team_1')['team1_won']
                .apply(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            ).fillna(0.5)

            df['recent_win_pct_team2'] = (
                df.groupby('team_2')['team2_won']
                .apply(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            ).fillna(0.5)

            df['recent_form_diff'] = df['recent_win_pct_team1'] - df['recent_win_pct_team2']

            df = df.drop(columns=['team1_won', 'team2_won'], errors='ignore')
        else:
            df['recent_form_diff'] = 0
    else:
        df['recent_form_diff'] = 0

    print(f"✅ Features engineered, {len(df.columns)} columns")

    return df


# ============================================================================
# ML TRAINING
# ============================================================================

def should_retrain_model(force_retrain=False):
    """Determine if model should be retrained"""
    if force_retrain:
        print("🔄 Force retrain requested")
        return True

    if not os.path.exists(Config.MODEL_FILE):
        print("🆕 No existing model found, training required")
        return True

    model_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(Config.MODEL_FILE))).days
    if model_age_days > Config.MODEL_RETRAIN_DAYS:
        print(f"⏰ Model is {model_age_days} days old, retraining")
        return True

    print(f"✅ Model is {model_age_days} days old, no retrain needed")
    return False


def train_model(training_df):
    """Train RandomForest model with GridSearch"""
    print("\n🤖 Training NHL ML model...")

    df = training_df.copy()

    # Drop metadata columns
    cols_to_drop = ['game_id', 'Sport', 'Start Time (CT)', 'data_quality']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Create binary winner column
    df['winner_binary'] = (df['winner'] == df['team_1']).astype(int)

    # Calculate upset factors
    team_upset_wins = {}
    team_underdog_games = {}

    for _, row in df.iterrows():
        if row['team_1_odds'] > row['team_2_odds']:
            team_underdog_games[row['team_1']] = team_underdog_games.get(row['team_1'], 0) + 1
            if row['team_1'] == row['winner']:
                team_upset_wins[row['team_1']] = team_upset_wins.get(row['team_1'], 0) + 1
        if row['team_2_odds'] > row['team_1_odds']:
            team_underdog_games[row['team_2']] = team_underdog_games.get(row['team_2'], 0) + 1
            if row['team_2'] == row['winner']:
                team_upset_wins[row['team_2']] = team_upset_wins.get(row['team_2'], 0) + 1

    team_upset_factors = {}
    for team, opportunities in team_underdog_games.items():
        wins = team_upset_wins.get(team, 0)
        team_upset_factors[team] = wins / opportunities

    df['Team1UpsetFactor'] = df['team_1'].map(team_upset_factors).fillna(0) * 0.1
    df['Team2UpsetFactor'] = df['team_2'].map(team_upset_factors).fillna(0) * 0.1

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    # Apply feature engineering
    df = feature_engineering(df)

    # Drop non-feature columns
    df = df.drop(columns=['date'], errors='ignore')

    # Prepare training data
    non_feature_cols = ['winner_binary', 'team_1', 'team_2', 'winner',
                        'home_team', 'away_team', 'team_1_odds', 'team_2_odds',
                        'Home Team', 'Away Team', 'Team 1', 'Team 2']

    X = df.drop(columns=[c for c in non_feature_cols if c in df.columns], errors='ignore')
    y = df['winner_binary']

    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])

    # Fill NaN values
    X = X.fillna(0)

    # Time-series split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"   Training samples: {len(X_train)}")
    print(f"   Features: {len(X.columns)}")

    # GridSearch
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [15, 20, 25],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [2, 3]
    }

    print("   Running GridSearchCV...")
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid,
        cv=tscv,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Evaluate
    test_accuracy = accuracy_score(y_test, best_model.predict(X_test))
    print(f"   Test Accuracy: {test_accuracy:.4f}")

    # Calibrate
    print("   Calibrating model...")
    calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=3)
    calibrated_model.fit(X_train, y_train)

    # Feature importance
    final_importances = best_model.feature_importances_
    sorted_idx = np.argsort(final_importances)[::-1]

    print("\n   Top 10 Features:")
    for idx in sorted_idx[:10]:
        print(f"      {X.columns[idx]}: {final_importances[idx]:.4f}")

    # Save model
    with open(Config.MODEL_FILE, 'wb') as f:
        pickle.dump(calibrated_model, f)

    print(f"\n✅ Model saved to {Config.MODEL_FILE}")

    return calibrated_model, X.columns


# ============================================================================
# ML PREDICTION
# ============================================================================

def predict_todays_games(today_df, model, feature_columns):
    """Make predictions for today's games"""
    print("\n🔮 Making predictions for today's NHL games...")

    if today_df.empty:
        print("   ⚠️  No games available for prediction")
        return pd.DataFrame()

    df = today_df.copy()

    # Rename columns
    rename_map = {
        'Team 1': 'team_1',
        'Team 2': 'team_2',
        'Home Team': 'home_team',
        'Away Team': 'away_team',
        'Odds 1': 'team_1_odds',
        'Odds 2': 'team_2_odds',
        'Date': 'date'
    }
    df = df.rename(columns=rename_map)

    # Save data quality
    data_quality_col = df.get('data_quality', pd.Series(['COMPLETE'] * len(df)))

    # Drop metadata columns
    cols_to_drop = ['game_id', 'Sport', 'Start Time (CT)', 'data_quality']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Apply feature engineering
    df = feature_engineering(df)

    # Save team info
    team_info_cols = ['team_1', 'team_2', 'home_team', 'away_team', 'team_1_odds', 'team_2_odds', 'date']
    team_info = df[[c for c in team_info_cols if c in df.columns]].copy()

    # Select only features used in training
    available_features = [f for f in feature_columns if f in df.columns]
    X = df[available_features].fillna(0)

    # Add missing features with zeros
    for f in feature_columns:
        if f not in X.columns:
            X[f] = 0

    X = X[feature_columns]

    # Predict
    predictions = model.predict(X)
    prediction_probs = model.predict_proba(X)

    team1_win_prob = prediction_probs[:, 1]
    team2_win_prob = prediction_probs[:, 0]

    # Create output DataFrame
    results = team_info.copy()
    results['predicted_winner'] = np.where(predictions == 1, team_info['team_1'], team_info['team_2'])
    results['confidence'] = prediction_probs.max(axis=1)

    # Home/away probabilities
    results['home_win_probability'] = np.where(
        team_info['home_team'] == team_info['team_1'],
        team1_win_prob,
        team2_win_prob
    )
    results['away_win_probability'] = np.where(
        team_info['away_team'] == team_info['team_1'],
        team1_win_prob,
        team2_win_prob
    )

    # Home ice adjustment (NHL home ice advantage is ~55%)
    HOME_ICE_ADJUSTMENT = 0.03
    results['home_win_probability'] = np.clip(results['home_win_probability'] + HOME_ICE_ADJUSTMENT, 0.01, 0.99)
    results['away_win_probability'] = np.clip(results['away_win_probability'] - HOME_ICE_ADJUSTMENT, 0.01, 0.99)

    # Normalize
    total_prob = results['home_win_probability'] + results['away_win_probability']
    results['home_win_probability'] = results['home_win_probability'] / total_prob
    results['away_win_probability'] = results['away_win_probability'] / total_prob

    # Value bet calculation
    results['implied_prob_team_1'] = 1 / results['team_1_odds']
    results['implied_prob_team_2'] = 1 / results['team_2_odds']

    # Update team probabilities with home adjustment
    team1_win_prob = np.where(
        team_info['team_1'] == team_info['home_team'],
        results['home_win_probability'],
        results['away_win_probability']
    )
    team2_win_prob = np.where(
        team_info['team_2'] == team_info['home_team'],
        results['home_win_probability'],
        results['away_win_probability']
    )

    results['value_bet'] = (
        ((team1_win_prob > team2_win_prob) & (team1_win_prob > results['implied_prob_team_1'])) |
        ((team2_win_prob > team1_win_prob) & (team2_win_prob > results['implied_prob_team_2']))
    )

    # Rename for output
    results = results.rename(columns={'date': 'game_date'})

    print(f"✅ Predictions made for {len(results)} games")
    print(f"   Value bets identified: {results['value_bet'].sum()}")

    # Save predictions
    results.to_csv(Config.PREDICTIONS_OUTPUT, index=False)
    print(f"💾 Saved predictions to {Config.PREDICTIONS_OUTPUT}")

    return results


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(force_retrain=False):
    """Main pipeline execution"""
    print("\n" + "="*60)
    print("NHL UNIFIED PREDICTION PIPELINE")
    print("="*60)

    start_time = datetime.now()

    # Initialize mapper
    mapper = TeamNameMapper(Config.TEAM_MAPPING_CSV)

    # ========== STEP 1: FETCH ODDS ==========
    print("\n" + "="*60)
    print("STEP 1: FETCH NHL ODDS")
    print("="*60)

    todays_odds_raw = fetch_todays_odds()

    if todays_odds_raw.empty:
        print("\n⚠️  No NHL odds available today")
        todays_odds = pd.DataFrame()
        todays_merged = pd.DataFrame()
        team_stats = pd.DataFrame()
    else:
        todays_odds = standardize_todays_odds(todays_odds_raw, mapper)
        print(f"\n✅ Found {len(todays_odds)} NHL games")

        # ========== STEP 2: FETCH STATS ==========
        print("\n" + "="*60)
        print("STEP 2: FETCH TEAM STATS")
        print("="*60)

        team_stats_raw = fetch_team_stats()
        team_stats = standardize_team_stats(team_stats_raw, mapper)

        # ========== STEP 3: MERGE ==========
        print("\n" + "="*60)
        print("STEP 3: MERGE ODDS + STATS")
        print("="*60)

        todays_merged = merge_today_with_stats(todays_odds, team_stats)

    # ========== STEP 4: HISTORICAL ODDS ==========
    print("\n" + "="*60)
    print("STEP 4: HISTORICAL ODDS")
    print("="*60)

    historical_odds_raw = fetch_historical_odds()
    historical_odds = standardize_historical_odds(historical_odds_raw, mapper)

    # ========== STEP 5: GAME RESULTS ==========
    print("\n" + "="*60)
    print("STEP 5: GAME RESULTS")
    print("="*60)

    game_results_raw = fetch_game_results()
    game_results = standardize_game_results(game_results_raw, mapper)

    # Update historical with winners
    historical_updated = update_historical_winners(historical_odds, game_results)

    # ========== STEP 6: TRAINING DATA ==========
    print("\n" + "="*60)
    print("STEP 6: TRAINING DATA")
    print("="*60)

    # Process archived games
    games_added = process_archived_games(game_results)

    # Load training data
    if os.path.exists(Config.TRAINING_DATA):
        training_data = pd.read_csv(Config.TRAINING_DATA)
        print(f"📂 Loaded training data: {len(training_data)} games")

        if 'game_id' in training_data.columns:
            training_data = training_data.drop_duplicates(subset=['game_id'], keep='first')

        if 'date' in training_data.columns:
            training_data['date'] = pd.to_datetime(training_data['date'])
            training_data = training_data.sort_values('date').reset_index(drop=True)
    else:
        # Build from historical odds
        if not historical_updated.empty:
            # Filter to games with winners
            training_data = historical_updated[
                historical_updated['winner'].notna() & (historical_updated['winner'] != '')
            ].copy()

            # Merge with stats
            if not team_stats.empty:
                training_data = training_data.merge(
                    team_stats, left_on='team_1', right_on='team', how='left', suffixes=('', '_team1')
                )
                training_data = training_data.merge(
                    team_stats, left_on='team_2', right_on='team', how='left', suffixes=('_team1', '_stats_team2')
                )
                training_data = training_data.drop(columns=['team_team1', 'team'], errors='ignore')

            print(f"📂 Created training data from historical: {len(training_data)} games")
            training_data.to_csv(Config.TRAINING_DATA, index=False)
        else:
            training_data = pd.DataFrame()
            print("📂 No training data available yet")

    # ========== STEP 7: ML TRAINING ==========
    print("\n" + "="*60)
    print("STEP 7: ML TRAINING")
    print("="*60)

    if len(training_data) < Config.MIN_TRAINING_SAMPLES:
        print(f"⚠️  Only {len(training_data)} games, need {Config.MIN_TRAINING_SAMPLES} minimum")
        if os.path.exists(Config.MODEL_FILE):
            print("📂 Loading existing model...")
            with open(Config.MODEL_FILE, 'rb') as f:
                model = pickle.load(f)

            # Get feature columns from model if possible
            if hasattr(model, 'estimators') and hasattr(model.estimators[0], 'feature_names_in_'):
                feature_columns = model.estimators[0].feature_names_in_
                print(f"   ✅ Loaded {len(feature_columns)} feature columns from model")
            elif hasattr(model, 'feature_names_in_'):
                feature_columns = model.feature_names_in_
                print(f"   ✅ Loaded {len(feature_columns)} feature columns from model")
            elif len(training_data) > 0:
                # Fallback: derive from training data
                temp_df = training_data.copy()
                temp_df['winner_binary'] = (temp_df['winner'] == temp_df['team_1']).astype(int)
                temp_df = feature_engineering(temp_df)

                non_feature_cols = ['winner_binary', 'team_1', 'team_2', 'winner',
                                    'home_team', 'away_team', 'team_1_odds', 'team_2_odds', 'date']
                feature_columns = [c for c in temp_df.columns if c not in non_feature_cols
                                   and temp_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
            else:
                print("   ⚠️  Cannot determine feature columns, skipping predictions")
                feature_columns = None
        else:
            print("❌ No model available - will archive games for future training")
            model = None
            feature_columns = None
    elif should_retrain_model(force_retrain):
        model, feature_columns = train_model(training_data)
    else:
        print("📂 Loading existing model...")
        with open(Config.MODEL_FILE, 'rb') as f:
            model = pickle.load(f)

        temp_df = training_data.copy()
        temp_df['winner_binary'] = (temp_df['winner'] == temp_df['team_1']).astype(int)
        temp_df = feature_engineering(temp_df)

        non_feature_cols = ['winner_binary', 'team_1', 'team_2', 'winner',
                            'home_team', 'away_team', 'team_1_odds', 'team_2_odds', 'date']
        feature_columns = [c for c in temp_df.columns if c not in non_feature_cols
                           and temp_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

    # ========== STEP 8: PREDICTIONS ==========
    print("\n" + "="*60)
    print("STEP 8: PREDICTIONS")
    print("="*60)

    if len(todays_merged) > 0:
        # Archive today's games regardless of whether predictions are made
        # This ensures training data accumulates even before model exists
        if 'Date' in todays_merged.columns:
            unique_dates = todays_merged['Date'].unique()
            for game_date in unique_dates:
                date_games = todays_merged[todays_merged['Date'] == game_date]
                archive_todays_merged_data(date_games, game_date)
                print(f"   📦 Archived {len(date_games)} games for {game_date}")

        # Make predictions only if model is available
        if model is not None and feature_columns is not None:
            predictions = predict_todays_games(todays_merged, model, feature_columns)

            print("\n📊 PREDICTION SUMMARY:")
            print(f"   Total games: {len(predictions)}")
            print(f"   Value bets: {predictions['value_bet'].sum()}")
        else:
            print("\n⚠️  No predictions made (model not ready)")
            print(f"   Games archived for future training")
    else:
        print("⚠️  No games to process")

    # Save unmapped teams report
    mapper.save_unmapped_teams_report("nhl_unmapped_teams.txt")

    # Completion
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "="*60)
    print(f"✅ PIPELINE COMPLETED in {elapsed:.1f} seconds")
    print("="*60)
    print(f"\nOutput files:")
    print(f"   Predictions: {Config.PREDICTIONS_OUTPUT}")
    print(f"   Training data: {Config.TRAINING_DATA}")
    print(f"   Model: {Config.MODEL_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NHL Unified Prediction Pipeline")
    parser.add_argument('--retrain', action='store_true', help='Force model retraining')
    parser.add_argument('--quick-odds', action='store_true', help='Quick odds refresh only')
    args = parser.parse_args()

    if args.quick_odds:
        print("\n" + "="*60)
        print("⚡ QUICK ODDS REFRESH MODE")
        print("="*60)

        import joblib
        import sys

        start_time = datetime.now()

        # Check required files
        required_files = {
            Config.MODEL_FILE: "Trained model",
            Config.TEAM_STATS_STANDARDIZED: "Team stats",
            Config.TEAM_MAPPING_CSV: "Team mapping"
        }

        missing = [f"{desc}: {path}" for path, desc in required_files.items() if not os.path.exists(path)]
        if missing:
            print("\n❌ Missing files:")
            for m in missing:
                print(f"   {m}")
            print("\nRun full pipeline first: python nhl_unified_pipeline.py")
            sys.exit(1)

        # Initialize
        mapper = TeamNameMapper(Config.TEAM_MAPPING_CSV)
        team_stats_df = pd.read_csv(Config.TEAM_STATS_STANDARDIZED)

        # Fetch odds
        todays_odds_raw = fetch_todays_odds()
        if todays_odds_raw.empty:
            print("\n⚠️  No NHL odds available")
            sys.exit(0)

        todays_odds = standardize_todays_odds(todays_odds_raw, mapper)
        todays_merged = merge_today_with_stats(todays_odds, team_stats_df)

        if todays_merged.empty:
            print("   ❌ No games could be merged")
            sys.exit(1)

        # Load model
        model = joblib.load(Config.MODEL_FILE)

        # Get feature columns
        if hasattr(model, 'estimators'):
            feature_columns = model.estimators[0].feature_names_in_
        else:
            feature_columns = model.feature_names_in_

        # Predict
        predictions_df = predict_todays_games(todays_merged, model, feature_columns)

        # Archive
        game_date = datetime.now().strftime('%Y-%m-%d')
        archive_todays_merged_data(todays_merged, game_date)

        mapper.save_unmapped_teams_report("nhl_unmapped_teams.txt")

        elapsed = (datetime.now() - start_time).total_seconds()
        print("\n" + "="*60)
        print(f"⚡ QUICK REFRESH COMPLETED in {elapsed:.1f} seconds")
        print("="*60)

        sys.exit(0)

    main(force_retrain=args.retrain)
