#!/usr/bin/env python3
"""
V12 Unified NCAA Basketball Betting Pipeline - With Incremental Training Data
==============================================================================
Streamlined all-in-one pipeline with temporally-correct training data

NEW IN V12:
- Incremental training data system (no look-ahead bias!)
- Today's predictions → Archive → Tomorrow's training data
- Each game uses stats from that game's date
- Training dataset grows daily with correct temporal alignment
- BETTING-FIRST APPROACH: Only fetch/predict games with DraftKings odds

Pipeline Flow:
    1. Fetch DraftKings odds FIRST (betting universe)
    2. If no odds → Skip predictions, process historical only
    3. If odds exist → Fetch team stats and merge
    4. Fetch historical odds (training data)
    5. Fetch game results ONLY for betting-relevant games (smart mode)
    6. Process archived games and update training data
    7. Train/load ML model
    8. Generate predictions (only for games with odds)
    9. Archive today's data for future training

Features:
- Prioritizes DraftKings odds (your betting universe)
- Only fetches game results for games you can bet on
- Fetches team stats from manual CSV (you update daily)
- Standardizes team names across all sources
- Archives games (with odds, no winners yet) for future training
- Processes archived games when results available
- Trains ML model with smart retraining
- Generates predictions with value bet identification

Usage:
    python v12_unified_pipeline.py              # Full pipeline (~10 min)
    python v12_unified_pipeline.py --retrain    # Force model retraining
    python v12_unified_pipeline.py --quick-odds # Quick odds refresh (~30 sec)

Daily Workflow:
    1. Export stats from CBB Reference → Save as 1.31ncaa_2025_team_stats.csv
    2. Run: python v12_unified_pipeline.py
    3. Check predictions in today_predictions.csv
    4. (Pipeline automatically archives today's data and processes past games)

Quick Refresh Workflow (throughout the day):
    1. Run: python v12_unified_pipeline.py --quick-odds
    2. Refreshes odds and predictions without re-scraping/training
    3. Use this to check for line movements or new games
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

# Scraper imports (automated data collection)
try:
    from scrape_team_stats_selenium import scrape_team_stats
    from scrape_game_results_selenium import scrape_game_results
    SCRAPERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Scraper modules not available: {e}")
    print("   Falling back to manual CSV workflow")
    SCRAPERS_AVAILABLE = False
from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration"""
    # API Keys - Use environment variables for security
    ODDS_API_KEY_TODAY = os.environ.get("ODDS_API_KEY_TODAY", "")
    ODDS_API_KEY_HISTORICAL = os.environ.get("ODDS_API_KEY_HISTORICAL", "")
    ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

    # Paths
    TEAM_MAPPING_CSV = "1.31team_name_mapping.csv"
    MODEL_FILE = "random_forest_binary_model.pkl"
    PREDICTIONS_OUTPUT = "today_predictions.csv"

    # Generated file paths (for debugging)
    TODAY_ODDS_RAW = "V12_today_odds_raw.csv"
    TODAY_ODDS_STANDARDIZED = "V12_today_odds_standardized.csv"
    TODAY_ODDS_WITH_STATS = "V12_today_odds_with_stats.csv"

    TEAM_STATS_RAW = "V12_team_stats_raw.csv"
    TEAM_STATS_STANDARDIZED = "V12_ncaa_2025_team_stats.csv"

    HISTORICAL_ODDS_RAW = "V12_historical_odds_raw.csv"
    HISTORICAL_ODDS_STANDARDIZED = "V12_historical_odds_standardized.csv"
    HISTORICAL_ODDS_WITH_STATS = "V12_historical_odds_with_stats.csv"

    GAME_RESULTS_RAW = "V12_game_results_raw.csv"
    GAME_RESULTS_STANDARDIZED = "V12_game_results_standardized.csv"

    TRAINING_DATA = "V12_training_data.csv"

    # Reference database for rolling feature calculation ONLY (not used for training)
    REFERENCE_DATABASE = "V12_training_data_LEAKED_BACKUP.csv"  # 5,273 games (Nov 2024 - Nov 2025)
    USE_REFERENCE_DB = False  # DISABLED: Contains leaked data (end-of-season stats)
    USE_REFERENCE_DB_FOR_TRAINING = False  # DISABLED: Avoid train/predict mismatch (re-enable mid-January)

    # V12 NEW: Incremental training data system
    MERGED_GAMES_ARCHIVE_DIR = "merged_games_archive"  # Games awaiting results
    BASELINE_2024 = "V11_historical_2024_baseline.csv"  # Frozen 2024 data (from V11)

    # Settings
    SEASON = "2026"  # NCAA season year (2025-2026 season uses end year)
    SEASON_START_DATE = "2025-11-03"  # When current season began (today!)
    HISTORICAL_DAYS_BACK = 150  # How far back to fetch historical odds
    WINNER_UPDATE_WINDOW = 2 # Only update winners for games in last N days
    CENTRAL_TZ = pytz.timezone('US/Central')

    # Model parameters
    MIN_TRAINING_SAMPLES = 200  # Minimum games needed to retrain (lowered for current-season-only training)
    MODEL_RETRAIN_DAYS = 5  # Retrain if model older than N days

    # Hybrid stats system (NEW for early season)
    MIN_GAMES_FOR_CURRENT_STATS = 1  # Use current stats if team has played ANY games
    BASELINE_STATS_FILE = "1.31ncaa_2025_team_stats.csv"  # Last year's complete stats (fallback only)

    # Weighted blending parameters (DISABLED - using current season only)
    USE_WEIGHTED_BLEND = False  # DISABLED: No blending, use current season stats only
    BLEND_MIN_GAMES = 5   # Start blending at 5 games (0% current stats below this)
    BLEND_MAX_GAMES = 15  # Full current stats at 15 games (100% current stats above this)

    # ========== FEATURE ENGINEERING CONTROL (Auto-adapts to data availability) ==========
    # These parameters control which features are enabled based on available training data

    # Auto-detect feature availability (recommended: True)
    AUTO_DETECT_FEATURE_AVAILABILITY = True

    # Thresholds for enabling rolling features (based on avg games per team)
    MIN_AVG_GAMES_FOR_3GAME_ROLLING = 2.5   # Enable 3-game rolling when avg >= 2.5 games/team
    MIN_AVG_GAMES_FOR_5GAME_ROLLING = 4.0   # Enable 5-game rolling when avg >= 4.0 games/team

    # Manual feature toggles (only used if AUTO_DETECT_FEATURE_AVAILABILITY = False)
    ENABLE_ROLLING_FEATURES = False  # Enable 5-game rolling features (recent_win_pct, etc.)
    ENABLE_RECENT_EFFICIENCY = False  # Enable recent efficiency features (recent_net_eff, etc.)

    # Static features are ALWAYS enabled (these work with any amount of data)
    # Examples: srs_diff, W-L%, net_eff_diff, shooting%, etc.

    @classmethod
    def get_historical_cutoff_date(cls):
        """
        Dynamic cutoff for fetching historical data
        Returns: Start of current season OR 150 days ago (whichever is more recent)
        """
        season_start = datetime.strptime(cls.SEASON_START_DATE, "%Y-%m-%d").date()
        days_back_cutoff = (datetime.now().date() - timedelta(days=cls.HISTORICAL_DAYS_BACK))
        return max(season_start, days_back_cutoff)

    @classmethod
    def get_winner_update_cutoff_date(cls):
        """
        Dynamic cutoff for updating winners
        Returns: Date from WINNER_UPDATE_WINDOW days ago
        Only games within this window will have winners updated
        """
        return datetime.now().date() - timedelta(days=cls.WINNER_UPDATE_WINDOW)

    # Scraping settings
    REQUEST_DELAY = 3  # Seconds between requests
    MAX_RETRIES = 3


# ============================================================================
# TEAM NAME MAPPING SYSTEM
# ============================================================================

class TeamNameMapper:
    """Handles team name standardization across all data sources"""

    def __init__(self, mapping_csv_path):
        self.mapping_csv_path = mapping_csv_path
        self.standard_to_odds = {}  # standard -> odds API variant
        self.odds_to_standard = {}  # odds API -> standard
        self.reference_to_standard = {}  # sports reference -> standard
        self.standard_to_reference = {}  # standard -> sports reference
        self.unmapped_teams = []  # Track unmapped teams for logging
        self.load_mapping()

    def load_mapping(self):
        """Load mapping from CSV file"""
        print(f"📋 Loading team name mapping from {self.mapping_csv_path}...")

        if not os.path.exists(self.mapping_csv_path):
            raise FileNotFoundError(f"Team mapping file not found: {self.mapping_csv_path}")

        try:
            df = pd.read_csv(self.mapping_csv_path)
            if df.empty or 'list1' not in df.columns or 'list2' not in df.columns:
                raise ValueError(f"Team mapping file is empty or missing required columns (list1, list2)")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Team mapping file is empty: {self.mapping_csv_path}")

        # CSV has: list1 (standard), list2 (sports reference variant)
        for _, row in df.iterrows():
            standard = str(row['list1']).strip()
            reference_variant = str(row['list2']).strip()

            # Skip empty rows
            if not standard or standard == 'nan':
                continue

            # Build bidirectional mappings
            self.standard_to_reference[standard] = reference_variant
            self.reference_to_standard[reference_variant] = standard

            # FIXED: DraftKings API uses SAME format as Sports Reference (list2)!
            # Example: DraftKings sends "VCU Rams", not "Virginia Commonwealth"
            self.standard_to_odds[standard] = reference_variant  # "Virginia Commonwealth" → "VCU Rams"
            self.odds_to_standard[reference_variant] = standard  # "VCU Rams" → "Virginia Commonwealth"

        print(f"✅ Loaded {len(self.standard_to_reference)} team mappings")

    def standardize(self, team_name, source='odds_api'):
        """
        Convert any team name variant to standard name

        Args:
            team_name: Raw team name from data source
            source: 'odds_api', 'sports_reference', or 'standard'

        Returns:
            Standardized team name
        """
        if pd.isna(team_name) or team_name == '':
            return None

        team_name = str(team_name).strip()

        # If already standard, return as-is
        if team_name in self.standard_to_reference:
            return team_name

        # Try exact match based on source
        if source == 'sports_reference':
            if team_name in self.reference_to_standard:
                return self.reference_to_standard[team_name]
        elif source == 'odds_api':
            if team_name in self.odds_to_standard:
                return self.odds_to_standard[team_name]

        # NO FUZZY MATCHING - return original name as fallback (like V10)
        print(f"⚠️  WARNING: Unmapped team '{team_name}' (source: {source})")
        print(f"   Using original name as fallback. Add to {Config.TEAM_MAPPING_CSV} for better mapping.")

        # Track unmapped team for summary report
        self.unmapped_teams.append({
            'team_name': team_name,
            'source': source
        })

        return team_name  # Fallback to original name, like V10 does

    def get_reference_name(self, standard_name):
        """Convert standard name to sports reference variant"""
        return self.standard_to_reference.get(standard_name, standard_name)

    def clean_team_name(self, name):
        """Remove rankings and extra characters"""
        if pd.isna(name):
            return None
        # Remove rankings like (15) or  (15)
        return re.sub(r"\s*\(\d+\)", "", str(name)).strip()

    def save_unmapped_teams_report(self, output_file="unmapped_teams.txt"):
        """Save report of all unmapped teams encountered"""
        if not self.unmapped_teams:
            return  # No unmapped teams to report

        # Get unique unmapped teams
        unique_teams = {}
        for entry in self.unmapped_teams:
            team = entry['team_name']
            source = entry['source']
            if team not in unique_teams:
                unique_teams[team] = []
            if source not in unique_teams[team]:
                unique_teams[team].append(source)

        # Create report
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("UNMAPPED TEAMS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total unmapped teams found: {len(unique_teams)}\n\n")
            f.write("These teams need to be added to: 1.31team_name_mapping.csv\n\n")
            f.write("=" * 80 + "\n")
            f.write("HOW TO ADD TEAMS TO MAPPING FILE:\n")
            f.write("=" * 80 + "\n\n")
            f.write("1. Open: 1.31team_name_mapping.csv\n")
            f.write("2. Add new row with two columns:\n")
            f.write("   - Column 'list1': Sports Reference name (WITHOUT mascot)\n")
            f.write("   - Column 'list2': DraftKings name (WITH mascot)\n\n")
            f.write("Example:\n")
            f.write("   list1,list2\n")
            f.write("   Virginia Commonwealth,VCU Rams\n")
            f.write("   East Texas A&M,East Texas A&M Lions\n\n")
            f.write("=" * 80 + "\n")
            f.write("UNMAPPED TEAMS LIST:\n")
            f.write("=" * 80 + "\n\n")

            for team, sources in sorted(unique_teams.items()):
                f.write(f"Team: {team}\n")
                f.write(f"   Source(s): {', '.join(sources)}\n")

                # Provide appropriate guidance based on source
                if 'sports_reference' in sources:
                    # Sports Reference names are WITHOUT mascot (this is list1)
                    f.write(f"   Add to CSV as:\n")
                    f.write(f"      list1: {team}\n")
                    f.write(f"      list2: [Look up DraftKings name with mascot]\n\n")
                elif 'odds_api' in sources:
                    # DraftKings names are WITH mascot (this is list2)
                    # Try to extract base name by removing common mascots
                    base_name = team
                    common_mascots = [
                        'Wildcats', 'Eagles', 'Tigers', 'Bears', 'Bulldogs', 'Lions',
                        'Panthers', 'Cougars', 'Rams', 'Hawks', 'Falcons', 'Knights',
                        'Warriors', 'Trojans', 'Rebels', 'Huskies', 'Aggies', 'Broncos',
                        'Cardinals', 'Terriers', 'Golden Eagles', 'Great Danes', 'Peacocks',
                        'Revolutionaries', 'Mountaineers', 'Gaels', 'Flames', 'Bruins',
                        'Aztecs', 'Spartans', 'Commodores', 'Crimson Tide', 'Razorbacks'
                    ]
                    for mascot in common_mascots:
                        if team.endswith(f' {mascot}'):
                            base_name = team[:-len(mascot)-1]
                            break

                    f.write(f"   Add to CSV as:\n")
                    f.write(f"      list1: {base_name}\n")
                    f.write(f"      list2: {team}\n\n")
                else:
                    # Unknown source
                    f.write(f"   Add to CSV as:\n")
                    f.write(f"      list1: {team} (or base name without mascot)\n")
                    f.write(f"      list2: {team} (or name with mascot)\n\n")

            f.write("=" * 80 + "\n")
            f.write("NOTES:\n")
            f.write("=" * 80 + "\n\n")
            f.write("- 'odds_api' source = DraftKings odds API\n")
            f.write("- 'sports_reference' source = Sports Reference website\n")
            f.write("- 'standard' source = Already in standard format\n\n")
            f.write("After adding teams, run the pipeline again to use the new mappings.\n")

        print(f"\n📝 Unmapped teams report saved to: {output_file}")
        print(f"   Found {len(unique_teams)} unmapped team(s)")
        print(f"   Please add these to {self.mapping_csv_path} for better data quality")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_game_id(date, team1, team2):
    """
    Generate unique game_id from date and team names (V10 pattern)

    Uses MD5 hash of sorted team names + date to ensure:
    - Consistency regardless of team order
    - Same format as 2024 baseline data
    - Uniqueness for duplicate detection

    Args:
        date: Game date (string or datetime.date)
        team1: First team name (standardized)
        team2: Second team name (standardized)

    Returns:
        32-character hex string (MD5 hash)
    """
    # Sort teams to ensure consistency (team1 vs team2 and team2 vs team1 = same ID)
    teams = sorted([str(team1), str(team2)])
    game_str = f"{date}_{teams[0]}_{teams[1]}"
    return hashlib.md5(game_str.encode()).hexdigest()


def blend_team_stats(baseline_row, current_row, games_played):
    """
    Smooth blend of baseline and current season stats

    Avoids train/predict distribution mismatch by using:
    - Games 0-4: 100% baseline (last year's complete stats)
    - Games 5-14: Linear blend (gradual transition)
    - Games 15+: 100% current (this year's stats)

    This ensures smooth transition and reduces feature distribution jumps.

    Args:
        baseline_row: Last year's complete season stats
        current_row: Current season stats (potentially small sample)
        games_played: Number of games played this season

    Returns:
        DataFrame row with blended statistics

    Examples:
        4 games: 0% current, 100% baseline
        10 games: 50% current, 50% baseline
        15+ games: 100% current, 0% baseline
    """
    # Smooth linear transition between BLEND_MIN_GAMES and BLEND_MAX_GAMES
    if games_played < Config.BLEND_MIN_GAMES:
        current_weight = 0.0
    elif games_played >= Config.BLEND_MAX_GAMES:
        current_weight = 1.0
    else:
        # Linear interpolation between min and max
        current_weight = (games_played - Config.BLEND_MIN_GAMES) / (Config.BLEND_MAX_GAMES - Config.BLEND_MIN_GAMES)

    baseline_weight = 1.0 - current_weight

    # Start with current row structure (preserves non-numeric columns like team name)
    blended = current_row.copy()

    # Define numeric columns to blend
    # These are the core statistics that benefit from blending
    numeric_cols_to_blend = [
        'G', 'W', 'L', 'SRS', 'SOS',  # Games and strength metrics
        'Tm.', 'Opp.',  # Points scored and allowed
        'FG', 'FGA', '3P', '3PA', 'FT', 'FTA',  # Shooting stats
        'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF'  # Box score stats
    ]

    # Blend each numeric column
    for col in numeric_cols_to_blend:
        # Check column exists in both datasets and has valid values
        if col in baseline_row.index and col in current_row.index:
            baseline_val = baseline_row[col]
            current_val = current_row[col]

            # Only blend if both values are non-null
            if pd.notna(baseline_val) and pd.notna(current_val):
                blended[col] = (
                    current_val * current_weight +
                    baseline_val * baseline_weight
                )

    return blended


# ============================================================================
# DATA COLLECTION - TEAM STATS
# ============================================================================

def fetch_team_stats(season=Config.SEASON):
    """
    Fetch current team stats with HYBRID APPROACH for early season

    Strategy:
    1. Load baseline stats (last year's complete data from BASELINE_STATS_FILE)
    2. Try to load current season stats (scraped or manual CSV)
    3. Create hybrid dataset:
       - Use current stats ONLY if team has played >= MIN_GAMES_FOR_CURRENT_STATS
       - Otherwise, use baseline stats from last year
    4. This ensures all teams have stats while transitioning to current season

    Returns:
        DataFrame with team stats (raw team names)
    """
    print(f"\n📊 Fetching team stats for {season} season...")
    if Config.USE_WEIGHTED_BLEND:
        print(f"   🔄 HARD CUTOVER MODE: 100% baseline → 100% current at {Config.BLEND_MAX_GAMES} games")
    else:
        print(f"   🔄 HYBRID MODE: Using {Config.MIN_GAMES_FOR_CURRENT_STATS}-game threshold")

    # ========== STEP 1: Load baseline stats (last year's complete data) ==========
    baseline_stats = None
    if os.path.exists(Config.BASELINE_STATS_FILE):
        try:
            baseline_stats = pd.read_csv(Config.BASELINE_STATS_FILE)
            if not baseline_stats.empty and 'team' in baseline_stats.columns and 'G' in baseline_stats.columns:
                print(f"   📂 Loaded baseline stats: {len(baseline_stats)} teams (last year)")
            else:
                print(f"   ⚠️  Baseline file exists but missing required columns")
                baseline_stats = None
        except Exception as e:
            print(f"   ⚠️  Could not load baseline stats: {e}")
            baseline_stats = None
    else:
        print(f"   ⚠️  Baseline stats file not found: {Config.BASELINE_STATS_FILE}")

    # ========== STEP 2: Try to get current season stats ==========
    current_stats = None

    # Option A: Try automated scraper (if available)
    if SCRAPERS_AVAILABLE:
        try:
            print("   🤖 Attempting automated scrape for current season...")
            df = scrape_team_stats(season=season, headless=True, max_retries=2)

            # Validate scraper results
            if df is not None and not df.empty and 'team' in df.columns and 'G' in df.columns:
                print(f"   ✅ Auto-scraped current season: {len(df)} teams")
                current_stats = df

                # Save scraped data for future reference
                current_stats.to_csv("V12_current_season_stats_scraped.csv", index=False)
            else:
                print("   ⚠️  Scraper returned insufficient data")
        except Exception as e:
            print(f"   ⚠️  Auto-scraping failed: {e}")

    # Option B: Try manually updated current season file
    current_season_file = "V12_current_season_stats_scraped.csv"  # Latest scraped/manual current season
    if current_stats is None and os.path.exists(current_season_file):
        try:
            df = pd.read_csv(current_season_file)
            if not df.empty and 'team' in df.columns and 'G' in df.columns:
                print(f"   📂 Loaded current season from {current_season_file}: {len(df)} teams")
                current_stats = df
        except Exception as e:
            print(f"   ⚠️  Could not load current season file: {e}")

    # ========== STEP 3: Create hybrid dataset ==========
    if baseline_stats is None:
        # No baseline available - must use current stats only (or fail)
        if current_stats is not None:
            print(f"   ⚠️  No baseline available, using current season stats only")
            print(f"   ⚠️  WARNING: Teams with few games will have unreliable stats!")
            df_final = current_stats
        else:
            raise FileNotFoundError(
                f"Cannot fetch team stats: No baseline or current stats found.\n"
                f"Please create: {Config.BASELINE_STATS_FILE} (last year's data)"
            )

    elif current_stats is None:
        # No current stats - use baseline only
        print(f"   ⚠️  No current season stats available, using baseline only")
        df_final = baseline_stats

    else:
        # HYBRID MODE: We have both baseline and current stats
        print(f"\n   🔀 Creating hybrid dataset...")
        if Config.USE_WEIGHTED_BLEND:
            print(f"   📏 Blending Method: Hard cutover (100% baseline → 100% current at {Config.BLEND_MAX_GAMES} games)")
        else:
            print(f"   📏 Threshold: {Config.MIN_GAMES_FOR_CURRENT_STATS} games")

        # Create lookup dictionary for current stats (team -> row data)
        current_stats_dict = {}
        for _, row in current_stats.iterrows():
            team = row['team']
            games_played = row['G']
            if pd.notna(team) and pd.notna(games_played):
                current_stats_dict[team] = row

        # Build hybrid dataset
        hybrid_rows = []
        teams_using_current = 0
        teams_using_baseline = 0
        teams_current_only = 0

        # Process all teams from baseline
        for _, baseline_row in baseline_stats.iterrows():
            team = baseline_row['team']

            # Check if team has current season data with enough games
            if team in current_stats_dict:
                current_row = current_stats_dict[team]
                games_played = current_row['G']

                if Config.USE_WEIGHTED_BLEND and games_played > 0:
                    # WEIGHTED BLEND: Progressive transition from baseline to current
                    # Uses KenPom-style gradual blending based on games played
                    blended_row = blend_team_stats(baseline_row, current_row, games_played)
                    hybrid_rows.append(blended_row)
                    teams_using_current += 1  # Count as "current" for reporting
                elif games_played >= Config.MIN_GAMES_FOR_CURRENT_STATS:
                    # HARD CUTOFF: Use current stats (team has played enough games)
                    # This branch only executes if USE_WEIGHTED_BLEND = False
                    hybrid_rows.append(current_row)
                    teams_using_current += 1
                else:
                    # Use baseline stats (not enough games yet and blending disabled)
                    hybrid_rows.append(baseline_row)
                    teams_using_baseline += 1
            else:
                # Team not in current stats, use baseline
                hybrid_rows.append(baseline_row)
                teams_using_baseline += 1

        # Add teams that are in current stats but NOT in baseline (new teams?)
        baseline_teams = set(baseline_stats['team'].dropna())
        for team, current_row in current_stats_dict.items():
            if team not in baseline_teams:
                hybrid_rows.append(current_row)
                teams_current_only += 1

        df_final = pd.DataFrame(hybrid_rows)

        print(f"   ✅ Hybrid dataset created:")
        print(f"      • {teams_using_current} teams using CURRENT stats ({Config.MIN_GAMES_FOR_CURRENT_STATS}+ games)")
        print(f"      • {teams_using_baseline} teams using BASELINE stats (last year)")
        if teams_current_only > 0:
            print(f"      • {teams_current_only} new teams (current only)")
        print(f"      • {len(df_final)} total teams")

    # ========== STEP 4: Save and return ==========
    if df_final.empty:
        raise ValueError("No team stats available (neither baseline nor current)")

    # Save RAW current stats (no blending) for archiving
    if current_stats is not None and not current_stats.empty:
        current_stats.to_csv("V12_current_season_stats_raw.csv", index=False)
        print(f"\n💾 Saved RAW current stats to V12_current_season_stats_raw.csv")

    # Save blended stats for predictions
    df_final.to_csv(Config.TEAM_STATS_RAW, index=False)
    print(f"💾 Saved blended stats to {Config.TEAM_STATS_RAW}")
    print(f"✅ Loaded stats for {len(df_final)} teams")

    # Return both: raw current stats (for archiving) and blended stats (for predictions)
    return current_stats, df_final


# ============================================================================
# DATA COLLECTION - TODAY'S ODDS
# ============================================================================

def fetch_todays_odds():
    """
    Fetch odds for upcoming games (today and tomorrow only) from The Odds API

    Returns:
        DataFrame with today's odds (raw team names)
    """
    print(f"\n💰 Fetching odds for today and tomorrow...")

    games = []
    current_time = datetime.now(pytz.utc)
    current_time_central = current_time.astimezone(Config.CENTRAL_TZ)
    today_date = current_time_central.date()

    sport = "basketball_ncaab"
    url = f"{Config.ODDS_API_BASE_URL}/sports/{sport}/odds/?apiKey={Config.ODDS_API_KEY_TODAY}&regions=us&markets=h2h"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if not data:
            print("⚠️  No odds available")
            return pd.DataFrame()

        for match in data:
            try:
                # Parse start time
                start_time = datetime.fromisoformat(match['commence_time'].replace("Z", "+00:00"))
                start_time_central = start_time.astimezone(Config.CENTRAL_TZ)
                match_date = start_time_central.date()

                # Only include games for today and tomorrow (0-1 days from now)
                days_ahead = (match_date - today_date).days
                if days_ahead < 0 or days_ahead > 1:
                    continue

                # Get DraftKings bookmaker
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
                    "Sport": sport,
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
                print(f"⚠️  Skipping match due to error: {e}")

        if not games:
            print("⚠️  No valid games found")
            return pd.DataFrame()

        df = pd.DataFrame(games)
        print(f"✅ Fetched odds for {len(df)} games")

        # Save raw data
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
    """
    Fetch historical odds from The Odds API

    Args:
        days_back: Number of days to fetch (from cutoff date)

    Returns:
        DataFrame with historical odds (raw team names)
    """
    print(f"\n📜 Fetching historical odds...")

    # Check for existing file to avoid re-fetching
    if os.path.exists(Config.HISTORICAL_ODDS_RAW):
        try:
            print(f"📂 Loading existing historical odds from {Config.HISTORICAL_ODDS_RAW}")
            df = pd.read_csv(Config.HISTORICAL_ODDS_RAW)
            if df.empty or 'game_id' not in df.columns:
                print("   ⚠️  File exists but is empty, starting fresh")
                existing_game_ids = set()
            else:
                existing_game_ids = set(df['game_id'])
                print(f"   Found {len(existing_game_ids)} existing games")
        except (pd.errors.EmptyDataError, KeyError):
            print("   ⚠️  File exists but can't be read, starting fresh")
            existing_game_ids = set()
    else:
        existing_game_ids = set()

    all_games = []
    central_tz = Config.CENTRAL_TZ
    utc_tz = pytz.utc

    # Use dynamic cutoff date
    cutoff_date_obj = Config.get_historical_cutoff_date()
    print(f"   Fetching games from {cutoff_date_obj} onward")

    now = datetime.now(central_tz)
    yesterday = now - timedelta(days=1)

    # Times to query (every 2 hours)
    times = [f"{hour:02}:00:00Z" for hour in range(0, 24, 2)]

    sport = "basketball_ncaab"
    new_games_count = 0

    for i in range(days_back):
        current_date = (yesterday - timedelta(days=i)).date()
        if current_date < cutoff_date_obj:
            break

        date_str = current_date.strftime("%Y-%m-%d")

        for time_str in times:
            query_timestamp = f"{date_str}T{time_str}"
            url = f"{Config.ODDS_API_BASE_URL}/historical/sports/{sport}/odds"
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
                time.sleep(0.5)  # Rate limiting
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

                                        # Skip if already exists
                                        if game_id in existing_game_ids:
                                            continue

                                        home_team = game.get('home_team', 'Unknown Home Team')
                                        away_team = game.get('away_team', 'Unknown Away Team')

                                        commence_utc = datetime.strptime(game["commence_time"], "%Y-%m-%dT%H:%M:%SZ")
                                        commence_utc = utc_tz.localize(commence_utc)
                                        commence_central = commence_utc.astimezone(central_tz)
                                        local_date = commence_central.strftime("%Y-%m-%d")

                                        last_update = datetime.strptime(bookmaker["last_update"], "%Y-%m-%dT%H:%M:%SZ")
                                        last_update = utc_tz.localize(last_update)

                                        all_games.append({
                                            "game_id": game_id,
                                            "date": local_date,
                                            "last_update": last_update,
                                            "team_1": outcomes[0]["name"],
                                            "team_1_odds": outcomes[0]["price"],
                                            "team_2": outcomes[1]["name"],
                                            "team_2_odds": outcomes[1]["price"],
                                            "home_team": home_team,
                                            "away_team": away_team,
                                            "winner": ""  # Will be filled later
                                        })
                                        new_games_count += 1
                                        existing_game_ids.add(game_id)

                elif response.status_code == 429:
                    print(f"⏳ Rate limited, waiting...")
                    time.sleep(60)

            except Exception as e:
                print(f"⚠️  Error fetching {query_timestamp}: {e}")

        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{days_back} days processed, {new_games_count} new games found")

    # Combine with existing data
    if os.path.exists(Config.HISTORICAL_ODDS_RAW):
        try:
            existing_df = pd.read_csv(Config.HISTORICAL_ODDS_RAW)
            if not existing_df.empty:
                if all_games:
                    new_df = pd.DataFrame(all_games)
                    df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    df = existing_df
            else:
                # Existing file is empty, use new games only
                df = pd.DataFrame(all_games)
        except (pd.errors.EmptyDataError, KeyError):
            # Can't read existing file, use new games only
            df = pd.DataFrame(all_games)
    else:
        df = pd.DataFrame(all_games)

    print(f"✅ Total historical games: {len(df)} ({new_games_count} new)")

    # Only save if there's data (don't create empty files)
    if not df.empty:
        df.to_csv(Config.HISTORICAL_ODDS_RAW, index=False)
        print(f"💾 Saved to {Config.HISTORICAL_ODDS_RAW}")
    else:
        print(f"⚠️  No data to save (season just started)")

    return df


# ============================================================================
# DATA COLLECTION - GAME RESULTS
# ============================================================================

def fetch_game_results(days_back=None):
    """
    Fetch game results from ESPN API (WINNERS ONLY - no stats)

    This function ONLY fetches who won each game. All team stats come from
    the manual CBB Reference export (1.31ncaa_2025_team_stats.csv).

    Args:
        days_back: Number of days back to fetch (defaults to WINNER_UPDATE_WINDOW + buffer)

    Returns:
        DataFrame with game results (raw team names): [Date, Team1, Team2, Winner]
    """
    # Default to update window + 5 day buffer for safety
    if days_back is None:
        days_back = Config.WINNER_UPDATE_WINDOW + 5

    print(f"\n🏀 Fetching game results from ESPN API (last {days_back} days)...")

    # Check for existing file
    if os.path.exists(Config.GAME_RESULTS_RAW):
        try:
            existing_df = pd.read_csv(Config.GAME_RESULTS_RAW)
            # Check if DataFrame is valid
            if existing_df.empty or 'Date' not in existing_df.columns:
                print(f"   ⚠️  File exists but is empty/invalid, starting fresh")
                existing_df = pd.DataFrame()
                existing_dates = set()
            else:
                existing_dates = set(existing_df['Date'].unique())
                print(f"📂 Found {len(existing_dates)} existing dates")
        except (pd.errors.EmptyDataError, KeyError):
            print(f"   ⚠️  File exists but can't be read, starting fresh")
            existing_df = pd.DataFrame()
            existing_dates = set()
    else:
        existing_df = pd.DataFrame()
        existing_dates = set()

    # Use dynamic cutoff
    cutoff_date = datetime.combine(Config.get_historical_cutoff_date(), datetime.min.time())
    print(f"   Fetching results from {cutoff_date.date()} onward")

    yesterday = datetime.today() - timedelta(days=1)
    current_date = yesterday

    all_results = []
    new_dates = 0

    for _ in range(days_back):
        if current_date < cutoff_date:
            break

        date_str = current_date.strftime("%Y-%m-%d")

        # Skip if already exists
        if date_str in existing_dates:
            current_date -= timedelta(days=1)
            continue

        try:
            # ESPN API endpoint (free, no authentication needed)
            espn_date_str = current_date.strftime("%Y%m%d")
            url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
            params = {'dates': espn_date_str}

            time.sleep(0.5)  # Brief rate limiting (ESPN is generous)
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            events = data.get('events', [])

            games_found = 0
            for event in events:
                try:
                    # Only process completed games
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

                    # Extract team names and winner
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

                    # Save result (ESPN gives us displayName which will be standardized later)
                    if home_team and away_team and winner:
                        all_results.append({
                            'Date': date_str,
                            'Team1': home_team,
                            'Team2': away_team,
                            'Winner': winner
                        })
                        games_found += 1

                except Exception as e:
                    # Skip individual game errors
                    continue

            if games_found > 0:
                new_dates += 1
                print(f"   {date_str}: {games_found} completed games")

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Error fetching {date_str}: {e}")
        except Exception as e:
            print(f"⚠️  Unexpected error for {date_str}: {e}")

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

    # Save (only if there's data to save)
    if not df.empty:
        df.to_csv(Config.GAME_RESULTS_RAW, index=False)
        print(f"💾 Saved to {Config.GAME_RESULTS_RAW}")
    else:
        print(f"⚠️  No game results to save (season just started)")

    return df


def fetch_game_results_auto(target_date=None, days_back=None):
    """
    Fetch game results with automated scraping + ESPN API + manual fallback

    Strategy:
    1. Try automated scraper for yesterday's games (high coverage)
    2. Supplement with ESPN API (covers ~20% of games)
    3. Check for manual CSV as final fallback

    Args:
        target_date: Specific date to scrape (defaults to yesterday)
        days_back: Number of days to fetch via ESPN API

    Returns:
        DataFrame with game results (raw team names): [Date, Team1, Team2, Winner]
    """
    print(f"\n🏀 Fetching game results (automated)...")

    all_results = []
    scraped_dates = set()

    # PRIORITY 1: Try automated scraper for yesterday's games
    if SCRAPERS_AVAILABLE:
        try:
            if target_date is None:
                target_date = (datetime.now() - timedelta(days=1)).date()
            elif isinstance(target_date, str):
                target_date = datetime.strptime(target_date, "%Y-%m-%d").date()

            print(f"   🤖 Attempting automated scrape for {target_date}...")
            scraped_df = scrape_game_results(target_date=target_date, headless=True, max_retries=4)

            # Validate scraper results
            if scraped_df is not None and not scraped_df.empty and len(scraped_df) > 0:
                print(f"✅ Auto-scraped {len(scraped_df)} games from Sports Reference")

                # Convert to standard format (Date, Team1, Team2, Winner)
                scraped_df = scraped_df.rename(columns={
                    'Away_Team': 'Team1',
                    'Home_Team': 'Team2'
                })

                # Ensure Date is string format
                scraped_df['Date'] = scraped_df['Date'].astype(str)

                all_results.append(scraped_df[['Date', 'Team1', 'Team2', 'Winner']])
                scraped_dates.add(str(target_date))

                # Save scraped results to manual CSV for backup
                manual_csv = f"manual_winners_{target_date}.csv"
                scraped_df[['Date', 'Team1', 'Team2', 'Winner']].to_csv(manual_csv, index=False)
                print(f"💾 Saved scraped results to {manual_csv}")
            else:
                print("   ⚠️  Scraper returned no data, falling back to ESPN API")
        except Exception as e:
            print(f"   ⚠️  Auto-scraping failed: {e}")
            print("   → Falling back to ESPN API")

    # PRIORITY 2: Fetch from ESPN API (supplements scraper, covers different games)
    print("\n   📡 Fetching from ESPN API...")
    espn_df = fetch_game_results(days_back=days_back)

    if not espn_df.empty:
        # Standardize column names to match scraper format
        if 'Team1' not in espn_df.columns and 'Home_Team' in espn_df.columns:
            espn_df = espn_df.rename(columns={
                'Away_Team': 'Team1',
                'Home_Team': 'Team2'
            })

        print(f"✅ ESPN API returned {len(espn_df)} games")
        all_results.append(espn_df[['Date', 'Team1', 'Team2', 'Winner']])

    # PRIORITY 3: Check for manual CSV files as fallback
    if target_date:
        manual_csv = f"manual_winners_{target_date}.csv"
        if os.path.exists(manual_csv):
            try:
                manual_df = pd.read_csv(manual_csv)
                if not manual_df.empty and 'Winner' in manual_df.columns:
                    print(f"   📂 Found manual CSV: {manual_csv} ({len(manual_df)} games)")

                    # Standardize column names
                    if 'Away_Team' in manual_df.columns:
                        manual_df = manual_df.rename(columns={
                            'Away_Team': 'Team1',
                            'Home_Team': 'Team2'
                        })

                    all_results.append(manual_df[['Date', 'Team1', 'Team2', 'Winner']])
            except Exception as e:
                print(f"   ⚠️  Could not load manual CSV: {e}")

    # Combine all results and deduplicate
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Remove duplicates (prefer scraper results, then ESPN, then manual)
        # Sort by date to ensure consistent deduplication
        combined_df = combined_df.sort_values('Date')

        # Create game key for deduplication (sorted teams + date)
        def make_game_key(row):
            teams = sorted([str(row['Team1']), str(row['Team2'])])
            return f"{row['Date']}_{teams[0]}_{teams[1]}"

        combined_df['_key'] = combined_df.apply(make_game_key, axis=1)
        combined_df = combined_df.drop_duplicates(subset=['_key'], keep='first')
        combined_df = combined_df.drop(columns=['_key'])

        print(f"\n✅ Total unique games: {len(combined_df)}")

        # Save combined results
        combined_df.to_csv(Config.GAME_RESULTS_RAW, index=False)
        print(f"💾 Saved to {Config.GAME_RESULTS_RAW}")

        return combined_df
    else:
        print("⚠️  No game results found from any source")
        return pd.DataFrame(columns=['Date', 'Team1', 'Team2', 'Winner'])


def fetch_game_results_smart(historical_odds_df, todays_merged_df):
    """
    Fetch game results ONLY for games we have odds for (betting-relevant games)

    Uses CBB Reference scraper for comprehensive coverage (gets ALL games for each date)

    Strategy:
    - Extract dates from historical odds and archived games
    - Use CBB Reference scraper for those specific dates
    - Much more comprehensive than ESPN API (ESPN only returns ~20% of games)

    Args:
        historical_odds_df: Historical games with odds (standardized)
        todays_merged_df: Today's games with odds (merged with stats)

    Returns:
        DataFrame with results only for betting-relevant games [Date, Team1, Team2, Winner]
    """
    print("\n🏀 Fetching game results (CBB Reference scraper - betting games only)...")

    # Extract unique dates from games we care about
    relevant_dates = set()

    # Add dates from historical odds
    if not historical_odds_df.empty and 'date' in historical_odds_df.columns:
        dates = pd.to_datetime(historical_odds_df['date'], errors='coerce').dt.date
        relevant_dates.update(dates.dropna().unique())

    # Add dates from archived games (games awaiting results)
    if os.path.exists(Config.MERGED_GAMES_ARCHIVE_DIR):
        archive_files = [f.replace('.csv', '') for f in os.listdir(Config.MERGED_GAMES_ARCHIVE_DIR) if f.endswith('.csv')]
        for date_str in archive_files:
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                relevant_dates.add(date_obj)
            except ValueError:
                continue

    if not relevant_dates:
        print("   No relevant dates to fetch results for")
        return pd.DataFrame(columns=['Date', 'Team1', 'Team2', 'Winner'])

    # Filter out future dates
    today = datetime.now().date()
    relevant_dates = {d for d in relevant_dates if d < today}

    # Filter to only recent dates based on WINNER_UPDATE_WINDOW
    winner_update_cutoff = Config.get_winner_update_cutoff_date()
    relevant_dates = {d for d in relevant_dates if d >= winner_update_cutoff}

    if not relevant_dates:
        print(f"   No dates within winner update window (last {Config.WINNER_UPDATE_WINDOW} days)")
        return pd.DataFrame(columns=['Date', 'Team1', 'Team2', 'Winner'])

    print(f"   Fetching results for {len(relevant_dates)} dates with betting activity (last {Config.WINNER_UPDATE_WINDOW} days)")

    # Fetch results only for these specific dates using CBB Reference scraper
    all_results = []

    for target_date in sorted(relevant_dates):
        try:
            date_str = target_date.strftime("%Y-%m-%d")

            # Use CBB Reference scraper (comprehensive coverage - gets ALL games)
            if SCRAPERS_AVAILABLE:
                scraped_df = scrape_game_results(target_date=target_date, headless=True, max_retries=4)

                if scraped_df is not None and not scraped_df.empty:
                    # Convert to standard format
                    scraped_df = scraped_df.rename(columns={
                        'Away_Team': 'Team1',
                        'Home_Team': 'Team2'
                    })
                    scraped_df['Date'] = scraped_df['Date'].astype(str)

                    games_found = len(scraped_df)
                    print(f"   {date_str}: {games_found} completed games")

                    # Add to results
                    for _, row in scraped_df.iterrows():
                        all_results.append({
                            'Date': row['Date'],
                            'Team1': row['Team1'],
                            'Team2': row['Team2'],
                            'Winner': row['Winner']
                        })
                else:
                    print(f"   {date_str}: No games found (scraper returned empty)")
            else:
                print(f"   ⚠️ Scraper not available, skipping {date_str}")

        except Exception as e:
            print(f"   ⚠️ Error scraping {target_date}: {e}")

    if not all_results:
        print("⚠️  No game results found for betting-relevant dates")
        return pd.DataFrame(columns=['Date', 'Team1', 'Team2', 'Winner'])

    df = pd.DataFrame(all_results)

    # Deduplicate
    def make_game_key(row):
        teams = sorted([str(row['Team1']), str(row['Team2'])])
        return f"{row['Date']}_{teams[0]}_{teams[1]}"

    df['_key'] = df.apply(make_game_key, axis=1)
    df = df.drop_duplicates(subset=['_key'], keep='first')
    df = df.drop(columns=['_key'])

    print(f"✅ Fetched results for {len(df)} games (betting-relevant only)")

    # Save to same location as original function
    df.to_csv(Config.GAME_RESULTS_RAW, index=False)
    print(f"💾 Saved to {Config.GAME_RESULTS_RAW}")

    return df


# ============================================================================
# DATA STANDARDIZATION
# ============================================================================

def standardize_team_stats(df_raw, mapper):
    """Standardize team names in team stats DataFrame"""
    print("\n🔄 Standardizing team stats...")

    if df_raw.empty:
        print("   ⚠️  No data to standardize, returning empty DataFrame")
        return pd.DataFrame()

    df = df_raw.copy()

    # CRITICAL: Rename columns to match 2024 baseline format
    # Scrapers create 'Tm' and 'Opp', but baseline has 'Tm.' and 'Opp.' (with periods)
    if 'Tm' in df.columns and 'Tm.' not in df.columns:
        df = df.rename(columns={'Tm': 'Tm.'})
    if 'Opp' in df.columns and 'Opp.' not in df.columns:
        df = df.rename(columns={'Opp': 'Opp.'})

    # Standardize team names
    df['team'] = df['team'].apply(lambda x: mapper.standardize(x, source='sports_reference'))

    # Remove any rows with None team names
    initial_count = len(df)
    df = df[df['team'].notna()]
    removed = initial_count - len(df)
    if removed > 0:
        print(f"   Removed {removed} rows with unmapped team names")

    # Remove duplicates created by incorrect name standardization
    # This happens when distinct teams (e.g., "UC San Diego" vs "San Diego")
    # are incorrectly mapped to the same standard name
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['team'], keep='first')
    after_dedup = len(df)
    if before_dedup > after_dedup:
        removed_dups = before_dedup - after_dedup
        print(f"   ⚠️  Removed {removed_dups} duplicate teams after standardization")
        print(f"   ⚠️  This indicates incorrect mappings in {Config.TEAM_MAPPING_CSV}")

    print(f"✅ Standardized {len(df)} teams")

    # Save standardized data
    df.to_csv(Config.TEAM_STATS_STANDARDIZED, index=False)
    print(f"💾 Saved to {Config.TEAM_STATS_STANDARDIZED}")

    return df


def standardize_todays_odds(df_raw, mapper):
    """Standardize team names in today's odds DataFrame"""
    print("\n🔄 Standardizing today's odds...")

    if df_raw.empty:
        print("   ⚠️  No data to standardize, returning empty DataFrame")
        return pd.DataFrame()

    df = df_raw.copy()

    # Standardize all team name columns
    df['Team 1'] = df['Team 1'].apply(lambda x: mapper.standardize(x, source='odds_api'))
    df['Team 2'] = df['Team 2'].apply(lambda x: mapper.standardize(x, source='odds_api'))
    df['Home Team'] = df['Home Team'].apply(lambda x: mapper.standardize(x, source='odds_api'))
    df['Away Team'] = df['Away Team'].apply(lambda x: mapper.standardize(x, source='odds_api'))

    # Generate game_id after standardization (V10 pattern)
    df['game_id'] = df.apply(
        lambda row: generate_game_id(row['Date'], row['Team 1'], row['Team 2']),
        axis=1
    )

    # Reorder columns (game_id first, like V10 and 2024 baseline)
    cols = ['game_id'] + [col for col in df.columns if col != 'game_id']
    df = df[cols]

    print(f"✅ Standardized {len(df)} games")

    # Save
    df.to_csv(Config.TODAY_ODDS_STANDARDIZED, index=False)
    print(f"💾 Saved to {Config.TODAY_ODDS_STANDARDIZED}")

    return df


def standardize_historical_odds(df_raw, mapper):
    """Standardize team names in historical odds DataFrame"""
    print("\n🔄 Standardizing historical odds...")

    if df_raw.empty:
        print("   ⚠️  No data to standardize, returning empty DataFrame")
        return pd.DataFrame()

    df = df_raw.copy()

    # Standardize all team name columns
    df['team_1'] = df['team_1'].apply(lambda x: mapper.standardize(x, source='odds_api'))
    df['team_2'] = df['team_2'].apply(lambda x: mapper.standardize(x, source='odds_api'))
    df['home_team'] = df['home_team'].apply(lambda x: mapper.standardize(x, source='odds_api'))
    df['away_team'] = df['away_team'].apply(lambda x: mapper.standardize(x, source='odds_api'))

    # Standardize winner if present
    if 'winner' in df.columns:
        df['winner'] = df['winner'].apply(
            lambda x: mapper.standardize(x, source='odds_api') if pd.notna(x) and x != '' else x
        )

    print(f"✅ Standardized {len(df)} games")

    # Save
    df.to_csv(Config.HISTORICAL_ODDS_STANDARDIZED, index=False)
    print(f"💾 Saved to {Config.HISTORICAL_ODDS_STANDARDIZED}")

    return df


def standardize_game_results(df_raw, mapper):
    """
    Standardize team names in game results DataFrame

    Note: Now uses ESPN API which returns full display names like "Michigan State Spartans"
    We try to match with and without the mascot (last word)
    """
    print("\n🔄 Standardizing game results...")

    if df_raw.empty:
        print("   ⚠️  No data to standardize, returning empty DataFrame")
        return pd.DataFrame()

    df = df_raw.copy()

    def standardize_espn_name(name, mapper):
        """
        Standardize ESPN display name (which includes mascot)

        Handles multi-word mascots and school names:
        - "Michigan State Blue Devils" (2-word mascot)
        - "UC-San Diego Tritons" (hyphenated school name)
        - "Texas A&M-San Antonio Roadrunners" (complex school name)
        """
        if pd.isna(name) or name == '':
            return None

        name = mapper.clean_team_name(name)

        # Try full name first with sports_reference source
        result = mapper.standardize(name, source='sports_reference')
        if result is not None:
            return result

        # Check if it's already a standard name (e.g., "Brigham Young")
        if name in mapper.standard_to_reference:
            return name

        # ESPN often adds mascot at end, but mascots can be 1-3 words:
        # "Michigan State Spartans" (1-word mascot)
        # "Wake Forest Demon Deacons" (2-word mascot)
        # Try removing 1, 2, or 3 words from the end
        words = name.split()
        for mascot_words in [1, 2, 3]:
            if len(words) > mascot_words:
                name_without_mascot = ' '.join(words[:-mascot_words])

                # Try as sports_reference variant (e.g., "BYU Cougars" → "BYU")
                result = mapper.standardize(name_without_mascot, source='sports_reference')
                if result is not None:
                    return result

                # Try as standard name (e.g., "Brigham Young Cougars" → "Brigham Young")
                if name_without_mascot in mapper.standard_to_reference:
                    return name_without_mascot

        # If still not found, return None (will be filtered out)
        return None

    # Clean and standardize
    df['Team1'] = df['Team1'].apply(lambda x: standardize_espn_name(x, mapper))
    df['Team2'] = df['Team2'].apply(lambda x: standardize_espn_name(x, mapper))
    df['Winner'] = df['Winner'].apply(lambda x: standardize_espn_name(x, mapper))

    # Remove rows with unmapped teams
    initial_count = len(df)
    df = df[df['Team1'].notna() & df['Team2'].notna() & df['Winner'].notna()]
    removed = initial_count - len(df)
    if removed > 0:
        print(f"   Removed {removed} games with unmapped team names")

    # Convert dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date

    print(f"✅ Standardized {len(df)} game results")

    # Save
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

    # Drop the 'team' column before second merge (important to avoid conflicts!)
    merged.drop(['team'], axis=1, inplace=True, errors='ignore')

    # Merge Team 2 stats
    merged = merged.merge(
        stats_df,
        left_on='Team 2',
        right_on='team',
        how='left',
        suffixes=('_team1', '_stats_team2')
    )

    # Drop the 'team' column from second merge
    merged.drop(['team'], axis=1, inplace=True, errors='ignore')

    # V10 COMPATIBILITY: Keep ALL games even if some stats are missing
    # (V10 has no dropna filter, includes all DraftKings games regardless of mapping quality)
    missing_team1 = merged['W_team1'].isna().sum()
    missing_team2 = merged['W_stats_team2'].isna().sum()

    if missing_team1 > 0 or missing_team2 > 0:
        print(f"   ⚠️  {missing_team1} games missing Team 1 stats, {missing_team2} missing Team 2 stats")
        print(f"   ℹ️  Keeping all {len(merged)} games (V10 behavior)")

    # Add data quality indicator for each game
    merged['data_quality'] = 'COMPLETE'
    incomplete_mask = merged['W_team1'].isna() | merged['W_stats_team2'].isna()
    merged.loc[incomplete_mask, 'data_quality'] = 'INCOMPLETE_STATS'

    incomplete_count = incomplete_mask.sum()
    if incomplete_count > 0:
        print(f"   📊 Data Quality: {len(merged) - incomplete_count} complete, {incomplete_count} incomplete")
        # List teams with incomplete stats
        incomplete_games = merged[incomplete_mask][['Team 1', 'Team 2']].drop_duplicates()
        print(f"   ⚠️  Games with incomplete stats:")
        for _, game in incomplete_games.iterrows():
            team1_missing = merged[(merged['Team 1'] == game['Team 1']) & merged['W_team1'].isna()].any().any()
            team2_missing = merged[(merged['Team 2'] == game['Team 2']) & merged['W_stats_team2'].isna()].any().any()
            if team1_missing:
                print(f"      - {game['Team 1']} (Team 1) vs {game['Team 2']}")
            if team2_missing:
                print(f"      - {game['Team 1']} vs {game['Team 2']} (Team 2)")

    print(f"✅ Merged {len(merged)} games")

    # Save
    merged.to_csv(Config.TODAY_ODDS_WITH_STATS, index=False)
    print(f"💾 Saved to {Config.TODAY_ODDS_WITH_STATS}")

    return merged


def merge_historical_with_stats(odds_df, stats_df):
    """Merge historical odds with team stats"""
    print("\n🔗 Merging historical odds with team stats...")

    # Handle empty DataFrames
    if odds_df.empty:
        print("   ⚠️  No historical odds to merge (season just started)")
        return pd.DataFrame()

    if stats_df.empty:
        print("   ⚠️  No team stats available")
        return pd.DataFrame()

    # Merge Team 1 stats
    merged = odds_df.merge(
        stats_df,
        left_on='team_1',
        right_on='team',
        how='left',
        suffixes=('_team1', '_stats_team1')
    )

    # Merge Team 2 stats
    merged = merged.merge(
        stats_df,
        left_on='team_2',
        right_on='team',
        how='left',
        suffixes=('_team1', '_stats_team2')
    )

    # Drop rows with missing stats
    initial_count = len(merged)
    merged = merged.dropna(subset=['W_team1', 'W_stats_team2'], how='any')
    merged = merged.dropna()  # Remove any remaining NaN
    removed = initial_count - len(merged)

    if removed > 0:
        print(f"   ⚠️  Removed {removed} games with missing team stats")

    print(f"✅ Merged {len(merged)} games")

    # Save only if there's data
    if not merged.empty:
        merged.to_csv(Config.HISTORICAL_ODDS_WITH_STATS, index=False)
        print(f"💾 Saved to {Config.HISTORICAL_ODDS_WITH_STATS}")

    return merged


def update_historical_winners(historical_df, results_df):
    """
    Update historical odds with actual game results using smart logic

    Strategy:
    - Games with existing non-empty winners AND older than update window: Skip (already validated)
    - Games with empty winners OR within update window: Try to update from scraped results
    - This prevents corrupting old data while keeping recent games fresh
    """
    print("\n🏆 Updating historical data with actual winners...")

    # Handle empty DataFrames
    if historical_df.empty:
        print("   ⚠️  No historical data to update (season just started)")
        return pd.DataFrame()

    if results_df.empty:
        print("   ⚠️  No game results available yet")
        return historical_df  # Return unchanged

    df = historical_df.copy()

    # Convert dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    results_df['Date'] = pd.to_datetime(results_df['Date'], errors='coerce').dt.date

    # Get dynamic cutoff - only update games within this window
    winner_update_cutoff = Config.get_winner_update_cutoff_date()
    print(f"   Update window: Games from {winner_update_cutoff} onward")
    print(f"   (Games older than {Config.WINNER_UPDATE_WINDOW} days with existing winners will be preserved)")

    # Create results lookup dictionary
    results_dict = {}
    for _, row in results_df.iterrows():
        key1 = (row['Date'], row['Team1'], row['Team2'])
        key2 = (row['Date'], row['Team2'], row['Team1'])
        results_dict[key1] = row['Winner']
        results_dict[key2] = row['Winner']

    # Update winners with smart logic
    updated_count = 0
    preserved_count = 0
    missing_count = 0

    today_games_skipped = 0

    for idx, row in df.iterrows():
        game_date = row['date']
        existing_winner = row.get('winner', '')

        # Smart logic: Only update if...
        should_update = (
            # Don't update today's games - they haven't been played yet!
            (game_date < datetime.now().date())
            and
            (
                # Game has no winner yet
                (pd.isna(existing_winner) or existing_winner == '')
                or
                # Game is within the update window (recent games)
                (game_date >= winner_update_cutoff)
            )
        )

        # Track today's games that are skipped
        if game_date == datetime.now().date() and (pd.isna(existing_winner) or existing_winner == ''):
            today_games_skipped += 1

        if not should_update:
            preserved_count += 1
            continue  # Preserve existing winner for old validated games

        # Try to find winner from results
        key1 = (game_date, row['team_1'], row['team_2'])
        key2 = (game_date, row['team_2'], row['team_1'])

        if key1 in results_dict:
            df.at[idx, 'winner'] = results_dict[key1]
            updated_count += 1
        elif key2 in results_dict:
            df.at[idx, 'winner'] = results_dict[key2]
            updated_count += 1
        else:
            # Game within window but no result found (probably hasn't been played yet)
            if game_date < datetime.now().date():
                missing_count += 1

    print(f"✅ Winners updated: {updated_count}")
    print(f"   Preserved (old validated data): {preserved_count}")
    if missing_count > 0:
        print(f"   ⚠️  Missing results for {missing_count} past games (check scraper)")
    if today_games_skipped > 0:
        print(f"   ℹ️  Skipped {today_games_skipped} games scheduled for today (not yet played)")

    return df


# ============================================================================
# V12 NEW: INCREMENTAL TRAINING DATA SYSTEM
# ============================================================================

def archive_todays_merged_data(merged_df, game_date, apply_features=True):
    """
    Save today's merged data (odds + stats) to archive with optional feature engineering

    Uses game_id deduplication to append new games if archive already exists
    (fixes issue where odds released later in day don't get archived)

    These games will be added to training data once results are available

    Args:
        merged_df: Today's games merged with stats
        game_date: Date of games (YYYY-MM-DD string)
        apply_features: Whether to apply feature engineering before archiving (default True)
    """
    if merged_df.empty:
        return

    # Ensure game_id column exists
    if 'game_id' not in merged_df.columns:
        print(f"   ⚠️  Warning: game_id column missing, cannot deduplicate")
        return

    # Apply feature engineering if requested
    if apply_features:
        print(f"   🔧 Applying feature engineering to archive...")

        # Make a proper copy to avoid SettingWithCopyWarning
        merged_df = merged_df.copy()

        # Standardize column names BEFORE feature engineering
        # Use rename to avoid creating duplicate columns
        column_renames = {}

        # Feature engineering expects lowercase 'date', but merge creates 'Date'
        if 'Date' in merged_df.columns and 'date' not in merged_df.columns:
            column_renames['Date'] = 'date'

        # Standardize team column names if needed
        if 'Team 1' in merged_df.columns and 'team_1' not in merged_df.columns:
            column_renames['Team 1'] = 'team_1'
        if 'Team 2' in merged_df.columns and 'team_2' not in merged_df.columns:
            column_renames['Team 2'] = 'team_2'

        # Standardize odds column names
        if 'Odds 1' in merged_df.columns and 'team_1_odds' not in merged_df.columns:
            column_renames['Odds 1'] = 'team_1_odds'
        if 'Odds 2' in merged_df.columns and 'team_2_odds' not in merged_df.columns:
            column_renames['Odds 2'] = 'team_2_odds'

        # Standardize home/away columns
        if 'Home Team' in merged_df.columns and 'home_team' not in merged_df.columns:
            column_renames['Home Team'] = 'home_team'
        if 'Away Team' in merged_df.columns and 'away_team' not in merged_df.columns:
            column_renames['Away Team'] = 'away_team'

        # Apply all renames at once
        if column_renames:
            merged_df = merged_df.rename(columns=column_renames)
            print(f"   ✓ Standardized {len(column_renames)} column names")

        # Add dummy winner column if not present (needed for feature engineering)
        if 'winner' not in merged_df.columns:
            merged_df['winner'] = None  # Will be filled in later by process_archived_games

        try:
            # Apply feature engineering (creates rolling features, win ratios, etc.)
            merged_df = feature_engineering(merged_df)

            # Count engineered features
            feature_count = len([c for c in merged_df.columns if 'recent_' in c or '_adj' in c or '_momentum' in c or '_diff' in c or '_ratio' in c])
            print(f"   ✅ Features engineered: {feature_count} feature columns added")

        except Exception as e:
            print(f"   ⚠️  Feature engineering failed: {e}")
            print(f"   Continuing with raw data only...")

    # Create archive directory if needed
    os.makedirs(Config.MERGED_GAMES_ARCHIVE_DIR, exist_ok=True)

    archive_file = os.path.join(Config.MERGED_GAMES_ARCHIVE_DIR, f"{game_date}.csv")

    # Check if archive already exists
    if os.path.exists(archive_file):
        try:
            # Read existing archive
            existing_df = pd.read_csv(archive_file)

            if 'game_id' not in existing_df.columns:
                print(f"   ⚠️  Existing archive missing game_id, overwriting")
                merged_df.to_csv(archive_file, index=False)
                print(f"📦 Archived {len(merged_df)} games to {archive_file}")
                return

            # Get existing game IDs
            existing_game_ids = set(existing_df['game_id'].dropna())

            # Filter to only NEW games (not in existing archive)
            new_games = merged_df[~merged_df['game_id'].isin(existing_game_ids)].copy()

            if new_games.empty:
                print(f"   ✓ Archive for {game_date} already has all {len(merged_df)} games")
                return

            # Append new games to existing archive
            combined_df = pd.concat([existing_df, new_games], ignore_index=True)
            combined_df.to_csv(archive_file, index=False)

            print(f"📦 Appended {len(new_games)} new games to {archive_file} (total: {len(combined_df)})")

        except Exception as e:
            print(f"   ⚠️  Error reading existing archive: {e}")
            print(f"   Overwriting archive file")
            merged_df.to_csv(archive_file, index=False)
            print(f"📦 Archived {len(merged_df)} games to {archive_file}")
    else:
        # No existing archive, create new one
        merged_df.to_csv(archive_file, index=False)
        print(f"📦 Archived {len(merged_df)} games to {archive_file}")


def process_archived_games(results_df):
    """
    Process archived games: add winners, append to training data

    Workflow:
    1. Find all archived game files
    2. For each archive, try to find results
    3. If results found, add winners and append to training data
    4. Remove from archive (now in training data)

    Args:
        results_df: Game results with Date, Team1, Team2, Winner

    Returns:
        Number of games added to training data
    """
    print("\n🔄 Processing archived games...")

    # Check if archive directory exists
    if not os.path.exists(Config.MERGED_GAMES_ARCHIVE_DIR):
        print("   No archive directory found")
        return 0

    # Get all archived files (exclude backup files)
    archive_files = sorted([
        f for f in os.listdir(Config.MERGED_GAMES_ARCHIVE_DIR)
        if f.endswith('.csv') and not 'backup' in f.lower()
    ])

    if not archive_files:
        print("   No archived games found")
        return 0

    print(f"   Found {len(archive_files)} archived date(s)")

    total_added = 0

    for archive_file in archive_files:
        archive_path = os.path.join(Config.MERGED_GAMES_ARCHIVE_DIR, archive_file)
        game_date_str = archive_file.replace('.csv', '')

        try:
            # Load archived data
            archived_df = pd.read_csv(archive_path)

            if archived_df.empty:
                print(f"   {game_date_str}: Empty archive, removing")
                os.remove(archive_path)
                continue

            # Convert date for matching
            game_date = pd.to_datetime(game_date_str).date()

            # CRITICAL VALIDATION: Only process archives for games that have ALREADY occurred
            from datetime import datetime
            today = datetime.now().date()

            if game_date >= today:
                print(f"   {game_date_str}: Skipping (future date, games haven't occurred yet)")
                continue

            # Check if archive ALREADY has winners (from manual CSV update)
            has_existing_winners = 'winner' in archived_df.columns and archived_df['winner'].notna().any() and (archived_df['winner'] != '').any()

            if not has_existing_winners:
                # No existing winners, try to fetch from ESPN results
                if results_df.empty:
                    print(f"   {game_date_str}: No results available yet ({len(archived_df)} games pending)")
                    continue

                results_for_date = results_df[results_df['Date'] == game_date]

                if results_for_date.empty:
                    print(f"   {game_date_str}: Results not available yet ({len(archived_df)} games pending)")
                    continue

                # Add winners to archived data from ESPN
                archived_df['winner'] = ''  # Initialize winner column if not exists

                # Determine archive column names (may have spaces: "Team 1" or underscores: "team_1")
                team1_col = 'team_1' if 'team_1' in archived_df.columns else 'Team 1'
                team2_col = 'team_2' if 'team_2' in archived_df.columns else 'Team 2'

                # Create results lookup
                results_dict = {}
                for _, row in results_for_date.iterrows():
                    key1 = (row['Team1'], row['Team2'])
                    key2 = (row['Team2'], row['Team1'])
                    results_dict[key1] = row['Winner']
                    results_dict[key2] = row['Winner']

                # DEBUG: Show what we're trying to match
                print(f"   {game_date_str}: Results available for matching:")
                for key, winner in results_dict.items():
                    if key[0] is not None and key[1] is not None:  # Only show valid matches
                        print(f"      {key[0]} vs {key[1]} → Winner: {winner}")

                # Update winners
                winners_found = 0
                for idx, row in archived_df.iterrows():
                    key1 = (row[team1_col], row[team2_col])
                    key2 = (row[team2_col], row[team1_col])

                    if key1 in results_dict:
                        archived_df.at[idx, 'winner'] = results_dict[key1]
                        winners_found += 1
                        print(f"      ✓ Matched: {key1[0]} vs {key1[1]}")
                    elif key2 in results_dict:
                        archived_df.at[idx, 'winner'] = results_dict[key2]
                        winners_found += 1
                        print(f"      ✓ Matched: {key2[0]} vs {key2[1]}")

                if winners_found == 0:
                    print(f"   {game_date_str}: No matching results found ({len(archived_df)} games pending)")
                    print(f"      Archive has: {list(zip(archived_df[team1_col][:3], archived_df[team2_col][:3]))}...")
                    continue
            # else: Winners already exist from manual update, proceed to processing

            # ONLY add games with winners to training data (supervised learning requires target variable)
            games_with_winners = archived_df[archived_df['winner'].notna() & (archived_df['winner'] != '')].copy()
            games_without_winners = archived_df[archived_df['winner'].isna() | (archived_df['winner'] == '')].copy()

            if len(games_with_winners) == 0:
                print(f"   {game_date_str}: No complete games (all missing winners, {len(archived_df)} games pending)")
                continue

            # Standardize column names before appending to training data
            # Archive has "Team 1", "Team 2" etc., training data needs "team_1", "team_2"
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

            # Drop columns that shouldn't be in training data
            # These are from the raw odds fetch and aren't needed for training
            cols_to_drop = ['Sport', 'Start Time (CT)']
            games_with_winners = games_with_winners.drop(columns=cols_to_drop, errors='ignore')

            # Generate game_id if not present (for archives created before game_id was added)
            if 'game_id' not in games_with_winners.columns:
                games_with_winners['game_id'] = games_with_winners.apply(
                    lambda row: generate_game_id(row['date'], row['team_1'], row['team_2']),
                    axis=1
                )

            # CRITICAL FIX: Check if archive has missing stat columns (from old incomplete archives)
            # If missing, fill them in from current team stats to prevent NaN in training data
            missing_stat_cols = ['G_team1', 'W_team1', 'L_team1', 'W-L%_team1', 'SRS_team1', 'SOS_team1']
            has_incomplete_stats = any(col not in games_with_winners.columns for col in missing_stat_cols)

            if has_incomplete_stats:
                print(f"   ⚠️  {game_date_str}: Archive has incomplete stats schema, repairing...")

                # Load RAW current team stats to fill in missing columns (not blended!)
                raw_stats_file = "V12_ncaa_2025_team_stats_RAW.csv"
                if os.path.exists(raw_stats_file):
                    stats_df = pd.read_csv(raw_stats_file)
                    print(f"   Using RAW current season stats for repair (not blended)")
                elif os.path.exists(Config.TEAM_STATS_STANDARDIZED):
                    # Fallback to blended if raw not available
                    stats_df = pd.read_csv(Config.TEAM_STATS_STANDARDIZED)
                    print(f"   ⚠️  WARNING: Using blended stats for repair (RAW stats file not found)")
                else:
                    print(f"   ❌ Cannot repair: No stats files found")
                    stats_df = None

                if stats_df is not None:

                    # Create stats lookup
                    stats_lookup = {}
                    for _, row in stats_df.iterrows():
                        team_name = row['team']
                        if pd.notna(team_name):
                            stats_lookup[team_name] = row

                    # Define stat columns to fill (mapping from training data column to stats file column)
                    stat_columns_to_fill = {
                        'G_team1': 'G', 'W_team1': 'W', 'L_team1': 'L', 'W-L%_team1': 'W-L%',
                        'SRS_team1': 'SRS', 'SOS_team1': 'SOS',
                        'Whome_team1': 'Whome', 'Lhome_team1': 'Lhome',
                        'Waway_team1': 'Waway', 'Laway_team1': 'Laway',
                        'Tm._team1': 'Tm.', 'MP_team1': 'MP',
                        'FG%_team1': 'FG%', '3P%_team1': '3P%', 'FT%_team1': 'FT%',
                        'TRB_team1': 'TRB',
                        'G_stats_team2': 'G', 'W_stats_team2': 'W', 'L_stats_team2': 'L', 'W-L%_stats_team2': 'W-L%',
                        'SRS_stats_team2': 'SRS', 'SOS_stats_team2': 'SOS',
                        'Whome_stats_team2': 'Whome', 'Lhome_stats_team2': 'Lhome',
                        'Waway_stats_team2': 'Waway', 'Laway_stats_team2': 'Laway',
                        'Tm._stats_team2': 'Tm.', 'MP_stats_team2': 'MP',
                        'FG%_stats_team2': 'FG%', '3P%_stats_team2': '3P%', 'FT%_stats_team2': 'FT%',
                        'TRB_stats_team2': 'TRB', 'ORB_stats_team2': 'ORB'
                    }

                    # Fill in missing columns for each game
                    for idx, row in games_with_winners.iterrows():
                        team1 = row['team_1']
                        team2 = row['team_2']

                        # Fill team1 stats
                        if pd.notna(team1) and team1 in stats_lookup:
                            team1_stats = stats_lookup[team1]
                            for train_col, stats_col in stat_columns_to_fill.items():
                                if '_team1' in train_col and train_col not in games_with_winners.columns:
                                    if stats_col in team1_stats.index:
                                        games_with_winners.at[idx, train_col] = team1_stats[stats_col]

                        # Fill team2 stats
                        if pd.notna(team2) and team2 in stats_lookup:
                            team2_stats = stats_lookup[team2]
                            for train_col, stats_col in stat_columns_to_fill.items():
                                if '_stats_team2' in train_col and train_col not in games_with_winners.columns:
                                    if stats_col in team2_stats.index:
                                        games_with_winners.at[idx, train_col] = team2_stats[stats_col]

                    print(f"   ✅ Repaired archive: filled in missing stat columns from current stats")
                else:
                    print(f"   ⚠️  Cannot repair: {Config.TEAM_STATS_STANDARDIZED} not found")

            # ========== CRITICAL: RECALCULATE FEATURES WITH WINNERS ==========
            # Archive had features calculated with winner=None, so rolling features were skipped
            # Now that we have winners, recalculate ALL features for training data
            print(f"   🔧 Recalculating features with winners for training data...")
            try:
                games_with_winners = feature_engineering(games_with_winners)
                feature_count = len([c for c in games_with_winners.columns if 'recent_' in c or '_diff' in c or '_momentum' in c])
                print(f"   ✅ Features recalculated: {feature_count} feature columns")
            except Exception as e:
                print(f"   ⚠️  Feature recalculation failed: {e}")
                print(f"   Continuing with existing features...")

            # Append to training data (ONLY games with winners)
            if os.path.exists(Config.TRAINING_DATA):
                training_df = pd.read_csv(Config.TRAINING_DATA)
                # Combine and remove duplicates by game_id (if exists)
                if 'game_id' in games_with_winners.columns and 'game_id' in training_df.columns:
                    combined = pd.concat([training_df, games_with_winners], ignore_index=True)
                    # Reset index to ensure it's unique before drop_duplicates
                    combined = combined.reset_index(drop=True)
                    combined = combined.drop_duplicates(subset=['game_id'], keep='last')
                else:
                    combined = pd.concat([training_df, games_with_winners], ignore_index=True)
                combined.to_csv(Config.TRAINING_DATA, index=False)
            else:
                # First time - just save archived data
                games_with_winners.to_csv(Config.TRAINING_DATA, index=False)

            print(f"   {game_date_str}: Added {len(games_with_winners)}/{len(archived_df)} games to training data")

            if len(games_without_winners) > 0:
                print(f"   ⚠️  {len(games_without_winners)} games still missing winners (kept in archive)")
                # Update archive with only games still missing winners
                games_without_winners.to_csv(archive_path, index=False)
            else:
                # All games have winners, remove archive
                os.remove(archive_path)

            total_added += len(games_with_winners)

        except Exception as e:
            print(f"   {game_date_str}: Error processing - {e}")
            continue

    if total_added > 0:
        print(f"✅ Added {total_added} games to training data")

    return total_added


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def assess_data_availability(df):
    """
    Analyze dataset to determine which features can be reliably calculated

    Returns:
        dict: Feature availability flags
            - avg_games_per_team: float
            - enable_3game_rolling: bool
            - enable_5game_rolling: bool
            - enable_recent_efficiency: bool
    """
    # Count games per team
    team1_counts = df['team_1'].value_counts()
    team2_counts = df['team_2'].value_counts()

    all_teams = set(df['team_1'].unique()) | set(df['team_2'].unique())
    team_games = {}
    for team in all_teams:
        count = team1_counts.get(team, 0) + team2_counts.get(team, 0)
        team_games[team] = count

    avg_games_per_team = np.mean(list(team_games.values()))
    median_games_per_team = np.median(list(team_games.values()))
    min_games = np.min(list(team_games.values()))
    max_games = np.max(list(team_games.values()))

    # Determine feature availability based on thresholds
    enable_3game = avg_games_per_team >= Config.MIN_AVG_GAMES_FOR_3GAME_ROLLING
    enable_5game = avg_games_per_team >= Config.MIN_AVG_GAMES_FOR_5GAME_ROLLING

    # Recent efficiency requires similar data as rolling features
    enable_recent_eff = enable_3game

    return {
        'avg_games_per_team': avg_games_per_team,
        'median_games_per_team': median_games_per_team,
        'min_games': min_games,
        'max_games': max_games,
        'total_teams': len(all_teams),
        'total_games': len(df),
        'enable_3game_rolling': enable_3game,
        'enable_5game_rolling': enable_5game,
        'enable_recent_efficiency': enable_recent_eff
    }


def feature_engineering(df):
    """
    Create derived features from team stats
    Adaptively enables/disables rolling features based on data availability
    """
    print("\n⚙️  Engineering features...")

    # ========== ASSESS DATA AVAILABILITY ==========
    if Config.AUTO_DETECT_FEATURE_AVAILABILITY:
        data_stats = assess_data_availability(df)
        print(f"   📊 Data availability: {data_stats['total_games']} games, "
              f"{data_stats['total_teams']} teams, "
              f"{data_stats['avg_games_per_team']:.1f} avg games/team")

        enable_rolling = data_stats['enable_5game_rolling']
        enable_3game_rolling = data_stats['enable_3game_rolling']
        enable_recent_eff = data_stats['enable_recent_efficiency']

        if enable_rolling:
            print(f"   ✅ Enabling 5-game rolling features (avg games/team: {data_stats['avg_games_per_team']:.1f} >= {Config.MIN_AVG_GAMES_FOR_5GAME_ROLLING})")
        elif enable_3game_rolling:
            print(f"   ⚙️  Enabling 3-game rolling features (avg games/team: {data_stats['avg_games_per_team']:.1f} >= {Config.MIN_AVG_GAMES_FOR_3GAME_ROLLING})")
        else:
            print(f"   ⚠️  Rolling features DISABLED - insufficient data (avg games/team: {data_stats['avg_games_per_team']:.1f} < {Config.MIN_AVG_GAMES_FOR_3GAME_ROLLING})")
            print(f"      Using static features only (SRS, W-L%, efficiency, shooting%, etc.)")
    else:
        # Manual control
        enable_rolling = Config.ENABLE_ROLLING_FEATURES
        enable_3game_rolling = Config.ENABLE_ROLLING_FEATURES
        enable_recent_eff = Config.ENABLE_RECENT_EFFICIENCY
        print(f"   ⚙️  Manual feature control: rolling={enable_rolling}, recent_eff={enable_recent_eff}")

    # Win Ratio Adjusted for Strength of Schedule (safe division)
    df["win_ratio_adj_team1"] = np.where(
        (df["W_team1"] + df["L_team1"]) > 0,
        df["W_team1"] / (df["W_team1"] + df["L_team1"]) * df["SOS_team1"],
        0
    )
    df["win_ratio_adj_team2"] = np.where(
        (df["W_stats_team2"] + df["L_stats_team2"]) > 0,
        df["W_stats_team2"] / (df["W_stats_team2"] + df["L_stats_team2"]) * df["SOS_stats_team2"],
        0
    )

    # ========== ROLLING FEATURE DATA PREPARATION ==========
    # Only prepare combined dataset if rolling features are enabled
    if enable_rolling or enable_3game_rolling:
        # Ensure date column is datetime
        df['_is_current'] = True
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        combined_df = df.copy()

        # Sort by date for rolling calculations
        combined_df = combined_df.sort_values(by=["date"]).reset_index(drop=True)
        print(f"   Sorted by date: {combined_df['date'].min()} to {combined_df['date'].max()}")
    else:
        # Skip rolling features - no need to prepare combined dataset
        combined_df = None

    # ========== ROLLING FEATURES (Conditional) ==========
    # Only calculate if enabled and combined_df is available
    if (enable_rolling or enable_3game_rolling) and combined_df is not None and "winner" in combined_df.columns and combined_df["winner"].notna().any():
        # TRAINING MODE: Create game-by-game result columns (1 = win, 0 = loss)
        # Use shifted rolling window to avoid look-ahead bias
        print("   Calculating recent form features...")
        print("      Training mode: calculating from historical data")
        combined_df["team1_won"] = (combined_df["winner"] == combined_df["team_1"]).astype(int)
        combined_df["team2_won"] = (combined_df["winner"] == combined_df["team_2"]).astype(int)

        # Calculate rolling win rate using SHIFTED window (excludes current game)
        # shift(1) moves data down by 1 row, so current game is excluded from its own calculation
        combined_df["recent_win_pct_team1"] = (
            combined_df.groupby("team_1")["team1_won"]
            .apply(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

        combined_df["recent_win_pct_team2"] = (
            combined_df.groupby("team_2")["team2_won"]
            .apply(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

        # Fill NaN values (first game for each team) with league average (0.5)
        combined_df["recent_win_pct_team1"] = combined_df["recent_win_pct_team1"].fillna(0.5)
        combined_df["recent_win_pct_team2"] = combined_df["recent_win_pct_team2"].fillna(0.5)

        # Drop temporary columns
        combined_df = combined_df.drop(columns=["team1_won", "team2_won", "_is_current"], errors='ignore')

        # Update df with the rolling features from combined_df
        df = combined_df
    else:
        # PREDICTION MODE: Calculate actual recent form from training data
        print("   ℹ️  Prediction mode: calculating recent form from training data")

        # Helper function to get recent win percentage for a team
        def get_recent_win_pct(team_name, training_file, n_games=5):
            """Look up team's recent win percentage from training data"""
            try:
                # Load training data
                training_df = pd.read_csv(training_file)

                # Find games where this team played (as team_1 or team_2)
                team_games = training_df[
                    (training_df['team_1'] == team_name) |
                    (training_df['team_2'] == team_name)
                ].copy()

                if len(team_games) == 0:
                    return 0.5  # Default if team has no history

                # Sort by date and take last N games
                team_games = team_games.sort_values('date').tail(n_games)

                # Count wins
                wins = sum(team_games['winner'] == team_name)

                return wins / len(team_games)
            except Exception as e:
                print(f"      Warning: Could not calculate recent form for {team_name}: {e}")
                return 0.5  # Default on error

        # Calculate recent win percentage for both teams
        df["recent_win_pct_team1"] = df["team_1"].apply(
            lambda x: get_recent_win_pct(x, Config.TRAINING_DATA)
        )
        df["recent_win_pct_team2"] = df["team_2"].apply(
            lambda x: get_recent_win_pct(x, Config.TRAINING_DATA)
        )

    # ========== DERIVED ROLLING FEATURES (Always needed) ==========
    # These use either rolling values (when enabled) or season averages (when disabled)

    # Initialize recent_* columns with season averages (always needed as baseline)
    # Note: Currently we only calculate recent_win_pct in rolling features,
    # so these other metrics still use season averages for now
    df["recent_pts_scored_team1"] = df["Tm._team1"]
    df["recent_pts_scored_team2"] = df["Tm._stats_team2"]
    df["recent_pts_allowed_team1"] = df["Opp._team1"]
    df["recent_pts_allowed_team2"] = df["Opp._stats_team2"]
    df["recent_fg_pct_team1"] = df["FG%_team1"]
    df["recent_fg_pct_team2"] = df["FG%_stats_team2"]
    df["recent_3p_pct_team1"] = df["3P%_team1"]
    df["recent_3p_pct_team2"] = df["3P%_stats_team2"]
    df["recent_tov_team1"] = df["TOV_team1"]
    df["recent_tov_team2"] = df["TOV_stats_team2"]

    # Safe division for recent win ratio
    df["recent_win_ratio"] = np.where(
        df["recent_win_pct_team1"] > 0,
        df["recent_win_pct_team2"] / df["recent_win_pct_team1"],
        0
    )

    # Scoring momentum differentials
    df["recent_scoring_diff"] = df["recent_pts_scored_team1"] - df["recent_pts_scored_team2"]
    df["recent_defense_diff"] = df["recent_pts_allowed_team1"] - df["recent_pts_allowed_team2"]

    # Shooting momentum (difference from season average)
    df["fg_momentum_team1"] = df["recent_fg_pct_team1"] - df["FG%_team1"]
    df["fg_momentum_team2"] = df["recent_fg_pct_team2"] - df["FG%_stats_team2"]
    df["three_pt_momentum_team1"] = df["recent_3p_pct_team1"] - df["3P%_team1"]
    df["three_pt_momentum_team2"] = df["recent_3p_pct_team2"] - df["3P%_stats_team2"]

    # Turnover trend (negative = improving ball security)
    df["tov_trend_team1"] = df["recent_tov_team1"] - df["TOV_team1"]
    df["tov_trend_team2"] = df["recent_tov_team2"] - df["TOV_stats_team2"]
    df["tov_trend_diff"] = df["tov_trend_team1"] - df["tov_trend_team2"]

    # Shooting efficiency differential (recent form comparison)
    df["recent_fg_diff"] = df["recent_fg_pct_team1"] - df["recent_fg_pct_team2"]
    df["recent_3p_diff"] = df["recent_3p_pct_team1"] - df["recent_3p_pct_team2"]

    # SRS & SOS Differences
    df["srs_diff"] = df["SRS_team1"] - df["SRS_stats_team2"]
    df["sos_diff"] = df["SOS_team1"] - df["SOS_stats_team2"]

    # ========================================================================
    # SOS-ENHANCED FEATURES (Simplified to prevent overconfidence)
    # ========================================================================
    # REMOVED problematic interaction features that caused multicollinearity:
    # - sos_adjusted_srs_diff (multiplicative interaction)
    # - competitive_sos_boost (unstable division)
    # - sos_diff_boosted (flawed conference detection)
    # - is_conference_game (crude heuristic)
    #
    # KEPT simple composite that doesn't create false certainty:
    # - sos_weighted_strength (linear combination, stable)
    # ========================================================================

    # SOS-weighted strength: Gives credit for performing well against tough schedules
    # Linear combination is less prone to overfitting than multiplicative interactions
    df["sos_weighted_strength"] = df["srs_diff"] + (df["sos_diff"] * 0.3)

    # Field Goal Efficiency Difference
    df["fg_eff_diff"] = df["FG%_team1"] - df["FG%_stats_team2"]

    # Three-Point Shooting Efficiency Difference
    df["three_pt_eff_diff"] = df["3P%_team1"] - df["3P%_stats_team2"]

    # Free Throw Efficiency Difference
    df["ft_eff_diff"] = df["FT%_team1"] - df["FT%_stats_team2"]

    # Rebounding Difference (Offensive & Total)
    df["orb_diff"] = df["ORB_team1"] - df["ORB_stats_team2"]
    df["trb_diff"] = df["TRB_team1"] - df["TRB_stats_team2"]

    # Assist & Turnover Efficiency
    df["ast_diff"] = df["AST_team1"] - df["AST_stats_team2"]
    df["tov_diff"] = df["TOV_team1"] - df["TOV_stats_team2"]

    # Fouls Committed Difference
    df["pf_diff"] = df["PF_team1"] - df["PF_stats_team2"]

    # Home/Away Performance Differences (safe division)
    df["home_win_pct_team1"] = np.where(
        (df["Whome_team1"] + df["Lhome_team1"]) > 0,
        df["Whome_team1"] / (df["Whome_team1"] + df["Lhome_team1"]),
        0
    )
    df["away_win_pct_team2"] = np.where(
        (df["Waway_stats_team2"] + df["Laway_stats_team2"]) > 0,
        df["Waway_stats_team2"] / (df["Waway_stats_team2"] + df["Laway_stats_team2"]),
        0
    )
    df["home_advantage"] = np.where(df["team_1"] == df["home_team"], 1, -1)

    # ========================================================================
    # NEW: ENHANCED HOME/AWAY SPLIT FEATURES (Added 2025-12-09)
    # These features help the model understand team performance in different contexts
    # ========================================================================

    # 1. Road win percentage for team 1
    df["road_win_pct_team1"] = np.where(
        (df["Waway_team1"] + df["Laway_team1"]) > 0,
        df["Waway_team1"] / (df["Waway_team1"] + df["Laway_team1"]),
        0
    )

    # 2. Road win percentage for team 2
    df["road_win_pct_team2"] = np.where(
        (df["Waway_stats_team2"] + df["Laway_stats_team2"]) > 0,
        df["Waway_stats_team2"] / (df["Waway_stats_team2"] + df["Laway_stats_team2"]),
        0
    )

    # 3. Home win percentage for team 2
    df["home_win_pct_team2"] = np.where(
        (df["Whome_stats_team2"] + df["Lhome_stats_team2"]) > 0,
        df["Whome_stats_team2"] / (df["Whome_stats_team2"] + df["Lhome_stats_team2"]),
        0
    )

    # 4. Home/away performance differential for team 1
    # Positive = better at home, Negative = better on road
    df["home_away_diff_team1"] = df["home_win_pct_team1"] - df["road_win_pct_team1"]

    # 5. Home/away performance differential for team 2
    df["home_away_diff_team2"] = df["home_win_pct_team2"] - df["road_win_pct_team2"]

    # 6. Situational win percentage differential
    # Compares the home team's home performance vs away team's road performance
    df["situational_win_pct_diff"] = np.where(
        df["home_advantage"] == 1,  # team_1 is home
        df["home_win_pct_team1"] - df["road_win_pct_team2"],
        df["road_win_pct_team1"] - df["home_win_pct_team2"]
    )

    # 7. Sample size indicators (for model to weight confidence)
    df["home_games_played_team1"] = df["Whome_team1"] + df["Lhome_team1"]
    df["away_games_played_team1"] = df["Waway_team1"] + df["Laway_team1"]
    df["home_games_played_team2"] = df["Whome_stats_team2"] + df["Lhome_stats_team2"]
    df["away_games_played_team2"] = df["Waway_stats_team2"] + df["Laway_stats_team2"]

    # ========================================================================
    # HOME COURT ADVANTAGE INTERACTION FEATURES
    # ========================================================================
    # The model isn't learning home court advantage well from the simple binary feature.
    # Create interaction terms between home_advantage and team strength metrics
    # This makes home court advantage more predictive and learnable for Random Forest

    print("   Creating home court advantage interaction features...")

    # Calculate win% for both teams (with safety for division by zero)
    team1_games = df["W_team1"] + df["L_team1"]
    team2_games = df["W_stats_team2"] + df["L_stats_team2"]

    team1_win_pct = np.where(team1_games > 0, df["W_team1"] / team1_games, 0.5)
    team2_win_pct = np.where(team2_games > 0, df["W_stats_team2"] / team2_games, 0.5)

    # Home court interactions with team strength
    # If team1 is home (home_advantage=1), these boost team1's metrics
    # If team1 is away (home_advantage=-1), these reduce team1's metrics
    df["home_adv_x_win_pct_diff"] = df["home_advantage"] * (team1_win_pct - team2_win_pct)
    df["home_adv_x_srs_diff"] = df["home_advantage"] * (df["SRS_team1"] - df["SRS_stats_team2"])
    df["home_adv_x_ppg_diff"] = df["home_advantage"] * (df["Tm._team1"] / df["G_team1"] - df["Tm._stats_team2"] / df["G_stats_team2"])

    # Specific home/away performance for the team playing at home
    # When team1 is home, use team1's home stats; when team1 is away, use team2's home stats
    df["home_team_home_win_pct"] = np.where(
        df["home_advantage"] == 1,
        df["home_win_pct_team1"],  # team1 is home, use their home win%
        df["home_win_pct_team2"]   # team2 is home, use their home win%
    )

    df["away_team_away_win_pct"] = np.where(
        df["home_advantage"] == 1,
        df["road_win_pct_team2"],  # team2 is away, use their road win%
        df["road_win_pct_team1"]   # team1 is away, use their road win%
    )

    # Home vs Away matchup advantage
    df["home_away_matchup_advantage"] = df["home_team_home_win_pct"] - df["away_team_away_win_pct"]

    # SRS-adjusted home court (stronger teams get bigger home court boost)
    df["srs_adjusted_home_advantage"] = np.where(
        df["home_advantage"] == 1,
        df["SRS_team1"] * 0.1,  # team1 home gets +10% of their SRS
        df["SRS_stats_team2"] * 0.1  # team2 home gets +10% of their SRS
    )

    # ========================================================================

    # Upset Factors (calculated from training data if available)
    # For now, initialize to 0 - will be calculated during training
    df['Team1UpsetFactor'] = 0.0
    df['Team2UpsetFactor'] = 0.0

    # ========================================================================
    # ADVANCED EFFICIENCY METRICS (Tier 1 & 2 Features)
    # ========================================================================
    print("   Calculating advanced efficiency metrics...")

    # ========== TIER 1: CRITICAL EFFICIENCY FEATURES ==========

    # 1. POSSESSIONS ESTIMATION (Dean Oliver formula)
    # Possessions ≈ FGA + 0.475 × FTA - ORB + TOV
    df["poss_team1"] = np.where(
        df["FGA_team1"].notna(),
        df["FGA_team1"] + 0.475 * df["FTA_team1"] - df["ORB_team1"] + df["TOV_team1"],
        0
    )
    df["poss_team2"] = np.where(
        df["FGA_stats_team2"].notna(),
        df["FGA_stats_team2"] + 0.475 * df["FTA_stats_team2"] - df["ORB_stats_team2"] + df["TOV_stats_team2"],
        0
    )

    # 2. OFFENSIVE EFFICIENCY (Points per 100 possessions)
    df["off_eff_team1"] = np.where(
        df["poss_team1"] > 0,
        (df["Tm._team1"] / df["poss_team1"]) * 100,
        0
    )
    df["off_eff_team2"] = np.where(
        df["poss_team2"] > 0,
        (df["Tm._stats_team2"] / df["poss_team2"]) * 100,
        0
    )

    # 3. DEFENSIVE EFFICIENCY (Points allowed per 100 possessions)
    df["def_eff_team1"] = np.where(
        df["poss_team1"] > 0,
        (df["Opp._team1"] / df["poss_team1"]) * 100,
        0
    )
    df["def_eff_team2"] = np.where(
        df["poss_team2"] > 0,
        (df["Opp._stats_team2"] / df["poss_team2"]) * 100,
        0
    )

    # 4. NET EFFICIENCY (The single most predictive metric in college basketball)
    df["net_eff_team1"] = df["off_eff_team1"] - df["def_eff_team1"]
    df["net_eff_team2"] = df["off_eff_team2"] - df["def_eff_team2"]
    df["net_eff_diff"] = df["net_eff_team1"] - df["net_eff_team2"]

    # 5. EFFECTIVE FIELD GOAL % (weights 3-pointers properly)
    # eFG% = (FG + 0.5 × 3P) / FGA
    df["efg_pct_team1"] = np.where(
        df["FGA_team1"] > 0,
        (df["FG_team1"] + 0.5 * df["3P_team1"]) / df["FGA_team1"],
        0
    )
    df["efg_pct_team2"] = np.where(
        df["FGA_stats_team2"] > 0,
        (df["FG_stats_team2"] + 0.5 * df["3P_stats_team2"]) / df["FGA_stats_team2"],
        0
    )
    df["efg_diff"] = df["efg_pct_team1"] - df["efg_pct_team2"]

    # 6. TURNOVER RATE (percentage of possessions ending in turnover)
    df["tov_rate_team1"] = np.where(
        df["poss_team1"] > 0,
        (df["TOV_team1"] / df["poss_team1"]) * 100,
        0
    )
    df["tov_rate_team2"] = np.where(
        df["poss_team2"] > 0,
        (df["TOV_stats_team2"] / df["poss_team2"]) * 100,
        0
    )
    df["tov_rate_diff"] = df["tov_rate_team1"] - df["tov_rate_team2"]

    # 7. PACE/TEMPO (possessions per game)
    df["pace_team1"] = np.where(
        df["G_team1"] > 0,
        df["poss_team1"] / df["G_team1"],
        0
    )
    df["pace_team2"] = np.where(
        df["G_stats_team2"] > 0,
        df["poss_team2"] / df["G_stats_team2"],
        0
    )
    df["pace_diff"] = df["pace_team1"] - df["pace_team2"]
    df["expected_pace"] = (df["pace_team1"] + df["pace_team2"]) / 2  # Expected game pace

    # ========== TIER 2: HIGH-VALUE FEATURES ==========

    # 8. TRUE SHOOTING % (best overall shooting efficiency metric)
    # TS% = Points / (2 × (FGA + 0.44 × FTA))
    df["ts_pct_team1"] = np.where(
        (df["FGA_team1"] + 0.44 * df["FTA_team1"]) > 0,
        df["Tm._team1"] / (2 * (df["FGA_team1"] + 0.44 * df["FTA_team1"])),
        0
    )
    df["ts_pct_team2"] = np.where(
        (df["FGA_stats_team2"] + 0.44 * df["FTA_stats_team2"]) > 0,
        df["Tm._stats_team2"] / (2 * (df["FGA_stats_team2"] + 0.44 * df["FTA_stats_team2"])),
        0
    )
    df["ts_diff"] = df["ts_pct_team1"] - df["ts_pct_team2"]

    # 9. OFFENSIVE REBOUNDING RATE
    # ORB% = ORB / (ORB_team + DRB_opponent)
    # DRB = TRB - ORB
    df["drb_team1"] = df["TRB_team1"] - df["ORB_team1"]
    df["drb_team2"] = df["TRB_stats_team2"] - df["ORB_stats_team2"]

    df["orb_pct_team1"] = np.where(
        (df["ORB_team1"] + df["drb_team2"]) > 0,
        df["ORB_team1"] / (df["ORB_team1"] + df["drb_team2"]),
        0
    )
    df["orb_pct_team2"] = np.where(
        (df["ORB_stats_team2"] + df["drb_team1"]) > 0,
        df["ORB_stats_team2"] / (df["ORB_stats_team2"] + df["drb_team1"]),
        0
    )
    df["orb_rate_diff"] = df["orb_pct_team1"] - df["orb_pct_team2"]

    # 10. DEFENSIVE REBOUNDING RATE
    # DRB% = DRB / (DRB_team + ORB_opponent)
    df["drb_pct_team1"] = np.where(
        (df["drb_team1"] + df["ORB_stats_team2"]) > 0,
        df["drb_team1"] / (df["drb_team1"] + df["ORB_stats_team2"]),
        0
    )
    df["drb_pct_team2"] = np.where(
        (df["drb_team2"] + df["ORB_team1"]) > 0,
        df["drb_team2"] / (df["drb_team2"] + df["ORB_team1"]),
        0
    )
    df["drb_rate_diff"] = df["drb_pct_team1"] - df["drb_pct_team2"]

    # 11. FREE THROW RATE (ability to get to the line)
    # FT Rate = FTA / FGA
    df["ft_rate_team1"] = np.where(
        df["FGA_team1"] > 0,
        df["FTA_team1"] / df["FGA_team1"],
        0
    )
    df["ft_rate_team2"] = np.where(
        df["FGA_stats_team2"] > 0,
        df["FTA_stats_team2"] / df["FGA_stats_team2"],
        0
    )
    df["ft_rate_diff"] = df["ft_rate_team1"] - df["ft_rate_team2"]

    # 12. ASSIST RATE (% of FG that are assisted)
    df["ast_rate_team1"] = np.where(
        df["FG_team1"] > 0,
        df["AST_team1"] / df["FG_team1"],
        0
    )
    df["ast_rate_team2"] = np.where(
        df["FG_stats_team2"] > 0,
        df["AST_stats_team2"] / df["FG_stats_team2"],
        0
    )
    df["ast_rate_diff"] = df["ast_rate_team1"] - df["ast_rate_team2"]

    # 13. AST/TO RATIO (ball security + playmaking)
    df["ast_to_ratio_team1"] = np.where(
        df["TOV_team1"] > 0,
        df["AST_team1"] / df["TOV_team1"],
        0
    )
    df["ast_to_ratio_team2"] = np.where(
        df["TOV_stats_team2"] > 0,
        df["AST_stats_team2"] / df["TOV_stats_team2"],
        0
    )
    df["ast_to_ratio_diff"] = df["ast_to_ratio_team1"] - df["ast_to_ratio_team2"]

    # 14. TWO-POINT PERCENTAGE
    # 2P% = (FG - 3P) / (FGA - 3PA)
    df["two_pt_pct_team1"] = np.where(
        (df["FGA_team1"] - df["3PA_team1"]) > 0,
        (df["FG_team1"] - df["3P_team1"]) / (df["FGA_team1"] - df["3PA_team1"]),
        0
    )
    df["two_pt_pct_team2"] = np.where(
        (df["FGA_stats_team2"] - df["3PA_stats_team2"]) > 0,
        (df["FG_stats_team2"] - df["3P_stats_team2"]) / (df["FGA_stats_team2"] - df["3PA_stats_team2"]),
        0
    )
    df["two_pt_diff"] = df["two_pt_pct_team1"] - df["two_pt_pct_team2"]

    # 15. THREE-POINT RATE (% of shots that are threes)
    df["three_pt_rate_team1"] = np.where(
        df["FGA_team1"] > 0,
        df["3PA_team1"] / df["FGA_team1"],
        0
    )
    df["three_pt_rate_team2"] = np.where(
        df["FGA_stats_team2"] > 0,
        df["3PA_stats_team2"] / df["FGA_stats_team2"],
        0
    )
    df["three_pt_rate_diff"] = df["three_pt_rate_team1"] - df["three_pt_rate_team2"]

    print(f"   ✅ Added {15} new advanced efficiency features (Tier 1 + 2)")

    # ========== RECENT FORM EFFICIENCY METRICS (5-game rolling windows) ==========
    # Only calculate if enabled
    if enable_recent_eff:
        print("   Calculating recent efficiency form...")

        # Only calculate recent form if we have training data (winner column exists)
        if "winner" in df.columns and df["winner"].notna().any():
            # Training mode - calculate from historical data with shifted windows

            # Recent Net Efficiency (last 5 games)
            df["recent_net_eff_team1"] = (
                df.groupby("team_1")["net_eff_team1"]
                .apply(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )
            df["recent_net_eff_team2"] = (
                df.groupby("team_2")["net_eff_team2"]
                .apply(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )

            # Recent Effective FG% (last 5 games)
            df["recent_efg_team1"] = (
                df.groupby("team_1")["efg_pct_team1"]
                .apply(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )
            df["recent_efg_team2"] = (
                df.groupby("team_2")["efg_pct_team2"]
                .apply(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )

            # Recent True Shooting % (last 5 games)
            df["recent_ts_team1"] = (
                df.groupby("team_1")["ts_pct_team1"]
                .apply(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )
            df["recent_ts_team2"] = (
                df.groupby("team_2")["ts_pct_team2"]
                .apply(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )

            # Recent Turnover Rate (last 5 games)
            df["recent_tov_rate_team1"] = (
                df.groupby("team_1")["tov_rate_team1"]
                .apply(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )
            df["recent_tov_rate_team2"] = (
                df.groupby("team_2")["tov_rate_team2"]
                .apply(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )

            # Recent Pace (last 5 games)
            df["recent_pace_team1"] = (
                df.groupby("team_1")["pace_team1"]
                .apply(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )
            df["recent_pace_team2"] = (
                df.groupby("team_2")["pace_team2"]
                .apply(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )

            # Fill NaN values with season averages
            df["recent_net_eff_team1"] = df["recent_net_eff_team1"].fillna(df["net_eff_team1"])
            df["recent_net_eff_team2"] = df["recent_net_eff_team2"].fillna(df["net_eff_team2"])
            df["recent_efg_team1"] = df["recent_efg_team1"].fillna(df["efg_pct_team1"])
            df["recent_efg_team2"] = df["recent_efg_team2"].fillna(df["efg_pct_team2"])
            df["recent_ts_team1"] = df["recent_ts_team1"].fillna(df["ts_pct_team1"])
            df["recent_ts_team2"] = df["recent_ts_team2"].fillna(df["ts_pct_team2"])
            df["recent_tov_rate_team1"] = df["recent_tov_rate_team1"].fillna(df["tov_rate_team1"])
            df["recent_tov_rate_team2"] = df["recent_tov_rate_team2"].fillna(df["tov_rate_team2"])
            df["recent_pace_team1"] = df["recent_pace_team1"].fillna(df["pace_team1"])
            df["recent_pace_team2"] = df["recent_pace_team2"].fillna(df["pace_team2"])

        else:
            # ==========================================
            # PREDICTION MODE: Look up efficiency metrics from training data
            # ==========================================
            print("   ℹ️  Prediction mode: looking up recent efficiency from training data")

            def get_recent_efficiency_stats(team_name, training_file, stat_col, n_games=5):
                """
                Look up team's recent efficiency metric from training data.
                If efficiency columns don't exist, calculate them on-the-fly from base stats.
                """
                try:
                    training_df = pd.read_csv(training_file)
    
                    # Find games where this team played (as team_1 or team_2)
                    team1_games = training_df[training_df['team_1'] == team_name].copy()
                    team2_games = training_df[training_df['team_2'] == team_name].copy()
    
                    # Determine what metric we're looking for
                    if '_team1' in stat_col:
                        team1_col = stat_col
                        team2_col = stat_col.replace('_team1', '_team2')
                    else:
                        team1_col = stat_col
                        team2_col = stat_col
    
                    # Try to use pre-calculated efficiency column if it exists
                    if team1_col in team1_games.columns and team2_col in team2_games.columns:
                        team1_games['stat_value'] = team1_games[team1_col]
                        team2_games['stat_value'] = team2_games[team2_col]
                    else:
                        # Calculate efficiency metric on-the-fly from base stats
                        # Determine which metric to calculate based on column name
    
                        if 'net_eff' in stat_col:
                            # Net Efficiency = Offensive Eff - Defensive Eff
                            # First calculate possessions
                            team1_games['poss'] = (team1_games['FGA_team1'] + 0.475 * team1_games['FTA_team1'] -
                                                  team1_games['ORB_team1'] + team1_games['TOV_team1'])
                            team2_games['poss'] = (team2_games['FGA_stats_team2'] + 0.475 * team2_games['FTA_stats_team2'] -
                                                  team2_games['ORB_stats_team2'] + team2_games['TOV_stats_team2'])
    
                            # Offensive Efficiency = (Points / Possessions) × 100
                            team1_games['off_eff'] = np.where(team1_games['poss'] > 0,
                                                              (team1_games['Tm._team1'] / team1_games['poss']) * 100, 0)
                            team2_games['off_eff'] = np.where(team2_games['poss'] > 0,
                                                              (team2_games['Tm._stats_team2'] / team2_games['poss']) * 100, 0)
    
                            # Defensive Efficiency = (Points Allowed / Possessions) × 100
                            team1_games['def_eff'] = np.where(team1_games['poss'] > 0,
                                                              (team1_games['Opp._team1'] / team1_games['poss']) * 100, 0)
                            team2_games['def_eff'] = np.where(team2_games['poss'] > 0,
                                                              (team2_games['Opp._stats_team2'] / team2_games['poss']) * 100, 0)
    
                            # Net Efficiency
                            team1_games['stat_value'] = team1_games['off_eff'] - team1_games['def_eff']
                            team2_games['stat_value'] = team2_games['off_eff'] - team2_games['def_eff']
    
                        elif 'efg_pct' in stat_col:
                            # eFG% = (FG + 0.5 × 3P) / FGA
                            team1_games['stat_value'] = np.where(team1_games['FGA_team1'] > 0,
                                                                 (team1_games['FG_team1'] + 0.5 * team1_games['3P_team1']) / team1_games['FGA_team1'], 0)
                            team2_games['stat_value'] = np.where(team2_games['FGA_stats_team2'] > 0,
                                                                 (team2_games['FG_stats_team2'] + 0.5 * team2_games['3P_stats_team2']) / team2_games['FGA_stats_team2'], 0)
    
                        elif 'ts_pct' in stat_col:
                            # TS% = Points / (2 × (FGA + 0.44 × FTA))
                            team1_games['stat_value'] = np.where((team1_games['FGA_team1'] + 0.44 * team1_games['FTA_team1']) > 0,
                                                                 team1_games['Tm._team1'] / (2 * (team1_games['FGA_team1'] + 0.44 * team1_games['FTA_team1'])), 0)
                            team2_games['stat_value'] = np.where((team2_games['FGA_stats_team2'] + 0.44 * team2_games['FTA_stats_team2']) > 0,
                                                                 team2_games['Tm._stats_team2'] / (2 * (team2_games['FGA_stats_team2'] + 0.44 * team2_games['FTA_stats_team2'])), 0)
    
                        elif 'tov_rate' in stat_col:
                            # TOV Rate = (TOV / Possessions) × 100
                            team1_games['poss'] = (team1_games['FGA_team1'] + 0.475 * team1_games['FTA_team1'] -
                                                  team1_games['ORB_team1'] + team1_games['TOV_team1'])
                            team2_games['poss'] = (team2_games['FGA_stats_team2'] + 0.475 * team2_games['FTA_stats_team2'] -
                                                  team2_games['ORB_stats_team2'] + team2_games['TOV_stats_team2'])
    
                            team1_games['stat_value'] = np.where(team1_games['poss'] > 0,
                                                                 (team1_games['TOV_team1'] / team1_games['poss']) * 100, 0)
                            team2_games['stat_value'] = np.where(team2_games['poss'] > 0,
                                                                 (team2_games['TOV_stats_team2'] / team2_games['poss']) * 100, 0)
    
                        elif 'pace' in stat_col:
                            # Pace = Possessions / Games
                            team1_games['poss'] = (team1_games['FGA_team1'] + 0.475 * team1_games['FTA_team1'] -
                                                  team1_games['ORB_team1'] + team1_games['TOV_team1'])
                            team2_games['poss'] = (team2_games['FGA_stats_team2'] + 0.475 * team2_games['FTA_stats_team2'] -
                                                  team2_games['ORB_stats_team2'] + team2_games['TOV_stats_team2'])
    
                            team1_games['stat_value'] = np.where(team1_games['G_team1'] > 0,
                                                                 team1_games['poss'] / team1_games['G_team1'], 0)
                            team2_games['stat_value'] = np.where(team2_games['G_stats_team2'] > 0,
                                                                 team2_games['poss'] / team2_games['G_stats_team2'], 0)
    
                        else:
                            # Unknown metric - return None
                            return None
    
                    # Combine and sort by date
                    all_games = pd.concat([team1_games, team2_games])
                    all_games = all_games.dropna(subset=['stat_value'])
    
                    if len(all_games) == 0:
                        return None
    
                    # Sort by date and take last N games
                    all_games = all_games.sort_values('date').tail(n_games)
                    return all_games['stat_value'].mean()
    
                except Exception as e:
                    # If any error, return None to fall back to season average
                    return None
    
            # Define mappings for efficiency stats (recent column -> source column)
            efficiency_mappings = [
                ('recent_net_eff_team1', 'net_eff_team1'),
                ('recent_efg_team1', 'efg_pct_team1'),
                ('recent_ts_team1', 'ts_pct_team1'),
                ('recent_tov_rate_team1', 'tov_rate_team1'),
                ('recent_pace_team1', 'pace_team1'),
            ]
    
            # Calculate recent efficiency for both teams
            for recent_col, stat_col in efficiency_mappings:
                # Team 1
                fallback_col = stat_col  # e.g., 'net_eff_team1'
                df[recent_col] = df["team_1"].apply(
                    lambda x: get_recent_efficiency_stats(x, Config.TRAINING_DATA, stat_col) or
                             (df[df["team_1"] == x][fallback_col].iloc[0] if len(df[df["team_1"] == x]) > 0 else 0)
                )
    
                # Team 2 (replace _team1 with _team2 in column names)
                recent_col_team2 = recent_col.replace('_team1', '_team2')
                stat_col_team2 = stat_col.replace('_team1', '_team2')
                fallback_col_team2 = stat_col_team2
    
                df[recent_col_team2] = df["team_2"].apply(
                    lambda x: get_recent_efficiency_stats(x, Config.TRAINING_DATA, stat_col_team2) or
                             (df[df["team_2"] == x][fallback_col_team2].iloc[0] if len(df[df["team_2"] == x]) > 0 else 0)
                )
    
            # Create differentials for recent form
            df["recent_net_eff_diff"] = df["recent_net_eff_team1"] - df["recent_net_eff_team2"]
            df["recent_efg_diff"] = df["recent_efg_team1"] - df["recent_efg_team2"]
            df["recent_ts_diff"] = df["recent_ts_team1"] - df["recent_ts_team2"]
            df["recent_tov_rate_diff"] = df["recent_tov_rate_team1"] - df["recent_tov_rate_team2"]
            df["recent_pace_diff"] = df["recent_pace_team1"] - df["recent_pace_team2"]
    
            # Momentum indicators (recent vs season average)
            df["net_eff_momentum_team1"] = df["recent_net_eff_team1"] - df["net_eff_team1"]
            df["net_eff_momentum_team2"] = df["recent_net_eff_team2"] - df["net_eff_team2"]
            df["efg_momentum_team1"] = df["recent_efg_team1"] - df["efg_pct_team1"]
            df["efg_momentum_team2"] = df["recent_efg_team2"] - df["efg_pct_team2"]
    
        print(f"   ✅ Added {14} recent efficiency form features")
    else:
        # Rolling features disabled - use season averages as "recent" form
        print(f"   ⚠️  Recent efficiency features DISABLED - using season averages as fallback")

        # Initialize all recent_* columns to match season averages
        df["recent_net_eff_team1"] = df["net_eff_team1"]
        df["recent_net_eff_team2"] = df["net_eff_team2"]
        df["recent_efg_team1"] = df["efg_pct_team1"]
        df["recent_efg_team2"] = df["efg_pct_team2"]
        df["recent_ts_team1"] = df["ts_pct_team1"]
        df["recent_ts_team2"] = df["ts_pct_team2"]
        df["recent_tov_rate_team1"] = df["tov_rate_team1"]
        df["recent_tov_rate_team2"] = df["tov_rate_team2"]
        df["recent_pace_team1"] = df["pace_team1"]
        df["recent_pace_team2"] = df["pace_team2"]

        # Create differential features (even if using season averages)
        df["recent_net_eff_diff"] = df["recent_net_eff_team1"] - df["recent_net_eff_team2"]
        df["recent_efg_diff"] = df["recent_efg_team1"] - df["recent_efg_team2"]
        df["recent_ts_diff"] = df["recent_ts_team1"] - df["recent_ts_team2"]
        df["recent_tov_rate_diff"] = df["recent_tov_rate_team1"] - df["recent_tov_rate_team2"]
        df["recent_pace_diff"] = df["recent_pace_team1"] - df["recent_pace_team2"]

        # Momentum indicators will be zero (recent = season average)
        df["net_eff_momentum_team1"] = 0.0
        df["net_eff_momentum_team2"] = 0.0
        df["efg_momentum_team1"] = 0.0
        df["efg_momentum_team2"] = 0.0

    # Drop redundant/intermediate features
    # NOTE: Keeping RAW stats (G, W, L, SRS, SOS, FG%, 3P%, FT%, Tm., TRB, MP, etc.) for training data
    # Only dropping intermediate calculation columns and redundant split stats
    columns_to_drop = [
        "_is_current",  # Internal tracking column, not a feature
        "recent_win_pct_team1", "recent_win_pct_team2",  # Redundant with win_ratio_adj
        "recent_pts_scored_team1", "recent_pts_scored_team2",  # Captured in rolling features
        "recent_pts_allowed_team1", "recent_pts_allowed_team2",  # Captured in rolling features
        "recent_fg_pct_team1", "recent_fg_pct_team2",  # Captured in rolling features
        "recent_3p_pct_team1", "recent_3p_pct_team2",  # Captured in rolling features
        "recent_tov_team1", "recent_tov_team2",  # Captured in rolling features
        "drb_team1", "drb_team2",  # Defensive rebounds (intermediate for rebounding rates)
    ]

    # Try to drop, but don't fail if column doesn't exist
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

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

    # Check model age
    model_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(Config.MODEL_FILE))).days
    if model_age_days > Config.MODEL_RETRAIN_DAYS:
        print(f"⏰ Model is {model_age_days} days old (threshold: {Config.MODEL_RETRAIN_DAYS}), retraining")
        return True

    print(f"✅ Model is {model_age_days} days old, no retrain needed")
    return False


def train_model(training_df):
    """Train RandomForest model with GridSearch"""
    print("\n🤖 Training ML model...")

    df = training_df.copy()

    # Drop columns not needed for training (same as V10)
    # Also drop Sport, Start Time (CT), and data_quality which are metadata columns
    df.drop(columns=["game_id", "team_team1", "team_stats_team2", "Rank_team1", "Rank_stats_team2", "game_day", "Sport", "Start Time (CT)", "data_quality"], errors="ignore", inplace=True)

    # Create binary winner column
    df["winner_binary"] = (df["winner"] == df["team_1"]).astype(int)

    # Calculate upset factors
    team_upset_wins = {}
    team_underdog_games = {}

    for _, row in df.iterrows():
        # For team_1: if team_1 is the underdog
        if row['team_1_odds'] > row['team_2_odds']:
            team_underdog_games[row['team_1']] = team_underdog_games.get(row['team_1'], 0) + 1
            if row['team_1'] == row['winner']:
                team_upset_wins[row['team_1']] = team_upset_wins.get(row['team_1'], 0) + 1
        # For team_2: if team_2 is the underdog
        if row['team_2_odds'] > row['team_1_odds']:
            team_underdog_games[row['team_2']] = team_underdog_games.get(row['team_2'], 0) + 1
            if row['team_2'] == row['winner']:
                team_upset_wins[row['team_2']] = team_upset_wins.get(row['team_2'], 0) + 1

    # Calculate upset factors
    team_upset_factors = {}
    for team, opportunities in team_underdog_games.items():
        wins = team_upset_wins.get(team, 0)
        team_upset_factors[team] = wins / opportunities

    # Apply to dataframe with scaling
    scaling_factor = 0.1
    df['Team1UpsetFactor'] = df['team_1'].map(team_upset_factors).fillna(0) * scaling_factor
    df['Team2UpsetFactor'] = df['team_2'].map(team_upset_factors).fillna(0) * scaling_factor

    # Sort by date for time-series split
    df = df.sort_values('date').reset_index(drop=True)
    print(f"   Sorted by date: {df['date'].min()} to {df['date'].max()}")

    # Apply feature engineering
    df = feature_engineering(df)

    # Drop date from features
    df = df.drop(columns=["date"], errors='ignore')

    # Drop removed SOS interaction features if they exist in training data CSV
    # (these were removed on 2026-01-08 to fix overconfidence issue)
    removed_features = ['sos_adjusted_srs_diff', 'competitive_sos_boost', 'is_conference_game', 'sos_diff_boosted']
    df = df.drop(columns=removed_features, errors='ignore')

    # Prepare training data
    X = df.drop(columns=["winner_binary", "team_1", "team_2", "winner", "home_team", "away_team", "team_1_odds", "team_2_odds"])
    y = df["winner_binary"]

    # TIME-SERIES AWARE SPLIT (chronological 80/20)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"   Training samples: {len(X_train)}")
    print(f"   Initial features: {len(X.columns)}")
    print(f"   Train set: {len(X_train)} games ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Test set: {len(X_test)} games ({len(X_test)/len(X)*100:.1f}%)")

    # ========================================================================
    # FEATURE SELECTION (Reduce overfitting by removing low-importance features)
    # ========================================================================
    print("\n   🔍 Running feature selection to reduce overfitting...")

    # Train a quick Random Forest to get feature importances
    rf_temp = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train, y_train)

    # Get feature importances
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_temp.feature_importances_
    }).sort_values('importance', ascending=False)

    # Critical CBB features that must be included (modern analytics essentials)
    critical_features = [
        'srs_diff', 'sos_weighted_strength', 'sos_diff',  # Rating systems
        'net_eff_diff', 'recent_net_eff_diff',  # Efficiency
        'efg_diff', 'ts_diff',  # Shooting efficiency
        'orb_rate_diff', 'drb_rate_diff',  # Rebounding
        'tov_rate_diff', 'ast_to_ratio_diff',  # Ball security
        'pace_diff', 'expected_pace',  # Tempo
        'situational_win_pct_diff', 'home_advantage',  # Context
        'recent_efg_diff', 'recent_ts_diff'  # Recent shooting
    ]

    # Strategy: Keep top N features by importance + force-include critical features
    importance_threshold = 0.005  # Keep features with >0.5% importance
    top_n_features = 75  # Maximum features to keep

    # Get features above importance threshold
    important_features = feature_importances[feature_importances['importance'] > importance_threshold]['feature'].tolist()

    # Combine: (top N by importance) + critical features
    selected_features_set = set(important_features[:top_n_features])

    # Add critical features that might have been missed
    for feat in critical_features:
        if feat in X.columns:
            selected_features_set.add(feat)

    # Convert to sorted list
    selected_features = [f for f in X.columns if f in selected_features_set]

    print(f"   ✂️  Reduced from {len(X.columns)} to {len(selected_features)} features")
    print(f"   📊 Importance threshold: {importance_threshold} | Top N: {top_n_features}")

    # Filter datasets to selected features
    X = X[selected_features]
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # Show removed vs kept features breakdown
    removed_count = len(feature_importances) - len(selected_features)
    print(f"   ✅ Kept: {len(selected_features)} features | ❌ Removed: {removed_count} low-importance features")

    # GridSearch
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [25, 30, 35],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [3, 4]
    }

    print("   Running GridSearchCV with TimeSeriesSplit (respects temporal order)...")
    # Use TimeSeriesSplit for cross-validation to respect time series nature
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_grid,
        cv=tscv,  # Time-series aware CV (trains on past, validates on future)
        scoring="accuracy",
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Evaluate
    test_accuracy = accuracy_score(y_test, best_model.predict(X_test))
    print(f"   Test Accuracy: {test_accuracy:.4f}")

    # Calibrate with TimeSeriesSplit
    print("   Calibrating model with TimeSeriesSplit...")
    tscv_calibration = TimeSeriesSplit(n_splits=3)
    calibrated_model = CalibratedClassifierCV(best_model, method="sigmoid", cv=tscv_calibration)
    calibrated_model.fit(X_train, y_train)

    # Cross-validation scores with TimeSeriesSplit
    tscv_cv = TimeSeriesSplit(n_splits=3)
    cv_scores = cross_val_score(calibrated_model, X_train, y_train, cv=tscv_cv, scoring="accuracy")
    print(f"   Mean CV Accuracy: {cv_scores.mean():.4f}")

    # Feature importance (final model with selected features)
    final_importances = best_model.feature_importances_
    features = X.columns
    sorted_idx = np.argsort(final_importances)[::-1]

    print("\n   Top 10 Features (after selection):")
    for idx in sorted_idx[:10]:
        print(f"      {features[idx]}: {final_importances[idx]:.4f}")

    # Save model
    with open(Config.MODEL_FILE, "wb") as f:
        pickle.dump(calibrated_model, f)

    print(f"\n✅ Model saved to {Config.MODEL_FILE}")

    return calibrated_model, X.columns


# ============================================================================
# ML PREDICTION
# ============================================================================

def predict_todays_games(today_df, model, feature_columns):
    """Make predictions for today's games"""
    print("\n🔮 Making predictions for today's games...")

    # Handle empty DataFrame
    if today_df.empty:
        print("   ⚠️  No games available for prediction")
        return pd.DataFrame()

    df = today_df.copy()

    # Validate team stats availability
    required_stat_cols = ['W_team1', 'L_team1', 'W_stats_team2', 'L_stats_team2',
                          'SRS_team1', 'SRS_stats_team2']
    missing_cols = [col for col in required_stat_cols if col not in df.columns]

    if missing_cols:
        print(f"   ⚠️  WARNING: Missing required stat columns: {missing_cols}")

    # Check for NaN values in critical stats
    if all(col in df.columns for col in required_stat_cols):
        critical_nan = df[required_stat_cols].isna().any(axis=1)
        incomplete_count = critical_nan.sum()

        if incomplete_count > 0:
            print(f"   ⚠️  WARNING: {incomplete_count}/{len(df)} games have incomplete team stats")
            print(f"   These predictions will use fallback values and may be less reliable")

            # List specific teams with missing stats
            incomplete_games = df[critical_nan][['Team 1', 'Team 2']]
            for idx, (_, game) in enumerate(incomplete_games.iterrows()):
                if idx < 5:  # Limit to first 5
                    print(f"      - {game['Team 1']} vs {game['Team 2']}")
            if len(incomplete_games) > 5:
                print(f"      ... and {len(incomplete_games) - 5} more")

    # Rename columns to match training data
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

    # Save data_quality before feature engineering (it's metadata, not a feature)
    data_quality_col = df["data_quality"].copy() if "data_quality" in df.columns else None

    # Drop same columns as training (same as V10)
    # Also drop Sport and Start Time (CT) which are metadata columns
    df.drop(columns=["game_id", "team_team1", "team_stats_team2", "Rank_team1", "Rank_stats_team2", "game_day", "Sport", "Start Time (CT)"], errors="ignore", inplace=True)

    # Drop data_quality before feature engineering (it's metadata, not a feature)
    df.drop(columns=["data_quality"], errors="ignore", inplace=True)

    # Apply feature engineering
    df = feature_engineering(df)

    # Save team info for output (include home/away for app compatibility)
    team_info_cols = ["team_1", "team_2", "home_team", "away_team", "team_1_odds", "team_2_odds", "date"]
    team_info = df[team_info_cols].copy()

    # Add data_quality back to team_info
    if data_quality_col is not None:
        team_info["data_quality"] = data_quality_col

    # Select only the features used in training
    X = df[feature_columns]

    X.to_csv("today_features_for_prediction.csv", index=False)

    # Predict
    predictions = model.predict(X)
    prediction_probs = model.predict_proba(X)
    confidence = prediction_probs.max(axis=1)

    # Get individual team probabilities (prediction_probs[:, 1] = team_1 wins, [:, 0] = team_2 wins)
    team1_win_prob = prediction_probs[:, 1]
    team2_win_prob = prediction_probs[:, 0]

    # Create output DataFrame
    results = team_info.copy()
    results["predicted_winner"] = np.where(predictions == 1, team_info["team_1"], team_info["team_2"])
    results["confidence_level"] = confidence
    results["Team1UpsetFactor"] = df["Team1UpsetFactor"].values
    results["Team2UpsetFactor"] = df["Team2UpsetFactor"].values

    # Calculate home/away win probabilities (map team1/team2 to home/away)
    # If team_1 is home team, use team1_prob for home, team2_prob for away
    # If team_2 is home team, use team2_prob for home, team1_prob for away
    results["home_win_probability"] = np.where(
        team_info["home_team"] == team_info["team_1"],
        team1_win_prob,
        team2_win_prob
    )
    results["away_win_probability"] = np.where(
        team_info["away_team"] == team_info["team_1"],
        team1_win_prob,
        team2_win_prob
    )

    # ========================================================================
    # DATA-DRIVEN HOME COURT ADVANTAGE ADJUSTMENT
    # ========================================================================
    # Empirical analysis shows:
    #   - Training data: Home teams win 62.6%
    #   - Model predicts: Home teams win 51.7%
    #   - Bet performance: Home picks 80.8% vs Away picks 58.0%
    #   - Required adjustment: +10.91 percentage points to home probability
    # This correction is based on 551 training games and 128 recent bets

    # Load empirically-calculated adjustment (calculated from training data)
    adjustment_file = "home_court_adjustment.txt"
    try:
        with open(adjustment_file, 'r') as f:
            HOME_COURT_ADJUSTMENT = float(f.read().strip())
        print(f"   ✅ Loaded home court adjustment: +{HOME_COURT_ADJUSTMENT*100:.2f}%")
    except FileNotFoundError:
        # Fallback to calculated value if file doesn't exist
        HOME_COURT_ADJUSTMENT = 0.00  # 10.91% from empirical analysis
        print(f"   ⚠️  Using default home court adjustment: +{HOME_COURT_ADJUSTMENT*100:.2f}%")

    # Apply adjustment to home probabilities
    results["home_win_probability"] = np.clip(
        results["home_win_probability"] + HOME_COURT_ADJUSTMENT,
        0.01,  # minimum 1%
        0.99   # maximum 99%
    )

    # Reduce away probabilities by same amount
    results["away_win_probability"] = np.clip(
        results["away_win_probability"] - HOME_COURT_ADJUSTMENT,
        0.01,
        0.99
    )

    # Normalize to ensure probabilities sum to 1.0
    total_prob = results["home_win_probability"] + results["away_win_probability"]
    results["home_win_probability"] = results["home_win_probability"] / total_prob
    results["away_win_probability"] = results["away_win_probability"] / total_prob

    # Recalculate team1/team2 probabilities with adjustment for confidence/value calculations
    team1_win_prob = np.where(
        team_info["team_1"] == team_info["home_team"],
        results["home_win_probability"],
        results["away_win_probability"]
    )
    team2_win_prob = np.where(
        team_info["team_2"] == team_info["home_team"],
        results["home_win_probability"],
        results["away_win_probability"]
    )

    # Update confidence to use adjusted probabilities
    confidence = np.maximum(team1_win_prob, team2_win_prob)

    # Update confidence_level in results with adjusted values
    results["confidence_level"] = confidence

    # Update predicted_winner based on adjusted probabilities
    # (adjustment might flip some predictions)
    results["predicted_winner"] = np.where(
        team1_win_prob > team2_win_prob,
        team_info["team_1"],
        team_info["team_2"]
    )

    # Model agreement (V12 uses single model, so use confidence as proxy)
    # Higher confidence = stronger "agreement" with itself
    results["model_agreement"] = confidence

    # Value bet calculation (using adjusted probabilities)
    results["implied_prob_team_1"] = 1 / results["team_1_odds"]
    results["implied_prob_team_2"] = 1 / results["team_2_odds"]

    # Use adjusted probabilities for value bet determination
    results["value_bet"] = (
        ((team1_win_prob > team2_win_prob) & (team1_win_prob > results["implied_prob_team_1"])) |
        ((team2_win_prob > team1_win_prob) & (team2_win_prob > results["implied_prob_team_2"]))
    )

    # Rename columns for app compatibility
    results = results.rename(columns={
        'date': 'game_date',
        'confidence_level': 'confidence'
    })

    print(f"✅ Predictions made for {len(results)} games")
    print(f"   Value bets identified: {results['value_bet'].sum()}")

    # Save predictions
    results.to_csv(Config.PREDICTIONS_OUTPUT, index=False)
    print(f"💾 Saved predictions to {Config.PREDICTIONS_OUTPUT}")

    return results


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================

def main(force_retrain=False):
    """Main pipeline execution"""
    print("\n" + "="*60)
    print("V11 UNIFIED NCAA BASKETBALL BETTING PIPELINE")
    print("="*60)

    start_time = datetime.now()

    # Initialize team name mapper
    mapper = TeamNameMapper(Config.TEAM_MAPPING_CSV)

    # ========== STEP 1: FETCH DRAFTKINGS ODDS FIRST (BETTING UNIVERSE) ==========
    print("\n" + "="*60)
    print("STEP 1: FETCH DRAFTKINGS ODDS (Betting Universe)")
    print("="*60)

    todays_odds_raw = fetch_todays_odds()

    # Check if we have any games to bet on today
    if todays_odds_raw.empty:
        print("\n⚠️  No DraftKings odds available today")
        print("   Skipping predictions, will only process historical data for training")
        todays_odds = pd.DataFrame()
        todays_merged = pd.DataFrame()
        team_stats = pd.DataFrame()
    else:
        # We have odds! Standardize them
        todays_odds = standardize_todays_odds(todays_odds_raw, mapper)
        print(f"\n✅ Found {len(todays_odds)} games with DraftKings odds")

        # ========== STEP 2: FETCH TEAM STATS (for games we can bet on) ==========
        print("\n" + "="*60)
        print("STEP 2: FETCH TEAM STATS")
        print("="*60)

        # Get current season stats (blending disabled via USE_WEIGHTED_BLEND = False)
        team_stats_raw_unblended, team_stats_raw_blended = fetch_team_stats()

        # Use UNBLENDED current season stats for predictions
        # Both variables contain current season data since USE_WEIGHTED_BLEND = False
        team_stats = standardize_team_stats(team_stats_raw_unblended if team_stats_raw_unblended is not None else team_stats_raw_blended, mapper)

        # Use RAW current season stats for archiving (what actually happened that day)
        # This ensures training data reflects actual performance, not blended with baseline
        if team_stats_raw_unblended is not None:
            team_stats_raw_for_archive = standardize_team_stats(team_stats_raw_unblended, mapper)
            # Save standardized raw stats for archive repair function
            team_stats_raw_for_archive.to_csv("V12_ncaa_2025_team_stats_RAW.csv", index=False)
            print(f"💾 Saved standardized RAW stats to V12_ncaa_2025_team_stats_RAW.csv")
        else:
            # Fallback to blended if raw scraping failed (shouldn't happen often)
            print("   ⚠️  WARNING: Raw stats unavailable, using blended stats for archiving")
            team_stats_raw_for_archive = team_stats

        # ========== STEP 3: MERGE ODDS + STATS ==========
        print("\n" + "="*60)
        print("STEP 3: MERGE BETTING GAMES WITH STATS")
        print("="*60)

        # Merge with current season stats for predictions (no blending)
        todays_merged = merge_today_with_stats(todays_odds, team_stats)
        print(f"✅ {len(todays_merged)} games ready for prediction")

        # Merge with raw stats for archiving (same as predictions since no blending)
        todays_merged_raw = merge_today_with_stats(todays_odds, team_stats_raw_for_archive)

    # ========== STEP 4: HISTORICAL ODDS (REMOVED) ==========
    # REMOVED: Historical odds fetch is redundant with archived games system
    # The archived games in merged_games_archive/ already provide all training data
    # Historical odds was fetching 0 games anyway and wasting 60+ seconds per run
    print("\n" + "="*60)
    print("STEP 4: HISTORICAL ODDS (SKIPPED - Using Archived Games)")
    print("="*60)
    print("✅ Training data from merged_games_archive/ (incremental system)")

    # Create empty DataFrames to maintain compatibility
    historical_odds = pd.DataFrame()
    historical_merged = pd.DataFrame()

    # ========== STEP 5: GAME RESULTS (only for betting-relevant games) ==========
    print("\n" + "="*60)
    print("STEP 5: GAME RESULTS (Betting Games Only)")
    print("="*60)

    # Only use today's merged (archived games handled separately by process_archived_games)
    game_results_raw = fetch_game_results_smart(pd.DataFrame(), todays_merged)
    game_results = standardize_game_results(game_results_raw, mapper)

    # Update historical with actual winners - REMOVED (historical_merged is empty)
    # historical_2025 = update_historical_winners(historical_merged, game_results)

    # ========== STEP 6: INCREMENTAL TRAINING DATA ==========
    print("\n" + "="*60)
    print("STEP 6: INCREMENTAL TRAINING DATA SYSTEM")
    print("="*60)

    # Process archived games (add winners, append to training data)
    games_added = process_archived_games(game_results)

    # Load or initialize training data (combine reference DB + current training data)
    training_data_list = []
    current_training = pd.DataFrame()
    reference_data = pd.DataFrame()

    # Load current training data
    if os.path.exists(Config.TRAINING_DATA):
        # Load existing training data (grows incrementally)
        current_training = pd.read_csv(Config.TRAINING_DATA)
        print(f"\n📂 Loaded current training data: {len(current_training)} games")

        # Deduplicate training data by game_id (prevents duplicate games from corrupting model)
        if 'game_id' in current_training.columns:
            before_count = len(current_training)
            current_training = current_training.drop_duplicates(subset=['game_id'], keep='first')
            after_count = len(current_training)
            if before_count != after_count:
                print(f"   🧹 Removed {before_count - after_count} duplicate games from current training")

        # Clean training data: drop columns that shouldn't be there
        cols_to_drop = ['Sport', 'Start Time (CT)']
        if any(col in current_training.columns for col in cols_to_drop):
            print(f"   🧹 Cleaning current training data (removing invalid columns)")
            current_training = current_training.drop(columns=cols_to_drop, errors='ignore')

        training_data_list.append(current_training)

    # Load reference database for training (if available and enabled)
    if Config.USE_REFERENCE_DB_FOR_TRAINING and os.path.exists(Config.REFERENCE_DATABASE):
        reference_data = pd.read_csv(Config.REFERENCE_DATABASE)
        print(f"📂 Loaded reference database: {len(reference_data)} games")

        # Deduplicate reference data
        if 'game_id' in reference_data.columns:
            before_count = len(reference_data)
            reference_data = reference_data.drop_duplicates(subset=['game_id'], keep='first')
            after_count = len(reference_data)
            if before_count != after_count:
                print(f"   🧹 Removed {before_count - after_count} duplicate games from reference DB")

        training_data_list.append(reference_data)

    # Combine datasets
    if training_data_list:
        # Combine all datasets
        training_data = pd.concat(training_data_list, ignore_index=True)

        # Remove duplicates (prefer current training over reference)
        if 'game_id' in training_data.columns:
            before_count = len(training_data)
            training_data = training_data.drop_duplicates(subset=['game_id'], keep='last')
            after_count = len(training_data)
            if before_count != after_count:
                print(f"   🧹 Removed {before_count - after_count} duplicate games between datasets")

        # Sort by date to maintain chronological order (critical for time-series!)
        if 'date' in training_data.columns:
            training_data['date'] = pd.to_datetime(training_data['date'])
            training_data = training_data.sort_values('date').reset_index(drop=True)
            print(f"📂 Combined training dataset: {len(training_data)} games (sorted chronologically)")
            print(f"   Date range: {training_data['date'].min().date()} to {training_data['date'].max().date()}")
        else:
            print(f"📂 Combined training dataset: {len(training_data)} games")
    else:
        # No training data found
        print(f"\n📂 No training data found - starting fresh")
        print(f"   Training data will accumulate from archived games with results")
        training_data = pd.DataFrame()

    # Show breakdown
    print(f"\n📊 Current training data: {len(training_data)} games")
    if len(reference_data) > 0:
        print(f"   Reference database: {len(reference_data)} games")
    if len(current_training) > 0:
        print(f"   Current training: {len(current_training)} games")

    # ========== STEP 7: ML TRAINING ==========
    print("\n" + "="*60)
    print("STEP 7: ML TRAINING")
    print("="*60)

    # Check if enough data to train
    if len(training_data) < Config.MIN_TRAINING_SAMPLES:
        print(f"⚠️  Only {len(training_data)} games, need {Config.MIN_TRAINING_SAMPLES} minimum")
        if os.path.exists(Config.MODEL_FILE):
            print("📂 Loading existing model instead...")
            with open(Config.MODEL_FILE, "rb") as f:
                model = pickle.load(f)
            # Get feature columns from existing training data
            if len(training_data) > 0:
                temp_df = training_data.copy()

                # Drop same columns as in training (same as V10)
                temp_df.drop(columns=["game_id", "team_team1", "team_stats_team2", "Rank_team1", "Rank_stats_team2", "game_day"], errors="ignore", inplace=True)

                temp_df["winner_binary"] = (temp_df["winner"] == temp_df["team_1"]).astype(int)
                temp_df = feature_engineering(temp_df)
                temp_df.drop(columns=["date"], errors='ignore', inplace=True)

                # Drop removed SOS interaction features if they exist in training data CSV
                removed_features = ['sos_adjusted_srs_diff', 'competitive_sos_boost', 'is_conference_game', 'sos_diff_boosted']
                temp_df.drop(columns=removed_features, errors='ignore', inplace=True)

                # Extract feature columns (same as training)
                feature_columns = temp_df.drop(columns=[
                    "winner_binary", "team_1", "team_2", "winner",
                    "home_team", "away_team", "team_1_odds", "team_2_odds"
                ], errors='ignore').columns
            print("✅ Model loaded")
        else:
            print("❌ No existing model found and not enough data to train")
            print("   Skipping training and predictions. Will still archive games.")
            print(f"   Progress: {len(training_data)}/{Config.MIN_TRAINING_SAMPLES} games collected")
            model = None
            feature_columns = None
    elif should_retrain_model(force_retrain):
        model, feature_columns = train_model(training_data)
    else:
        print("📂 Loading existing model...")
        with open(Config.MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        # Get feature columns from training data (must match training process exactly)
        temp_df = training_data.copy()

        # Drop same columns as in training (same as V10)
        # Also drop Sport, Start Time (CT), and data_quality which are metadata columns that may exist in training data
        temp_df.drop(columns=["game_id", "team_team1", "team_stats_team2", "Rank_team1", "Rank_stats_team2", "game_day", "Sport", "Start Time (CT)", "data_quality"], errors="ignore", inplace=True)

        temp_df["winner_binary"] = (temp_df["winner"] == temp_df["team_1"]).astype(int)
        temp_df = feature_engineering(temp_df)
        temp_df.drop(columns=["date"], errors='ignore', inplace=True)

        # Drop removed SOS interaction features if they exist in training data CSV
        removed_features = ['sos_adjusted_srs_diff', 'competitive_sos_boost', 'is_conference_game', 'sos_diff_boosted']
        temp_df.drop(columns=removed_features, errors='ignore', inplace=True)

        # Extract feature columns (same as training)
        feature_columns = temp_df.drop(columns=[
            "winner_binary", "team_1", "team_2", "winner",
            "home_team", "away_team", "team_1_odds", "team_2_odds"
        ], errors='ignore').columns
        print("✅ Model loaded")

    # ========== STEP 8: PREDICTIONS ==========
    print("\n" + "="*60)
    print("STEP 8: PREDICTIONS")
    print("="*60)

    if len(todays_merged) > 0:
        # Only make predictions if we have a trained model
        if model is not None and feature_columns is not None:
            predictions = predict_todays_games(todays_merged, model, feature_columns)

            # Print summary
            print("\n📊 PREDICTION SUMMARY:")
            print(f"   Total games: {len(predictions)}")
            print(f"   Value bets: {predictions['value_bet'].sum()}")
            if predictions['value_bet'].sum() > 0:
                print("\n   Top Value Bets:")
                value_bets = predictions[predictions['value_bet']].sort_values('confidence', ascending=False)
                for _, row in value_bets.head(5).iterrows():
                    print(f"      {row['predicted_winner']} ({row['confidence']:.1%} confidence)")
        else:
            print("\n⚠️  Skipping predictions - no trained model available")
            print("   Model will be trained once enough games are collected")

        # Archive today's merged data for future training (using RAW stats)
        print("\n" + "="*60)
        print("STEP 9: ARCHIVING TODAY'S DATA (RAW STATS)")
        print("="*60)

        # Get today's date from the merged data
        if 'Date' in todays_merged_raw.columns:
            # Use the date from the data (may have multiple dates if fetching ahead)
            unique_dates = todays_merged_raw['Date'].unique()
            for game_date in unique_dates:
                date_games = todays_merged_raw[todays_merged_raw['Date'] == game_date]
                archive_todays_merged_data(date_games, game_date, apply_features=False)
                print(f"   📦 Archived {len(date_games)} games for {game_date} (raw stats only)")
        else:
            # Fallback to today's date
            today_str = datetime.now(Config.CENTRAL_TZ).strftime("%Y-%m-%d")
            archive_todays_merged_data(todays_merged_raw, today_str, apply_features=False)
            print(f"   📦 Archived {len(todays_merged_raw)} games for {today_str} (raw stats only)")
    else:
        print("\n⚠️  No DraftKings odds available today")
        print("   Skipping predictions and archiving")
        print("   (Historical data was still processed for training)")

    # ========== UNMAPPED TEAMS REPORT ==========
    # Save report of any unmapped teams found during this run
    mapper.save_unmapped_teams_report("unmapped_teams.txt")

    # ========== COMPLETION ==========
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "="*60)
    print(f"✅ PIPELINE COMPLETED in {elapsed:.1f} seconds")
    print("="*60)
    print(f"\nOutput files:")
    print(f"   Predictions: {Config.PREDICTIONS_OUTPUT}")
    print(f"   Training data: {Config.TRAINING_DATA} (grows incrementally)")
    print(f"   Archive: {Config.MERGED_GAMES_ARCHIVE_DIR}/ (pending results)")
    print(f"   Model: {Config.MODEL_FILE}")
    if mapper.unmapped_teams:
        print(f"   Unmapped teams: unmapped_teams.txt (ACTION REQUIRED)")
    print(f"\nRun Streamlit app: streamlit run V10app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V12 Unified NCAA Basketball Betting Pipeline - With Incremental Training Data")
    parser.add_argument('--retrain', action='store_true', help='Force model retraining')
    parser.add_argument('--quick-odds', action='store_true',
                       help='Quick mode: only refresh odds and predict (skip scraping/training)')
    args = parser.parse_args()

    # ========== QUICK ODDS REFRESH MODE ==========
    if args.quick_odds:
        print("\n" + "="*60)
        print("⚡ QUICK ODDS REFRESH MODE")
        print("="*60)
        print("Refreshing odds and predictions without re-scraping/training")

        import joblib
        import sys

        start_time = datetime.now()

        # Check required files exist
        required_files = {
            Config.MODEL_FILE: "Trained model",
            Config.TEAM_STATS_STANDARDIZED: "Team stats",
            Config.TRAINING_DATA: "Training data",
            Config.TEAM_MAPPING_CSV: "Team name mapping"
        }

        missing_files = []
        for file_path, file_desc in required_files.items():
            if not os.path.exists(file_path):
                missing_files.append(f"   ❌ {file_desc}: {file_path}")

        if missing_files:
            print("\n❌ Error: Required files not found. Run full pipeline first:")
            for msg in missing_files:
                print(msg)
            print("\nRun: python v12_unified_pipeline.py")
            sys.exit(1)

        print("✅ All required files found")

        # Initialize team name mapper
        print("\n📋 Loading team name mapper...")
        mapper = TeamNameMapper(Config.TEAM_MAPPING_CSV)

        # Load team stats
        print("📊 Loading team stats...")
        team_stats_df = pd.read_csv(Config.TEAM_STATS_STANDARDIZED)
        print(f"   Loaded stats for {len(team_stats_df)} teams")

        # Fetch today's odds
        print("\n🔄 Fetching today's odds from DraftKings...")
        todays_odds_raw = fetch_todays_odds()

        if todays_odds_raw.empty:
            print("\n⚠️  No DraftKings odds available today")
            print("   No games to predict")
            sys.exit(0)

        # Standardize odds
        print("🔧 Standardizing odds...")
        todays_odds = standardize_todays_odds(todays_odds_raw, mapper)
        print(f"   Standardized {len(todays_odds)} games")

        # Merge odds with stats
        print("\n🎲 Merging odds with team stats...")
        todays_merged = merge_today_with_stats(todays_odds, team_stats_df)

        if todays_merged.empty:
            print("   ❌ No games could be merged (team stats missing)")
            sys.exit(1)

        print(f"   Merged {len(todays_merged)} games")

        # Load model
        print("\n🤖 Loading trained model...")
        model = joblib.load(Config.MODEL_FILE)
        print(f"   ✅ Model loaded from {Config.MODEL_FILE}")

        # Get feature columns from the model itself (definitive source)
        print("📋 Extracting feature columns from trained model...")
        # For CalibratedClassifierCV, need to access the base estimator
        if hasattr(model, 'estimators'):  # CalibratedClassifierCV
            feature_columns = model.estimators[0].feature_names_in_
        else:  # Regular model
            feature_columns = model.feature_names_in_
        print(f"   ✅ Model expects {len(feature_columns)} feature columns")

        # Make predictions
        print("\n🔮 Making predictions...")
        predictions_df = predict_todays_games(todays_merged, model, feature_columns)

        # Archive today's data (raw stats only)
        # Feature engineering will be applied dynamically during training
        print("\n📦 Archiving today's data...")
        game_date = datetime.now().strftime('%Y-%m-%d')
        archive_todays_merged_data(todays_merged, game_date, apply_features=False)

        # Save unmapped teams report
        mapper.save_unmapped_teams_report("unmapped_teams.txt")

        # Completion
        elapsed = (datetime.now() - start_time).total_seconds()
        print("\n" + "="*60)
        print(f"⚡ QUICK REFRESH COMPLETED in {elapsed:.1f} seconds")
        print("="*60)
        print(f"\nOutput files:")
        print(f"   Predictions: {Config.PREDICTIONS_OUTPUT}")
        print(f"\n💡 To run full pipeline (scrape + train): python v12_unified_pipeline.py")

        sys.exit(0)

    # ========== FULL PIPELINE MODE ==========
    main(force_retrain=args.retrain)
