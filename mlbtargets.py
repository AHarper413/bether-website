# enhanced_mlb_betting_model.py
# Version 5.0 - Major improvements: XGBoost, walk-forward validation, feature selection, bullpen analysis

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import statsapi
from pybaseball import cache, standings, team_pitching, team_batting, pitching_stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings
import time
import pytz
from geopy.distance import geodesic

# NEW IMPORTS for enhanced models
try:
    import xgboost as xgb
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost not available, falling back to Random Forest")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("⚠️ LightGBM not available")
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings('ignore')
cache.enable()

# -------------------------
# Config
# -------------------------
CURRENT_YEAR = datetime.now().year
TODAY = datetime.now().strftime("%Y-%m-%d")
TOMORROW = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

output_dir = f"mlb_improved_data_{CURRENT_YEAR}"
os.makedirs(output_dir, exist_ok=True)
models_dir = os.path.join(output_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# -------------------------
# Stadium Factors and Environmental Data (Enhanced)
# -------------------------
STADIUM_RUN_FACTORS = {
    'Colorado Rockies': 1.15,      # Coors Field - thin air
    'Boston Red Sox': 1.08,        # Fenway - Green Monster
    'Cincinnati Reds': 1.06,       # GABP - hitter friendly
    'Texas Rangers': 1.05,         # Globe Life - hot weather
    'Arizona Diamondbacks': 1.04,  # Chase Field - dry air
    'Baltimore Orioles': 1.03,     # Camden Yards
    'Chicago Cubs': 1.02,          # Wrigley - wind
    'New York Yankees': 1.01,      # Yankee Stadium - short RF
    'Kansas City Royals': 1.01,    # Kauffman - big OF
    'Minnesota Twins': 1.00,       # Target Field - dome
    'Houston Astros': 0.99,        # Minute Maid - dome
    'Toronto Blue Jays': 0.99,     # Rogers Centre - dome
    'Tampa Bay Rays': 0.98,        # Tropicana - dome
    'Seattle Mariners': 0.98,      # T-Mobile - marine air
    'San Diego Padres': 0.97,      # Petco - marine air, big OF
    'Los Angeles Angels': 0.96,    # Angel Stadium - marine air
    'San Francisco Giants': 0.95,  # Oracle Park - marine air, cold
    'Los Angeles Dodgers': 0.94,   # Dodger Stadium - pitcher friendly
    'Miami Marlins': 0.93,         # loanDepot - pitcher park
    'Oakland Athletics': 0.95,     # Coliseum - foul territory
    'Atlanta Braves': 1.03,        # Truist Park
    'Chicago White Sox': 1.00,     # Guaranteed Rate
    'Cleveland Guardians': 0.98,   # Progressive Field
    'Detroit Tigers': 0.99,        # Comerica Park
    'Milwaukee Brewers': 1.00,     # American Family Field
    'New York Mets': 0.98,         # Citi Field - pitcher friendly
    'Philadelphia Phillies': 1.02, # Citizens Bank Park
    'Pittsburgh Pirates': 0.97,    # PNC Park - big OF
    'St. Louis Cardinals': 1.01,   # Busch Stadium
    'Washington Nationals': 0.99   # Nationals Park
}

DOME_STADIUMS = {
    'Houston Astros', 'Tampa Bay Rays', 'Toronto Blue Jays', 
    'Minnesota Twins', 'Arizona Diamondbacks', 'Miami Marlins'
}

WEST_COAST_TEAMS = {'Los Angeles Dodgers', 'Los Angeles Angels', 'San Francisco Giants', 
                   'Oakland Athletics', 'San Diego Padres', 'Seattle Mariners'}
EAST_COAST_TEAMS = {'Boston Red Sox', 'New York Yankees', 'New York Mets', 'Tampa Bay Rays',
                   'Baltimore Orioles', 'Philadelphia Phillies', 'Washington Nationals', 
                   'Miami Marlins', 'Atlanta Braves', 'Toronto Blue Jays'}

# NEW: Umpire factors (you would need to populate this with real data)
UMPIRE_RUN_FACTORS = {
    # Example - you'd need to research actual umpire tendencies
    'Angel Hernandez': 1.02,  # Higher run environment
    'Joe West': 0.98,         # Lower run environment
    # Add more umpires as you gather data
}

# -------------------------
# Team Mapping and Stadium Locations
# -------------------------
team_mapping = {
    'LAD': 'Los Angeles Dodgers', 'NYY': 'New York Yankees', 'ATL': 'Atlanta Braves',
    'PHI': 'Philadelphia Phillies', 'SD': 'San Diego Padres', 'TOR': 'Toronto Blue Jays',
    'BOS': 'Boston Red Sox', 'PIT': 'Pittsburgh Pirates', 'STL': 'St. Louis Cardinals',
    'CIN': 'Cincinnati Reds', 'TBR': 'Tampa Bay Rays', 'TEX': 'Texas Rangers',
    'SEA': 'Seattle Mariners', 'KC': 'Kansas City Royals', 'AZ': 'Arizona Diamondbacks',
    'SF': 'San Francisco Giants', 'HOU': 'Houston Astros', 'COL': 'Colorado Rockies',
    'DET': 'Detroit Tigers', 'WSH': 'Washington Nationals', 'ATH': 'Oakland Athletics',
    'CWS': 'Chicago White Sox', 'LAA': 'Los Angeles Angels', 'MIA': 'Miami Marlins',
    'MIL': 'Milwaukee Brewers', 'MIN': 'Minnesota Twins', 'NYM': 'New York Mets',
    'CLE': 'Cleveland Guardians', 'CHC': 'Chicago Cubs', 'BAL': 'Baltimore Orioles'
}

# Stadium locations for travel distance calculation
stadium_locations = {
    'Arizona Diamondbacks': (33.4455, -112.0667),
    'Atlanta Braves': (33.8906, -84.4677),
    'Baltimore Orioles': (39.2840, -76.6218),
    'Boston Red Sox': (42.3467, -71.0972),
    'Chicago Cubs': (41.9484, -87.6553),
    'Chicago White Sox': (41.8300, -87.6338),
    'Cincinnati Reds': (39.0974, -84.5063),
    'Cleveland Guardians': (41.4962, -81.6852),
    'Colorado Rockies': (39.7559, -104.9942),
    'Detroit Tigers': (42.3390, -83.0485),
    'Houston Astros': (29.7572, -95.3553),
    'Kansas City Royals': (39.0517, -94.4803),
    'Los Angeles Angels': (33.8003, -117.8827),
    'Los Angeles Dodgers': (34.0739, -118.2400),
    'Miami Marlins': (25.7781, -80.2197),
    'Milwaukee Brewers': (43.0280, -87.9712),
    'Minnesota Twins': (44.9817, -93.2776),
    'New York Mets': (40.7571, -73.8458),
    'New York Yankees': (40.8296, -73.9262),
    'Oakland Athletics': (37.7516, -122.2005),
    'Philadelphia Phillies': (39.9061, -75.1665),
    'Pittsburgh Pirates': (40.4469, -80.0058),
    'San Diego Padres': (32.7073, -117.1566),
    'San Francisco Giants': (37.7786, -122.3893),
    'Seattle Mariners': (47.5914, -122.3326),
    'St. Louis Cardinals': (38.6226, -90.1928),
    'Tampa Bay Rays': (27.7682, -82.6534),
    'Texas Rangers': (32.7511, -97.0821),
    'Toronto Blue Jays': (43.6414, -79.3894),
    'Washington Nationals': (38.8730, -77.0074)
}

teams_info = statsapi.get('teams', {'sportIds':'1'})['teams']
abbr_to_id = {team['abbreviation']: team['id'] for team in teams_info}

# -------------------------
# Enhanced Schedule Fetching (Keep existing)
# -------------------------
def fetch_enhanced_schedule_data():
    """Fetch schedule data incrementally - only get missing games"""
    schedule_path = os.path.join(output_dir, f"enhanced_schedule_{CURRENT_YEAR}.csv")
    
    if os.path.exists(schedule_path):
        print("📁 Loading existing schedule cache...")
        schedule_df = pd.read_csv(schedule_path, parse_dates=["game_date"])
        
        # Ensure score columns are numeric
        if 'home_score' in schedule_df.columns:
            schedule_df['home_score'] = pd.to_numeric(schedule_df['home_score'], errors='coerce')
        if 'away_score' in schedule_df.columns:
            schedule_df['away_score'] = pd.to_numeric(schedule_df['away_score'], errors='coerce')
        
        # Find the most recent game date in cache
        most_recent_date = schedule_df['game_date'].max()
        print(f"📅 Most recent cached game: {most_recent_date.strftime('%Y-%m-%d')}")
        
        # Fetch only games since the most recent date
        fetch_start_date = most_recent_date.strftime('%Y-%m-%d')
        
        print(f"🔄 Fetching games from {fetch_start_date} to {TODAY}...")
        
        new_games = []
        for abbrev, team_name in team_mapping.items():
            team_id = abbr_to_id.get(abbrev)
            if not team_id:
                continue
            
            try:
                schedule = statsapi.schedule(
                    start_date=fetch_start_date, 
                    end_date=TODAY, 
                    team=team_id
                )
                for game in schedule:
                    game['team_abbrev'] = abbrev
                    game['team_name'] = team_name
                new_games.extend(schedule)
            except Exception as e:
                print(f"⚠️ Error fetching schedule for {team_name}: {e}")
                continue
        
        if new_games:
            new_games_df = pd.DataFrame(new_games)
            new_games_df['game_date'] = pd.to_datetime(new_games_df['game_date'])
            
            # Remove duplicates and games already in cache
            existing_game_ids = set(schedule_df['game_id']) if 'game_id' in schedule_df.columns else set()
            new_games_df = new_games_df[~new_games_df['game_id'].isin(existing_game_ids)]
            
            if not new_games_df.empty:
                # Ensure score columns are numeric for new games
                if 'home_score' in new_games_df.columns:
                    new_games_df['home_score'] = pd.to_numeric(new_games_df['home_score'], errors='coerce')
                if 'away_score' in new_games_df.columns:
                    new_games_df['away_score'] = pd.to_numeric(new_games_df['away_score'], errors='coerce')
                
                print(f"📈 Found {len(new_games_df)} new games to add")
                
                # Append new games to existing schedule
                schedule_df = pd.concat([schedule_df, new_games_df], ignore_index=True)
                schedule_df = schedule_df.drop_duplicates(subset=['game_id'], keep='last')
                schedule_df = schedule_df.sort_values('game_date')
                
                # Save updated schedule
                schedule_df.to_csv(schedule_path, index=False)
                print(f"✅ Updated schedule cache with {len(new_games_df)} new games")
            else:
                print("📅 No new games found since last update")
        else:
            print("📅 No new games to fetch")
            
    else:
        print("📅 No existing cache - fetching full season schedule...")
        all_games = []
        
        # Fetch full season schedule (first time only)
        for abbrev, team_name in team_mapping.items():
            team_id = abbr_to_id.get(abbrev)
            if not team_id:
                continue
            
            try:
                schedule = statsapi.schedule(
                    start_date=f"{CURRENT_YEAR}-01-01", 
                    end_date=TODAY, 
                    team=team_id
                )
                for game in schedule:
                    game['team_abbrev'] = abbrev
                    game['team_name'] = team_name
                all_games.extend(schedule)
            except Exception as e:
                print(f"⚠️ Error fetching schedule for {team_name}: {e}")
                continue

        schedule_df = pd.DataFrame(all_games)
        if not schedule_df.empty:
            schedule_df['game_date'] = pd.to_datetime(schedule_df['game_date'])
            schedule_df.dropna(subset=['home_name', 'away_name'], inplace=True)
            
            # Ensure score columns are numeric from the start
            if 'home_score' in schedule_df.columns:
                schedule_df['home_score'] = pd.to_numeric(schedule_df['home_score'], errors='coerce')
            if 'away_score' in schedule_df.columns:
                schedule_df['away_score'] = pd.to_numeric(schedule_df['away_score'], errors='coerce')
            
            schedule_df.to_csv(schedule_path, index=False)
            print(f"✅ Created initial schedule cache with {len(schedule_df)} games")
    
    # Clean up and return final dataset
    # Clean up and return final dataset
    if not schedule_df.empty:
        schedule_df.dropna(subset=['home_name', 'away_name'], inplace=True)
        
        # NEW: Filter out games with 0-0 scores (likely unplayed games)
        before_filter = len(schedule_df)
        schedule_df = schedule_df[
            ~((schedule_df['home_score'] == 0) & (schedule_df['away_score'] == 0))
        ]
        after_filter = len(schedule_df)
        
        completed_games = len(schedule_df.dropna(subset=['home_score', 'away_score']))
        total_games = len(schedule_df)
        
        print(f"📊 Final dataset: {total_games} total games, {completed_games} completed")
        print(f"📊 Filtered out {before_filter - after_filter} games with 0-0 scores (likely unplayed)")

    return schedule_df

# -------------------------
# Enhanced Recent Form (Keep existing)
# -------------------------
def calculate_weighted_recent_form(schedule_df, team_name, as_of_date, days_back=15):
    """Calculate recent form with exponential weighting (more recent = more important)"""
    try:
        cutoff_date = as_of_date - timedelta(days=days_back)
        
        recent_games = schedule_df[
            ((schedule_df['home_name'] == team_name) | (schedule_df['away_name'] == team_name)) &
            (schedule_df['game_date'] >= cutoff_date) &
            (schedule_df['game_date'] < as_of_date) &
            (schedule_df['home_score'].notna()) &
            (schedule_df['away_score'].notna())
        ].copy().sort_values('game_date')
        
        if len(recent_games) == 0:
            return 0.5
        
        # Calculate wins with exponential weighting (recent games matter more)
        total_weight = 0
        weighted_wins = 0
        
        for i, (_, game) in enumerate(recent_games.iterrows()):
            # More recent games get higher weight
            days_ago = (as_of_date - game['game_date']).days
            weight = np.exp(-days_ago / 5.0)  # Exponential decay
            
            total_weight += weight
            
            # Determine if this team won
            if game['home_name'] == team_name:
                won = game['home_score'] > game['away_score']
            else:
                won = game['away_score'] > game['home_score']
            
            if won:
                weighted_wins += weight
        
        if total_weight > 0:
            weighted_form = weighted_wins / total_weight
            return np.clip(weighted_form, 0.1, 0.9)
        else:
            return 0.5
            
    except Exception as e:
        return 0.5

# -------------------------
# NEW: Enhanced Bullpen Fatigue Analysis
# -------------------------
def calculate_bullpen_fatigue(schedule_df, team_name, game_date, days_back=3):
    """Track bullpen usage based on recent game patterns"""
    
    recent_games = schedule_df[
        ((schedule_df['home_name'] == team_name) | (schedule_df['away_name'] == team_name)) &
        (schedule_df['game_date'] >= game_date - timedelta(days=days_back)) &
        (schedule_df['game_date'] < game_date) &
        (schedule_df['home_score'].notna()) &
        (schedule_df['away_score'].notna())
    ].copy()
    
    if len(recent_games) == 0:
        return {
            'bullpen_fatigue': 0.0,
            'games_last_3_days': 0,
            'close_games_recent': 0,
            'high_scoring_recent': 0
        }
    
    bullpen_fatigue = 0
    close_games = 0
    high_scoring_games = 0
    
    for _, game in recent_games.iterrows():
        total_runs = game['home_score'] + game['away_score']
        margin = abs(game['home_score'] - game['away_score'])
        
        # Close games = heavy bullpen usage
        if margin <= 3:
            bullpen_fatigue += 0.3
            close_games += 1
        
        # High-scoring games = heavy bullpen usage
        if total_runs > 12:
            bullpen_fatigue += 0.2
            high_scoring_games += 1
        
        # Extra innings (would need to detect this - approximation)
        if total_runs > 15:  # Likely extra innings
            bullpen_fatigue += 0.4
        
        # Blowouts = less bullpen fatigue for winning team
        if margin > 6:
            if game['home_name'] == team_name:
                won_big = game['home_score'] > game['away_score']
            else:
                won_big = game['away_score'] > game['home_score']
            
            if won_big:
                bullpen_fatigue -= 0.1  # Starters went longer
    
    return {
        'bullpen_fatigue': max(0, min(bullpen_fatigue, 1.0)),  # Cap at 1.0
        'games_last_3_days': len(recent_games),
        'close_games_recent': close_games,
        'high_scoring_recent': high_scoring_games
    }

# -------------------------
# Enhanced Form Momentum (Keep existing)
# -------------------------
def calculate_enhanced_form_momentum(schedule_df, home_team, away_team, game_date, days_back=10):
    """Fixed form momentum that actually captures game margins and clutch performance"""
    
    def get_team_recent_margins(team_name, as_of_date):
        recent_games = schedule_df[
            ((schedule_df['home_name'] == team_name) | (schedule_df['away_name'] == team_name)) &
            (schedule_df['game_date'] >= as_of_date - timedelta(days=days_back)) &
            (schedule_df['game_date'] < as_of_date) &
            (schedule_df['home_score'].notna()) &
            (schedule_df['away_score'].notna())
        ].copy()
        
        if len(recent_games) == 0:
            return 0, 0.5, 0
        
        margins = []
        close_games = 0
        close_wins = 0
        
        for _, game in recent_games.iterrows():
            if game['home_name'] == team_name:
                margin = game['home_score'] - game['away_score']
                won = margin > 0
            else:
                margin = game['away_score'] - game['home_score']  
                won = margin > 0
            
            margins.append(margin)
            
            # Track clutch performance (games decided by 3 runs or less)
            if abs(margin) <= 3:
                close_games += 1
                if won:
                    close_wins += 1
        
        avg_margin = np.mean(margins)
        clutch_rate = close_wins / close_games if close_games > 0 else 0.5
        recent_games_count = len(recent_games)
        
        return avg_margin, clutch_rate, recent_games_count
    
    try:
        home_margin, home_clutch, home_games = get_team_recent_margins(home_team, game_date)
        away_margin, away_clutch, away_games = get_team_recent_margins(away_team, game_date)
        
        # Weight by sample size
        min_games = min(home_games, away_games)
        if min_games < 3:
            return 0  # Not enough data
        
        # Margin momentum (normalized)
        margin_momentum = np.tanh((home_margin - away_margin) / 8.0)  # Smooth scaling
        
        # Clutch momentum 
        clutch_momentum = (home_clutch - away_clutch) * 0.5
        
        # Combined with more weight on margins
        enhanced_momentum = (margin_momentum * 0.7) + (clutch_momentum * 0.3)
        
        return enhanced_momentum
        
    except Exception as e:
        return 0

# -------------------------
# Series Context (Keep existing)
# -------------------------
def get_series_context_features(schedule_df, home_team, away_team, game_date):
    """Add series and head-to-head context"""
    
    # Find recent games between these teams (last 2 years)
    h2h_games = schedule_df[
        (((schedule_df['home_name'] == home_team) & (schedule_df['away_name'] == away_team)) |
         ((schedule_df['home_name'] == away_team) & (schedule_df['away_name'] == home_team))) &
        (schedule_df['game_date'] < game_date) &
        (schedule_df['game_date'] >= game_date - timedelta(days=730)) &  # Last 2 years
        (schedule_df['home_score'].notna())
    ].copy()
    
    # Head-to-head record
    home_wins = 0
    total_h2h = len(h2h_games)
    
    for _, game in h2h_games.iterrows():
        if game['home_name'] == home_team:
            if game['home_score'] > game['away_score']:
                home_wins += 1
        else:  # away_team was home
            if game['away_score'] > game['home_score']:
                home_wins += 1
    
    h2h_advantage = (home_wins / total_h2h - 0.5) if total_h2h > 0 else 0
    
    # Series game number (find current series)
    current_series_games = schedule_df[
        (((schedule_df['home_name'] == home_team) & (schedule_df['away_name'] == away_team)) |
         ((schedule_df['home_name'] == away_team) & (schedule_df['away_name'] == home_team))) &
        (schedule_df['game_date'] >= game_date - timedelta(days=7)) &  # Within a week
        (schedule_df['game_date'] <= game_date + timedelta(days=7))
    ].sort_values('game_date')
    
    series_game_number = 1
    for i, (_, series_game) in enumerate(current_series_games.iterrows()):
        if series_game['game_date'] == game_date:
            series_game_number = i + 1
            break
    
    return {
        'h2h_advantage': h2h_advantage,
        'h2h_games_sample': min(total_h2h, 10),  # Cap influence of small samples
        'series_game_number': series_game_number,
        'series_fatigue': 0.02 * (series_game_number - 1)  # Slight fatigue in long series
    }

# -------------------------
# Stadium Features (Keep existing)
# -------------------------
def get_stadium_features(home_team, away_team, game_date):
    """Add stadium-specific factors"""
    
    home_run_factor = STADIUM_RUN_FACTORS.get(home_team, 1.0)
    is_dome = 1 if home_team in DOME_STADIUMS else 0
    
    # Time of year factors (basic seasonality)
    day_of_year = game_date.timetuple().tm_yday
    
    # Spring/Summer/Fall adjustments
    if day_of_year < 120:  # April-early May (cold)
        weather_run_factor = 0.95
    elif day_of_year < 180:  # Late May-June (warming up)
        weather_run_factor = 1.02
    elif day_of_year < 240:  # July-August (hot)
        weather_run_factor = 1.05
    else:  # September+ (cooling)
        weather_run_factor = 0.98
    
    # Dome games not affected by weather
    if is_dome:
        effective_run_factor = home_run_factor
    else:
        effective_run_factor = home_run_factor * weather_run_factor
    
    # Travel factor for away team
    cross_country_travel = 0
    if away_team in WEST_COAST_TEAMS and home_team in EAST_COAST_TEAMS:
        cross_country_travel = 0.02  # Slight disadvantage
    elif away_team in EAST_COAST_TEAMS and home_team in WEST_COAST_TEAMS:
        cross_country_travel = 0.02
    
    return {
        'stadium_run_factor': effective_run_factor,
        'is_dome_game': is_dome,
        'cross_country_travel': cross_country_travel,
        'seasonal_offense_boost': weather_run_factor - 1.0
    }

# -------------------------
# Temporal Features (Keep existing)
# -------------------------
def get_temporal_features(game_date, game_time=None):
    """Add day of week and time of day factors"""
    
    day_of_week = game_date.weekday()  # 0=Monday, 6=Sunday
    
    # Weekend games (different atmosphere)
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Friday games (start of weekend)
    is_friday = 1 if day_of_week == 4 else 0
    
    # Day of season (fatigue factor)
    season_start = datetime(game_date.year, 3, 28)  # Approximate season start
    days_into_season = (game_date - season_start).days
    season_fatigue = min(days_into_season / 180.0, 1.0)  # 0 to 1 over 6 months
    
    return {
        'is_weekend_game': is_weekend,
        'is_friday_game': is_friday, 
        'season_fatigue_factor': season_fatigue,
        'late_season_boost': 0.02 if days_into_season > 140 else 0  # September push
    }

# -------------------------
# Rest Days (Keep existing)
# -------------------------
def calculate_rest_days(schedule_df, team_name, game_date):
    """Calculate days of rest before this game"""
    try:
        # Find the most recent completed game before this date
        previous_games = schedule_df[
            ((schedule_df['home_name'] == team_name) | (schedule_df['away_name'] == team_name)) &
            (schedule_df['game_date'] < game_date) &
            (schedule_df['home_score'].notna()) &
            (schedule_df['away_score'].notna())
        ].sort_values('game_date')
        
        if len(previous_games) == 0:
            return 1  # Default to 1 day rest
        
        last_game_date = previous_games.iloc[-1]['game_date']
        rest_days = (game_date - last_game_date).days
        return min(rest_days, 4)  # Cap at 4 days
        
    except Exception as e:
        return 1  # Default

# -------------------------
# Enhanced Home Field Advantage (Keep existing)
# -------------------------
def calculate_enhanced_home_advantage(home_stats, away_stats, schedule_df, home_team, away_team):
    """More nuanced home advantage calculation that allows more variation"""
    base_advantage = 0.535  # MLB average
    
    try:
        # Team-specific home performance with LARGER influence
        home_games = schedule_df[
            (schedule_df['home_name'] == home_team) &
            (schedule_df['home_score'].notna())
        ]
        
        if len(home_games) >= 8:  # Need reasonable sample
            home_wins = len(home_games[home_games['home_score'] > home_games['away_score']])
            home_rate = home_wins / len(home_games)
            team_factor = (home_rate - 0.5) * 0.6  # INCREASED from 0.3 to 0.6
        else:
            team_factor = 0
        
        # Away team road struggles with LARGER influence
        away_games = schedule_df[
            (schedule_df['away_name'] == away_team) &
            (schedule_df['away_score'].notna())
        ]
        
        if len(away_games) >= 8:
            away_wins = len(away_games[away_games['away_score'] > away_games['home_score']])
            away_rate = away_wins / len(away_games)
            road_factor = (0.5 - away_rate) * 0.4  # INCREASED from 0.2 to 0.4
        else:
            road_factor = 0
        
        # Team quality adjustment - allows elite vs poor teams to have bigger home advantage
        quality_diff = home_stats['pythagorean_exp'] - away_stats['pythagorean_exp']
        quality_factor = quality_diff * 0.15  # NEW: Quality matters for home advantage
        
        enhanced_advantage = base_advantage + team_factor + road_factor + quality_factor
        return np.clip(enhanced_advantage, 0.38, 0.72)  # WIDER bounds for more variation
        
    except:
        return base_advantage

# -------------------------
# ENHANCED Team Season Stats with Better Error Handling
# -------------------------
def get_enhanced_team_season_stats():
    print("📊 Fetching enhanced team season statistics...")
    
    stats_cache_path = os.path.join(output_dir, f"enhanced_team_stats_{CURRENT_YEAR}.csv")
    if os.path.exists(stats_cache_path):
        cache_age_hours = (time.time() - os.path.getmtime(stats_cache_path)) / 3600
        if cache_age_hours < 24:  # Less than 24 hours old
            print("📁 Loading cached team stats...")
            return pd.read_csv(stats_cache_path)
        else:
            print("🔄 Team stats cache expired, refreshing...")
    
    team_stats = []
    
    # Get team batting and pitching data
    batting_data = pd.DataFrame()
    pitching_data = pd.DataFrame()
    
    try:
        batting_data = team_batting(CURRENT_YEAR)
        if batting_data is not None and not batting_data.empty:
            print("✅ Got team batting data")
            print(f"📊 Batting columns available: {list(batting_data.columns)}")  # DEBUG
        else:
            batting_data = pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Team batting failed: {e}")
        batting_data = pd.DataFrame()
    
    try:
        pitching_data = team_pitching(CURRENT_YEAR)
        if pitching_data is not None and not pitching_data.empty:
            print("✅ Got team pitching data")
            print(f"📊 Pitching columns available: {list(pitching_data.columns)}")  # DEBUG
        else:
            pitching_data = pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Team pitching failed: {e}")
        pitching_data = pd.DataFrame()
    
    # Team abbreviation mapping for data
    abbrev_to_full_name = {
        'ARI': 'Arizona Diamondbacks', 'ATH': 'Oakland Athletics', 'ATL': 'Atlanta Braves',
        'BAL': 'Baltimore Orioles', 'BOS': 'Boston Red Sox', 'CHC': 'Chicago Cubs',
        'CHW': 'Chicago White Sox', 'CIN': 'Cincinnati Reds', 'CLE': 'Cleveland Guardians',
        'COL': 'Colorado Rockies', 'DET': 'Detroit Tigers', 'HOU': 'Houston Astros',
        'KCR': 'Kansas City Royals', 'LAA': 'Los Angeles Angels', 'LAD': 'Los Angeles Dodgers',
        'MIA': 'Miami Marlins', 'MIL': 'Milwaukee Brewers', 'MIN': 'Minnesota Twins',
        'NYM': 'New York Mets', 'NYY': 'New York Yankees', 'PHI': 'Philadelphia Phillies',
        'PIT': 'Pittsburgh Pirates', 'SDP': 'San Diego Padres', 'SEA': 'Seattle Mariners',
        'SFG': 'San Francisco Giants', 'STL': 'St. Louis Cardinals', 'TBR': 'Tampa Bay Rays',
        'TEX': 'Texas Rangers', 'TOR': 'Toronto Blue Jays', 'WSN': 'Washington Nationals'
    }
    
    # Compile stats for each team
    for team_name in team_mapping.values():
        team_row = {'team_name': team_name}
        
        # Find abbreviation
        data_team_abbrev = None
        for abbrev, full_name in abbrev_to_full_name.items():
            if full_name == team_name:
                data_team_abbrev = abbrev
                break
        
        # From batting data with enhanced error handling
        if not batting_data.empty and data_team_abbrev:
            batting_team = batting_data[batting_data['Team'] == data_team_abbrev]
            if not batting_team.empty:
                batting_stats = batting_team.iloc[0]
                team_row.update({
                    'runs_scored': batting_stats.get('R', 400),
                    'team_obp': batting_stats.get('OBP', 0.320),
                    'team_slg': batting_stats.get('SLG', 0.400),
                    'team_ops': batting_stats.get('OPS', 0.720),
                    'team_hrs': batting_stats.get('HR', 150),
                    # Use get() with defaults for advanced stats that might not exist
                    'team_iso': batting_stats.get('ISO', batting_stats.get('SLG', 0.400) - batting_stats.get('AVG', 0.260)),
                    'team_wrc_plus': batting_stats.get('wRC+', 100),
                    'team_bb_rate': batting_stats.get('BB%', 0.08),
                    'team_k_rate': batting_stats.get('K%', 0.22),
                })
        
        # From pitching data with enhanced error handling
        if not pitching_data.empty and data_team_abbrev:
            pitching_team = pitching_data[pitching_data['Team'] == data_team_abbrev]
            if not pitching_team.empty:
                pitching_stats = pitching_team.iloc[0]
                team_row.update({
                    'wins': pitching_stats.get('W', 40),
                    'losses': pitching_stats.get('L', 40),
                    'runs_allowed': pitching_stats.get('R', 400),
                    'team_era': pitching_stats.get('ERA', 4.50),
                    'team_whip': pitching_stats.get('WHIP', 1.30),
                    # Use get() with defaults for advanced stats that might not exist
                    'team_k9': pitching_stats.get('K/9', 8.0),
                    'team_bb9': pitching_stats.get('BB/9', 3.0),
                    'team_fip': pitching_stats.get('FIP', 4.00),
                    'team_h9': pitching_stats.get('H/9', 9.0),
                    'team_hr9': pitching_stats.get('HR/9', 1.2),
                })
        
        # Calculate derived stats
        games_played = team_row.get('wins', 40) + team_row.get('losses', 40)
        team_row['games_played'] = games_played
        team_row['win_pct'] = team_row.get('wins', 40) / games_played if games_played > 0 else 0.5
        
        runs_scored = team_row.get('runs_scored', 400)
        runs_allowed = team_row.get('runs_allowed', 400)
        team_row['run_diff'] = runs_scored - runs_allowed
        
        # Pythagorean expectation
        if runs_scored > 0 and runs_allowed > 0:
            rs_sq = runs_scored ** 2
            ra_sq = runs_allowed ** 2
            team_row['pythagorean_exp'] = rs_sq / (rs_sq + ra_sq)
        else:
            team_row['pythagorean_exp'] = 0.5
        
        # Fill defaults for missing values
        defaults = {
            'wins': 40, 'losses': 40, 'win_pct': 0.5, 'runs_scored': 400, 'runs_allowed': 400,
            'run_diff': 0, 'games_played': 80, 'team_obp': 0.320, 'team_slg': 0.400,
            'team_ops': 0.720, 'team_iso': 0.150, 'team_era': 4.50, 'team_whip': 1.30,
            'team_k9': 8.0, 'team_bb9': 3.0, 'pythagorean_exp': 0.5, 'team_fip': 4.00,
            'team_hrs': 150, 'team_wrc_plus': 100, 'team_bb_rate': 0.08, 'team_k_rate': 0.22,
            'team_h9': 9.0, 'team_hr9': 1.2
        }
        
        for key, default_value in defaults.items():
            if key not in team_row:
                team_row[key] = default_value
        
        team_stats.append(team_row)
    
    team_stats_df = pd.DataFrame(team_stats)
    team_stats_df.to_csv(stats_cache_path, index=False)
    print(f"✅ Enhanced stats compiled for {len(team_stats_df)} teams")
    
    return team_stats_df

# -------------------------
# ENHANCED Pitcher Context with Better Field Handling
# -------------------------
def get_enhanced_pitcher_context():
    print("⚾ Fetching enhanced pitcher context...")
    
    pitcher_cache_path = os.path.join(output_dir, f"enhanced_pitcher_context_{CURRENT_YEAR}.csv")
    
    if os.path.exists(pitcher_cache_path):
        cache_time = os.path.getmtime(pitcher_cache_path)
        if time.time() - cache_time < 86400:
            print("📁 Loading cached pitcher context...")
            return pd.read_csv(pitcher_cache_path)
    
    try:
        pitcher_data = pitching_stats(CURRENT_YEAR, qual=0)
        
        if pitcher_data is not None and not pitcher_data.empty:
            print(f"📊 Pitcher columns available: {list(pitcher_data.columns)}")  # DEBUG
            
            # Start with basic fields that should exist
            pitcher_context = pitcher_data[['Name', 'ERA', 'IP', 'Age']].copy()
            
            # Add enhanced fields - these DO exist based on the output
            pitcher_context['FIP'] = pitcher_data['FIP']
            pitcher_context['xFIP'] = pitcher_data['xFIP']
            pitcher_context['K_rate'] = pitcher_data['K%']
            pitcher_context['BB_rate'] = pitcher_data['BB%']
            pitcher_context['K_9'] = pitcher_data['K/9']
            pitcher_context['BB_9'] = pitcher_data['BB/9']
            pitcher_context['H_9'] = pitcher_data['H/9']
            pitcher_context['HR_9'] = pitcher_data['HR/9']
            
            # Create last name for matching
            pitcher_context['last_name'] = pitcher_context['Name'].str.split().str[-1]
            
            pitcher_context.to_csv(pitcher_cache_path, index=False)
            print(f"✅ Got enhanced context for {len(pitcher_context)} pitchers")
            return pitcher_context
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"⚠️ Error fetching pitcher context: {e}")
        return pd.DataFrame()

# -------------------------
# ENHANCED Pitcher Features with Advanced Metrics
# -------------------------
def calculate_enhanced_pitcher_features(home_pitcher, away_pitcher, pitcher_context, schedule_df, home_team, away_team, game_date):
    """More sophisticated pitcher evaluation with advanced metrics"""
    
    def get_pitcher_advanced_stats(pitcher_name, context_df):
        if not isinstance(pitcher_name, str) or context_df.empty:
            return {
                'era': 4.50, 'fip': 4.00, 'xfip': 4.00, 'k_rate': 0.20, 
                'bb_rate': 0.08, 'ip': 50, 'age': 28
            }
        
        cleaned_name = pitcher_name.strip()
        if not cleaned_name:
            return {
                'era': 4.50, 'fip': 4.00, 'xfip': 4.00, 'k_rate': 0.20, 
                'bb_rate': 0.08, 'ip': 50, 'age': 28
            }
        
        name_parts = cleaned_name.split()
        if not name_parts:
            return {
                'era': 4.50, 'fip': 4.00, 'xfip': 4.00, 'k_rate': 0.20, 
                'bb_rate': 0.08, 'ip': 50, 'age': 28
            }
        
        last_name = name_parts[-1]
        matches = context_df[context_df['last_name'].str.contains(last_name, case=False, na=False)]
        
        if matches.empty:
            return {
                'era': 4.50, 'fip': 4.00, 'xfip': 4.00, 'k_rate': 0.20, 
                'bb_rate': 0.08, 'ip': 50, 'age': 28
            }
        
        pitcher = matches.nlargest(1, 'IP').iloc[0] if len(matches) > 1 else matches.iloc[0]
        
        return {
            'era': pitcher.get('ERA', 4.50),
            'fip': pitcher.get('FIP', 4.00),
            'xfip': pitcher.get('xFIP', 4.00),
            'k_rate': pitcher.get('K_rate', 0.20),
            'bb_rate': pitcher.get('BB_rate', 0.08),
            'ip': pitcher.get('IP', 50),
            'age': pitcher.get('Age', 28),
            'k9': pitcher.get('K_9', 8.0),
            'bb9': pitcher.get('BB_9', 3.0),
            'h9': pitcher.get('H_9', 9.0),
            'hr9': pitcher.get('HR_9', 1.2)
        }
    
    home_stats = get_pitcher_advanced_stats(home_pitcher, pitcher_context)
    away_stats = get_pitcher_advanced_stats(away_pitcher, pitcher_context)
    
    # Enhanced pitcher quality assessment
    def calculate_pitcher_quality_score(stats):
        # Weighted combination of multiple metrics
        era_score = max(0, (5.00 - stats['era']) / 2.5)  # 0-1 scale
        fip_score = max(0, (5.00 - stats['fip']) / 2.5)   # 0-1 scale
        k_score = min(1, stats['k_rate'] / 0.30)          # 0-1 scale
        bb_score = max(0, (0.12 - stats['bb_rate']) / 0.08)  # 0-1 scale
        ip_score = min(1, stats['ip'] / 150.0)            # Experience factor
        
        # Weighted average
        quality = (era_score * 0.25 + fip_score * 0.25 + k_score * 0.25 + 
                  bb_score * 0.15 + ip_score * 0.10)
        
        return np.clip(quality, 0, 1)
    
    home_quality = calculate_pitcher_quality_score(home_stats)
    away_quality = calculate_pitcher_quality_score(away_stats)
    
    # Specific metric advantages
    era_advantage = (away_stats['era'] - home_stats['era']) / 2.0
    fip_advantage = (away_stats['fip'] - home_stats['fip']) / 2.0
    k_rate_advantage = (home_stats['k_rate'] - away_stats['k_rate']) * 2.0
    control_advantage = (away_stats['bb_rate'] - home_stats['bb_rate']) * 5.0
    experience_advantage = min(home_stats['ip'], 200) - min(away_stats['ip'], 200)
    experience_advantage = experience_advantage / 100.0  # Normalize
    
    return {
        'pitcher_era_advantage': era_advantage,
        'pitcher_fip_advantage': fip_advantage,
        'pitcher_k_rate_advantage': k_rate_advantage,
        'pitcher_control_advantage': control_advantage,
        'pitcher_experience_advantage': experience_advantage,
        'pitcher_quality_advantage': home_quality - away_quality,
        'pitcher_combined_advantage': (
            era_advantage * 0.25 + fip_advantage * 0.25 + k_rate_advantage * 0.20 + 
            control_advantage * 0.15 + (home_quality - away_quality) * 0.15
        )
    }

# -------------------------
# IMPROVED Create Targets with Better Options
# -------------------------
def create_improved_targets(features_df):
    print("🎯 Creating improved target variables...")
    
    print(f"📊 Input data: {len(features_df)} total games")
    
    # Check initial data
    initial_completed = features_df.dropna(subset=['home_score', 'away_score'])
    print(f"📊 Games with non-null scores: {len(initial_completed)}")
    
    if initial_completed.empty:
        print("⚠️ No games with scores found!")
        return features_df
    
    completed = initial_completed.copy()
    
    # ENHANCED: Check data types and values BEFORE conversion
    print(f"\n🔍 PRE-CONVERSION ANALYSIS:")
    print(f"   home_score dtype: {completed['home_score'].dtype}")
    print(f"   away_score dtype: {completed['away_score'].dtype}")
    print(f"   Sample home_scores: {completed['home_score'].head().tolist()}")
    print(f"   Sample away_scores: {completed['away_score'].head().tolist()}")
    print(f"   Unique home_score values: {completed['home_score'].nunique()}")
    print(f"   Unique away_score values: {completed['away_score'].nunique()}")
    
    # Check for non-numeric values
    def check_non_numeric(series, name):
        non_numeric = []
        for val in series.dropna().unique():
            try:
                float(val)
            except (ValueError, TypeError):
                non_numeric.append(val)
        if non_numeric:
            print(f"   🚨 Non-numeric {name} values: {non_numeric[:10]}")  # Show first 10
        return len(non_numeric)
    
    home_non_numeric = check_non_numeric(completed['home_score'], 'home_score')
    away_non_numeric = check_non_numeric(completed['away_score'], 'away_score')
    
    # Convert to numeric with detailed tracking
    print(f"\n🔧 CONVERTING TO NUMERIC:")
    original_home_count = completed['home_score'].notna().sum()
    original_away_count = completed['away_score'].notna().sum()
    
    completed['home_score'] = pd.to_numeric(completed['home_score'], errors='coerce')
    completed['away_score'] = pd.to_numeric(completed['away_score'], errors='coerce')
    
    converted_home_count = completed['home_score'].notna().sum()
    converted_away_count = completed['away_score'].notna().sum()
    
    print(f"   Home scores: {original_home_count} → {converted_home_count} (lost {original_home_count - converted_home_count})")
    print(f"   Away scores: {original_away_count} → {converted_away_count} (lost {original_away_count - converted_away_count})")
    
    # Remove rows where conversion failed
    before_final_filter = len(completed)
    completed = completed.dropna(subset=['home_score', 'away_score'])
    after_final_filter = len(completed)
    
    print(f"   Final games after dropna: {before_final_filter} → {after_final_filter} (lost {before_final_filter - after_final_filter})")
    
    if completed.empty:
        print("⚠️ No valid completed games found after cleaning")
        return completed
    
    # ENHANCED: Check date range and distribution
    print(f"\n📅 DATE RANGE ANALYSIS:")
    print(f"   Date range: {completed['game_date'].min().strftime('%Y-%m-%d')} to {completed['game_date'].max().strftime('%Y-%m-%d')}")
    
    # Create targets FIRST
    completed['total_runs'] = completed['home_score'] + completed['away_score']
    completed['home_wins'] = (completed['home_score'] > completed['away_score']).astype(int)
    completed['over_8_5'] = (completed['total_runs'] > 8.5).astype(int)
    
    # NEW: Improved total runs targets (more predictable)
    completed['total_runs_binned'] = pd.cut(
        completed['total_runs'], 
        bins=[0, 6.5, 8.5, 10.5, 20], 
        labels=['low', 'medium', 'high', 'very_high']
    )
    
    # Convert to numeric for modeling
    label_encoder = LabelEncoder()
    completed['total_runs_category'] = label_encoder.fit_transform(completed['total_runs_binned'])
    
    # Alternative totals targets
    completed['over_7_5'] = (completed['total_runs'] > 7.5).astype(int)
    completed['over_9_5'] = (completed['total_runs'] > 9.5).astype(int)
    completed['under_7_5'] = (completed['total_runs'] <= 7.5).astype(int)
    
    # NEW: First 5 innings approximation (typically ~60% of total runs)
    completed['f5_total'] = completed['total_runs'] * 0.6
    completed['f5_over_4_5'] = (completed['f5_total'] > 4.5).astype(int)
    completed['f5_over_5_5'] = (completed['f5_total'] > 5.5).astype(int)
    
    # Game margin categories
    completed['margin'] = abs(completed['home_score'] - completed['away_score'])
    completed['close_game'] = (completed['margin'] <= 3).astype(int)
    completed['blowout'] = (completed['margin'] >= 6).astype(int)
    
    # ENHANCED: Target distribution analysis
    print(f"\n🎯 TARGET DISTRIBUTIONS:")
    print(f"   Total games: {len(completed)}")
    print(f"   Average total runs: {completed['total_runs'].mean():.1f}")
    print(f"   Home team win rate: {completed['home_wins'].mean():.1%}")
    print(f"   Over 8.5 rate: {completed['over_8_5'].mean():.1%}")
    print(f"   Over 7.5 rate: {completed['over_7_5'].mean():.1%}")
    print(f"   F5 Over 4.5 rate: {completed['f5_over_4_5'].mean():.1%}")
    print(f"   Close game rate: {completed['close_game'].mean():.1%}")
    
    # NOW check recent games AFTER targets are created
    recent_cutoff = pd.to_datetime('2025-07-15')
    recent_games = completed[completed['game_date'] >= recent_cutoff]
    print(f"   Games since July 15: {len(recent_games)}")
    
    if len(recent_games) > 0:
        print(f"   Recent score ranges:")
        print(f"     Home: {recent_games['home_score'].min():.0f}-{recent_games['home_score'].max():.0f}")
        print(f"     Away: {recent_games['away_score'].min():.0f}-{recent_games['away_score'].max():.0f}")
        print(f"     Total: {(recent_games['home_score'] + recent_games['away_score']).min():.0f}-{(recent_games['home_score'] + recent_games['away_score']).max():.0f}")
        
        # CRITICAL: Check recent targets - NOW THIS WILL WORK
        print(f"\n🎯 RECENT TARGET DISTRIBUTIONS (July 15+):")
        print(f"   Recent games: {len(recent_games)}")
        print(f"   Recent home win rate: {recent_games['home_wins'].mean():.1%}")
        print(f"   Recent over 8.5 rate: {recent_games['over_8_5'].mean():.1%}")
        print(f"   Recent F5 over 4.5 rate: {recent_games['f5_over_4_5'].mean():.1%}")
        
        # Show actual recent games
        print(f"\n📋 SAMPLE RECENT GAMES:")
        sample_cols = ['game_date', 'home_team', 'away_team', 'home_score', 'away_score', 'total_runs', 'home_wins']
        if len(recent_games) <= 10:
            print(recent_games[sample_cols].to_string(index=False))
        else:
            print(recent_games[sample_cols].tail(10).to_string(index=False))
    
    return completed

# -------------------------
# NEW: Feature Selection Function
# -------------------------
def select_best_features(X, y, target_type='classification', max_features=25):
    """Automated feature selection to reduce noise"""
    
    print(f"🔍 Selecting best features for {target_type}...")
    print(f"   Starting with {X.shape[1]} features")
    
    # Step 1: Remove highly correlated features
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_features = [
        column for column in upper_triangle.columns 
        if any(upper_triangle[column] > 0.85)
    ]
    
    print(f"   Removing {len(high_corr_features)} highly correlated features")
    X_reduced = X.drop(columns=high_corr_features)
    
    # Step 2: Select K best features
    if target_type == 'classification':
        selector = SelectKBest(score_func=f_classif, k=min(max_features, X_reduced.shape[1]))
    else:
        selector = SelectKBest(score_func=f_regression, k=min(max_features, X_reduced.shape[1]))
    
    X_selected = selector.fit_transform(X_reduced, y)
    selected_features = X_reduced.columns[selector.get_support()].tolist()
    
    print(f"   Selected {len(selected_features)} best features")
    
    # Return additional info for consistent application
    return X_selected, selected_features, selector, high_corr_features, X_reduced.columns.tolist()

# -------------------------
# NEW: Walk-Forward Validation
# -------------------------
def walk_forward_validation(df, models_config, feature_cols, target_col, n_splits=4):
    """More realistic validation for time series data"""
    
    print(f"📈 Running walk-forward validation with {n_splits} splits...")
    
    df_sorted = df.sort_values('game_date')
    n = len(df_sorted)
    
    fold_size = n // (n_splits + 1)
    results = []
    
    for i in range(n_splits):
        print(f"   Fold {i+1}/{n_splits}...")
        
        train_end = (i + 1) * fold_size
        test_start = train_end
        test_end = min(train_end + fold_size, n)
        
        train_idx = df_sorted.index[:train_end]
        test_idx = df_sorted.index[test_start:test_end]
        
        if len(test_idx) == 0:
            continue
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[target_col]
        
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]
        
        # Train model for this fold
        if target_col in ['total_runs', 'f5_total']:
            if XGBOOST_AVAILABLE:
                model = XGBRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    random_state=42, n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
                )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'fold': i+1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'mse': mse,
                'r2': r2,
                'mean_actual': y_test.mean()
            })
            
        else:  # Classification
            if XGBOOST_AVAILABLE:
                model = XGBClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    random_state=42, n_jobs=-1
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
                )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            logloss = log_loss(y_test, y_proba)
            
            results.append({
                'fold': i+1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'accuracy': acc,
                'logloss': logloss,
                'baseline': y_test.mean()
            })
    
    # Aggregate results
    if results:
        avg_results = {}
        for key in results[0].keys():
            if key in ['fold', 'train_size', 'test_size']:
                continue
            avg_results[f'avg_{key}'] = np.mean([r[key] for r in results])
            avg_results[f'std_{key}'] = np.std([r[key] for r in results])
        
        print(f"   Walk-forward validation completed:")
        for key, value in avg_results.items():
            print(f"     {key}: {value:.4f}")
    
    return results

# -------------------------
# ENHANCED Feature Creation (Updated)
# -------------------------
def create_enhanced_signal_features(schedule_df, team_stats_df):
    print("🔧 Creating ENHANCED signal features with ALL improvements...")
    
    pitcher_context = get_enhanced_pitcher_context()
    game_features = []
    
    for game in schedule_df.itertuples():
        home_team = game.home_name
        away_team = game.away_name
        game_date = game.game_date
        
        home_stats = team_stats_df[team_stats_df['team_name'] == home_team]
        away_stats = team_stats_df[team_stats_df['team_name'] == away_team]
        
        if home_stats.empty or away_stats.empty:
            continue
            
        home_stats = home_stats.iloc[0]
        away_stats = away_stats.iloc[0]
        
        # Original features
        home_recent_form = calculate_weighted_recent_form(schedule_df, home_team, game_date)
        away_recent_form = calculate_weighted_recent_form(schedule_df, away_team, game_date)
        home_field_adv = calculate_enhanced_home_advantage(home_stats, away_stats, schedule_df, home_team, away_team)
        
        # Enhanced features
        enhanced_momentum = calculate_enhanced_form_momentum(schedule_df, home_team, away_team, game_date)
        series_features = get_series_context_features(schedule_df, home_team, away_team, game_date)
        stadium_features = get_stadium_features(home_team, away_team, game_date)
        temporal_features = get_temporal_features(game_date)
        
        # NEW: Bullpen fatigue
        home_bullpen = calculate_bullpen_fatigue(schedule_df, home_team, game_date)
        away_bullpen = calculate_bullpen_fatigue(schedule_df, away_team, game_date)
        
        # Rest days
        home_rest_days = calculate_rest_days(schedule_df, home_team, game_date)
        away_rest_days = calculate_rest_days(schedule_df, away_team, game_date)
        
        # Enhanced pitcher features
        home_pitcher = getattr(game, 'home_probable_pitcher', None)
        away_pitcher = getattr(game, 'away_probable_pitcher', None)
        pitcher_features = calculate_enhanced_pitcher_features(
            home_pitcher, away_pitcher, pitcher_context, schedule_df, home_team, away_team, game_date
        )
        
        features = {
            'game_id': game.game_id,
            'game_date': game_date,
            'home_team': home_team,
            'away_team': away_team,
            
            # Core team stats
            'home_win_pct': home_stats['win_pct'],
            'away_win_pct': away_stats['win_pct'],
            'home_pythagorean': home_stats['pythagorean_exp'],
            'away_pythagorean': away_stats['pythagorean_exp'],
            'home_ops': home_stats['team_ops'],
            'away_ops': away_stats['team_ops'],
            'home_era': home_stats['team_era'],
            'away_era': away_stats['team_era'],
            'home_fip': home_stats['team_fip'],
            'away_fip': away_stats['team_fip'],
            
            # NEW: Advanced team stats
            'home_k_rate': home_stats['team_k_rate'],
            'away_k_rate': away_stats['team_k_rate'],
            'home_bb_rate': home_stats['team_bb_rate'],
            'away_bb_rate': away_stats['team_bb_rate'],
            'home_iso': home_stats['team_iso'],
            'away_iso': away_stats['team_iso'],
            
            # Form features
            'home_recent_form': home_recent_form,
            'away_recent_form': away_recent_form,
            'recent_form_diff': home_recent_form - away_recent_form,
            'enhanced_form_momentum': enhanced_momentum,
            
            # Core differences
            'win_pct_diff': home_stats['win_pct'] - away_stats['win_pct'],
            'pythagorean_diff': home_stats['pythagorean_exp'] - away_stats['pythagorean_exp'],
            'run_diff_diff': home_stats['run_diff'] - away_stats['run_diff'],
            'era_diff': away_stats['team_era'] - home_stats['team_era'],
            'ops_diff': home_stats['team_ops'] - away_stats['team_ops'],
            'fip_diff': away_stats['team_fip'] - home_stats['team_fip'],
            
            # NEW: Advanced differences
            'k_rate_diff': home_stats['team_k_rate'] - away_stats['team_k_rate'],
            'bb_rate_diff': away_stats['team_bb_rate'] - home_stats['team_bb_rate'],  # Lower is better
            'iso_diff': home_stats['team_iso'] - away_stats['team_iso'],
            
            # Situational factors
            'home_field_centered': home_field_adv - 0.5,
            'rest_advantage': np.tanh((home_rest_days - away_rest_days) / 2.0),
            
            # NEW: Bullpen factors
            'bullpen_fatigue_diff': away_bullpen['bullpen_fatigue'] - home_bullpen['bullpen_fatigue'],
            'home_bullpen_fatigue': home_bullpen['bullpen_fatigue'],
            'away_bullpen_fatigue': away_bullpen['bullpen_fatigue'],
            'games_played_diff': away_bullpen['games_last_3_days'] - home_bullpen['games_last_3_days'],
            
            # Interaction features
            'quality_home_advantage': (home_stats['pythagorean_exp'] - away_stats['pythagorean_exp']) * (home_field_adv - 0.5),
            'offensive_edge': (home_stats['team_ops'] - away_stats['team_ops']) * (away_stats['team_era'] - home_stats['team_era']),
            
            # Combined metrics
            'home_off_def_combo': home_stats['team_ops'] - home_stats['team_era'] + 4.5,
            'away_off_def_combo': away_stats['team_ops'] - away_stats['team_era'] + 4.5,
            
            # Store scores
            'home_score': getattr(game, 'home_score', None),
            'away_score': getattr(game, 'away_score', None)
        }
        
        # Add all feature groups
        features.update(series_features)
        features.update(stadium_features) 
        features.update(pitcher_features)
        features.update(temporal_features)
        
        # Convert scores to numeric
        for score_col in ['home_score', 'away_score']:
            if features[score_col] is not None:
                try:
                    features[score_col] = float(features[score_col])
                except (ValueError, TypeError):
                    features[score_col] = None
        
        game_features.append(features)
    
    features_df = pd.DataFrame(game_features)
    print(f"✅ Created ENHANCED features for {len(features_df)} games")
    print("🎯 Added: bullpen fatigue, advanced team stats, enhanced pitcher analysis")
    
    return features_df

# -------------------------
# NEW: Enhanced Model Training with XGBoost and Ensemble
# -------------------------
def train_enhanced_models_with_validation(df):
    """Enhanced models with XGBoost, feature selection, and EXPANDING WINDOW validation"""
    print("🤖 Training ENHANCED models v5.0 (XGBoost + EXPANDING WINDOW)...")
    
    # Get all possible features
    feature_cols = [col for col in df.columns if col not in [
        'game_id', 'game_date', 'home_team', 'away_team', 'home_score', 'away_score',
        'total_runs', 'home_wins', 'over_8_5', 'over_7_5', 'over_9_5', 'under_7_5',
        'f5_total', 'f5_over_4_5', 'f5_over_5_5', 'total_runs_binned', 'total_runs_category',
        'margin', 'close_game', 'blowout'
    ]]
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    
    # Create chronological split with EXPANDING WINDOW
    df_sorted = df.sort_values('game_date')
    n = len(df_sorted)
    
    # NEW: Use expanding window - train on MORE recent data, smaller holdout
    # Instead of fixed 70%, use 92% for training (expanding window)
    # This way model learns from almost all available data including recent patterns
    min_training_games = max(200, int(0.85 * n))  # At least 85% or 200 games
    
    # Keep slightly larger holdout to avoid single-class issues (last 8% of games)
    holdout_size = min(max(50, int(0.08 * n)), 100)  # 8% of data, 50-100 games
    
    train_end = n - holdout_size
    holdout_idx = df_sorted.index[train_end:]
    train_idx = df_sorted.index[:train_end]
    
    print(f"📊 EXPANDING WINDOW - Train={len(train_idx)} ({len(train_idx)/n:.1%}), Holdout={len(holdout_idx)} ({len(holdout_idx)/n:.1%})")
    print(f"📊 Training includes games up to: {df_sorted.iloc[train_end-1]['game_date'].strftime('%Y-%m-%d')}")
    print(f"📊 Holdout includes games from: {df_sorted.iloc[train_end]['game_date'].strftime('%Y-%m-%d')} onward")
    print(f"📊 Holdout size increased to {len(holdout_idx)} games to ensure class diversity")
    print(f"📊 Starting with {len(feature_cols)} features")
    
    models = {}
    results = {}
    selected_features = {}
    feature_selectors = {}  # Store selectors for prediction
    feature_transforms = {}  # Store transformation info
    
    # Define targets to train
    targets_config = {
        'total_runs': {'type': 'regression', 'target': 'total_runs'},
        'f5_over_4_5': {'type': 'classification', 'target': 'f5_over_4_5'},  # NEW: F5 target
        'home_wins': {'type': 'classification', 'target': 'home_wins'},
        'over_8_5': {'type': 'classification', 'target': 'over_8_5'},
        'over_7_5': {'type': 'classification', 'target': 'over_7_5'},  # NEW: Alternative total
    }
    
    for model_name, config in targets_config.items():
        print(f"\n📊 Training {model_name} model with expanding window...")
        
        target_col = config['target']
        if target_col not in df.columns:
            print(f"   ⚠️ Target {target_col} not found, skipping...")
            continue
        
        y = df[target_col]
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_holdout, y_holdout = X.loc[holdout_idx], y.loc[holdout_idx]
        
        # Feature selection - FIXED to handle both train and holdout properly
        X_selected, best_features, selector, removed_corr_features, remaining_after_corr = select_best_features(
            X_train, y_train, 
            target_type=config['type'], 
            max_features=30
        )
        
        selected_features[model_name] = best_features
        feature_selectors[model_name] = selector  # Store for predictions
        feature_transforms[model_name] = {
            'removed_corr_features': removed_corr_features,
            'remaining_after_corr': remaining_after_corr
        }
        
        # Apply same transformations to holdout set
        # Step 1: Remove the same correlated features
        X_holdout_reduced = X_holdout.drop(columns=removed_corr_features, errors='ignore')
        
        # Step 2: Keep only the features that remained after correlation removal in training
        holdout_cols_to_keep = [col for col in remaining_after_corr if col in X_holdout_reduced.columns]
        X_holdout_reduced = X_holdout_reduced[holdout_cols_to_keep]
        
        # Step 3: Apply selector transform
        X_holdout_selected = selector.transform(X_holdout_reduced)
        
        # Train model with MORE CONSERVATIVE parameters to avoid overfitting to larger training set
        if config['type'] == 'regression':
            # Try a different approach for total runs - use classification instead
            if model_name == 'total_runs':
                # Convert to classification problem: Low (≤7), Medium (8-10), High (11+)
                # First, handle any potential NaN values
                y_train_clean = y_train.fillna(y_train.median())
                y_holdout_clean = y_holdout.fillna(y_holdout.median())
                
                # Create bins and convert to categorical
                y_train_class = pd.cut(y_train_clean, bins=[0, 7.5, 10.5, 25], labels=[0, 1, 2])
                y_holdout_class = pd.cut(y_holdout_clean, bins=[0, 7.5, 10.5, 25], labels=[0, 1, 2])
                
                # Handle any remaining NaN values by assigning to medium category
                y_train_class = y_train_class.fillna(1).astype(int)
                y_holdout_class = y_holdout_class.fillna(1).astype(int)
                
                if XGBOOST_AVAILABLE:
                    # MORE CONSERVATIVE parameters with larger training set
                    model = XGBClassifier(
                        n_estimators=300,  # Reduced from 400
                        max_depth=6,       # Reduced from 8  
                        learning_rate=0.08, # Reduced from 0.1
                        subsample=0.7,     # Reduced from 0.8
                        colsample_bytree=0.7, # Reduced from 0.8
                        reg_alpha=0.1,     # NEW: L1 regularization
                        reg_lambda=0.1,    # NEW: L2 regularization
                        random_state=42,
                        n_jobs=-1,
                        eval_metric='mlogloss'
                    )
                else:
                    model = RandomForestClassifier(
                        n_estimators=200,    # Reduced from 300
                        max_depth=10,        # Reduced from 12
                        min_samples_split=8, # Increased from 5
                        min_samples_leaf=3,  # NEW: minimum samples per leaf
                        max_features=0.7,    # Reduced from 0.8
                        random_state=42,
                        n_jobs=-1
                    )
                
                model.fit(X_selected, y_train_class)
                holdout_pred_class = model.predict(X_holdout_selected)
                holdout_proba = model.predict_proba(X_holdout_selected)
                
                # Convert back to runs prediction (expected value)
                expected_runs = (holdout_proba[:, 0] * 6.5 + 
                               holdout_proba[:, 1] * 9.0 + 
                               holdout_proba[:, 2] * 12.0)
                
                # Use original y_holdout for comparison (not cleaned version)
                holdout_mse = mean_squared_error(y_holdout.fillna(y_holdout.median()), expected_runs)
                holdout_r2 = r2_score(y_holdout.fillna(y_holdout.median()), expected_runs)
                
                results[model_name] = {
                    'holdout_mse': holdout_mse,
                    'holdout_r2': holdout_r2,
                    'mean_actual': y_holdout.mean(),
                    'features_used': len(best_features),
                    'method': 'classification_converted',
                    'training_window': 'expanding'
                }
            else:
                # Regular regression for other targets
                if XGBOOST_AVAILABLE:
                    model = XGBRegressor(
                        n_estimators=300,    # Reduced from 400
                        max_depth=6,         # Reduced from 8
                        learning_rate=0.08,  # Reduced from 0.1
                        subsample=0.7,       # Reduced from 0.8
                        colsample_bytree=0.7, # Reduced from 0.8
                        reg_alpha=0.1,       # NEW: L1 regularization
                        reg_lambda=0.1,      # NEW: L2 regularization
                        random_state=42,
                        n_jobs=-1
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=200,     # Reduced from 300
                        max_depth=10,         # Reduced from 12
                        min_samples_split=8,  # Increased from 5
                        min_samples_leaf=3,   # NEW: minimum samples per leaf
                        max_features=0.7,     # Reduced from 0.8
                        random_state=42,
                        n_jobs=-1
                    )
                
                model.fit(X_selected, y_train)
                holdout_pred = model.predict(X_holdout_selected)
                
                holdout_mse = mean_squared_error(y_holdout, holdout_pred)
                holdout_r2 = r2_score(y_holdout, holdout_pred)
                
                results[model_name] = {
                    'holdout_mse': holdout_mse,
                    'holdout_r2': holdout_r2,
                    'mean_actual': y_holdout.mean(),
                    'features_used': len(best_features),
                    'training_window': 'expanding'
                }
            
        else:  # Classification
            if XGBOOST_AVAILABLE:
                base_model = XGBClassifier(
                    n_estimators=300,     # Reduced from 400
                    max_depth=6,          # Reduced from 8
                    learning_rate=0.08,   # Reduced from 0.1
                    subsample=0.7,        # Reduced from 0.8
                    colsample_bytree=0.7, # Reduced from 0.8
                    reg_alpha=0.1,        # NEW: L1 regularization
                    reg_lambda=0.1,       # NEW: L2 regularization
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss'
                )
            else:
                base_model = RandomForestClassifier(
                    n_estimators=200,     # Reduced from 300
                    max_depth=10,         # Reduced from 12
                    min_samples_split=8,  # Increased from 5
                    min_samples_leaf=3,   # NEW: minimum samples per leaf
                    max_features=0.7,     # Reduced from 0.8
                    random_state=42,
                    n_jobs=-1
                )
            
            # Use calibration to reduce overconfidence
            model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
            model.fit(X_selected, y_train)
            
            holdout_pred = model.predict(X_holdout_selected)
            holdout_proba = model.predict_proba(X_holdout_selected)[:, 1]
            
            holdout_acc = accuracy_score(y_holdout, holdout_pred)
            
            # FIXED: Handle case where holdout has only one class
            unique_labels = len(set(y_holdout))
            if unique_labels > 1:
                holdout_logloss = log_loss(y_holdout, holdout_proba)
            else:
                # If only one class in holdout, use a penalty score
                holdout_logloss = 1.0  # High penalty for poor generalization
                print(f"   ⚠️ Warning: Holdout set for {model_name} contains only one class")
            
            # Check probability spread
            prob_range = holdout_proba.max() - holdout_proba.min()
            prob_std = holdout_proba.std()
            high_conf_games = sum((holdout_proba > 0.65) | (holdout_proba < 0.35))
            very_high_conf = sum((holdout_proba > 0.70) | (holdout_proba < 0.30))
            
            print(f"   Accuracy: {holdout_acc:.3f}")
            print(f"   Unique labels in holdout: {unique_labels}")
            print(f"   Probability range: {prob_range:.3f}")
            print(f"   High confidence games: {high_conf_games}/{len(holdout_proba)}")
            
            results[model_name] = {
                'holdout_accuracy': holdout_acc,
                'holdout_logloss': holdout_logloss,
                'baseline': y_holdout.mean(),
                'prob_range': prob_range,
                'prob_std': prob_std,
                'high_conf_games': high_conf_games,
                'very_high_conf': very_high_conf,
                'features_used': len(best_features),
                'calibrated': True,
                'training_window': 'expanding',
                'holdout_classes': unique_labels  # Track this for debugging
            }
        
        models[model_name] = model
        
        # Run walk-forward validation for key models (optional)
        if model_name in ['home_wins', 'over_8_5']:
            print(f"   Running additional walk-forward validation for {model_name}...")
            walk_forward_validation(df, config, best_features, target_col)
    
    print(f"\n✅ EXPANDING WINDOW training completed!")
    print(f"📈 Model now trained on {len(train_idx)} games including recent patterns")
    
    return models, results, selected_features, feature_selectors, feature_transforms

# -------------------------
# Show Enhanced Feature Importance
# -------------------------
def show_feature_importance(models, selected_features):
    print("\n📈 Feature Importance Analysis:")
    
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            features = selected_features[model_name]
            
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print(f"\n{model_name.upper()} - Top 15 Features:")
            print(feature_importance.head(15).round(4).to_string(index=False))

# -------------------------
# Enhanced Save Models
# -------------------------
def save_enhanced_models(models, selected_features, feature_selectors, feature_transforms):
    print("\n💾 Saving enhanced models...")
    
    for model_name, model in models.items():
        model_path = os.path.join(models_dir, f"{model_name}_enhanced_v5_model.joblib")
        joblib.dump(model, model_path)
        print(f"Saved {model_name} model to {model_path}")
    
    # Save feature selections, selectors, and transforms
    features_path = os.path.join(models_dir, "selected_features_v5.joblib")
    selectors_path = os.path.join(models_dir, "feature_selectors_v5.joblib")
    transforms_path = os.path.join(models_dir, "feature_transforms_v5.joblib")
    
    joblib.dump(selected_features, features_path)
    joblib.dump(feature_selectors, selectors_path)
    joblib.dump(feature_transforms, transforms_path)
    print("✅ All enhanced models, features, selectors, and transforms saved!")

# -------------------------
# Enhanced Predictions with Fixed Feature Selection
# -------------------------
def predict_and_save_enhanced_predictions(models, selected_features, feature_selectors, feature_transforms, team_stats_df, schedule_df):
    print("\n🔮 Making ENHANCED predictions v5.0 for today's games...")

    try:
        # Get today's schedule
        today_schedule = []
        for abbrev, team_name in team_mapping.items():
            team_id = abbr_to_id.get(abbrev)
            if not team_id:
                continue
            schedule = statsapi.schedule(start_date=TODAY, end_date=TODAY, team=team_id)
            for game in schedule:
                game['team_abbrev'] = abbrev
                game['team_name'] = team_name
            today_schedule.extend(schedule)

        if not today_schedule:
            print("📅 No games scheduled for today")
            return

        today_df = pd.DataFrame(today_schedule)
        today_df['game_date'] = pd.to_datetime(today_df['game_date'])
        
        # Handle timezone
        def convert_to_cst(series):
            if series.dt.tz is None:
                series = series.dt.tz_localize('UTC')
            else:
                series = series.dt.tz_convert('UTC')
            central = pytz.timezone('America/Chicago')
            return series.dt.tz_convert(central)

        today_df['game_datetime'] = pd.to_datetime(today_df['game_datetime'], errors='coerce')
        today_df['game_datetime'] = convert_to_cst(today_df['game_datetime'])
        today_df = today_df.drop_duplicates(subset=['game_id'])

        print(f"📅 Found {len(today_df)} unique games for {TODAY}")

        pitcher_context = get_enhanced_pitcher_context()
        prediction_data = []

        for game in today_df.itertuples():
            home_team = game.home_name
            away_team = game.away_name
            game_date = pd.to_datetime(TODAY)

            home_pitcher = getattr(game, 'home_probable_pitcher', None)
            away_pitcher = getattr(game, 'away_probable_pitcher', None)

            if not home_pitcher or not away_pitcher:
                print(f"⚠️ Skipping {away_team} @ {home_team} - missing pitcher info")
                continue

            home_stats = team_stats_df[team_stats_df['team_name'] == home_team]
            away_stats = team_stats_df[team_stats_df['team_name'] == away_team]

            if home_stats.empty or away_stats.empty:
                print(f"⚠️ Missing stats for {away_team} @ {home_team}")
                continue

            home_stats = home_stats.iloc[0]
            away_stats = away_stats.iloc[0]

            # Calculate ALL features (similar to training)
            home_recent_form = calculate_weighted_recent_form(schedule_df, home_team, game_date)
            away_recent_form = calculate_weighted_recent_form(schedule_df, away_team, game_date)
            home_field_adv = calculate_enhanced_home_advantage(
                home_stats, away_stats, schedule_df, home_team, away_team
            )
            
            enhanced_momentum = calculate_enhanced_form_momentum(schedule_df, home_team, away_team, game_date)
            series_features = get_series_context_features(schedule_df, home_team, away_team, game_date)
            stadium_features = get_stadium_features(home_team, away_team, game_date)
            temporal_features = get_temporal_features(game_date)
            
            # NEW: Bullpen features
            home_bullpen = calculate_bullpen_fatigue(schedule_df, home_team, game_date)
            away_bullpen = calculate_bullpen_fatigue(schedule_df, away_team, game_date)
            
            pitcher_features = calculate_enhanced_pitcher_features(
                home_pitcher, away_pitcher, pitcher_context, schedule_df, home_team, away_team, game_date
            )
            
            home_rest_days = calculate_rest_days(schedule_df, home_team, game_date)
            away_rest_days = calculate_rest_days(schedule_df, away_team, game_date)

            # Create complete feature set
            game_features = {
                'game_id': game.game_id,
                'game_date': game.game_datetime if pd.notnull(game.game_datetime) else game.game_date,
                'away_team': away_team,
                'home_team': home_team,
                'home_pitcher': home_pitcher,
                'away_pitcher': away_pitcher,

                # Core features
                'home_win_pct': home_stats['win_pct'],
                'away_win_pct': away_stats['win_pct'],
                'home_pythagorean': home_stats['pythagorean_exp'],
                'away_pythagorean': away_stats['pythagorean_exp'],
                'home_ops': home_stats['team_ops'],
                'away_ops': away_stats['team_ops'],
                'home_era': home_stats['team_era'],
                'away_era': away_stats['team_era'],
                'home_fip': home_stats['team_fip'],
                'away_fip': away_stats['team_fip'],
                
                # NEW: Advanced stats
                'home_k_rate': home_stats['team_k_rate'],
                'away_k_rate': away_stats['team_k_rate'],
                'home_bb_rate': home_stats['team_bb_rate'],
                'away_bb_rate': away_stats['team_bb_rate'],
                'home_iso': home_stats['team_iso'],
                'away_iso': away_stats['team_iso'],

                'home_recent_form': home_recent_form,
                'away_recent_form': away_recent_form,
                'recent_form_diff': home_recent_form - away_recent_form,
                'enhanced_form_momentum': enhanced_momentum,

                'win_pct_diff': home_stats['win_pct'] - away_stats['win_pct'],
                'pythagorean_diff': home_stats['pythagorean_exp'] - away_stats['pythagorean_exp'],
                'run_diff_diff': home_stats['run_diff'] - away_stats['run_diff'],
                'era_diff': away_stats['team_era'] - home_stats['team_era'],
                'ops_diff': home_stats['team_ops'] - away_stats['team_ops'],
                'fip_diff': away_stats['team_fip'] - home_stats['team_fip'],
                
                # NEW: Advanced differences
                'k_rate_diff': home_stats['team_k_rate'] - away_stats['team_k_rate'],
                'bb_rate_diff': away_stats['team_bb_rate'] - home_stats['team_bb_rate'],
                'iso_diff': home_stats['team_iso'] - away_stats['team_iso'],

                'home_field_centered': home_field_adv - 0.5,
                'rest_advantage': np.tanh((home_rest_days - away_rest_days) / 2.0),
                
                # NEW: Bullpen features
                'bullpen_fatigue_diff': away_bullpen['bullpen_fatigue'] - home_bullpen['bullpen_fatigue'],
                'home_bullpen_fatigue': home_bullpen['bullpen_fatigue'],
                'away_bullpen_fatigue': away_bullpen['bullpen_fatigue'],
                'games_played_diff': away_bullpen['games_last_3_days'] - home_bullpen['games_last_3_days'],

                'quality_home_advantage': (home_stats['pythagorean_exp'] - away_stats['pythagorean_exp']) * (home_field_adv - 0.5),
                'offensive_edge': (home_stats['team_ops'] - away_stats['team_ops']) * (away_stats['team_era'] - home_stats['team_era']),

                'home_off_def_combo': home_stats['team_ops'] - home_stats['team_era'] + 4.5,
                'away_off_def_combo': away_stats['team_ops'] - away_stats['team_era'] + 4.5,
            }
            
            # Add all feature groups
            game_features.update(series_features)
            game_features.update(stadium_features)
            game_features.update(pitcher_features)
            game_features.update(temporal_features)

            # Make predictions with each model using proper feature selection
            predictions = {}
            
            for model_name, model in models.items():
                if model_name in feature_selectors and model_name in feature_transforms:
                    try:
                        # Get transformation info for this model
                        transforms = feature_transforms[model_name]
                        removed_corr_features = transforms['removed_corr_features']
                        remaining_after_corr = transforms['remaining_after_corr']
                        selector = feature_selectors[model_name]
                        
                        # Create feature vector with all possible features
                        all_feature_names = [col for col in game_features.keys() if col not in [
                            'game_id', 'game_date', 'home_team', 'away_team', 'home_pitcher', 'away_pitcher'
                        ]]
                        
                        full_features = {f: game_features.get(f, 0) for f in all_feature_names}
                        X_full = pd.DataFrame([full_features])
                        
                        # Apply same transformations as training:
                        # Step 1: Remove correlated features
                        X_reduced = X_full.drop(columns=removed_corr_features, errors='ignore')
                        
                        # Step 2: Keep only features that remained after correlation removal in training
                        prediction_cols_to_keep = [col for col in remaining_after_corr if col in X_reduced.columns]
                        X_reduced = X_reduced[prediction_cols_to_keep]
                        
                        # Step 3: Apply selector transform
                        X_selected = selector.transform(X_reduced)
                        
                        # Make prediction
                        if model_name in ['total_runs']:
                            # Handle total_runs as classification converted to expected value
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(X_selected)
                                if proba.shape[1] == 3:  # 3 classes: Low, Medium, High
                                    expected_runs = (proba[0][0] * 6.5 + 
                                                   proba[0][1] * 9.0 + 
                                                   proba[0][2] * 12.0)
                                    predictions[model_name] = expected_runs
                                else:
                                    # Fallback if something went wrong
                                    predictions[model_name] = 8.5
                            else:
                                # If for some reason it's not a classifier, use direct prediction
                                predictions[model_name] = model.predict(X_selected)[0]
                        else:
                            predictions[f'{model_name}_prob'] = model.predict_proba(X_selected)[0][1]
                            
                    except Exception as e:
                        print(f"⚠️ Error predicting {model_name}: {e}")
                        # Use defaults for failed predictions
                        if model_name in ['total_runs']:
                            predictions[model_name] = 8.5
                        else:
                            predictions[f'{model_name}_prob'] = 0.5

            # Create enhanced output
            game_features.update({
                'predicted_total_runs': round(predictions.get('total_runs', 8.5), 1),
                'home_win_probability': round(predictions.get('home_wins_prob', 0.5), 3),
                'away_win_probability': round(1 - predictions.get('home_wins_prob', 0.5), 3),
                'over_8_5_probability': round(predictions.get('over_8_5_prob', 0.5), 3),
                'over_7_5_probability': round(predictions.get('over_7_5_prob', 0.5), 3),
                'f5_over_4_5_probability': round(predictions.get('f5_over_4_5_prob', 0.5), 3),

                # Enhanced betting signals with more conservative thresholds
                'strong_home_favorite': 'YES' if predictions.get('home_wins_prob', 0.5) > 0.58 else 'NO',  # Reduced from 0.62
                'strong_away_favorite': 'YES' if predictions.get('home_wins_prob', 0.5) < 0.42 else 'NO',  # Reduced from 0.38
                'lean_over_8_5': 'YES' if predictions.get('over_8_5_prob', 0.5) > 0.55 else 'NO',  # Reduced from 0.58
                'lean_over_7_5': 'YES' if predictions.get('over_7_5_prob', 0.5) > 0.55 else 'NO',  # Reduced from 0.58
                'f5_over_signal': 'YES' if predictions.get('f5_over_4_5_prob', 0.5) > 0.55 else 'NO',  # Reduced from 0.58

                # Context
                'home_field_advantage': round(home_field_adv, 3),
                'bullpen_advantage': 'HOME' if home_bullpen['bullpen_fatigue'] < away_bullpen['bullpen_fatigue'] else 'AWAY',
                'stadium_factor': f"{stadium_features['stadium_run_factor']:.2f}x",
                'model_version': 'v5.0_XGBoost',

                'confidence_tier': 'HIGH' if max(predictions.get('home_wins_prob', 0.5), 1-predictions.get('home_wins_prob', 0.5)) > 0.58 else 
                                 'MEDIUM' if max(predictions.get('home_wins_prob', 0.5), 1-predictions.get('home_wins_prob', 0.5)) > 0.52 else 'LOW'
            })

            prediction_data.append(game_features)

        if not prediction_data:
            print("⚠️ No valid games found for prediction")
            return

        predictions_df = pd.DataFrame(prediction_data)
        predictions_df = predictions_df.sort_values('home_win_probability', ascending=False)

        csv_filename = os.path.join(output_dir, f"mlb_predictions_v5_{TODAY.replace('-', '_')}.csv")
        predictions_df.to_csv(csv_filename, index=False)

        print(f"\n✅ ENHANCED v5.0 predictions saved to: {csv_filename}")
        print(f"📊 Saved {len(predictions_df)} games with XGBoost + feature selection")
        
        # Enhanced betting recommendations
        print(f"\n🎯 BETTING RECOMMENDATIONS (v5.0 - XGBoost Enhanced):")
        high_conf = predictions_df[predictions_df['confidence_tier'] == 'HIGH']
        strong_bets = predictions_df[
            (predictions_df['strong_home_favorite'] == 'YES') | 
            (predictions_df['strong_away_favorite'] == 'YES')
        ]
        
        print(f"   High confidence games: {len(high_conf)}")
        print(f"   Strong betting signals: {len(strong_bets)}")
        
        if len(strong_bets) > 0:
            print(f"   RECOMMENDED BETS:")
            for _, game in strong_bets.iterrows():
                prob = max(game['home_win_probability'], game['away_win_probability'])
                favorite = game['home_team'] if game['home_win_probability'] > 0.5 else game['away_team']
                factors = []
                if game['f5_over_signal'] == 'YES':
                    factors.append('F5_OVER')
                if game['bullpen_advantage'] in ['HOME', 'AWAY']:
                    factors.append(f"BULLPEN_{game['bullpen_advantage']}")
                if abs(game['enhanced_form_momentum']) > 0.1:
                    factors.append('MOMENTUM')
                factor_str = f" ({','.join(factors)})" if factors else ""
                print(f"     {favorite} ({prob:.1%}){factor_str}")
        
        # Show new metrics
        f5_signals = len(predictions_df[predictions_df['f5_over_signal'] == 'YES'])
        print(f"   F5 Over 4.5 signals: {f5_signals}")
        print(f"   Over 7.5 alternative: {len(predictions_df[predictions_df['lean_over_7_5'] == 'YES'])}")
        
        return predictions_df

    except Exception as e:
        print(f"⚠️ Could not make today's predictions: {e}")
        return None

def debug_schedule_data(schedule_df):
    """Debug schedule data quality issues"""
    print("\n🔍 SCHEDULE DATA QUALITY CHECK:")
    print("="*50)
    
    # Basic info
    print(f"📊 Total games in schedule: {len(schedule_df)}")
    print(f"📅 Date range: {schedule_df['game_date'].min()} to {schedule_df['game_date'].max()}")
    
    # Check score availability
    has_home_score = schedule_df['home_score'].notna().sum()
    has_away_score = schedule_df['away_score'].notna().sum()
    has_both_scores = schedule_df[['home_score', 'away_score']].notna().all(axis=1).sum()
    
    print(f"📊 Score availability:")
    print(f"   Games with home_score: {has_home_score}")
    print(f"   Games with away_score: {has_away_score}")
    print(f"   Games with both scores: {has_both_scores}")
    
    # Check recent games specifically
    recent_schedule = schedule_df[schedule_df['game_date'] >= pd.to_datetime('2025-07-15')].copy()
    print(f"\n📅 Recent games (July 15+): {len(recent_schedule)}")
    
    if len(recent_schedule) > 0:
        recent_completed = recent_schedule[['home_score', 'away_score']].notna().all(axis=1).sum()
        print(f"   Recent completed games: {recent_completed}")
        
        # Check score types in recent games
        recent_with_scores = recent_schedule.dropna(subset=['home_score', 'away_score'])
        if len(recent_with_scores) > 0:
            print(f"   Recent score samples:")
            print(f"     Home scores: {recent_with_scores['home_score'].head().tolist()}")
            print(f"     Away scores: {recent_with_scores['away_score'].head().tolist()}")
            print(f"     Score types: home={recent_with_scores['home_score'].dtype}, away={recent_with_scores['away_score'].dtype}")
        else:
            print("   ⚠️ NO recent games with scores!")
            
            # Show what recent games look like
            print(f"   Sample recent games without scores:")
            sample_cols = ['game_date', 'home_name', 'away_name', 'status', 'home_score', 'away_score']
            available_cols = [col for col in sample_cols if col in recent_schedule.columns]
            print(recent_schedule[available_cols].head().to_string(index=False))
    
    # Check for games that might be completed but missing scores
    print(f"\n🔍 GAME STATUS ANALYSIS:")
    if 'status' in schedule_df.columns:
        status_counts = schedule_df['status'].value_counts()
        print(f"   Game statuses: {dict(status_counts)}")
        
        # Look for completed games without scores
        completed_status_games = schedule_df[
            (schedule_df['status'].str.contains('Final|Completed', case=False, na=False)) |
            (schedule_df['status'] == 'F')
        ]
        completed_without_scores = completed_status_games[
            completed_status_games[['home_score', 'away_score']].isna().any(axis=1)
        ]
        
        print(f"   Games marked complete: {len(completed_status_games)}")
        print(f"   Complete games missing scores: {len(completed_without_scores)}")
        
        if len(completed_without_scores) > 0:
            print(f"   🚨 ISSUE: {len(completed_without_scores)} completed games missing scores!")
    
    print("="*50)
    
# Add this function to debug the data issue
def debug_data_processing(schedule_df, features_df, completed_df):
    """Debug why holdout games all have same outcome"""
    
    print("\n🔍 DEBUGGING DATA PROCESSING ISSUE:")
    print("="*60)
    
    # 1. Check raw schedule data
    print("📊 RAW SCHEDULE DATA:")
    recent_schedule = schedule_df[schedule_df['game_date'] >= pd.to_datetime('2025-07-15')].copy()
    print(f"   Games from July 15+: {len(recent_schedule)}")
    print(f"   Completed games from July 15+: {len(recent_schedule.dropna(subset=['home_score', 'away_score']))}")
    
    # Check score types
    print(f"\n📋 SCORE DATA TYPES:")
    print(f"   home_score type: {recent_schedule['home_score'].dtype}")
    print(f"   away_score type: {recent_schedule['away_score'].dtype}")
    print(f"   Sample home_scores: {recent_schedule['home_score'].dropna().head().tolist()}")
    print(f"   Sample away_scores: {recent_schedule['away_score'].dropna().head().tolist()}")
    
    # 2. Check features data
    print(f"\n📊 FEATURES DATA:")
    recent_features = features_df[features_df['game_date'] >= pd.to_datetime('2025-07-15')].copy()
    print(f"   Feature games from July 15+: {len(recent_features)}")
    print(f"   Completed feature games: {len(recent_features.dropna(subset=['home_score', 'away_score']))}")
    
    # 3. Check completed data (after target creation)
    print(f"\n📊 COMPLETED DATA (after target creation):")
    recent_completed = completed_df[completed_df['game_date'] >= pd.to_datetime('2025-07-15')].copy()
    print(f"   Completed games from July 15+: {len(recent_completed)}")
    
    if len(recent_completed) > 0:
        print(f"\n🎯 TARGET DISTRIBUTIONS (July 15+):")
        print(f"   Home wins: {recent_completed['home_wins'].value_counts().to_dict()}")
        print(f"   Over 8.5: {recent_completed['over_8_5'].value_counts().to_dict()}")
        if 'f5_over_4_5' in recent_completed.columns:
            print(f"   F5 Over 4.5: {recent_completed['f5_over_4_5'].value_counts().to_dict()}")
        
        print(f"\n📋 RECENT GAMES SAMPLE:")
        sample_cols = ['game_date', 'home_team', 'away_team', 'home_score', 'away_score', 'home_wins', 'over_8_5']
        sample_cols = [col for col in sample_cols if col in recent_completed.columns]
        print(recent_completed[sample_cols].head(10).to_string(index=False))
        
        print(f"\n📊 SCORE STATISTICS (July 15+):")
        print(f"   Home score range: {recent_completed['home_score'].min():.1f} to {recent_completed['home_score'].max():.1f}")
        print(f"   Away score range: {recent_completed['away_score'].min():.1f} to {recent_completed['away_score'].max():.1f}")
        print(f"   Total runs range: {(recent_completed['home_score'] + recent_completed['away_score']).min():.1f} to {(recent_completed['home_score'] + recent_completed['away_score']).max():.1f}")
        
        # Check for suspicious patterns
        total_runs = recent_completed['home_score'] + recent_completed['away_score']
        print(f"\n🚨 SUSPICIOUS PATTERNS CHECK:")
        print(f"   All home scores same? {recent_completed['home_score'].nunique() == 1}")
        print(f"   All away scores same? {recent_completed['away_score'].nunique() == 1}")
        print(f"   All total runs same? {total_runs.nunique() == 1}")
        print(f"   All home wins same? {recent_completed['home_wins'].nunique() == 1}")
        
        # Show unique combinations
        print(f"\n📈 UNIQUE SCORE COMBINATIONS:")
        score_combos = recent_completed[['home_score', 'away_score']].drop_duplicates()
        print(score_combos.head(10).to_string(index=False))
        
    else:
        print("   ⚠️ NO completed games found in recent period!")
    
    # 4. Check data conversion issues
    print(f"\n🔧 DATA CONVERSION CHECK:")
    test_conversion = recent_schedule[['home_score', 'away_score']].copy()
    
    # Before conversion
    print(f"   Before pd.to_numeric:")
    print(f"     Non-null home_scores: {test_conversion['home_score'].notna().sum()}")
    print(f"     Non-null away_scores: {test_conversion['away_score'].notna().sum()}")
    
    # After conversion
    test_conversion['home_score_numeric'] = pd.to_numeric(test_conversion['home_score'], errors='coerce')
    test_conversion['away_score_numeric'] = pd.to_numeric(test_conversion['away_score'], errors='coerce')
    
    print(f"   After pd.to_numeric:")
    print(f"     Non-null home_scores: {test_conversion['home_score_numeric'].notna().sum()}")
    print(f"     Non-null away_scores: {test_conversion['away_score_numeric'].notna().sum()}")
    
    # Show conversion failures
    conversion_failures = test_conversion[
        (test_conversion['home_score'].notna()) & 
        (test_conversion['home_score_numeric'].isna())
    ]
    
    if len(conversion_failures) > 0:
        print(f"   🚨 CONVERSION FAILURES ({len(conversion_failures)} games):")
        print(f"     Sample failed home_scores: {conversion_failures['home_score'].head().tolist()}")
    
    print("="*60)
    return recent_completed

# -------------------------
# Main Function
# -------------------------
def main():
    print("🚀 Starting ENHANCED MLB Betting Model v5.0...")
    print("🎯 Major upgrades:")
    print("   • XGBoost models (better than Random Forest)")
    print("   • Automated feature selection (reduces noise)")
    print("   • Walk-forward validation (realistic performance)")
    print("   • Bullpen fatigue tracking")
    print("   • F5 innings totals (more predictable)")
    print("   • Enhanced pitcher analysis with FIP, K%, BB%")
    print("   • Alternative totals (7.5, 9.5)")
    print("   • Advanced team stats (ISO, K%, BB%)")
    
    # Fetch data
    schedule_df = fetch_enhanced_schedule_data()
    debug_schedule_data(schedule_df)
    team_stats_df = get_enhanced_team_season_stats()
    
    # Create enhanced features
    features_df = create_enhanced_signal_features(schedule_df, team_stats_df)
    completed_df = create_improved_targets(features_df)
    debug_data = debug_data_processing(schedule_df, features_df, completed_df)
    
    if len(completed_df) < 50:
        print(f"⚠️ Only {len(completed_df)} completed games found. Need more data for reliable training.")
        features_df.to_csv(os.path.join(output_dir, f"enhanced_features_v5_{CURRENT_YEAR}.csv"), index=False)
        return
    
    # Train enhanced models
    models, results, selected_features, feature_selectors, feature_transforms = train_enhanced_models_with_validation(completed_df)
    
    # Show results
    print(f"\n🎯 ENHANCED Model v5.0 Performance:")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
    
    show_feature_importance(models, selected_features)
    save_enhanced_models(models, selected_features, feature_selectors, feature_transforms)
    
    # Save training data
    completed_df.to_csv(os.path.join(output_dir, f"enhanced_training_data_v5_{CURRENT_YEAR}.csv"), index=False)
    features_df.to_csv(os.path.join(output_dir, f"enhanced_features_v5_{CURRENT_YEAR}.csv"), index=False)
    
    # Make today's predictions
    predictions_df = predict_and_save_enhanced_predictions(models, selected_features, feature_selectors, feature_transforms, team_stats_df, schedule_df)
    

if __name__ == "__main__":
    main()