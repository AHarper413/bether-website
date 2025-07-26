import os
import sys
import json
import pandas as pd
from datetime import datetime, date
import subprocess
import glob
import requests
import time

def run_mlb_model():
    """Run the main MLB model and return the predictions"""
    print("🚀 Running MLB betting model...")
    
    try:
        # Run your main model file (now in same directory)
        result = subprocess.run([sys.executable, 'mlbtargets.py'], 
                              capture_output=True, text=True, timeout=1800)
        
        if result.returncode != 0:
            print(f"❌ Model failed with error: {result.stderr}")
            return None
            
        print("✅ Model completed successfully")
        return True
        
    except subprocess.TimeoutExpired:
        print("⏰ Model timed out after 30 minutes")
        return None
    except Exception as e:
        print(f"❌ Error running model: {e}")
        return None

def find_latest_predictions_file():
    """Find the most recent predictions CSV file"""
    today_str = date.today().strftime("%Y_%m_%d")
    current_year = datetime.now().year
    
    # Look for today's file first in the output directory
    expected_filename = f"./mlb_improved_data_{current_year}/mlb_predictions_v5_{today_str}.csv"
    
    if os.path.exists(expected_filename):
        return expected_filename
    
    # Look for any recent predictions file
    pattern = f"./mlb_improved_data_{current_year}/mlb_predictions_v5_*.csv"
    files = glob.glob(pattern)
    
    if files:
        # Return the most recent file
        return max(files, key=os.path.getmtime)
    
    # If no files found, return None
    print("❌ No predictions file found")
    return None

# === NEW: Odds API functions ===
def fetch_draftkings_odds():
    """Fetch current MLB odds from DraftKings via The Odds API"""
    api_key = "704d21dd7f686383fffd15d45a6d05c8"  # Your API key from Streamlit app
    
    url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    params = {
        'api_key': api_key,
        'regions': 'us',
        'markets': 'h2h,totals',  # head-to-head (moneyline) and totals
        'bookmakers': 'draftkings',
        'oddsFormat': 'american',
        'dateFormat': 'iso'
    }
    
    try:
        print("📡 Fetching DraftKings odds...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Fetched odds for {len(data)} games")
        
        # Check remaining requests
        remaining_requests = response.headers.get('x-requests-remaining')
        if remaining_requests:
            print(f"📊 API requests remaining: {remaining_requests}")
        
        return data
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching odds: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def normalize_team_name(team_name):
    """Normalize team names for better matching between API and predictions"""
    # Common team name mappings
    name_mappings = {
        'Arizona Diamondbacks': 'Arizona Diamondbacks',
        'Atlanta Braves': 'Atlanta Braves', 
        'Baltimore Orioles': 'Baltimore Orioles',
        'Boston Red Sox': 'Boston Red Sox',
        'Chicago Cubs': 'Chicago Cubs',
        'Chicago White Sox': 'Chicago White Sox',
        'Cincinnati Reds': 'Cincinnati Reds',
        'Cleveland Guardians': 'Cleveland Guardians',
        'Colorado Rockies': 'Colorado Rockies',
        'Detroit Tigers': 'Detroit Tigers',
        'Houston Astros': 'Houston Astros',
        'Kansas City Royals': 'Kansas City Royals',
        'Los Angeles Angels': 'Los Angeles Angels',
        'Los Angeles Dodgers': 'Los Angeles Dodgers',
        'Miami Marlins': 'Miami Marlins',
        'Milwaukee Brewers': 'Milwaukee Brewers',
        'Minnesota Twins': 'Minnesota Twins',
        'New York Mets': 'New York Mets',
        'New York Yankees': 'New York Yankees',
        'Oakland Athletics': 'Oakland Athletics',
        'Philadelphia Phillies': 'Philadelphia Phillies',
        'Pittsburgh Pirates': 'Pittsburgh Pirates',
        'San Diego Padres': 'San Diego Padres',
        'San Francisco Giants': 'San Francisco Giants',
        'Seattle Mariners': 'Seattle Mariners',
        'St. Louis Cardinals': 'St. Louis Cardinals',
        'Tampa Bay Rays': 'Tampa Bay Rays',
        'Texas Rangers': 'Texas Rangers',
        'Toronto Blue Jays': 'Toronto Blue Jays',
        'Washington Nationals': 'Washington Nationals'
    }
    
    team_name = team_name.strip()
    
    # Direct mapping first
    if team_name in name_mappings:
        return name_mappings[team_name]
    
    # Try partial matching for common variations
    for full_name in name_mappings.values():
        if team_name in full_name or full_name.split()[-1] in team_name:
            return full_name
    
    return team_name

def american_to_decimal(odds):
    """Convert American odds to decimal odds"""
    if odds > 0:
        return 1 + odds / 100
    else:
        return 1 + 100 / abs(odds)

def calculate_expected_value(model_prob, american_odds):
    """Calculate expected value percentage"""
    decimal_odds = american_to_decimal(american_odds)
    ev = (decimal_odds * model_prob) - 1
    return ev * 100

def match_odds_with_predictions(odds_data, predictions_df):
    """Match DraftKings odds with model predictions and calculate EV"""
    if not odds_data:
        print("⚠️ No odds data to match")
        return predictions_df
    
    print(f"🔄 Matching {len(odds_data)} odds with {len(predictions_df)} predictions...")
    
    # Add new columns for odds and EV
    predictions_df['home_ml_odds'] = None
    predictions_df['away_ml_odds'] = None
    predictions_df['over_8_5_odds'] = None
    predictions_df['under_8_5_odds'] = None
    predictions_df['total_line'] = None
    predictions_df['home_ml_ev'] = None
    predictions_df['away_ml_ev'] = None
    predictions_df['over_8_5_ev'] = None
    predictions_df['under_8_5_ev'] = None
    predictions_df['best_ev_bet'] = None
    predictions_df['best_ev_value'] = None
    
    matched_count = 0
    
    for game in odds_data:
        home_team = normalize_team_name(game['home_team'])
        away_team = normalize_team_name(game['away_team'])
        
        # Find DraftKings odds
        dk_odds = None
        for bookmaker in game['bookmakers']:
            if bookmaker['key'] == 'draftkings':
                dk_odds = bookmaker
                break
        
        if not dk_odds:
            continue
        
        # Extract moneyline and totals odds
        h2h_odds = None
        totals_odds = None
        
        for market in dk_odds['markets']:
            if market['key'] == 'h2h':
                h2h_odds = market
            elif market['key'] == 'totals':
                totals_odds = market
        
        # Find matching prediction
        pred_idx = None
        for idx, row in predictions_df.iterrows():
            pred_home_norm = normalize_team_name(row['home_team'])
            pred_away_norm = normalize_team_name(row['away_team'])
            
            if pred_home_norm == home_team and pred_away_norm == away_team:
                pred_idx = idx
                break
        
        if pred_idx is None:
            print(f"⚠️ No prediction match for: {away_team} @ {home_team}")
            continue
        
        # Store odds
        if h2h_odds:
            for outcome in h2h_odds['outcomes']:
                if outcome['name'] == game['home_team']:
                    predictions_df.loc[pred_idx, 'home_ml_odds'] = outcome['price']
                elif outcome['name'] == game['away_team']:
                    predictions_df.loc[pred_idx, 'away_ml_odds'] = outcome['price']
        
        if totals_odds:
            for outcome in totals_odds['outcomes']:
                if outcome['name'] == 'Over':
                    predictions_df.loc[pred_idx, 'over_8_5_odds'] = outcome['price']
                    predictions_df.loc[pred_idx, 'total_line'] = outcome['point']
                elif outcome['name'] == 'Under':
                    predictions_df.loc[pred_idx, 'under_8_5_odds'] = outcome['price']
        
        # Calculate EV if we have both odds and probabilities
        row = predictions_df.loc[pred_idx]
        ev_bets = []
        
        if pd.notna(row['home_ml_odds']):
            home_ev = calculate_expected_value(row['home_win_probability'], row['home_ml_odds'])
            predictions_df.loc[pred_idx, 'home_ml_ev'] = home_ev
            ev_bets.append(('Home ML', home_ev))
        
        if pd.notna(row['away_ml_odds']):
            away_ev = calculate_expected_value(row['away_win_probability'], row['away_ml_odds'])
            predictions_df.loc[pred_idx, 'away_ml_ev'] = away_ev
            ev_bets.append(('Away ML', away_ev))
        
        if pd.notna(row['over_8_5_odds']) and row['total_line'] == 8.5:
            over_ev = calculate_expected_value(row['over_8_5_probability'], row['over_8_5_odds'])
            predictions_df.loc[pred_idx, 'over_8_5_ev'] = over_ev
            ev_bets.append(('Over 8.5', over_ev))
        
        if pd.notna(row['under_8_5_odds']) and row['total_line'] == 8.5:
            under_prob = 1 - row['over_8_5_probability']  # Calculate under probability
            under_ev = calculate_expected_value(under_prob, row['under_8_5_odds'])
            predictions_df.loc[pred_idx, 'under_8_5_ev'] = under_ev
            ev_bets.append(('Under 8.5', under_ev))
        
        # Find best EV bet for this game
        if ev_bets:
            best_bet = max(ev_bets, key=lambda x: x[1])
            if best_bet[1] > 2:  # Only show if EV > 2%
                predictions_df.loc[pred_idx, 'best_ev_bet'] = best_bet[0]
                predictions_df.loc[pred_idx, 'best_ev_value'] = best_bet[1]
        
        matched_count += 1
    
    print(f"✅ Successfully matched {matched_count} games with odds")
    return predictions_df

def create_picks_json(predictions_df):
    """Convert predictions to JSON format for website - ENHANCED with EV data"""
    today = date.today().strftime("%B %d, %Y")
    
    # Count signals
    strong_home_count = len(predictions_df[predictions_df['strong_home_favorite'] == 'YES'])
    strong_away_count = len(predictions_df[predictions_df['strong_away_favorite'] == 'YES']) 
    total_signals = strong_home_count + strong_away_count
    
    over_8_5_count = len(predictions_df[predictions_df['lean_over_8_5'] == 'YES'])
    f5_signals_count = len(predictions_df[predictions_df['f5_over_signal'] == 'YES'])
    
    # Count EV opportunities
    ev_opportunities = len(predictions_df[predictions_df['best_ev_value'] > 2]) if 'best_ev_value' in predictions_df.columns else 0
    
    # Get best bets (highest confidence)
    best_ml_bets = []
    for _, game in predictions_df.iterrows():
        home_prob = game['home_win_probability']
        away_prob = game['away_win_probability']
        
        if home_prob > 0.58:  # Strong home favorite
            confidence = "HIGH" if home_prob > 0.62 else "MEDIUM"
            
            # Add EV info if available
            ev_info = ""
            if pd.notna(game.get('home_ml_ev')):
                ev_info = f" (EV: {game['home_ml_ev']:.1f}%)"
            
            best_ml_bets.append({
                "game": f"{game['away_team']} @ {game['home_team']}",
                "pick": f"{game['home_team']} ML{ev_info}",
                "probability": f"{home_prob:.1%}",
                "confidence": confidence,
                "reasoning": f"Model favors home team with {home_prob:.1%} win probability",
                "odds": game.get('home_ml_odds'),
                "ev": game.get('home_ml_ev')
            })
        elif away_prob > 0.58:  # Strong away favorite  
            confidence = "HIGH" if away_prob > 0.62 else "MEDIUM"
            
            # Add EV info if available
            ev_info = ""
            if pd.notna(game.get('away_ml_ev')):
                ev_info = f" (EV: {game['away_ml_ev']:.1f}%)"
            
            best_ml_bets.append({
                "game": f"{game['away_team']} @ {game['home_team']}",
                "pick": f"{game['away_team']} ML{ev_info}",
                "probability": f"{away_prob:.1%}",
                "confidence": confidence,
                "reasoning": f"Model favors away team with {away_prob:.1%} win probability",
                "odds": game.get('away_ml_odds'),
                "ev": game.get('away_ml_ev')
            })
    
    # Get total bets with EV
    total_bets = []
    for _, game in predictions_df.iterrows():
        if game['lean_over_8_5'] == 'YES':
            prob = game['over_8_5_probability']
            confidence = "HIGH" if prob > 0.60 else "MEDIUM"
            
            # Add EV info if available
            ev_info = ""
            if pd.notna(game.get('over_8_5_ev')):
                ev_info = f" (EV: {game['over_8_5_ev']:.1f}%)"
            
            total_bets.append({
                "game": f"{game['away_team']} @ {game['home_team']}",
                "pick": f"Over 8.5 Runs{ev_info}",
                "probability": f"{prob:.1%}",
                "confidence": confidence,
                "predicted_total": game['predicted_total_runs'],
                "odds": game.get('over_8_5_odds'),
                "ev": game.get('over_8_5_ev')
            })
    
    # NEW: Get best EV bets regardless of signals
    ev_bets = []
    if 'best_ev_bet' in predictions_df.columns:
        for _, game in predictions_df.iterrows():
            if pd.notna(game['best_ev_bet']) and game['best_ev_value'] > 2:
                ev_bets.append({
                    "game": f"{game['away_team']} @ {game['home_team']}",
                    "bet": game['best_ev_bet'],
                    "ev": f"{game['best_ev_value']:.1f}%",
                    "confidence": "POSITIVE EV" if game['best_ev_value'] > 5 else "SLIGHT EDGE"
                })
    
    # Create JSON structure
    picks_data = {
        "date": today,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S EST"),
        "model_version": "v5.0_XGBoost",
        "summary": {
            "total_games": len(predictions_df),
            "moneyline_signals": total_signals,
            "over_8_5_signals": over_8_5_count,
            "f5_signals": f5_signals_count,
            "ev_opportunities": ev_opportunities
        },
        "best_moneyline_bets": best_ml_bets[:5],  # Top 5
        "best_total_bets": total_bets[:3],  # Top 3
        "best_ev_bets": sorted(ev_bets, key=lambda x: float(x['ev'].rstrip('%')), reverse=True)[:5],  # Top 5 EV
        "season_record": {"wins": 156, "losses": 100, "percentage": 61.0}  # Update this manually
    }
    
    return picks_data

def create_picks_html(picks_data, predictions_df):
    """Create enhanced HTML page with EV betting section"""
    
    # Convert predictions_df to JSON for JavaScript (including new EV columns)
    import json
    games_data = []
    for _, game in predictions_df.iterrows():
        game_dict = {
            'game_id': str(game['game_id']),
            'away_team': game['away_team'],
            'home_team': game['home_team'],
            'home_pitcher': game.get('home_pitcher', 'TBD'),
            'away_pitcher': game.get('away_pitcher', 'TBD'),
            'home_win_probability': float(game['home_win_probability']),
            'away_win_probability': float(game['away_win_probability']),
            'predicted_total_runs': float(game['predicted_total_runs']),
            'over_8_5_probability': float(game['over_8_5_probability']),
            'confidence_tier': game['confidence_tier'],
            
            # Team Stats
            'home_era': float(game.get('home_era', 4.50)),
            'away_era': float(game.get('away_era', 4.50)),
            'home_ops': float(game.get('home_ops', 0.720)),
            'away_ops': float(game.get('away_ops', 0.720)),
            'home_win_pct': float(game.get('home_win_pct', 0.500)),
            'away_win_pct': float(game.get('away_win_pct', 0.500)),
            'home_pythagorean': float(game.get('home_pythagorean', 0.500)),
            'away_pythagorean': float(game.get('away_pythagorean', 0.500)),
            'home_recent_form': float(game.get('home_recent_form', 0.500)),
            'away_recent_form': float(game.get('away_recent_form', 0.500)),
            
            # Advanced Stats
            'home_fip': float(game.get('home_fip', 4.00)),
            'away_fip': float(game.get('away_fip', 4.00)),
            'home_k_rate': float(game.get('home_k_rate', 0.22)),
            'away_k_rate': float(game.get('away_k_rate', 0.22)),
            'pitcher_era_advantage': float(game.get('pitcher_era_advantage', 0.0)),
            'pitcher_k_rate_advantage': float(game.get('pitcher_k_rate_advantage', 0.0)),
            
            # Situational
            'stadium_factor': game.get('stadium_factor', '1.00x'),
            'bullpen_advantage': game.get('bullpen_advantage', 'EVEN'),
            'home_field_advantage': float(game.get('home_field_advantage', 0.535)),
            
            # Signals
            'strong_home_favorite': game.get('strong_home_favorite', 'NO') == 'YES',
            'strong_away_favorite': game.get('strong_away_favorite', 'NO') == 'YES',
            'lean_over_8_5': game.get('lean_over_8_5', 'NO') == 'YES',
            'f5_over_signal': game.get('f5_over_signal', 'NO') == 'YES',
            
            # NEW: EV data
            'home_ml_odds': game.get('home_ml_odds'),
            'away_ml_odds': game.get('away_ml_odds'),
            'over_8_5_odds': game.get('over_8_5_odds'),
            'under_8_5_odds': game.get('under_8_5_odds'),
            'home_ml_ev': game.get('home_ml_ev'),
            'away_ml_ev': game.get('away_ml_ev'),
            'over_8_5_ev': game.get('over_8_5_ev'),
            'under_8_5_ev': game.get('under_8_5_ev'),
            'best_ev_bet': game.get('best_ev_bet'),
            'best_ev_value': game.get('best_ev_value')
        }
        games_data.append(game_dict)
    
    games_json = json.dumps(games_data, indent=2)
    
    # Count EV opportunities for display
    ev_opportunities = len([g for g in games_data if g.get('best_ev_value') and g['best_ev_value'] > 2])
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Free MLB Picks Today - {picks_data['date']} | Expected Value Betting | BetHer</title>
    <meta name="description" content="Free MLB picks today with Expected Value analysis from our advanced XGBoost model. Expert baseball predictions with confidence ratings and EV calculations.">
    <meta name="keywords" content="free mlb picks today, expected value betting, mlb EV bets, baseball predictions, expert mlb picks, positive EV betting">
    
    <!-- Favicon -->
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHZpZXdCb3g9IjAgMCAzMiAzMiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZGVmcz4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0iZmF2aWNvbkdyYWRpZW50IiB4MT0iMCUiIHkxPSIwJSIgeDI9IjEwMCUiIHkyPSIxMDAlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3R5bGU9InN0b3AtY29sb3I6I2VkNjQ5NiIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0eWxlPSJzdG9wLWNvbG9yOiNmODcxNzEiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgPC9kZWZzPgogIDxjaXJjbGUgY3g9IjE2IiBjeT0iMTYiIHI9IjE0IiBmaWxsPSJ3aGl0ZSIgb3BhY2l0eT0iMC4yIi8+CiAgPGNpcmNsZSBjeD0iMTYiIGN5PSIxNiIgcj0iMTEiIGZpbGw9InVybCgjZmF2aWNvbkdyYWRpZW50KSIvPgogIDxwYXRoIGQ9Ik0xMS41IDE0IEwxMy41IDE2IEwyMC41IDkiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+CiAgPGNpcmNsZSBjeD0iMTYiIGN5PSIyMSIgcj0iMi41IiBmaWxsPSJ3aGl0ZSIvPgogIDxjaXJjbGUgY3g9IjEyIiBjeT0iMjEiIHI9IjEuNSIgZmlsbD0id2hpdGUiIG9wYWNpdHk9IjAuOCIvPgogIDxjaXJjbGUgY3g9IjIwIiBjeT0iMjEiIHI9IjEuNSIgZmlsbD0id2hpdGUiIG9wYWNpdHk9IjAuOCIvPgo8L3N2Zz4K">
    
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            color: #1e293b;
            background: #f8fafc;
            margin: 0;
            padding: 0;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            min-height: 100vh;
        }}
        
        /* Header Branding */
        .brand-header {{
            background: #ffffff;
            padding: 2rem 2.5rem;
            border-bottom: 1px solid #e2e8f0;
        }}
        .brand-content {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }}
        .brand-section {{
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }}
        .logo {{
            width: 48px;
            height: 48px;
            cursor: pointer;
            transition: transform 0.2s ease;
        }}
        .logo:hover {{
            transform: scale(1.05);
        }}
        .brand-text h1 {{
            font-family: 'Playfair Display', serif;
            font-size: 2.5rem;
            font-weight: 600;
            color: #e11d48;
            margin-bottom: 0.25rem;
            letter-spacing: -0.02em;
        }}
        .brand-text p {{
            font-size: 0.95rem;
            color: #64748b;
            font-weight: 400;
            margin: 0;
        }}
        .nav-links {{
            display: flex;
            gap: 1.5rem;
            align-items: center;
        }}
        .nav-link {{
            color: #64748b;
            text-decoration: none;
            font-weight: 500;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            transition: all 0.2s ease;
        }}
        .nav-link:hover {{
            color: #e11d48;
            background: #fdf2f8;
        }}
        .nav-link.current {{
            color: #e11d48;
            background: #fdf2f8;
            font-weight: 600;
        }}
        
        /* Main Content */
        .main-content {{
            padding: 2rem 2.5rem;
        }}
        
        .page-header {{
            text-align: center;
            margin-bottom: 2rem;
            border-bottom: 2px solid #e11d48;
            padding-bottom: 1rem;
        }}
        .page-header h1 {{
            color: #e11d48;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        
        /* User Guide Section */
        .user-guide {{
            background: linear-gradient(135deg, #fdf2f8 0%, #f8fafc 100%);
            border: 2px solid #e11d48;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
        }}
        .user-guide h2 {{
            color: #e11d48;
            font-size: 1.5rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .guide-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }}
        .guide-item {{
            background: white;
            padding: 1.25rem;
            border-radius: 8px;
            border-left: 4px solid #e11d48;
        }}
        .guide-item h3 {{
            color: #1e293b;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }}
        .guide-item p {{
            color: #64748b;
            font-size: 0.95rem;
            margin: 0;
        }}
        .guide-highlight {{
            background: #dcfce7;
            border-left-color: #10b981;
        }}
        
        /* NEW: EV Section Styles */
        .ev-section {{
            background: linear-gradient(135deg, #0d9488 0%, #10b981 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }}
        .ev-section h2 {{
            margin: 0 0 1rem 0;
            font-size: 1.8rem;
        }}
        .ev-explanation {{
            background: rgba(255,255,255,0.1);
            padding: 1.5rem;
            border-radius: 8px;
            margin-top: 1rem;
        }}
        .ev-bets {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }}
        .ev-bet-card {{
            background: rgba(255,255,255,0.15);
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #fbbf24;
        }}
        .ev-bet-card.high-ev {{
            border-left-color: #10b981;
        }}
        .ev-value {{
            font-size: 1.2rem;
            font-weight: bold;
            color: #fbbf24;
        }}
        .ev-value.high {{
            color: #86efac;
        }}
        
        .model-info {{
            background: linear-gradient(135deg, #e11d48 0%, #8b5cf6 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 2rem;
        }}
        .model-info h2 {{
            margin: 0 0 0.5rem 0;
            font-size: 1.5rem;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: #f1f5f9;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card.ev-highlight {{
            background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
            border: 2px solid #10b981;
        }}
        .stat-number {{
            font-size: 2rem;
            font-weight: bold;
            color: #e11d48;
        }}
        .stat-number.ev {{
            color: #059669;
        }}
        
        /* Game Cards */
        .games-grid {{
            display: grid;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .game-card {{
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
        }}
        .game-card:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}
        .game-card.high-confidence {{
            border-left: 4px solid #10b981;
        }}
        .game-card.medium-confidence {{
            border-left: 4px solid #f59e0b;
        }}
        .game-card.low-confidence {{
            border-left: 4px solid #6b7280;
        }}
        .game-card.positive-ev {{
            border: 2px solid #059669;
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        }}
        
        .game-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}
        .matchup {{
            font-size: 1.2rem;
            font-weight: 600;
            color: #1e293b;
        }}
        .confidence-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .confidence-high {{ background: #d1fae5; color: #065f46; }}
        .confidence-medium {{ background: #fef3c7; color: #92400e; }}
        .confidence-low {{ background: #f3f4f6; color: #4b5563; }}
        
        /* NEW: EV display in game cards */
        .ev-display {{
            display: flex;
            gap: 0.5rem;
            justify-content: center;
            margin-top: 1rem;
            flex-wrap: wrap;
        }}
        .ev-bet {{
            background: #f1f5f9;
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            font-size: 0.85rem;
            border: 1px solid #e2e8f0;
        }}
        .ev-bet.positive {{
            background: #d1fae5;
            border-color: #10b981;
            color: #065f46;
            font-weight: 600;
        }}
        .ev-bet.negative {{
            background: #fee2e2;
            border-color: #ef4444;
            color: #991b1b;
        }}
        
        .teams-section {{
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 1rem;
            align-items: center;
            margin-bottom: 1rem;
        }}
        .team {{
            text-align: center;
            cursor: pointer;
            padding: 0.75rem;
            border-radius: 8px;
            transition: all 0.2s ease;
            border: 2px solid transparent;
        }}
        .team:hover {{
            background: #f8fafc;
            transform: scale(1.02);
            border-color: #e11d48;
        }}
        .team-name {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #e11d48;
            margin-bottom: 0.25rem;
        }}
        .team-pitcher {{
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 0.5rem;
        }}
        .win-probability {{
            font-size: 1.3rem;
            font-weight: bold;
            color: #1e293b;
        }}
        .vs-section {{
            text-align: center;
            padding: 0 1rem;
        }}
        .predicted-total {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #8b5cf6;
            margin-bottom: 0.5rem;
        }}
        
        .signals {{
            display: flex;
            gap: 0.5rem;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 1rem;
        }}
        .signal {{
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 500;
        }}
        .signal-home {{ background: #dbeafe; color: #1e40af; }}
        .signal-away {{ background: #fef3c7; color: #92400e; }}
        .signal-over {{ background: #dcfce7; color: #166534; }}
        .signal-f5 {{ background: #e0e7ff; color: #3730a3; }}
        .signal-ev {{ background: #fbbf24; color: #92400e; font-weight: 700; }}
        
        /* Modal for team stats */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
        }}
        .modal-content {{
            background-color: white;
            margin: 5% auto;
            padding: 2rem;
            border-radius: 12px;
            width: 90%;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
        }}
        .close {{
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }}
        .close:hover {{
            color: #e11d48;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        .stat-item {{
            background: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-label {{
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 0.5rem;
        }}
        .stat-value {{
            font-size: 1.2rem;
            font-weight: bold;
            color: #1e293b;
        }}
        
        .back-link {{
            text-align: center;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e2e8f0;
        }}
        .back-link a {{
            color: #e11d48;
            text-decoration: none;
            font-weight: 600;
        }}
        
        @media (max-width: 768px) {{
            .brand-header {{ padding: 1.5rem; }}
            .brand-content {{ 
                flex-direction: column; 
                text-align: center;
                gap: 1.5rem;
            }}
            .brand-text h1 {{ font-size: 2rem; }}
            .nav-links {{ 
                flex-wrap: wrap; 
                justify-content: center;
            }}
            .main-content {{ padding: 1rem; }}
            .page-header h1 {{ font-size: 2rem; }}
            .teams-section {{ grid-template-columns: 1fr; gap: 0.5rem; }}
            .vs-section {{ order: -1; }}
            .ev-bets {{ grid-template-columns: 1fr; }}
            .guide-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Brand Header -->
        <div class="brand-header">
            <div class="brand-content">
                <div class="brand-section">
                    <a href="/index.html" style="text-decoration: none;">
                        <svg class="logo" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
                            <defs>
                                <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" style="stop-color:#e11d48"/>
                                    <stop offset="100%" style="stop-color:#8b5cf6"/>
                                </linearGradient>
                            </defs>
                            <circle cx="100" cy="100" r="90" fill="white" opacity="0.1"/>
                            <circle cx="100" cy="100" r="70" fill="url(#logoGradient)"/>
                            <path d="M70 85 L85 100 L130 55" stroke="white" stroke-width="8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
                            <circle cx="100" cy="130" r="15" fill="white"/>
                            <circle cx="75" cy="130" r="8" fill="white" opacity="0.8"/>
                            <circle cx="125" cy="130" r="8" fill="white" opacity="0.8"/>
                        </svg>
                    </a>
                    <div class="brand-text">
                        <h1>BetHer</h1>
                        <p>Free MLB picks and stats</p>
                    </div>
                </div>
                <div class="nav-links">
                    <a href="/index.html" class="nav-link">Home</a>
                    <a href="/free-mlb-picks.html" class="nav-link">About Our Analysis</a>
                    <a href="/calculators.html" class="nav-link">Calculators</a>
                    <a href="#" class="nav-link current">Today's Picks</a>
                </div>
            </div>
        </div>

        <div class="main-content">
            <div class="page-header">
                <h1>Free MLB Picks - {picks_data['date']}</h1>
                <p>Advanced XGBoost predictions with Expected Value analysis</p>
                <p><strong>Last Updated:</strong> {picks_data['last_updated']}</p>
            </div>

            <!-- User Guide Section -->
            <div class="user-guide">
                <h2>📋 How to Use This Page</h2>
                <div class="guide-grid">
                    <div class="guide-item">
                        <h3>🎯 Confidence Levels</h3>
                        <p>Each game is rated HIGH, MEDIUM, or LOW confidence based on our model's certainty. Focus on higher confidence picks for better results.</p>
                    </div>
                    <div class="guide-item guide-highlight">
                        <h3>💰 Expected Value (EV)</h3>
                        <p>Green EV percentages show mathematically profitable bets. Positive EV means our model gives you an edge over the sportsbook.</p>
                    </div>
                    <div class="guide-item">
                        <h3>📊 Team Stats</h3>
                        <p><strong>Click any team name</strong> to see detailed season statistics, recent form, and advanced metrics in a popup window.</p>
                    </div>
                    <div class="guide-item">
                        <h3>🏷️ Signal Tags</h3>
                        <p>Colored tags show our strongest recommendations: HOME FAV, AWAY FAV, OVER 8.5, F5 OVER, and +EV opportunities.</p>
                    </div>
                </div>
            </div>
        
        /* NEW: EV Section Styles */
        .ev-section {{
            background: linear-gradient(135deg, #0d9488 0%, #10b981 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }}
        .ev-section h2 {{
            margin: 0 0 1rem 0;
            font-size: 1.8rem;
        }}
        .ev-explanation {{
            background: rgba(255,255,255,0.1);
            padding: 1.5rem;
            border-radius: 8px;
            margin-top: 1rem;
        }}
        .ev-bets {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }}
        .ev-bet-card {{
            background: rgba(255,255,255,0.15);
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #fbbf24;
        }}
        .ev-bet-card.high-ev {{
            border-left-color: #10b981;
        }}
        .ev-value {{
            font-size: 1.2rem;
            font-weight: bold;
            color: #fbbf24;
        }}
        .ev-value.high {{
            color: #86efac;
        }}
        
        .model-info {{
            background: linear-gradient(135deg, #e11d48 0%, #8b5cf6 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 2rem;
        }}
        .model-info h2 {{
            margin: 0 0 0.5rem 0;
            font-size: 1.5rem;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: #f1f5f9;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card.ev-highlight {{
            background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
            border: 2px solid #10b981;
        }}
        .stat-number {{
            font-size: 2rem;
            font-weight: bold;
            color: #e11d48;
        }}
        .stat-number.ev {{
            color: #059669;
        }}
        
        /* Game Cards */
        .games-grid {{
            display: grid;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .game-card {{
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
        }}
        .game-card:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}
        .game-card.high-confidence {{
            border-left: 4px solid #10b981;
        }}
        .game-card.medium-confidence {{
            border-left: 4px solid #f59e0b;
        }}
        .game-card.low-confidence {{
            border-left: 4px solid #6b7280;
        }}
        .game-card.positive-ev {{
            border: 2px solid #059669;
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        }}
        
        .game-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}
        .matchup {{
            font-size: 1.2rem;
            font-weight: 600;
            color: #1e293b;
        }}
        .confidence-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .confidence-high {{ background: #d1fae5; color: #065f46; }}
        .confidence-medium {{ background: #fef3c7; color: #92400e; }}
        .confidence-low {{ background: #f3f4f6; color: #4b5563; }}
        
        /* NEW: EV display in game cards */
        .ev-display {{
            display: flex;
            gap: 0.5rem;
            justify-content: center;
            margin-top: 1rem;
            flex-wrap: wrap;
        }}
        .ev-bet {{
            background: #f1f5f9;
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            font-size: 0.85rem;
            border: 1px solid #e2e8f0;
        }}
        .ev-bet.positive {{
            background: #d1fae5;
            border-color: #10b981;
            color: #065f46;
            font-weight: 600;
        }}
        .ev-bet.negative {{
            background: #fee2e2;
            border-color: #ef4444;
            color: #991b1b;
        }}
        
        .teams-section {{
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 1rem;
            align-items: center;
            margin-bottom: 1rem;
        }}
        .team {{
            text-align: center;
            cursor: pointer;
            padding: 0.75rem;
            border-radius: 8px;
            transition: all 0.2s ease;
        }}
        .team:hover {{
            background: #f8fafc;
            transform: scale(1.02);
        }}
        .team-name {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #e11d48;
            margin-bottom: 0.25rem;
        }}
        .team-pitcher {{
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 0.5rem;
        }}
        .win-probability {{
            font-size: 1.3rem;
            font-weight: bold;
            color: #1e293b;
        }}
        .vs-section {{
            text-align: center;
            padding: 0 1rem;
        }}
        .predicted-total {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #8b5cf6;
            margin-bottom: 0.5rem;
        }}
        
        .signals {{
            display: flex;
            gap: 0.5rem;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 1rem;
        }}
        .signal {{
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 500;
        }}
        .signal-home {{ background: #dbeafe; color: #1e40af; }}
        .signal-away {{ background: #fef3c7; color: #92400e; }}
        .signal-over {{ background: #dcfce7; color: #166534; }}
        .signal-f5 {{ background: #e0e7ff; color: #3730a3; }}
        .signal-ev {{ background: #fbbf24; color: #92400e; font-weight: 700; }}
        
        /* Modal for team stats */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
        }}
        .modal-content {{
            background-color: white;
            margin: 5% auto;
            padding: 2rem;
            border-radius: 12px;
            width: 90%;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
        }}
        .close {{
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }}
        .close:hover {{
            color: #e11d48;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        .stat-item {{
            background: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-label {{
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 0.5rem;
        }}
        .stat-value {{
            font-size: 1.2rem;
            font-weight: bold;
            color: #1e293b;
        }}
        
        .back-link {{
            text-align: center;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e2e8f0;
        }}
        .back-link a {{
            color: #e11d48;
            text-decoration: none;
            font-weight: 600;
        }}
        
        @media (max-width: 768px) {{
            .container {{ padding: 1rem; }}
            .header h1 {{ font-size: 2rem; }}
            .teams-section {{ grid-template-columns: 1fr; gap: 0.5rem; }}
            .vs-section {{ order: -1; }}
            .ev-bets {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Free MLB Picks - {picks_data['date']}</h1>
            <p>Advanced XGBoost predictions with Expected Value analysis</p>
            <p><strong>Last Updated:</strong> {picks_data['last_updated']}</p>
        </div>

        <!-- NEW: Expected Value Section -->
        <div class="ev-section">
            <h2>💰 Expected Value (EV) Betting Today</h2>
            <p>Expected Value betting is a mathematical approach that identifies bets where the odds offer better payouts than the true probability suggests. A positive EV bet means you have a long-term advantage over the sportsbook.</p>
            
            <div class="ev-explanation">
                <h3>🎯 How EV Works:</h3>
                <p><strong>Positive EV (+2% or higher):</strong> Our model believes the true probability is higher than what the odds suggest. These are mathematically profitable bets over time.</p>
                <p><strong>Negative EV:</strong> The sportsbook has the advantage on this bet. Avoid these.</p>
                <p><strong>Example:</strong> If our model gives a team a 60% chance to win, but the odds imply only 55% chance, that's a +EV opportunity.</p>
            </div>
            
            {"<div class='ev-bets'>" + "".join([
                f"""<div class='ev-bet-card {"high-ev" if bet.get("ev") and float(bet["ev"].rstrip("%")) > 5 else ""}'>
                    <div style='font-weight: 600;'>{bet['game']}</div>
                    <div style='font-size: 1.1rem; margin: 0.5rem 0;'>{bet['bet']}</div>
                    <div class='ev-value {"high" if bet.get("ev") and float(bet["ev"].rstrip("%")) > 5 else ""}'>{bet['ev']} EV</div>
                    <div style='font-size: 0.9rem; opacity: 0.9;'>{bet['confidence']}</div>
                </div>"""
                for bet in picks_data.get('best_ev_bets', [])
            ]) + "</div>" if picks_data.get('best_ev_bets') else "<p style='text-align: center; font-style: italic;'>No positive EV opportunities found with current odds. Check back as lines move throughout the day.</p>"}
        </div>

        <div class="model-info">
            <h2>🤖 XGBoost Model with Expected Value Analysis</h2>
            <p>Advanced machine learning algorithm analyzing 50+ factors plus real-time DraftKings odds to identify profitable betting opportunities</p>
        </div>

        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-number">{picks_data['summary']['total_games']}</div>
                <div>Games Today</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{picks_data['summary']['moneyline_signals']}</div>
                <div>Strong Picks</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{picks_data['summary']['over_8_5_signals']}</div>
                <div>Over 8.5 Signals</div>
            </div>
            <div class="stat-card ev-highlight">
                <div class="stat-number ev">{ev_opportunities}</div>
                <div>+EV Opportunities</div>
            </div>
        </div>"""

    # Add games grid (modified to include EV data)
    html_content += """
        <div class="games-grid">
"""

    # Add each game with EV information
    for game_data in games_data:
        confidence_class = game_data['confidence_tier'].lower() + '-confidence'
        
        # Check if this game has positive EV
        has_positive_ev = (game_data.get('best_ev_value') and game_data['best_ev_value'] > 2)
        if has_positive_ev:
            confidence_class += ' positive-ev'
        
        # Determine signals
        signals = []
        if game_data['strong_home_favorite']:
            signals.append(('HOME FAV', 'signal-home'))
        if game_data['strong_away_favorite']:
            signals.append(('AWAY FAV', 'signal-away'))
        if game_data['lean_over_8_5']:
            signals.append(('OVER 8.5', 'signal-over'))
        if game_data['f5_over_signal']:
            signals.append(('F5 OVER', 'signal-f5'))
        if has_positive_ev:
            signals.append((f"+EV: {game_data['best_ev_bet']}", 'signal-ev'))
        
        signals_html = ''.join([f'<span class="signal {signal_class}">{signal_text}</span>' 
                               for signal_text, signal_class in signals])
        
        # Create EV display for all bet types
        ev_display_html = ""
        ev_bets = []
        
        if game_data.get('home_ml_ev') is not None:
            ev_class = 'positive' if game_data['home_ml_ev'] > 2 else 'negative'
            ev_bets.append(f'<span class="ev-bet {ev_class}">Home ML: {game_data["home_ml_ev"]:.1f}%</span>')
        
        if game_data.get('away_ml_ev') is not None:
            ev_class = 'positive' if game_data['away_ml_ev'] > 2 else 'negative'
            ev_bets.append(f'<span class="ev-bet {ev_class}">Away ML: {game_data["away_ml_ev"]:.1f}%</span>')
        
        if game_data.get('over_8_5_ev') is not None:
            ev_class = 'positive' if game_data['over_8_5_ev'] > 2 else 'negative'
            ev_bets.append(f'<span class="ev-bet {ev_class}">Over 8.5: {game_data["over_8_5_ev"]:.1f}%</span>')
        
        if game_data.get('under_8_5_ev') is not None:
            ev_class = 'positive' if game_data['under_8_5_ev'] > 2 else 'negative'
            ev_bets.append(f'<span class="ev-bet {ev_class}">Under 8.5: {game_data["under_8_5_ev"]:.1f}%</span>')
        
        if ev_bets:
            ev_display_html = f'<div class="ev-display">{"".join(ev_bets)}</div>'
        
        html_content += f"""
            <div class="game-card {confidence_class}">
                <div class="game-header">
                    <div class="matchup">{game_data['away_team']} @ {game_data['home_team']}</div>
                    <span class="confidence-badge confidence-{game_data['confidence_tier'].lower()}">{game_data['confidence_tier']}</span>
                </div>
                
                <div class="teams-section">
                    <div class="team" onclick="showTeamStats('{game_data['game_id']}', 'away')">
                        <div class="team-name">{game_data['away_team']}</div>
                        <div class="team-pitcher">{game_data['away_pitcher']}</div>
                        <div class="win-probability">{game_data['away_win_probability']:.1%}</div>
                    </div>
                    
                    <div class="vs-section">
                        <div style="color: #64748b; font-weight: 600;">VS</div>
                        <div class="predicted-total">{game_data['predicted_total_runs']:.1f} runs</div>
                        <div style="font-size: 0.9rem; color: #8b5cf6;">O8.5: {game_data['over_8_5_probability']:.0%}</div>
                    </div>
                    
                    <div class="team" onclick="showTeamStats('{game_data['game_id']}', 'home')">
                        <div class="team-name">{game_data['home_team']}</div>
                        <div class="team-pitcher">{game_data['home_pitcher']}</div>
                        <div class="win-probability">{game_data['home_win_probability']:.1%}</div>
                    </div>
                </div>
                
                {ev_display_html}
                
                {f'<div class="signals">{signals_html}</div>' if signals else ''}
            </div>
"""

    # Add modal and JavaScript (same as before, but with EV data)
    html_content += f"""
        </div>

        <!-- Team Stats Modal -->
        <div id="teamModal" class="modal">
            <div class="modal-content">
                <span class="close" onclick="closeModal()">&times;</span>
                <h2 id="modalTitle"></h2>
                <div id="modalContent"></div>
            </div>
        </div>

        <div class="back-link">
            <p><strong>Disclaimer:</strong> Expected Value calculations are for educational purposes only. Past performance does not guarantee future results. Always bet responsibly and within your means.</p>
            <p><a href="/index.html">← Back to BetHer Home</a> | <a href="/free-mlb-picks.html">← About Our Analysis</a> | <a href="/calculators.html">← Betting Calculators</a></p>
        </div>
    </div>
    </div>

    <script>
        // Games data with EV information
        const gamesData = {games_json};
        
        function showTeamStats(gameId, teamType) {{
            const game = gamesData.find(g => g.game_id === gameId);
            if (!game) return;
            
            const isHome = teamType === 'home';
            const teamName = isHome ? game.home_team : game.away_team;
            const pitcher = isHome ? game.home_pitcher : game.away_pitcher;
            
            document.getElementById('modalTitle').textContent = `${{teamName}} - Current Season Stats`;
            
            // Add EV information to modal
            let evInfo = '';
            if (isHome && game.home_ml_ev !== null) {{
                const evClass = game.home_ml_ev > 2 ? 'color: #059669; font-weight: bold;' : 'color: #dc2626;';
                evInfo = `<p><strong>Moneyline EV:</strong> <span style="${{evClass}}">${{game.home_ml_ev.toFixed(1)}}%</span> ${{game.home_ml_ev > 2 ? '(Positive Edge!)' : '(Negative Edge)'}}</p>`;
            }} else if (!isHome && game.away_ml_ev !== null) {{
                const evClass = game.away_ml_ev > 2 ? 'color: #059669; font-weight: bold;' : 'color: #dc2626;';
                evInfo = `<p><strong>Moneyline EV:</strong> <span style="${{evClass}}">${{game.away_ml_ev.toFixed(1)}}%</span> ${{game.away_ml_ev > 2 ? '(Positive Edge!)' : '(Negative Edge)'}}</p>`;
            }}
            
            const modalContent = `
                <div style="text-align: center; margin-bottom: 1.5rem; padding: 1rem; background: #f8fafc; border-radius: 8px;">
                    <h3 style="color: #e11d48; margin-bottom: 0.5rem;">Starting Pitcher: ${{pitcher}}</h3>
                    ${{evInfo}}
                    <p style="font-size: 0.9rem; color: #64748b; margin: 0;">📊 Click-to-view current season statistics</p>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">Win Probability</div>
                        <div class="stat-value">${{isHome ? game.home_win_probability.toFixed(1) : game.away_win_probability.toFixed(1)}}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Season Win %</div>
                        <div class="stat-value">${{(isHome ? game.home_win_pct : game.away_win_pct).toFixed(3)}}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Team ERA</div>
                        <div class="stat-value">${{(isHome ? game.home_era : game.away_era).toFixed(2)}}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Team OPS</div>
                        <div class="stat-value">${{(isHome ? game.home_ops : game.away_ops).toFixed(3)}}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Pythagorean Win %</div>
                        <div class="stat-value">${{(isHome ? game.home_pythagorean : game.away_pythagorean).toFixed(3)}}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Recent Form (L10)</div>
                        <div class="stat-value">${{(isHome ? game.home_recent_form : game.away_recent_form).toFixed(3)}}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Team FIP</div>
                        <div class="stat-value">${{(isHome ? game.home_fip : game.away_fip).toFixed(2)}}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Strikeout Rate</div>
                        <div class="stat-value">${{(isHome ? game.home_k_rate : game.away_k_rate * 100).toFixed(1)}}%</div>
                    </div>
                </div>
                
                <div style="margin-top: 1.5rem; padding: 1rem; background: #f8fafc; border-radius: 8px;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #e11d48;">Today's Game Context</h4>
                    <p><strong>Stadium Factor:</strong> ${{game.stadium_factor}} (run environment)</p>
                    <p><strong>Bullpen Advantage:</strong> ${{game.bullpen_advantage}}</p>
                    <p><strong>Home Field Advantage:</strong> ${{(game.home_field_advantage * 100).toFixed(1)}}%</p>
                    ${{isHome ? '' : `<p><strong>Pitcher Matchup:</strong> ${{game.pitcher_era_advantage > 0 ? 'Favorable' : 'Unfavorable'}} (${{game.pitcher_era_advantage.toFixed(2)}} ERA diff)</p>`}}
                </div>
                
                <div style="text-align: center; margin-top: 1rem; padding: 0.75rem; background: #fdf2f8; border-radius: 6px;">
                    <p style="font-size: 0.85rem; color: #64748b; margin: 0;"><strong>💡 Tip:</strong> Higher Pythagorean Win % indicates better run differential. Recent form shows last 10 games performance.</p>
                </div>
            `;
            
            document.getElementById('modalContent').innerHTML = modalContent;
            document.getElementById('teamModal').style.display = 'block';
        }}
        
        function closeModal() {{
            document.getElementById('teamModal').style.display = 'none';
        }}
        
        // Close modal when clicking outside
        window.onclick = function(event) {{
            const modal = document.getElementById('teamModal');
            if (event.target === modal) {{
                modal.style.display = 'none';
            }}
        }}
        
        // Smooth scroll to sections
        document.addEventListener('DOMContentLoaded', function() {{
            // Highlight team cards on hover to show they're clickable
            const teamCards = document.querySelectorAll('.team');
            teamCards.forEach(card => {{
                card.addEventListener('mouseenter', function() {{
                    this.style.background = '#fdf2f8';
                    this.style.borderColor = '#e11d48';
                }});
                card.addEventListener('mouseleave', function() {{
                    this.style.background = '';
                    this.style.borderColor = 'transparent';
                }});
            }});
        }});
    </script>
</body>
</html>
"""
    
    return html_content

def main():
    print("🚀 Starting MLB picks automation with EV analysis...")
    
    # Run the model
    model_success = run_mlb_model()
    if not model_success:
        print("❌ Model execution failed, cannot generate picks")
        return
    
    # Find the predictions file
    predictions_file = find_latest_predictions_file()
    if not predictions_file:
        print("❌ No predictions file found")
        return
    
    print(f"📊 Found predictions file: {predictions_file}")
    
    # Load predictions
    try:
        predictions_df = pd.read_csv(predictions_file)
        print(f"✅ Loaded {len(predictions_df)} game predictions")
    except Exception as e:
        print(f"❌ Error loading predictions: {e}")
        return
    
    # NEW: Fetch odds and calculate EV
    odds_data = fetch_draftkings_odds()
    if odds_data:
        predictions_df = match_odds_with_predictions(odds_data, predictions_df)
        
        # Save enhanced predictions with EV data
        enhanced_file = predictions_file.replace('.csv', '_with_ev.csv')
        predictions_df.to_csv(enhanced_file, index=False)
        print(f"✅ Saved enhanced predictions with EV data: {enhanced_file}")
    else:
        print("⚠️ No odds data fetched, proceeding without EV calculations")
    
    # Create JSON data (now includes EV information)
    picks_data = create_picks_json(predictions_df)
    
    # Save JSON file
    with open('daily_picks.json', 'w') as f:
        json.dump(picks_data, f, indent=2)
    print("✅ Created daily_picks.json with EV data")
    
    # Create HTML page with EV section
    html_content = create_picks_html(picks_data, predictions_df)
    with open('mlb-picks-today.html', 'w') as f:
        f.write(html_content)
    print("✅ Created mlb-picks-today.html with Expected Value analysis")
    
    print(f"🎯 Summary for {picks_data['date']}:")
    print(f"   📊 {picks_data['summary']['total_games']} games analyzed")
    print(f"   🎯 {picks_data['summary']['moneyline_signals']} ML signals")
    print(f"   📈 {picks_data['summary']['over_8_5_signals']} total signals")
    print(f"   💰 {picks_data['summary']['ev_opportunities']} +EV opportunities")

if __name__ == "__main__":
    main()