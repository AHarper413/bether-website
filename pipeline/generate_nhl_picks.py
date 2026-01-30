#!/usr/bin/env python3
"""
Generate NHL Picks for SmartBetCalcs.com
========================================
Runs the NHL ML pipeline and generates the daily picks HTML page.

Usage:
    python generate_nhl_picks.py              # Full pipeline
    python generate_nhl_picks.py --quick-odds # Quick mode (API only, no scraping)
    python generate_nhl_picks.py --retrain    # Force model retraining
"""

import subprocess
import pandas as pd
import json
from datetime import datetime
import pytz
import os
import sys

# Configuration
PIPELINE_SCRIPT = "nhl_unified_pipeline.py"
PREDICTIONS_CSV = "nhl_today_predictions.csv"
OUTPUT_JSON = "../nhl_daily_picks.json"
OUTPUT_HTML = "../nhl-picks-today.html"
MODEL_FILE = "nhl_random_forest_model.pkl"
TEAM_STATS_FILE = "nhl_team_stats.csv"

def run_pipeline(retrain=False, quick_odds=False):
    """Run the NHL ML pipeline."""
    # Check if quick_odds is requested but model doesn't exist
    # In that case, fall back to full pipeline
    if quick_odds and not os.path.exists(MODEL_FILE):
        print(f"⚠️  Model not found ({MODEL_FILE}), falling back to full pipeline")
        quick_odds = False

    # Also check for team stats file
    if quick_odds and not os.path.exists(TEAM_STATS_FILE):
        print(f"⚠️  Team stats not found ({TEAM_STATS_FILE}), falling back to full pipeline")
        quick_odds = False

    cmd = ["python3", PIPELINE_SCRIPT]
    if quick_odds:
        cmd.append("--quick-odds")
    elif retrain:
        cmd.append("--retrain")

    print(f"Running pipeline: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # Increased timeout for full pipeline

    if result.returncode != 0:
        print(f"Pipeline error: {result.stderr}")
        print(f"Pipeline stdout: {result.stdout}")
        return False

    print(result.stdout)
    return True

def load_predictions():
    """Load predictions from CSV."""
    if not os.path.exists(PREDICTIONS_CSV):
        print(f"No predictions file found: {PREDICTIONS_CSV}")
        return None

    df = pd.read_csv(PREDICTIONS_CSV)
    return df

def generate_json(df):
    """Generate JSON summary of picks."""
    central = pytz.timezone('America/Chicago')
    now = datetime.now(central)

    picks = []
    for _, row in df.iterrows():
        # Get the win probability for the predicted winner
        predicted_winner = row.get("predicted_winner", "")
        home_team = row.get("home_team", "")
        away_team = row.get("away_team", "")

        # Determine win probability based on who was predicted to win
        if predicted_winner == home_team:
            win_prob = row.get("home_win_probability", row.get("confidence", 0.5))
        else:
            win_prob = row.get("away_win_probability", row.get("confidence", 0.5))

        # Ensure win_prob is a float between 0 and 1
        if isinstance(win_prob, (int, float)):
            win_prob = float(win_prob)
        else:
            win_prob = 0.5

        # Calculate confidence tier from probability
        if win_prob >= 0.65:
            confidence_tier = "High"
        elif win_prob >= 0.55:
            confidence_tier = "Medium"
        else:
            confidence_tier = "Low"

        # Determine EV rating from value_bet flag
        is_value_bet = row.get("value_bet", False)
        if is_value_bet:
            ev_rating = "Positive EV"
        else:
            ev_rating = "Neutral"

        # Get moneylines - convert decimal odds to American
        team_1_odds = row.get("team_1_odds", 2.0)
        team_2_odds = row.get("team_2_odds", 2.0)
        team_1 = row.get("team_1", "")
        team_2 = row.get("team_2", "")

        # Map team_1/team_2 odds to home/away
        if team_1 == home_team:
            home_decimal = team_1_odds
            away_decimal = team_2_odds
        else:
            home_decimal = team_2_odds
            away_decimal = team_1_odds

        # Convert decimal to American odds
        def decimal_to_american(decimal_odds):
            if decimal_odds >= 2.0:
                return int((decimal_odds - 1) * 100)
            else:
                return int(-100 / (decimal_odds - 1))

        try:
            moneyline_home = decimal_to_american(float(home_decimal))
            moneyline_away = decimal_to_american(float(away_decimal))
        except (ValueError, ZeroDivisionError):
            moneyline_home = -110
            moneyline_away = -110

        pick = {
            "away_team": away_team,
            "home_team": home_team,
            "predicted_winner": predicted_winner,
            "win_probability": round(win_prob * 100, 1),
            "confidence": confidence_tier,
            "ev_rating": ev_rating,
            "moneyline_away": moneyline_away,
            "moneyline_home": moneyline_home,
        }
        picks.append(pick)

    output = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S CST"),
        "date": now.strftime("%B %d, %Y"),
        "total_games": len(picks),
        "picks": picks
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Generated {OUTPUT_JSON} with {len(picks)} picks")
    return output

def generate_html(data):
    """Generate the HTML page for today's NHL picks."""
    central = pytz.timezone('America/Chicago')
    now = datetime.now(central)
    date_str = now.strftime("%B %d, %Y")

    # Generate pick cards HTML
    pick_cards = ""
    for pick in data["picks"]:
        confidence_class = "high" if pick["confidence"] == "High" else "medium" if pick["confidence"] == "Medium" else "low"
        ev_class = "positive" if "Positive" in pick["ev_rating"] else "neutral" if "Neutral" in pick["ev_rating"] else "negative"

        pick_cards += f'''
            <div class="pick-card">
                <div class="matchup">
                    <div class="team away">{pick["away_team"]}</div>
                    <div class="vs">@</div>
                    <div class="team home">{pick["home_team"]}</div>
                </div>
                <div class="prediction">
                    <div class="predicted-winner">
                        <span class="label">Model Pick:</span>
                        <span class="winner">{pick["predicted_winner"]}</span>
                    </div>
                    <div class="confidence {confidence_class}">
                        <span class="prob">{pick["win_probability"]}%</span>
                        <span class="tier">{pick["confidence"]} Confidence</span>
                    </div>
                </div>
                <div class="odds-display">
                    <div class="odds-item">
                        <span class="odds-label">{pick["away_team"]}</span>
                        <span class="odds-value">{pick["moneyline_away"]:+d}</span>
                    </div>
                    <div class="odds-item">
                        <span class="odds-label">{pick["home_team"]}</span>
                        <span class="odds-value">{pick["moneyline_home"]:+d}</span>
                    </div>
                </div>
                <div class="ev-badge {ev_class}">{pick["ev_rating"]}</div>
            </div>
'''

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Free NHL Picks Today - {date_str} | SmartBetCalcs</title>
    <meta name="description" content="Free NHL hockey picks for {date_str}. Algorithm-driven NHL predictions with confidence ratings and expected value analysis.">
    <meta name="google-site-verification" content="JusfWhcDMEBbB2sXK78E611obvJ_UkdAVMkrq-mS71c" />
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHZpZXdCb3g9IjAgMCAzMiAzMiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZGVmcz4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0iZmF2aWNvbkdyYWRpZW50IiB4MT0iMCUiIHkxPSIwJSIgeDI9IjEwMCUiIHkyPSIxMDAlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3R5bGU9InN0b3AtY29sb3I6I2UxMWQ0OCIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0eWxlPSJzdG9wLWNvbG9yOiNmOTczMTYiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgPC9kZWZzPgogIDxjaXJjbGUgY3g9IjE2IiBjeT0iMTYiIHI9IjE0IiBmaWxsPSIjMGYxNzJhIiBvcGFjaXR5PSIwLjkiLz4KICA8Y2lyY2xlIGN4PSIxNiIgY3k9IjE2IiByPSIxMSIgZmlsbD0idXJsKCNmYXZpY29uR3JhZGllbnQpIi8+CiAgPHBhdGggZD0iTTExLjUgMTQgTDEzLjUgMTYgTDIwLjUgOSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KICA8Y2lyY2xlIGN4PSIxNiIgY3k9IjIxIiByPSIyLjUiIGZpbGw9IndoaXRlIi8+CiAgPGNpcmNsZSBjeD0iMTIiIGN5PSIyMSIgcj0iMS41IiBmaWxsPSJ3aGl0ZSIgb3BhY2l0eT0iMC44Ii8+CiAgPGNpcmNsZSBjeD0iMjAiIGN5PSIyMSIgcj0iMS41IiBmaWxsPSJ3aGl0ZSIgb3BhY2l0eT0iMC44Ii8+Cjwvc3ZnPgo=">
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-HZ2MWQF940"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());
      gtag('config', 'G-HZ2MWQF940');
    </script>
    <link rel="stylesheet" href="/styles.css">
    <style>
        .picks-container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        .page-header {{
            text-align: center;
            margin-bottom: 2rem;
            padding: 2rem;
            background: linear-gradient(135deg, #0891b2, #06b6d4);
            border-radius: 12px;
            color: white;
        }}
        .page-header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        .page-header .date {{
            font-size: 1.25rem;
            opacity: 0.9;
        }}
        .generated-time {{
            font-size: 0.9rem;
            color: var(--text-secondary);
            text-align: center;
            margin-bottom: 2rem;
        }}
        .picks-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 1.5rem;
        }}
        .pick-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-dark);
            border-radius: 12px;
            padding: 1.5rem;
        }}
        .matchup {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }}
        .team {{
            font-weight: 600;
            color: var(--text-primary);
        }}
        .vs {{
            color: var(--text-muted);
            font-size: 0.9rem;
        }}
        .prediction {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding: 1rem;
            background: var(--bg-tertiary);
            border-radius: 8px;
        }}
        .predicted-winner .label {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            display: block;
        }}
        .predicted-winner .winner {{
            font-weight: 700;
            color: #0891b2;
            font-size: 1.1rem;
        }}
        .confidence {{
            text-align: right;
        }}
        .confidence .prob {{
            font-size: 1.5rem;
            font-weight: 700;
            display: block;
        }}
        .confidence.high .prob {{ color: #10b981; }}
        .confidence.medium .prob {{ color: #f59e0b; }}
        .confidence.low .prob {{ color: var(--text-secondary); }}
        .confidence .tier {{
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
        .odds-display {{
            display: flex;
            justify-content: space-around;
            margin-bottom: 1rem;
            padding: 0.75rem;
            background: var(--bg-primary);
            border-radius: 6px;
        }}
        .odds-item {{
            text-align: center;
        }}
        .odds-label {{
            display: block;
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
        .odds-value {{
            font-weight: 600;
            color: var(--text-primary);
        }}
        .ev-badge {{
            text-align: center;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .ev-badge.positive {{
            background: rgba(16, 185, 129, 0.2);
            color: #10b981;
        }}
        .ev-badge.neutral {{
            background: rgba(148, 163, 184, 0.2);
            color: var(--text-secondary);
        }}
        .ev-badge.negative {{
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
        }}
        .disclaimer {{
            background: var(--bg-tertiary);
            border-left: 4px solid #f59e0b;
            padding: 1rem;
            margin-top: 2rem;
            border-radius: 0 8px 8px 0;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }}
        .no-picks {{
            text-align: center;
            padding: 3rem;
            color: var(--text-secondary);
        }}
    </style>
</head>
<body>
    <div class="disclaimer-banner">
        <strong>Disclaimer:</strong> All picks are for entertainment and educational purposes only. Sports betting involves risk. Never bet more than you can afford to lose.
    </div>

    <div class="header">
        <div class="header-content">
            <div class="brand-section">
                <a href="/index.html" style="text-decoration: none;">
                    <svg class="logo" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                            <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" style="stop-color:#e11d48"/>
                                <stop offset="100%" style="stop-color:#f97316"/>
                            </linearGradient>
                        </defs>
                        <circle cx="100" cy="100" r="90" fill="#1e293b"/>
                        <circle cx="100" cy="100" r="70" fill="url(#logoGradient)"/>
                        <path d="M70 85 L85 100 L130 55" stroke="white" stroke-width="8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
                        <circle cx="100" cy="130" r="15" fill="white"/>
                        <circle cx="75" cy="130" r="8" fill="white" opacity="0.8"/>
                        <circle cx="125" cy="130" r="8" fill="white" opacity="0.8"/>
                    </svg>
                </a>
                <div class="brand-text">
                    <h1>SmartBetCalcs</h1>
                    <p>Data-driven sports betting education</p>
                </div>
            </div>
        </div>
    </div>

    <nav class="nav-section">
        <div class="nav-buttons">
            <a href="/index.html" class="nav-btn">Home</a>
            <a href="/sports-betting-guide.html" class="nav-btn">Betting Guides</a>
            <a href="/free-cbb-picks.html" class="nav-btn">CBB Picks</a>
            <a href="/free-nhl-picks.html" class="nav-btn nhl-highlight">NHL Picks</a>
            <a href="/free-mlb-picks.html" class="nav-btn">MLB Picks</a>
            <a href="/calculators.html" class="nav-btn">Calculators</a>
            <a href="/articles.html" class="nav-btn">Betting News</a>
        </div>
    </nav>

    <div class="picks-container">
        <div class="page-header">
            <h1>Free NHL Picks Today</h1>
            <div class="date">{date_str}</div>
        </div>

        <div class="generated-time">
            Last updated: {data["generated_at"]} | {data["total_games"]} games analyzed
        </div>

        {"<div class='picks-grid'>" + pick_cards + "</div>" if data["total_games"] > 0 else "<div class='no-picks'><h2>No Picks Available</h2><p>Our NHL prediction model is still training. Check back soon for daily picks!</p></div>"}

        <div class="disclaimer">
            <strong>Important:</strong> These picks are generated by a machine learning model for educational and entertainment purposes only.
            Past performance does not guarantee future results. Always do your own research and never bet more than you can afford to lose.
            Gambling can be addictive - if you need help, call 1-800-GAMBLER.
        </div>
    </div>
</body>
</html>
'''

    with open(OUTPUT_HTML, 'w') as f:
        f.write(html_content)

    print(f"Generated {OUTPUT_HTML}")

def main():
    # Change to pipeline directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Check for flags
    quick_odds = "--quick-odds" in sys.argv
    retrain = "--retrain" in sys.argv

    # Run the pipeline
    print("=" * 60)
    print("SmartBetCalcs NHL Picks Generator")
    print("=" * 60)

    if quick_odds:
        print("Mode: Quick Odds (API only, no scraping)")
    elif retrain:
        print("Mode: Full pipeline with model retraining")
    else:
        print("Mode: Full pipeline")

    if not run_pipeline(retrain=retrain, quick_odds=quick_odds):
        print("Pipeline failed, checking for existing predictions...")

    # Load predictions
    df = load_predictions()

    if df is None or df.empty:
        print("No predictions available")
        # Generate empty page
        data = {
            "generated_at": datetime.now(pytz.timezone('America/Chicago')).strftime("%Y-%m-%d %H:%M:%S CST"),
            "date": datetime.now(pytz.timezone('America/Chicago')).strftime("%B %d, %Y"),
            "total_games": 0,
            "picks": []
        }
    else:
        # Generate JSON
        data = generate_json(df)

    # Generate HTML
    generate_html(data)

    print("=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
