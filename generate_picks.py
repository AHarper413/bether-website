import os
import sys
import json
import pandas as pd
from datetime import datetime, date
import subprocess
import glob

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

# [Rest of the functions remain the same as before - create_picks_json, create_picks_html, etc.]

def create_picks_json(predictions_df):
    """Convert predictions to JSON format for website"""
    today = date.today().strftime("%B %d, %Y")
    
    # Count signals
    strong_home_count = len(predictions_df[predictions_df['strong_home_favorite'] == 'YES'])
    strong_away_count = len(predictions_df[predictions_df['strong_away_favorite'] == 'YES']) 
    total_signals = strong_home_count + strong_away_count
    
    over_8_5_count = len(predictions_df[predictions_df['lean_over_8_5'] == 'YES'])
    f5_signals_count = len(predictions_df[predictions_df['f5_over_signal'] == 'YES'])
    
    # Get best bets (highest confidence)
    best_ml_bets = []
    for _, game in predictions_df.iterrows():
        home_prob = game['home_win_probability']
        away_prob = game['away_win_probability']
        
        if home_prob > 0.58:  # Strong home favorite
            confidence = "HIGH" if home_prob > 0.62 else "MEDIUM"
            best_ml_bets.append({
                "game": f"{game['away_team']} @ {game['home_team']}",
                "pick": f"{game['home_team']} ML",
                "probability": f"{home_prob:.1%}",
                "confidence": confidence,
                "reasoning": f"Model favors home team with {home_prob:.1%} win probability"
            })
        elif away_prob > 0.58:  # Strong away favorite  
            confidence = "HIGH" if away_prob > 0.62 else "MEDIUM"
            best_ml_bets.append({
                "game": f"{game['away_team']} @ {game['home_team']}",
                "pick": f"{game['away_team']} ML", 
                "probability": f"{away_prob:.1%}",
                "confidence": confidence,
                "reasoning": f"Model favors away team with {away_prob:.1%} win probability"
            })
    
    # Get total bets
    total_bets = []
    for _, game in predictions_df.iterrows():
        if game['lean_over_8_5'] == 'YES':
            prob = game['over_8_5_probability']
            confidence = "HIGH" if prob > 0.60 else "MEDIUM"
            total_bets.append({
                "game": f"{game['away_team']} @ {game['home_team']}",
                "pick": "Over 8.5 Runs",
                "probability": f"{prob:.1%}",
                "confidence": confidence,
                "predicted_total": game['predicted_total_runs']
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
            "f5_signals": f5_signals_count
        },
        "best_moneyline_bets": best_ml_bets[:5],  # Top 5
        "best_total_bets": total_bets[:3],  # Top 3
        "season_record": {"wins": 156, "losses": 100, "percentage": 61.0}  # Update this manually
    }
    
    return picks_data

def create_picks_html(picks_data, predictions_df):
    """Create enhanced HTML page with interactive team stats"""
    
    # Convert predictions_df to JSON for JavaScript
    import json
    games_data = []
    for _, game in predictions_df.iterrows():
        games_data.append({
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
            'f5_over_signal': game.get('f5_over_signal', 'NO') == 'YES'
        })
    
    games_json = json.dumps(games_data, indent=2)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Free MLB Picks Today - {picks_data['date']} | BetHer</title>
    <meta name="description" content="Free MLB picks today from our advanced XGBoost model. Expert baseball predictions with confidence ratings and detailed team analysis.">
    <meta name="keywords" content="free mlb picks today, baseball predictions, mlb betting tips, expert mlb picks">
    
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            color: #1e293b;
            background: #f8fafc;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 2rem;
            border-bottom: 2px solid #e11d48;
            padding-bottom: 1rem;
        }}
        .header h1 {{
            color: #e11d48;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
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
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: #f1f5f9;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 2rem;
            font-weight: bold;
            color: #e11d48;
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
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Free MLB Picks - {picks_data['date']}</h1>
            <p>Advanced XGBoost predictions with detailed analysis</p>
            <p><strong>Last Updated:</strong> {picks_data['last_updated']}</p>
        </div>

        <div class="model-info">
            <h2>🤖 XGBoost Model with Feature Selection</h2>
            <p>Advanced machine learning algorithm analyzing 50+ factors including team stats, pitcher matchups, bullpen fatigue, stadium factors, and recent form</p>
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
            <div class="stat-card">
                <div class="stat-number">{picks_data['summary']['f5_signals']}</div>
                <div>F5 Signals</div>
            </div>
        </div>"""

    # Add games grid
    html_content += """
        <div class="games-grid">
"""

    # Add each game
    for game_data in games_data:
        confidence_class = game_data['confidence_tier'].lower() + '-confidence'
        
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
        
        signals_html = ''.join([f'<span class="signal {signal_class}">{signal_text}</span>' 
                               for signal_text, signal_class in signals])
        
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
                
                {f'<div class="signals">{signals_html}</div>' if signals else ''}
            </div>
"""

    # Add modal and JavaScript
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
            <p><strong>Disclaimer:</strong> These are model predictions for educational purposes. Always bet responsibly.</p>
            <p><a href="/index.html">← Back to BetHer Home</a> | <a href="/free-mlb-picks.html">← SEO Page</a></p>
        </div>
    </div>

    <script>
        // Games data
        const gamesData = {games_json};
        
        function showTeamStats(gameId, teamType) {{
            const game = gamesData.find(g => g.game_id === gameId);
            if (!game) return;
            
            const isHome = teamType === 'home';
            const teamName = isHome ? game.home_team : game.away_team;
            const pitcher = isHome ? game.home_pitcher : game.away_pitcher;
            
            document.getElementById('modalTitle').textContent = `${{teamName}} - Today's Analysis`;
            
            const modalContent = `
                <div style="text-align: center; margin-bottom: 1rem;">
                    <h3>Starting Pitcher: ${{pitcher}}</h3>
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
                        <div class="stat-label">Recent Form</div>
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
                    <h4 style="margin: 0 0 0.5rem 0;">Game Context</h4>
                    <p><strong>Stadium Factor:</strong> ${{game.stadium_factor}}</p>
                    <p><strong>Bullpen Advantage:</strong> ${{game.bullpen_advantage}}</p>
                    <p><strong>Home Field Advantage:</strong> ${{(game.home_field_advantage * 100).toFixed(1)}}%</p>
                    ${{isHome ? '' : `<p><strong>Pitcher ERA Advantage:</strong> ${{game.pitcher_era_advantage > 0 ? 'Favorable' : 'Unfavorable'}} (${{game.pitcher_era_advantage.toFixed(2)}})</p>`}}
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
    </script>
</body>
</html>
"""
    
    return html_content

def main():
    print("🚀 Starting MLB picks automation...")
    
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
    
    # Create JSON data
    picks_data = create_picks_json(predictions_df)
    
    # Save JSON file
    with open('daily_picks.json', 'w') as f:
        json.dump(picks_data, f, indent=2)
    print("✅ Created daily_picks.json")
    
    # Create HTML page - UPDATED TO PASS predictions_df
    html_content = create_picks_html(picks_data, predictions_df)
    with open('mlb-picks-today.html', 'w') as f:
        f.write(html_content)
    print("✅ Created mlb-picks-today.html")
    
    print(f"🎯 Summary for {picks_data['date']}:")
    print(f"   📊 {picks_data['summary']['total_games']} games analyzed")
    print(f"   🎯 {picks_data['summary']['moneyline_signals']} ML signals")
    print(f"   📈 {picks_data['summary']['over_8_5_signals']} total signals")
    print(f"   ⚾ {picks_data['summary']['f5_signals']} F5 signals")

if __name__ == "__main__":
    main()