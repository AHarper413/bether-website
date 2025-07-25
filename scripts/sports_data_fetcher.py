#!/usr/bin/env python3
"""
Women's Sports Data Fetcher for Netlify Deployment
Save as: sports_data_fetcher.py
Location on PC: /your-project-folder/scripts/sports_data_fetcher.py
"""

import requests
import json
import os
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SportsDataFetcher:
    def __init__(self):
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports"
        
    def fetch_wnba_data(self):
        """Fetch WNBA games and standings"""
        logger.info("Fetching WNBA data...")
        wnba_data = {
            'games': [],
            'standings': [],
            'top_players': []
        }
        
        try:
            # Get current games
            games_url = f"{self.base_url}/basketball/wnba/scoreboard"
            response = requests.get(games_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for event in data.get('events', [])[:5]:  # Limit to 5 games
                    if len(event.get('competitions', [])) > 0:
                        comp = event['competitions'][0]
                        if len(comp.get('competitors', [])) >= 2:
                            game_info = {
                                'date': event.get('date'),
                                'status': event.get('status', {}).get('type', {}).get('description', 'Scheduled'),
                                'home_team': {
                                    'name': comp['competitors'][0]['team']['displayName'],
                                    'abbreviation': comp['competitors'][0]['team']['abbreviation'],
                                    'score': comp['competitors'][0].get('score', '0')
                                },
                                'away_team': {
                                    'name': comp['competitors'][1]['team']['displayName'],
                                    'abbreviation': comp['competitors'][1]['team']['abbreviation'],
                                    'score': comp['competitors'][1].get('score', '0')
                                }
                            }
                            wnba_data['games'].append(game_info)
            
            # Get standings
            standings_url = f"{self.base_url}/basketball/wnba/standings"
            standings_response = requests.get(standings_url, timeout=10)
            
            if standings_response.status_code == 200:
                standings_data = standings_response.json()
                
                for group in standings_data.get('children', []):
                    conference = group.get('name', 'Conference')
                    for standing in group.get('standings', {}).get('entries', [])[:6]:  # Top 6 per conference
                        team_data = {
                            'conference': conference,
                            'team_name': standing['team']['displayName'],
                            'abbreviation': standing['team']['abbreviation'],
                            'wins': standing['stats'][0]['value'] if len(standing['stats']) > 0 else 0,
                            'losses': standing['stats'][1]['value'] if len(standing['stats']) > 1 else 0,
                            'win_percentage': standing['stats'][2]['displayValue'] if len(standing['stats']) > 2 else '0.000'
                        }
                        wnba_data['standings'].append(team_data)
            
            # Add popular players (static for now since ESPN doesn't have easy player stats API)
            wnba_data['top_players'] = [
                {'name': 'Caitlin Clark', 'team': 'Indiana Fever', 'ppg': '19.2', 'apg': '8.4'},
                {'name': "A'ja Wilson", 'team': 'Las Vegas Aces', 'ppg': '27.3', 'rpg': '11.9'},
                {'name': 'Breanna Stewart', 'team': 'New York Liberty', 'ppg': '20.4', 'rpg': '8.5'}
            ]
            
        except Exception as e:
            logger.error(f"Error fetching WNBA data: {e}")
            
        return wnba_data
    
    def fetch_tennis_data(self):
        """Fetch WTA tennis data"""
        logger.info("Fetching Tennis data...")
        tennis_data = {
            'tournaments': [],
            'rankings': []
        }
        
        try:
            # Get WTA tournaments
            tennis_url = f"{self.base_url}/tennis/wta/scoreboard"
            response = requests.get(tennis_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for event in data.get('events', [])[:5]:  # Limit to 5 matches
                    if len(event.get('competitions', [])) > 0:
                        comp = event['competitions'][0]
                        if len(comp.get('competitors', [])) >= 2:
                            tournament_info = {
                                'tournament_name': event.get('name', 'Tournament'),
                                'date': event.get('date'),
                                'status': event.get('status', {}).get('type', {}).get('description', 'Scheduled'),
                                'player1': comp['competitors'][0]['athlete']['fullName'],
                                'player2': comp['competitors'][1]['athlete']['fullName']
                            }
                            tennis_data['tournaments'].append(tournament_info)
            
            # Add top WTA players (static)
            tennis_data['rankings'] = [
                {'rank': 1, 'name': 'Iga Świątek', 'country': 'Poland'},
                {'rank': 2, 'name': 'Aryna Sabalenka', 'country': 'Belarus'},
                {'rank': 3, 'name': 'Coco Gauff', 'country': 'USA'},
                {'rank': 4, 'name': 'Elena Rybakina', 'country': 'Kazakhstan'},
                {'rank': 5, 'name': 'Jessica Pegula', 'country': 'USA'}
            ]
            
        except Exception as e:
            logger.error(f"Error fetching tennis data: {e}")
            
        return tennis_data
    
    def fetch_soccer_data(self):
        """Fetch NWSL soccer data"""
        logger.info("Fetching Soccer data...")
        soccer_data = {
            'games': [],
            'standings': []
        }
        
        try:
            # NWSL games
            soccer_url = f"{self.base_url}/soccer/usa.nwsl/scoreboard"
            response = requests.get(soccer_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for event in data.get('events', [])[:5]:  # Limit to 5 games
                    if len(event.get('competitions', [])) > 0:
                        comp = event['competitions'][0]
                        if len(comp.get('competitors', [])) >= 2:
                            game_info = {
                                'date': event.get('date'),
                                'status': event.get('status', {}).get('type', {}).get('description', 'Scheduled'),
                                'home_team': {
                                    'name': comp['competitors'][0]['team']['displayName'],
                                    'abbreviation': comp['competitors'][0]['team']['abbreviation'],
                                    'score': comp['competitors'][0].get('score', '0')
                                },
                                'away_team': {
                                    'name': comp['competitors'][1]['team']['displayName'],
                                    'abbreviation': comp['competitors'][1]['team']['abbreviation'],
                                    'score': comp['competitors'][1].get('score', '0')
                                }
                            }
                            soccer_data['games'].append(game_info)
            
        except Exception as e:
            logger.error(f"Error fetching soccer data: {e}")
            
        return soccer_data
    
    def generate_summary(self, wnba_data, tennis_data, soccer_data):
        """Generate summary data for homepage"""
        return {
            'wnba': {
                'games_today': len([g for g in wnba_data['games'] if self.is_today(g.get('date'))]),
                'featured_game': wnba_data['games'][0] if wnba_data['games'] else None,
                'top_player': wnba_data['top_players'][0] if wnba_data['top_players'] else None,
                'standings_leaders': wnba_data['standings'][:3]
            },
            'tennis': {
                'active_tournaments': len(tennis_data['tournaments']),
                'featured_match': tennis_data['tournaments'][0] if tennis_data['tournaments'] else None,
                'top_players': tennis_data['rankings'][:3]
            },
            'soccer': {
                'games_this_week': len(soccer_data['games']),
                'featured_game': soccer_data['games'][0] if soccer_data['games'] else None,
                'standings_top3': soccer_data['standings'][:3]
            }
        }
    
    def is_today(self, date_str):
        """Check if date is today"""
        if not date_str:
            return False
        try:
            game_date = datetime.fromisoformat(date_str.replace('Z', '+00:00')).date()
            return game_date == datetime.now().date()
        except:
            return False
    
    def run(self):
        """Main execution function"""
        logger.info("Starting sports data fetch...")
        
        # Fetch all data
        wnba_data = self.fetch_wnba_data()
        tennis_data = self.fetch_tennis_data()
        soccer_data = self.fetch_soccer_data()
        
        # Generate summary
        summary = self.generate_summary(wnba_data, tennis_data, soccer_data)
        
        # Create final data structure
        final_data = {
            'summary': summary,
            'detailed': {
                'wnba': wnba_data,
                'tennis': tennis_data,
                'soccer': soccer_data
            },
            'last_updated': datetime.now().isoformat(),
            'next_update': (datetime.now() + timedelta(hours=2)).isoformat()
        }
        
        # Create output directory
        os.makedirs('data', exist_ok=True)
        
        # Save summary for homepage (public)
        with open('data/sports_summary.json', 'w') as f:
            json.dump({
                'summary': summary,
                'last_updated': final_data['last_updated']
            }, f, indent=2)
        
        # Save detailed data (for premium page)
        with open('data/sports_detailed.json', 'w') as f:
            json.dump(final_data, f, indent=2)
        
        logger.info("✅ Sports data generated successfully!")
        logger.info(f"WNBA games: {len(wnba_data['games'])}")
        logger.info(f"Tennis matches: {len(tennis_data['tournaments'])}")
        logger.info(f"Soccer games: {len(soccer_data['games'])}")
        
        return final_data

if __name__ == "__main__":
    fetcher = SportsDataFetcher()
    fetcher.run()