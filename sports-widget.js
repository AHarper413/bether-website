/**
 * Women's Sports Data Widget for Smart Bet Calcs
 * Save as: sports-widget.js  
 * Location on PC: /your-project-folder/public/js/sports-widget.js
 * Upload to Netlify: This file goes in your public/js/ folder
 */

class SportsDataWidget {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.warn(`Sports widget container '${containerId}' not found`);
            return;
        }
        
        this.options = {
            dataUrl: options.dataUrl || '/data/sports_summary.json',
            updateInterval: options.updateInterval || 300000, // 5 minutes
            showDetailed: options.showDetailed || false,
            ...options
        };
        
        this.data = null;
        this.lastUpdated = null;
        this.isLoading = false;
        
        this.init();
    }
    
    async init() {
        this.showLoading();
        await this.fetchData();
        this.render();
        
        // Set up auto-refresh
        setInterval(async () => {
            await this.fetchData();
            this.render();
        }, this.options.updateInterval);
    }
    
    async fetchData() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        try {
            const response = await fetch(this.options.dataUrl + '?t=' + Date.now());
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            this.data = data;
            this.lastUpdated = new Date();
            
            console.log('Sports data updated:', this.lastUpdated);
            
        } catch (error) {
            console.error('Error fetching sports data:', error);
            this.showError('Unable to load sports data. Please try again later.');
        } finally {
            this.isLoading = false;
        }
    }
    
    showLoading() {
        this.container.innerHTML = `
            <div class="sports-data-widget loading">
                <div class="loading-spinner">
                    <div class="spinner"></div>
                    <p>Loading live sports data...</p>
                </div>
            </div>
        `;
    }
    
    showError(message) {
        this.container.innerHTML = `
            <div class="sports-data-widget error">
                <div class="error-message">
                    <span class="error-icon">⚠️</span>
                    <span>${message}</span>
                </div>
                <button onclick="location.reload()" class="retry-btn">Retry</button>
            </div>
        `;
    }
    
    render() {
        if (!this.data || !this.data.summary) {
            this.showError('No sports data available');
            return;
        }
        
        const summary = this.data.summary;
        
        this.container.innerHTML = `
            <div class="sports-data-widget">
                <div class="widget-header">
                    <h3>🏀 Live Women's Sports</h3>
                    <div class="last-updated">
                        Updated: ${this.formatTime(this.lastUpdated)}
                    </div>
                </div>
                
                <div class="sports-grid">
                    ${this.renderWNBA(summary.wnba)}
                    ${this.renderTennis(summary.tennis)}
                    ${this.renderSoccer(summary.soccer)}
                </div>
                
                <div class="widget-footer">
                    <button onclick="window.open('/live-stats.html', '_blank')" class="view-all-btn">
                        📊 View Detailed Stats & Analysis
                    </button>
                </div>
            </div>
        `;
    }
    
    renderWNBA(wnbaData) {
        if (!wnbaData) return '<div class="sport-section"><p>WNBA data unavailable</p></div>';
        
        return `
            <div class="sport-section wnba-section">
                <div class="sport-header">
                    <h4>🏀 WNBA</h4>
                    <span class="games-count">${wnbaData.games_today || 0} games today</span>
                </div>
                
                ${wnbaData.featured_game ? `
                    <div class="featured-game">
                        <div class="game-teams">
                            <div class="team">
                                <span class="team-name">${wnbaData.featured_game.away_team.abbreviation}</span>
                                <span class="score">${wnbaData.featured_game.away_team.score}</span>
                            </div>
                            <div class="vs">@</div>
                            <div class="team">
                                <span class="team-name">${wnbaData.featured_game.home_team.abbreviation}</span>
                                <span class="score">${wnbaData.featured_game.home_team.score}</span>
                            </div>
                        </div>
                        <div class="game-status">${this.formatStatus(wnbaData.featured_game.status)}</div>
                    </div>
                ` : '<div class="no-games">No games scheduled</div>'}
                
                ${wnbaData.top_player ? `
                    <div class="top-player">
                        <span class="player-name">⭐ ${wnbaData.top_player.name}</span>
                        <span class="player-stats">${wnbaData.top_player.ppg} PPG, ${wnbaData.top_player.apg} APG</span>
                    </div>
                ` : ''}
            </div>
        `;
    }
    
    renderTennis(tennisData) {
        if (!tennisData) return '<div class="sport-section"><p>Tennis data unavailable</p></div>';
        
        return `
            <div class="sport-section tennis-section">
                <div class="sport-header">
                    <h4>🎾 WTA Tennis</h4>
                    <span class="tournaments-count">${tennisData.active_tournaments || 0} active matches</span>
                </div>
                
                ${tennisData.featured_match ? `
                    <div class="featured-match">
                        <div class="match-players">
                            <div class="player">
                                <span class="player-name">${this.truncateName(tennisData.featured_match.player1)}</span>
                            </div>
                            <div class="vs">vs</div>
                            <div class="player">
                                <span class="player-name">${this.truncateName(tennisData.featured_match.player2)}</span>
                            </div>
                        </div>
                        <div class="match-status">${this.formatStatus(tennisData.featured_match.status)}</div>
                    </div>
                ` : '<div class="no-matches">No active matches</div>'}
                
                ${tennisData.top_players && tennisData.top_players.length > 0 ? `
                    <div class="top-ranking">
                        <span class="ranking-info">👑 #1 ${tennisData.top_players[0].name}</span>
                        <span class="country">${tennisData.top_players[0].country}</span>
                    </div>
                ` : ''}
            </div>
        `;
    }
    
    renderSoccer(soccerData) {
        if (!soccerData) return '<div class="sport-section"><p>Soccer data unavailable</p></div>';
        
        return `
            <div class="sport-section soccer-section">
                <div class="sport-header">
                    <h4>⚽ NWSL</h4>
                    <span class="games-count">${soccerData.games_this_week || 0} recent games</span>
                </div>
                
                ${soccerData.featured_game ? `
                    <div class="featured-game">
                        <div class="game-teams">
                            <div class="team">
                                <span class="team-name">${soccerData.featured_game.away_team.abbreviation}</span>
                                <span class="score">${soccerData.featured_game.away_team.score}</span>
                            </div>
                            <div class="vs">@</div>
                            <div class="team">
                                <span class="team-name">${soccerData.featured_game.home_team.abbreviation}</span>
                                <span class="score">${soccerData.featured_game.home_team.score}</span>
                            </div>
                        </div>
                        <div class="game-status">${this.formatStatus(soccerData.featured_game.status)}</div>
                    </div>
                ` : '<div class="no-games">No recent games</div>'}
            </div>
        `;
    }
    
    truncateName(name) {
        if (!name) return '';
        const parts = name.split(' ');
        if (parts.length <= 2) return name;
        return `${parts[0]} ${parts[parts.length - 1]}`;
    }
    
    formatStatus(status) {
        if (!status) return '';
        
        // Clean up common status messages
        const statusMap = {
            'Final': '✅ Final',
            'In Progress': '🔴 Live',
            'Halftime': '⏸️ Halftime',
            'Scheduled': '📅 Upcoming',
            'Postponed': '⏰ Postponed',
            'Canceled': '❌ Canceled'
        };
        
        return statusMap[status] || status;
    }
    
    formatTime(date) {
        if (!date) return 'Never';
        return date.toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit',
            hour12: true 
        });
    }
}

// Auto-initialize if container exists
document.addEventListener('DOMContentLoaded', function() {
    // Try to find sports widget containers and initialize them
    const containers = [
        'sports-data-container',
        'sports-widget-container', 
        'live-sports-widget'
    ];
    
    containers.forEach(containerId => {
        const element = document.getElementById(containerId);
        if (element && !element.dataset.initialized) {
            element.dataset.initialized = 'true';
            new SportsDataWidget(containerId);
        }
    });
});

// Export for manual initialization
window.SportsDataWidget = SportsDataWidget;