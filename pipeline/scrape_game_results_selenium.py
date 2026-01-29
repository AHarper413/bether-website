"""
Automated Game Results Scraper using Selenium + Stealth

Scrapes game results from Sports Reference to eliminate manual CSV entry.
Uses browser automation with anti-detection to bypass 403 blocking.

Replaces manual_winners_YYYY-MM-DD.csv creation.

Rate Limit: 20 requests/minute (enforced via delays)
Anti-Detection: selenium-stealth + realistic browser fingerprints
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium_stealth import stealth
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from datetime import datetime, timedelta
import time
import random
import sys

def create_stealth_driver(headless=True):
    """Create a Chrome driver with stealth settings to avoid detection"""

    options = webdriver.ChromeOptions()

    if headless:
        options.add_argument("--headless=new")

    # Anti-detection settings
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")

    # Realistic user agent
    options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    try:
        # Use webdriver-manager to automatically download correct ChromeDriver version
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    except Exception as e:
        print(f"❌ Error creating Chrome driver: {e}")
        print("   Make sure Chrome/Chromium is installed")
        return None

    # Apply stealth settings
    stealth(driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="MacIntel",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
    )

    return driver

def scrape_game_results(target_date=None, headless=True, max_retries=3):
    """
    Scrape game results for a specific date from Sports Reference

    Args:
        target_date: Date to scrape (datetime.date object). If None, uses yesterday.
        headless: Run browser in headless mode
        max_retries: Number of retry attempts on failure

    Returns:
        pandas.DataFrame with columns: Date, Away_Team, Home_Team, Winner
        Or None on failure
    """

    # Default to yesterday if no date specified
    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).date()

    # Ensure target_date is a date object
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()

    # Build URL
    url = f"https://www.sports-reference.com/cbb/boxscores/index.cgi?month={target_date.month}&day={target_date.day}&year={target_date.year}"

    print(f"\n{'='*70}")
    print(f"GAME RESULTS SCRAPER - {target_date}")
    print(f"{'='*70}")
    print(f"\n🌐 Target URL: {url}")
    print(f"🤖 Mode: {'Headless' if headless else 'Visible Browser'}")
    print(f"🔄 Max retries: {max_retries}")

    for attempt in range(1, max_retries + 1):
        driver = None

        try:
            print(f"\n📍 Attempt {attempt}/{max_retries}")

            # Create stealth driver
            print("   🚀 Launching browser...")
            driver = create_stealth_driver(headless=headless)

            if driver is None:
                print("   ❌ Failed to create driver")
                continue

            # Add random delay to appear human (respect 20 req/min = 3+ sec delay)
            delay = random.uniform(3.5, 5.5)
            print(f"   ⏳ Waiting {delay:.1f} seconds (rate limit compliance)...")
            time.sleep(delay)

            # Set page load timeout to prevent indefinite hanging
            driver.set_page_load_timeout(45)  # 45 seconds max for page load

            # Navigate to page
            print(f"   📥 Fetching page...")
            driver.get(url)

            # Increased wait time for games to load (20 seconds instead of 10)
            print("   ⏳ Waiting for game summaries to load...")
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "game_summary"))
                )
                print("   ✅ Game summaries found!")
            except:
                print("   ⚠️  Timeout waiting for games (page may have no games)...")

            # Small delay to ensure full render
            time.sleep(2)

            # Check for blocking indicators
            page_source = driver.page_source
            if "403 Forbidden" in page_source or "Access Denied" in page_source:
                print("   ❌ 403 Forbidden detected!")
                if attempt < max_retries:
                    backoff = 15 * (2 ** (attempt - 1))  # Exponential backoff: 15s, 30s, 60s
                    print(f"   ⏳ Backing off for {backoff} seconds...")
                    time.sleep(backoff)
                continue

            # Find all game summary divs
            print("   🔍 Extracting game data...")
            game_elements = driver.find_elements(By.CSS_SELECTOR, ".game_summary.nohover.gender-m")

            if not game_elements:
                print("   ⚠️  No games found for this date")
                print("   This could mean: (1) No games scheduled, (2) Page structure changed, (3) Still loading")

                # Check if there's a "No games" message
                if "No games" in page_source or len(page_source) < 5000:
                    print("   ℹ️  Confirmed: No games on this date")
                    return pd.DataFrame(columns=['Date', 'Away_Team', 'Home_Team', 'Winner'])

                continue

            print(f"   📊 Found {len(game_elements)} games")

            # Extract game data
            results = []

            for i, game_el in enumerate(game_elements, 1):
                try:
                    # Get team rows (away team first, home team second)
                    team_rows = game_el.find_elements(By.CSS_SELECTOR, "tr")

                    if len(team_rows) < 2:
                        print(f"   ⚠️  Game {i}: Not enough team rows, skipping")
                        continue

                    # Away team (first row)
                    away_row = team_rows[0]
                    away_team_el = away_row.find_element(By.CSS_SELECTOR, "td a")
                    away_team = away_team_el.text.strip()
                    away_is_winner = "winner" in away_row.get_attribute("class")

                    # Home team (second row)
                    home_row = team_rows[1]
                    home_team_el = home_row.find_element(By.CSS_SELECTOR, "td a")
                    home_team = home_team_el.text.strip()
                    home_is_winner = "winner" in home_row.get_attribute("class")

                    # Determine winner
                    if away_is_winner:
                        winner = away_team
                    elif home_is_winner:
                        winner = home_team
                    else:
                        print(f"   ⚠️  Game {i}: No winner indicator found")
                        winner = None

                    if winner:
                        results.append({
                            'Date': target_date,
                            'Away_Team': away_team,
                            'Home_Team': home_team,
                            'Winner': winner
                        })
                        print(f"   ✓ Game {i}: {away_team} @ {home_team} → Winner: {winner}")

                except Exception as e:
                    print(f"   ⚠️  Game {i}: Error extracting data - {e}")
                    continue

            if not results:
                print("   ❌ No valid games extracted")
                continue

            # Create DataFrame
            df = pd.DataFrame(results)

            print(f"\n{'='*70}")
            print(f"✅ SUCCESS - Scraped {len(df)} games")
            print(f"{'='*70}")

            return df

        except Exception as e:
            print(f"   ❌ Error: {e}")

            if attempt < max_retries:
                backoff = 15 * (2 ** (attempt - 1))  # Exponential backoff: 15s, 30s, 60s
                print(f"   ⏳ Retrying in {backoff} seconds...")
                time.sleep(backoff)
            else:
                print(f"\n{'='*70}")
                print(f"❌ FAILED after {max_retries} attempts")
                print(f"{'='*70}")

        finally:
            if driver:
                driver.quit()
                print("   🛑 Browser closed")

    return None

def save_game_results(df, target_date=None, output_file=None):
    """Save scraped game results to CSV file"""

    if df is None:
        print("\n❌ No data to save")
        return False

    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).date()

    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()

    if output_file is None:
        output_file = f"manual_winners_{target_date}.csv"

    try:
        df.to_csv(output_file, index=False)
        print(f"\n💾 Saved to: {output_file}")
        print(f"   Games: {len(df)}")
        print(f"   Format: Date, Away_Team, Home_Team, Winner")
        return True
    except Exception as e:
        print(f"\n❌ Error saving file: {e}")
        return False

if __name__ == "__main__":
    """
    Usage:
        python scrape_game_results_selenium.py                  # Scrape yesterday's games
        python scrape_game_results_selenium.py 2025-11-03       # Scrape specific date
        python scrape_game_results_selenium.py 2025-11-03 visible # Show browser
    """

    # Parse command line arguments
    target_date = None
    headless = True

    if len(sys.argv) > 1:
        try:
            target_date = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        except:
            print(f"Invalid date format: {sys.argv[1]}")
            print("Use YYYY-MM-DD format (e.g., 2025-11-03)")
            sys.exit(1)

    if len(sys.argv) > 2 and sys.argv[2].lower() in ['visible', 'show', 'gui']:
        headless = False

    # Run scraper
    df = scrape_game_results(target_date=target_date, headless=headless)

    if df is not None:
        # Show results
        if len(df) > 0:
            print(f"\n📋 Game Results for {target_date or (datetime.now() - timedelta(days=1)).date()}:")
            print(df.to_string(index=False))

            # Save to file
            if save_game_results(df, target_date):
                print("\n✅ Scraping completed successfully!")
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            print(f"\nℹ️  No games found for {target_date or (datetime.now() - timedelta(days=1)).date()}")
            print("   This is normal if no games were scheduled on this date.")
            sys.exit(0)
    else:
        print("\n❌ Scraping failed - please use manual CSV entry as backup")
        print(f"   Fallback: https://www.sports-reference.com/cbb/boxscores/index.cgi?month=11&day=3&year=2025")
        sys.exit(1)
