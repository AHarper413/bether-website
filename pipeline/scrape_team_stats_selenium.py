"""
Automated Team Stats Scraper using Selenium + Stealth

Scrapes team statistics from Sports Reference to eliminate manual CSV export.
Uses browser automation with anti-detection to bypass 403 blocking.

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

def scrape_team_stats(season=2026, headless=True, max_retries=3):
    """
    Scrape team statistics for given season from Sports Reference

    Args:
        season: NCAA season year (e.g., 2025)
        headless: Run browser in headless mode
        max_retries: Number of retry attempts on failure

    Returns:
        pandas.DataFrame with team stats or None on failure
    """

    url = f"https://www.sports-reference.com/cbb/seasons/men/{season}-school-stats.html"

    print(f"\n{'='*70}")
    print(f"TEAM STATS SCRAPER - Season {season}")
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

            # Navigate to page
            print(f"   📥 Fetching page...")
            driver.get(url)

            # Wait for table to load
            print("   ⏳ Waiting for table to load...")
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "basic_school_stats"))
                )
                print("   ✅ Table found!")
            except:
                print("   ⚠️  Timeout waiting for table (trying anyway)...")

            # Small delay to ensure full render
            time.sleep(2)

            # Get page source
            html = driver.page_source

            # Check for blocking indicators
            if "403 Forbidden" in html or "Access Denied" in html:
                print("   ❌ 403 Forbidden detected!")
                if attempt < max_retries:
                    backoff = 10 * (2 ** (attempt - 1))  # Exponential backoff: 10s, 20s, 40s
                    print(f"   ⏳ Backing off for {backoff} seconds...")
                    time.sleep(backoff)
                continue

            # Extract the stats table by ID using BeautifulSoup
            print("   🔍 Extracting stats table...")
            from bs4 import BeautifulSoup
            from io import StringIO

            soup = BeautifulSoup(html, 'html.parser')
            table_element = soup.find('table', {'id': 'basic_school_stats'})

            if not table_element:
                print("   ❌ Table with ID 'basic_school_stats' not found")
                continue

            # Convert table to string and parse with pandas
            table_html = str(table_element)
            tables = pd.read_html(StringIO(table_html))

            if not tables or len(tables) == 0:
                print("   ❌ Failed to parse table with pandas")
                continue

            stats_df = tables[0]
            print(f"   ✅ Found stats table: {stats_df.shape[0]} teams, {stats_df.shape[1]} columns")

            # Clean up the dataframe
            print("   🧹 Cleaning data...")

            # Flatten multi-level column names (if present)
            if isinstance(stats_df.columns, pd.MultiIndex):
                # Flatten by intelligently combining the two levels
                new_cols = []
                for col in stats_df.columns:
                    # If first level has meaningful name, use it
                    level0 = str(col[0]) if col[0] and not str(col[0]).startswith('Unnamed') else ''
                    # If second level has meaningful name, use it
                    level1 = str(col[1]) if col[1] and not str(col[1]).startswith('Unnamed') else ''

                    # Combine intelligently
                    if level0 and level1:
                        new_cols.append(f"{level0}_{level1}")
                    elif level1:
                        new_cols.append(level1)
                    elif level0:
                        new_cols.append(level0)
                    else:
                        new_cols.append('Unknown')

                stats_df.columns = new_cols

            # Clean up column names - remove redundant prefixes
            stats_df.columns = [col.replace('Totals_', '').replace('Overall_', '').replace('Points_', '') for col in stats_df.columns]

            # Rename key columns to match expected format (2024 baseline compatibility)
            rename_map = {
                'Rk': 'Rank',
                'School': 'team',
                # KEEP Tm. and Opp. WITH PERIODS (don't rename them)
                # Conference records
                'Conf._W': 'Wconference',
                'Conf._L': 'Lconference',
                # Home/Away records (match 2024 baseline format)
                'Home_W': 'Whome',
                'Home_L': 'Lhome',
                'Away_W': 'Waway',
                'Away_L': 'Laway'
            }
            stats_df = stats_df.rename(columns=rename_map)

            # Remove header/separator rows that appear in data
            if 'Rank' in stats_df.columns:
                # Remove rows where Rank is 'Rank' or 'Rk' (header rows)
                stats_df = stats_df[stats_df['Rank'] != 'Rank']
                stats_df = stats_df[stats_df['Rank'] != 'Rk']

                # Remove rows where Rank is NaN (separator rows from pagination)
                stats_df = stats_df[stats_df['Rank'].notna()]

                # Remove rows where Rank is empty string or 'nan' string
                stats_df = stats_df[stats_df['Rank'] != '']
                stats_df = stats_df[stats_df['Rank'] != 'nan']

                # Remove 'NCAA' suffix from rank column
                stats_df['Rank'] = stats_df['Rank'].astype(str).str.replace('NCAA', '').str.strip()

            # Reset index
            stats_df = stats_df.reset_index(drop=True)

            # Drop columns named "Unknown" (separator columns from website)
            stats_df = stats_df.loc[:, ~stats_df.columns.str.contains('Unknown')]

            # Convert numeric columns to proper types (prevents "can't divide str by str" errors)
            numeric_columns = [
                'Rank', 'G', 'W', 'L', 'W-L%', 'SRS', 'SOS',
                'Wconference', 'Lconference', 'Whome', 'Lhome', 'Waway', 'Laway',
                'Tm.', 'Opp.', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%',
                'FT', 'FTA', 'FT%', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF'
            ]
            for col in numeric_columns:
                if col in stats_df.columns:
                    stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')

            print(f"\n{'='*70}")
            print(f"✅ SUCCESS - Scraped {len(stats_df)} teams")
            print(f"{'='*70}")
            print(f"\n📊 Sample data (first 3 teams):")
            print(stats_df.head(3).to_string())

            return stats_df

        except Exception as e:
            print(f"   ❌ Error: {e}")

            if attempt < max_retries:
                backoff = 10 * (2 ** (attempt - 1))
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

def save_team_stats(df, output_file="1.31ncaa_2025_team_stats.csv"):
    """Save scraped stats to CSV file"""

    if df is None:
        print("\n❌ No data to save")
        return False

    try:
        df.to_csv(output_file, index=False)
        print(f"\n💾 Saved to: {output_file}")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        return True
    except Exception as e:
        print(f"\n❌ Error saving file: {e}")
        return False

if __name__ == "__main__":
    """
    Usage:
        python scrape_team_stats_selenium.py              # Scrape current season (2026 = 2025-26)
        python scrape_team_stats_selenium.py 2025         # Scrape 2024-25 season
        python scrape_team_stats_selenium.py 2026 visible # Show browser
    """

    # Parse command line arguments
    season = 2026  # Default to current season (2025-26)
    headless = True

    if len(sys.argv) > 1:
        try:
            season = int(sys.argv[1])
        except:
            print(f"Invalid season: {sys.argv[1]}")
            sys.exit(1)

    if len(sys.argv) > 2 and sys.argv[2].lower() in ['visible', 'show', 'gui']:
        headless = False

    # Run scraper
    df = scrape_team_stats(season=season, headless=headless)

    if df is not None:
        # Save to standard filename (pipeline expects this exact name)
        output_file = "1.31ncaa_2025_team_stats.csv"  # Keep standard name for pipeline compatibility
        if save_team_stats(df, output_file):
            print("\n✅ Scraping completed successfully!")
            print(f"   Season: {season-1}-{str(season)[-2:]}")
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        print("\n❌ Scraping failed - please use manual CSV export as backup")
        print(f"   Fallback: https://www.sports-reference.com/cbb/seasons/men/{season}-school-stats.html")
        sys.exit(1)
