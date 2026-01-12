"""
Web scraping module for BizBuySell business listings.
Uses ScraperAPI to bypass anti-bot protection.
"""

import re
import time
import urllib3
import requests
import httpx
from bs4 import BeautifulSoup
from typing import Optional
from dataclasses import dataclass, field

# Disable SSL warnings for fallback methods
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class BusinessData:
    """Data class to hold scraped business information."""
    url: str
    name: str = "Unknown Business"
    asking_price: Optional[float] = None
    cash_flow: Optional[float] = None
    gross_revenue: Optional[float] = None
    ebitda: Optional[float] = None
    inventory: Optional[float] = None
    ffe: Optional[float] = None  # Furniture, Fixtures & Equipment
    real_estate: Optional[str] = None
    lease_info: Optional[str] = None
    year_established: Optional[int] = None
    employees: Optional[int] = None
    description: str = ""
    reason_for_selling: str = ""
    location: str = ""
    category: str = ""
    highlights: list = field(default_factory=list)
    raw_html: str = ""


class BizBuySellScraper:
    """Scraper for BizBuySell website."""

    BASE_URL = "https://www.bizbuysell.com"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    def __init__(self, delay: float = 1.5, scraper_api_key: str = None):
        """
        Initialize the scraper.

        Args:
            delay: Delay between requests in seconds (be polite!)
            scraper_api_key: Optional ScraperAPI key for bypassing blocks
        """
        # Reduce delay when using ScraperAPI (they handle rate limiting)
        self.delay = 0.5 if scraper_api_key else delay
        self.scraper_api_key = scraper_api_key

    def _parse_currency(self, text: str) -> Optional[float]:
        """Parse currency string to float."""
        if not text:
            return None
        # Remove currency symbols, commas, and whitespace
        cleaned = re.sub(r'[^\d.]', '', text)
        try:
            return float(cleaned) if cleaned else None
        except ValueError:
            return None

    def _parse_number(self, text: str) -> Optional[int]:
        """Parse number string to int."""
        if not text:
            return None
        cleaned = re.sub(r'[^\d]', '', text)
        try:
            return int(cleaned) if cleaned else None
        except ValueError:
            return None

    def _fetch_url(self, url: str) -> str:
        """Fetch URL with ScraperAPI support for bypassing blocks."""
        from urllib.parse import urlencode
        errors = []

        # If ScraperAPI key is provided, try it first with retries
        if self.scraper_api_key:
            for attempt in range(2):
                try:
                    # Use render=false for faster responses (no JS rendering needed)
                    params = {
                        'api_key': self.scraper_api_key,
                        'url': url,
                        'render': 'false',
                        'country_code': 'us'
                    }
                    api_url = f"http://api.scraperapi.com?{urlencode(params)}"
                    response = requests.get(api_url, timeout=45)
                    response.raise_for_status()
                    return response.text
                except Exception as e:
                    if attempt < 1:
                        time.sleep(1)
                        continue
                    errors.append(f"scraperapi: {str(e)[:80]}")
                    # Fall through to other methods if all retries fail

        # Fallback Method 1: httpx (no SSL verify)
        try:
            with httpx.Client(
                verify=False,
                timeout=30.0,
                follow_redirects=True
            ) as client:
                response = client.get(url, headers=self.HEADERS)
                response.raise_for_status()
                return response.text
        except Exception as e1:
            errors.append(f"httpx: {str(e1)[:80]}")

        # Fallback Method 2: requests (no SSL verify)
        try:
            response = requests.get(
                url,
                headers=self.HEADERS,
                timeout=30,
                verify=False
            )
            response.raise_for_status()
            return response.text
        except Exception as e2:
            errors.append(f"requests: {str(e2)[:80]}")

        raise Exception(f"All methods failed: {'; '.join(errors)}")

    def get_listings_from_search_page(self, parent_url: str, limit: int = 10) -> list[BusinessData]:
        """
        Extract business data directly from search results page (faster, more reliable).

        Args:
            parent_url: URL of the BizBuySell search results page
            limit: Maximum number of listings to return

        Returns:
            List of BusinessData objects with data from search results
        """
        try:
            html = self._fetch_url(parent_url)
        except Exception as e:
            raise Exception(f"Failed to fetch search page: {e}")

        soup = BeautifulSoup(html, 'lxml')
        results = []
        seen_urls = set()

        # BizBuySell uses Angular components - try multiple selectors
        # app-listing-diamond for main listings, app-listing-basic for filtered/paginated results
        listing_cards = soup.select('app-listing-diamond, app-listing-basic')

        # Fallback: find parent containers of business-opportunity links
        if not listing_cards:
            for link in soup.find_all('a', href=lambda h: h and '/business-opportunity/' in h):
                parent = link.find_parent(['app-listing-diamond', 'app-listing-basic', 'div', 'article'])
                if parent and parent not in listing_cards:
                    listing_cards.append(parent)

        for card in listing_cards:
            if len(results) >= limit:
                break

            try:
                # Extract URL
                link = card.find('a', href=lambda h: h and '/business-opportunity/' in h)
                if not link:
                    continue

                url = link.get('href', '')
                if not url:
                    continue
                if url.startswith('/'):
                    url = self.BASE_URL + url

                # Skip duplicates
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                # Get all text from the card
                card_text = card.get_text(separator=' | ', strip=True)

                # Extract name/title - usually first part before location
                parts = card_text.split(' | ')
                name = parts[0] if parts else "Unknown Business"
                name = name[:100]

                data = BusinessData(url=url, name=name)

                # Extract location (usually "City, ST" format)
                loc_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2})', card_text)
                if loc_match:
                    data.location = loc_match.group(1)

                # Extract price (first dollar amount is usually asking price)
                price_matches = re.findall(r'\$([\d,]+)', card_text)
                if price_matches:
                    data.asking_price = self._parse_currency(price_matches[0])

                # Extract cash flow
                cf_match = re.search(r'cash\s*flow[:\s]*\$?([\d,]+)', card_text, re.IGNORECASE)
                if cf_match:
                    data.cash_flow = self._parse_currency(cf_match.group(1))

                # Extract revenue
                rev_match = re.search(r'(?:gross\s*)?revenue[:\s]*\$?([\d,]+)', card_text, re.IGNORECASE)
                if rev_match:
                    data.gross_revenue = self._parse_currency(rev_match.group(1))

                # Extract description (usually after location)
                if len(parts) > 2:
                    # Find the description part (usually the longest text segment)
                    for part in parts[2:]:
                        if len(part) > 50 and not part.startswith('$'):
                            data.description = part[:500]
                            break

                results.append(data)

            except Exception:
                continue

        return results


def scrape_listings(parent_url: str, limit: int = 10, progress_callback=None, scraper_api_key: str = None) -> list[BusinessData]:
    """
    Main function to scrape multiple business listings.

    Uses fast extraction from search results page instead of visiting each listing
    (individual pages often timeout with ScraperAPI free tier).

    Args:
        parent_url: URL of the search results page
        limit: Maximum number of listings to scrape
        progress_callback: Optional callback function(current, total, message)
        scraper_api_key: Optional ScraperAPI key for bypassing blocks

    Returns:
        List of BusinessData objects
    """
    scraper = BizBuySellScraper(scraper_api_key=scraper_api_key)

    if progress_callback:
        progress_callback(0, limit, "Fetching search results...")

    # Use fast extraction from search page (individual pages timeout)
    results = scraper.get_listings_from_search_page(parent_url, limit)

    if not results:
        raise Exception("No business listings found on the page. The page structure may have changed.")

    if progress_callback:
        progress_callback(len(results), len(results), f"Found {len(results)} listings!")

    return results
