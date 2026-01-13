"""
Web scraping module for BizBuySell business listings.
Uses ScraperAPI to bypass anti-bot protection.
"""

import os
import re
import time
import json
import urllib3
import requests
import httpx
from bs4 import BeautifulSoup
from typing import Optional
from dataclasses import dataclass, field
from urllib.parse import urlencode
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# Disable SSL warnings for fallback methods
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Business types to exclude from results
EXCLUDED_KEYWORDS = [
    # Dental
    "dental", "dentist", "orthodont",
    # Legal
    "lawyer", "attorney", "law firm", "legal practice", "law office",
    # Accounting
    "cpa", "accounting firm", "accountant", "tax practice", "bookkeeping",
    # Medical
    "medical center", "medical practice", "clinic", "healthcare", "physician",
    "chiropractic", "chiropractor", "physical therapy", "optometry", "veterinar",
    # Restaurants & Food Service (comprehensive)
    "restaurant", "cafe", "café", "diner", "eatery", "food service", "pizzeria",
    "pizza", "bistro", "grill", "kitchen", "dining", "bakery", "coffee shop",
    "bar ", "tavern", "pub ", "brewery", "catering", "food truck", "ice cream",
    "sandwich", "sushi", "taco", "burger", "bbq", "barbecue", "steakhouse",
    "seafood", "thai", "chinese", "mexican", "italian", "indian", "japanese",
    "vietnamese", "korean", "mediterranean", "greek", "french cuisine",
    "fast food", "quick service", "qsr", "juice bar", "smoothie", "acai",
    "donut", "bagel", "deli", "sub shop", "wing", "chicken", "noodle",
    "ramen", "pho", "curry", "buffet", "food court", "cantina", "trattoria",
]


def _get_excluded_keyword(name: str, description: str) -> Optional[str]:
    """Check if business matches excluded keywords and return the matched keyword."""
    # Normalize text: lowercase and replace accented chars
    text = f"{name} {description}".lower()
    text = text.replace("é", "e").replace("è", "e").replace("ê", "e")
    text = text.replace("á", "a").replace("à", "a").replace("ñ", "n")
    for kw in EXCLUDED_KEYWORDS:
        if kw in text:
            return kw
    return None


def _is_excluded_by_keywords(name: str, description: str) -> bool:
    """Quick keyword check to filter obvious excluded business types."""
    return _get_excluded_keyword(name, description) is not None


def _classify_with_ai(name: str, description: str) -> dict:
    """
    Use OpenAI to classify if business should be excluded.
    Returns dict with 'exclude' (bool) and 'reason' (str).

    ALWAYS checks keywords first - if keyword match found, skip AI call.
    """
    # ALWAYS check keywords first - this is fast and reliable
    matched_keyword = _get_excluded_keyword(name, description)
    if matched_keyword:
        # Categorize the keyword for a cleaner display
        if matched_keyword in ["restaurant", "cafe", "coffee", "bakery", "pizza", "bar", "grill",
                                "food truck", "catering", "diner", "eatery", "tavern", "bistro",
                                "sandwich", "sushi", "taco", "burger", "bbq", "barbecue", "steakhouse",
                                "seafood", "thai", "chinese", "mexican", "italian", "indian", "japanese",
                                "vietnamese", "korean", "mediterranean", "greek", "french cuisine",
                                "fast food", "quick service", "qsr", "juice bar", "smoothie", "acai",
                                "donut", "bagel", "deli", "sub shop", "wing", "chicken", "noodle",
                                "ramen", "pho", "curry", "buffet", "food court", "cantina", "trattoria"]:
            return {"exclude": True, "reason": f"Restaurant/Food service ({matched_keyword})"}
        elif matched_keyword in ["dental", "dentist", "orthodontic", "orthodontist", "oral surgery"]:
            return {"exclude": True, "reason": f"Dental/Orthodontic practice ({matched_keyword})"}
        elif matched_keyword in ["law firm", "attorney", "legal practice", "lawyer", "legal services"]:
            return {"exclude": True, "reason": f"Law/Legal practice ({matched_keyword})"}
        elif matched_keyword in ["cpa", "accounting firm", "tax practice", "bookkeeping"]:
            return {"exclude": True, "reason": f"CPA/Accounting firm ({matched_keyword})"}
        elif matched_keyword in ["medical", "clinic", "physician", "doctor", "healthcare",
                                  "chiropractic", "chiropractor", "optometry", "optometrist",
                                  "physical therapy", "urgent care", "med spa", "dermatology",
                                  "veterinary", "vet clinic", "animal hospital", "pharmacy"]:
            return {"exclude": True, "reason": f"Medical/Healthcare ({matched_keyword})"}
        else:
            return {"exclude": True, "reason": f"Excluded category ({matched_keyword})"}

    # If no keyword match, try AI classification for edge cases
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("your_"):
        return {"exclude": False, "reason": ""}

    try:
        from openai import OpenAI
        # Use SSL workaround for environments with certificate issues
        http_client = httpx.Client(verify=False)
        client = OpenAI(api_key=api_key, http_client=http_client)

        prompt = f"""Classify this business. Should it be EXCLUDED from an investment analysis?

EXCLUDE these types (be strict):
- Dental/Orthodontic practices
- Law firms/Attorney practices
- CPA/Accounting/Bookkeeping firms
- Medical centers/Clinics/Healthcare practices
- ANY food/beverage business: restaurants, cafes, pizzerias, fast food, ethnic food (Indian, Chinese, Thai, etc.), juice bars, coffee shops, bakeries, delis, food trucks, bars, breweries, ice cream shops, sandwich shops, etc.

Business: {name}
Description: {description[:300] if description else 'No description'}

Respond JSON only: {{"exclude": true/false, "reason": "brief reason if excluded"}}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Classify businesses. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=80
        )

        content = response.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```json")[-1].split("```")[0].strip()

        return json.loads(content)

    except Exception:
        # AI failed, but we already checked keywords above
        return {"exclude": False, "reason": ""}


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


@dataclass
class SkippedListing:
    """Data class to track skipped listings and reasons."""
    url: str
    name: str
    reason: str  # e.g., "Excluded business type (restaurant)", "No cash flow data"


class BizBuySellScraper:
    """Scraper for BizBuySell website."""

    BASE_URL = "https://m.bizbuysell.com"  # Use mobile site for better scraping

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

    def _to_mobile_url(self, url: str) -> str:
        """Convert any BizBuySell URL to mobile version for better scraping."""
        return url.replace('www.bizbuysell.com', 'm.bizbuysell.com')

    def _get_paginated_url(self, base_url: str, page_num: int) -> str:
        """
        Generate a paginated URL for BizBuySell.

        URL pattern:
        - Page 1: /california/businesses-for-sale/?q=... OR /california/businesses-for-sale/1/?q=...
        - Page 2: /california/businesses-for-sale/2/?q=...

        The page number goes before the query string.
        """
        # Split URL into path and query
        if '?' in base_url:
            path, query = base_url.split('?', 1)
            query = '?' + query
        else:
            path = base_url
            query = ''

        # Remove trailing slash for easier manipulation
        path = path.rstrip('/')

        # Check if path already has a page number at the end
        path_parts = path.split('/')
        if path_parts and path_parts[-1].isdigit():
            # Replace existing page number
            path_parts[-1] = str(page_num)
            new_path = '/'.join(path_parts)
        else:
            # Append page number
            new_path = f"{path}/{page_num}"

        return new_path + '/' + query

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

    def _fetch_url(self, url: str) -> str:
        """Fetch URL with ScraperAPI support for bypassing blocks."""
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

    def fetch_full_description(self, listing_url: str) -> str:
        """
        Fetch the full description from the mobile site.
        The mobile site has complete descriptions in JSON-LD data.
        """
        if not self.scraper_api_key:
            return ""

        # Convert desktop URL to mobile URL
        mobile_url = self._to_mobile_url(listing_url)

        try:
            params = {
                'api_key': self.scraper_api_key,
                'url': mobile_url,
                'render': 'false'
            }
            api_url = f"http://api.scraperapi.com?{urlencode(params)}"

            session = requests.Session()
            session.trust_env = False  # Disable proxy from env

            response = session.get(api_url, timeout=35)

            if response.status_code != 200 or len(response.text) < 1000:
                return ""

            soup = BeautifulSoup(response.text, 'lxml')

            # Get description from JSON-LD (most reliable source)
            for script in soup.find_all('script', type='application/ld+json'):
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict) and 'description' in data:
                        desc = data['description']
                        if len(desc) > 100 and not desc.rstrip().endswith('...'):
                            # Replace newlines with spaces to preserve word separation
                            desc = re.sub(r'\s*\n\s*', ' ', desc)
                            # Clean up multiple spaces
                            desc = re.sub(r' +', ' ', desc)
                            return desc.strip()
                except:
                    pass

            return ""
        except Exception:
            return ""

    def scrape_single_listing(self, listing_url: str) -> Optional[BusinessData]:
        """
        Scrape a single business listing page directly.

        Args:
            listing_url: Direct URL to a business listing (e.g., /business-opportunity/...)

        Returns:
            BusinessData object with full details, or None if scraping fails
        """
        if not self.scraper_api_key:
            return None

        try:
            # Convert to mobile URL for better data extraction
            mobile_url = self._to_mobile_url(listing_url)

            params = {
                'api_key': self.scraper_api_key,
                'url': mobile_url,
                'render': 'false'
            }
            api_url = f"http://api.scraperapi.com?{urlencode(params)}"

            session = requests.Session()
            session.trust_env = False

            response = session.get(api_url, timeout=45)

            if response.status_code != 200 or len(response.text) < 1000:
                return None

            soup = BeautifulSoup(response.text, 'lxml')
            page_text = soup.get_text(separator=' | ', strip=True)

            # Initialize business data
            data = BusinessData(url=listing_url, name="Unknown Business")

            # Extract data from JSON-LD (most reliable source)
            for script in soup.find_all('script', type='application/ld+json'):
                try:
                    json_data = json.loads(script.string)
                    if isinstance(json_data, dict):
                        if 'name' in json_data:
                            data.name = json_data['name']
                        if 'description' in json_data:
                            desc = json_data['description']
                            desc = re.sub(r'\s*\n\s*', ' ', desc)
                            desc = re.sub(r' +', ' ', desc)
                            data.description = desc.strip()
                        if 'address' in json_data:
                            addr = json_data['address']
                            if isinstance(addr, dict):
                                city = addr.get('addressLocality', '')
                                state = addr.get('addressRegion', '')
                                if city and state:
                                    data.location = f"{city}, {state}"
                        # Try to extract price from JSON-LD (Product schema)
                        if 'offers' in json_data:
                            offers = json_data['offers']
                            if isinstance(offers, dict) and 'price' in offers:
                                try:
                                    data.asking_price = float(offers['price'])
                                except (ValueError, TypeError):
                                    pass
                        elif 'price' in json_data:
                            try:
                                data.asking_price = float(str(json_data['price']).replace(',', '').replace('$', ''))
                            except (ValueError, TypeError):
                                pass
                except:
                    pass

            # Extract financial data from page text using multiple patterns
            # Asking price - try multiple patterns
            price_patterns = [
                r'asking\s*price[:\s]*\$?([\d,]+)',
                r'\$\s*([\d,]+)\s*(?:asking|list)',
                r'price[:\s]*\$?([\d,]+)',
                r'listed\s*(?:at|for)[:\s]*\$?([\d,]+)',
            ]
            for pattern in price_patterns:
                price_match = re.search(pattern, page_text, re.IGNORECASE)
                if price_match:
                    data.asking_price = self._parse_currency(price_match.group(1))
                    break

            # Cash flow - try multiple patterns
            cf_patterns = [
                r'cash\s*flow[:\s]*\$?([\d,]+)',
                r'cf[:\s]*\$?([\d,]+)',
                r'discretionary\s*(?:earnings?|cash)[:\s]*\$?([\d,]+)',
                r'sde[:\s]*\$?([\d,]+)',
                r'owner(?:\'?s?)?\s*(?:benefit|earnings?)[:\s]*\$?([\d,]+)',
            ]
            for pattern in cf_patterns:
                cf_match = re.search(pattern, page_text, re.IGNORECASE)
                if cf_match:
                    data.cash_flow = self._parse_currency(cf_match.group(1))
                    break

            # Gross revenue - try multiple patterns
            rev_patterns = [
                r'gross\s*revenue[:\s]*\$?([\d,]+)',
                r'revenue[:\s]*\$?([\d,]+)',
                r'gross\s*sales[:\s]*\$?([\d,]+)',
                r'annual\s*(?:sales|revenue)[:\s]*\$?([\d,]+)',
                r'sales[:\s]*\$?([\d,]+)',
            ]
            for pattern in rev_patterns:
                rev_match = re.search(pattern, page_text, re.IGNORECASE)
                if rev_match:
                    data.gross_revenue = self._parse_currency(rev_match.group(1))
                    break

            # If still no financial data, try to find labeled dollar amounts in structured format
            # BizBuySell often uses format like "Asking Price | $XXX,XXX" or "Asking Price$XXX,XXX"
            if not data.asking_price:
                alt_price = re.search(r'asking\s*price\s*\|?\s*\$?([\d,]+)', page_text, re.IGNORECASE)
                if alt_price:
                    data.asking_price = self._parse_currency(alt_price.group(1))

            if not data.cash_flow:
                alt_cf = re.search(r'cash\s*flow\s*\|?\s*\$?([\d,]+)', page_text, re.IGNORECASE)
                if alt_cf:
                    data.cash_flow = self._parse_currency(alt_cf.group(1))

            if not data.gross_revenue:
                alt_rev = re.search(r'(?:gross\s*)?revenue\s*\|?\s*\$?([\d,]+)', page_text, re.IGNORECASE)
                if alt_rev:
                    data.gross_revenue = self._parse_currency(alt_rev.group(1))

            # Last resort: look for dollar amounts near keywords
            if not data.asking_price or not data.cash_flow:
                # Find all dollar amounts
                all_amounts = re.findall(r'\$([\d,]+)', page_text)
                parsed_amounts = [self._parse_currency(a) for a in all_amounts if self._parse_currency(a)]

                if parsed_amounts and not data.asking_price:
                    # The largest amount near the top is likely the asking price
                    # Filter for reasonable asking prices (> $10k)
                    valid_prices = [a for a in parsed_amounts if a >= 10000]
                    if valid_prices:
                        # Look for a value that appears multiple times (common for asking price)
                        from collections import Counter
                        counts = Counter(valid_prices)
                        most_common = counts.most_common(1)
                        if most_common and most_common[0][1] >= 2:
                            data.asking_price = most_common[0][0]
                        elif valid_prices:
                            # Otherwise take the first large value
                            data.asking_price = valid_prices[0]

            # Category/Industry
            cat_match = re.search(r'(?:category|industry|type)[:\s]*([A-Za-z][A-Za-z\s&/]+?)(?:\s*\||$)', page_text, re.IGNORECASE)
            if cat_match:
                data.category = cat_match.group(1).strip()

            # Employees
            emp_match = re.search(r'(?:employees?|# of employees?)[:\s]*(\d+)', page_text, re.IGNORECASE)
            if emp_match:
                data.employees = int(emp_match.group(1))

            # Year established
            year_match = re.search(r'(?:established|year established|founded)[:\s]*(\d{4})', page_text, re.IGNORECASE)
            if year_match:
                data.year_established = int(year_match.group(1))

            return data

        except Exception as e:
            print(f"Error scraping single listing: {e}")
            return None

    def get_listings_from_search_page(self, parent_url: str, limit: int = 10, filter_excluded: bool = True, filter_callback=None) -> tuple[list[BusinessData], list[SkippedListing]]:
        """
        Extract business data directly from search results page (faster, more reliable).
        Supports pagination - automatically fetches additional pages if needed.
        Filters out excluded business types (dental, legal, medical, restaurants, etc.)

        Args:
            parent_url: URL of the BizBuySell search results page
            limit: Maximum number of listings to return
            filter_excluded: Whether to filter out excluded business types
            filter_callback: Optional callback(name, total_checked, excluded_count) for progress

        Returns:
            Tuple of (List of BusinessData objects, List of SkippedListing objects)
        """
        results = []
        skipped = []  # Track skipped listings with reasons
        seen_urls = set()
        seen_names = set()  # Track business names to avoid duplicates
        excluded_count = 0  # Business type exclusions (restaurants, dental, etc.)
        no_cashflow_count = 0  # Skipped due to missing cash flow
        total_checked = 0
        current_page = 1
        max_pages = 10  # Safety limit to prevent infinite loops
        consecutive_empty_pages = 0

        while len(results) < limit and current_page <= max_pages:
            # Generate paginated URL
            if current_page == 1:
                # First page - use original URL (might or might not have /1/)
                page_url = parent_url
            else:
                page_url = self._get_paginated_url(parent_url, current_page)

            # Always use mobile URL for better scraping reliability
            mobile_url = self._to_mobile_url(page_url)

            try:
                html = self._fetch_url(mobile_url)
            except Exception as e:
                if current_page == 1:
                    raise Exception(f"Failed to fetch search page: {e}")
                else:
                    # Failed to fetch subsequent page, stop pagination
                    break

            soup = BeautifulSoup(html, 'lxml')

            # BizBuySell uses Angular components - try multiple selectors
            listing_cards = soup.select('app-listing-diamond, app-listing-basic')

            # Fallback: find parent containers of business-opportunity links
            if not listing_cards:
                for link in soup.find_all('a', href=lambda h: h and '/business-opportunity/' in h):
                    parent = link.find_parent(['app-listing-diamond', 'app-listing-basic', 'div', 'article'])
                    if parent and parent not in listing_cards:
                        listing_cards.append(parent)

            if not listing_cards:
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= 2:
                    # No more pages available
                    break
                current_page += 1
                continue

            consecutive_empty_pages = 0
            cards_processed_this_page = 0

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

                    # Skip duplicate URLs (important for pagination)
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    # Get all text from the card
                    card_text = card.get_text(separator=' | ', strip=True)

                    # Extract name/title - usually first part before location
                    parts = card_text.split(' | ')
                    name = parts[0] if parts else "Unknown Business"

                    # Skip duplicate business names (same listing appearing multiple times)
                    name_normalized = name.lower().strip()
                    if name_normalized in seen_names:
                        continue
                    seen_names.add(name_normalized)

                    # Extract description for filtering
                    description = ""
                    if len(parts) > 2:
                        for part in parts[2:]:
                            if len(part) > 50 and not part.startswith('$'):
                                description = part[:2000]
                                break

                    total_checked += 1
                    cards_processed_this_page += 1

                    # Filter out excluded business types
                    if filter_excluded:
                        if filter_callback:
                            total_skipped = excluded_count + no_cashflow_count
                            filter_callback(name[:30], total_checked, total_skipped)

                        classification = _classify_with_ai(name, description)
                        if classification.get("exclude", False):
                            reason = classification.get("reason", "Excluded business type")
                            skipped.append(SkippedListing(url=url, name=name, reason=f"Excluded: {reason}"))
                            excluded_count += 1
                            continue

                    data = BusinessData(url=url, name=name)
                    data.description = description

                    # Extract location (usually "City, ST" format)
                    loc_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2})', card_text)
                    if loc_match:
                        data.location = loc_match.group(1)

                    # First, find cash flow if labeled
                    cf_match = re.search(r'cash\s*flow(?:\s*\([^)]+\))?[:\s]*\$?([\d,]+)', card_text, re.IGNORECASE)
                    cash_flow_value = None
                    if cf_match:
                        cash_flow_value = self._parse_currency(cf_match.group(1))

                    # Find all dollar amounts
                    price_matches = re.findall(r'\$([\d,]+)', card_text)
                    all_amounts = [self._parse_currency(p) for p in price_matches if self._parse_currency(p)]

                    # Determine asking price
                    asking_price_match = re.search(r'asking\s*price[:\s]*\$?([\d,]+)', card_text, re.IGNORECASE)
                    if asking_price_match:
                        data.asking_price = self._parse_currency(asking_price_match.group(1))
                    elif all_amounts:
                        price_candidates = [a for a in all_amounts if a != cash_flow_value]

                        if price_candidates:
                            from collections import Counter
                            counts = Counter(price_candidates)
                            most_common = counts.most_common(1)[0]

                            if most_common[1] >= 2:
                                data.asking_price = most_common[0]
                            else:
                                data.asking_price = price_candidates[-1]
                        elif all_amounts:
                            data.asking_price = all_amounts[0]

                    # Set cash flow (with sanity check)
                    if cash_flow_value:
                        if data.asking_price and cash_flow_value <= data.asking_price * 2:
                            data.cash_flow = cash_flow_value
                        elif not data.asking_price:
                            data.cash_flow = cash_flow_value

                    # Extract revenue
                    rev_match = re.search(r'(?:gross\s*)?revenue[:\s]*\$?([\d,]+)', card_text, re.IGNORECASE)
                    if rev_match:
                        data.gross_revenue = self._parse_currency(rev_match.group(1))

                    # Extract employee count
                    emp_match = re.search(r'(?:#\s*)?employees?[:\s]*(\d+)', card_text, re.IGNORECASE)
                    if emp_match:
                        try:
                            data.employees = int(emp_match.group(1))
                        except ValueError:
                            pass

                    # Extract description
                    if len(parts) > 2:
                        for part in parts[2:]:
                            if len(part) > 50 and not part.startswith('$'):
                                data.description = part[:2000]
                                break

                    # Skip listings without cash flow (required for ROI calculation)
                    if not data.cash_flow:
                        skipped.append(SkippedListing(url=url, name=name, reason="No cash flow data"))
                        no_cashflow_count += 1
                        continue

                    results.append(data)

                except Exception:
                    continue

            # Move to next page if we need more results
            if len(results) >= limit:
                break

            current_page += 1
            time.sleep(0.5)  # Small delay between page requests

        return results, skipped


def scrape_listings(
    parent_url: str,
    limit: int = 10,
    progress_callback=None,
    scraper_api_key: str = None,
    filter_excluded: bool = True,
    fetch_full_details: bool = True  # Fetch full descriptions from mobile site
) -> tuple[list[BusinessData], list[SkippedListing]]:
    """
    Main function to scrape multiple business listings.

    Uses fast extraction from search results page, then optionally fetches
    full descriptions from individual listing pages.

    Automatically filters out excluded business types:
    - Dental/Orthodontic practices
    - Law firms/Legal practices
    - CPA/Accounting firms
    - Medical centers/Clinics
    - Restaurants/Cafes/Food service

    Args:
        parent_url: URL of the search results page
        limit: Maximum number of listings to scrape
        progress_callback: Optional callback function(current, total, message)
        scraper_api_key: Optional ScraperAPI key for bypassing blocks
        filter_excluded: Whether to filter out excluded business types (default True)
        fetch_full_details: Whether to fetch full descriptions from listing pages (default True)

    Returns:
        Tuple of (List of BusinessData objects, List of SkippedListing objects)
    """
    scraper = BizBuySellScraper(scraper_api_key=scraper_api_key)

    if progress_callback:
        progress_callback(0, limit, "Fetching search results...")

    # Create filter callback for progress updates
    def filter_callback(name, total_checked, excluded_count):
        if progress_callback:
            progress_callback(0, limit, f"Filtering: checked {total_checked}, skipped {excluded_count} (checking '{name}...')")

    # Use fast extraction from search page (individual pages timeout)
    results, skipped = scraper.get_listings_from_search_page(
        parent_url,
        limit,
        filter_excluded=filter_excluded,
        filter_callback=filter_callback
    )

    if not results:
        raise Exception("No business listings found on the page. The page structure may have changed or all listings were filtered out.")

    # Fetch full descriptions from individual listing pages (parallel)
    if fetch_full_details and scraper_api_key:
        completed = 0
        total = len(results)

        def fetch_description(business):
            """Fetch full description for a single business."""
            full_desc = scraper.fetch_full_description(business.url)
            if full_desc and len(full_desc) > len(business.description or ""):
                business.description = full_desc
            return business

        if progress_callback:
            progress_callback(0, total, f"Fetching details (0/{total})...")

        # Use parallel fetching with max 5 concurrent requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_description, b): b for b in results}

            for future in as_completed(futures):
                completed += 1
                if progress_callback:
                    progress_callback(completed, total, f"Fetching details ({completed}/{total})...")

    if progress_callback:
        progress_callback(len(results), len(results), f"Found {len(results)} qualifying listings!")

    return results, skipped


def is_direct_listing_url(url: str) -> bool:
    """Check if the URL is a direct business listing (not a search page)."""
    return '/business-opportunity/' in url.lower()


def scrape_single_listing(
    listing_url: str,
    progress_callback=None,
    scraper_api_key: str = None
) -> Optional[BusinessData]:
    """
    Scrape a single business listing page.

    Args:
        listing_url: Direct URL to a business listing
        progress_callback: Optional callback function(current, total, message)
        scraper_api_key: ScraperAPI key for bypassing blocks

    Returns:
        BusinessData object with full details, or None if scraping fails
    """
    if not scraper_api_key:
        raise Exception("ScraperAPI key is required for scraping")

    scraper = BizBuySellScraper(scraper_api_key=scraper_api_key)

    if progress_callback:
        progress_callback(0, 1, "Fetching business listing...")

    business = scraper.scrape_single_listing(listing_url)

    if not business:
        raise Exception("Failed to scrape business listing. The page may be unavailable.")

    if progress_callback:
        progress_callback(1, 1, f"Fetched: {business.name}")

    return business
