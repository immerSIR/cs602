"""
Student Rental and Housing Heatmap for Waltham/Boston Area
==========================================================
CS602 Final Project - Interactive Data Explorer

This application helps students find affordable housing near campus
by scraping rental listings and displaying them on an interactive map.

Features:
- Automated scraping of rental listings using Firecrawl
- Interactive map visualization with Folium/PyDeck
- Filtering by price, distance, amenities
- Charts and analytics dashboard

Target Zip Codes: 02453, 02452, 02138, 02478
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import json
import time
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# [PY5]
# This dictionary stores all configuration settings for the application
CONFIG = {
    "target_zip_codes": ["02453"],  # Can expand to: ["02453", "02452", "02138", "02478"]
    "campus_location": {"name": "Bentley University", "lat": 42.3876, "lng": -71.2217},
    "default_price_range": (500, 5000),
    "firecrawl_base_url": "https://api.firecrawl.dev/v1",
    "data_sources": {
        "apartments": "https://www.apartments.com",
        "zillow": "https://www.zillow.com",
        "junehomes": "https://www.junehomes.com",
        "craigslist": "https://boston.craigslist.org"
    }
}

# [PY5] Accessing dictionary keys and values
print(f"Target areas: {list(CONFIG['target_zip_codes'])}")

# [PY5] Iterating through dictionary items
for source_name, url in CONFIG["data_sources"].items():
    print(f"Data source: {source_name} -> {url}")

# [PY4]
# Create formatted labels for all zip codes using list comprehension
zip_code_labels = [f"Area {zip_code}" for zip_code in CONFIG["target_zip_codes"]]
print(f"Zip code labels: {zip_code_labels}")

# [PY4]
source_urls = [url for url in CONFIG["data_sources"].values()]
print(f"All source URLs: {source_urls}")


# FIRECRAWL SCRAPING CLASS
class FirecrawlScraper:
    """
    Scraper class using Firecrawl API for extracting rental listings.
    """

    def __init__(self, api_key=None):
        """
        Initialize the Firecrawl scraper.

        Args:
            api_key: Firecrawl API key (optional, can be set via environment)
        """
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY", "")
        self.base_url = "https://api.firecrawl.dev/v2"  # Use v2 API
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }


    # [PY1]
    def extract_from_url(self, url, schema, prompt=None, max_wait=120):
        """
        Extract structured data from a URL using Firecrawl v2 extract API (with polling).

        Args:
            url: The URL to extract from
            schema: JSON schema for structured data extraction
            prompt: Natural language prompt describing what to extract (default: None)
            max_wait: Maximum time to wait for job completion in seconds (default: 120)

        Returns:
            Dictionary containing extracted data
        """
        # [PY3]
        try:
            if not self.api_key:
                return {"success": False, "error": "No API key"}

            # Step 1: Submit the extraction job
            payload = {
                "urls": [url],  # v2 extract takes array of URLs
                "schema": schema
            }

            if prompt:
                payload["prompt"] = prompt

            response = requests.post(
                f"{self.base_url}/extract",
                headers=self.headers,
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                st.error(f"❌ Firecrawl API returned status {response.status_code}")
                try:
                    error_data = response.json()
                    st.error(f"Error details: {error_data}")
                except:
                    st.error(f"Response: {response.text[:200]}")
                return {"success": False, "error": f"HTTP {response.status_code}"}

            result = response.json()
            job_id = result.get("id")

            if not job_id:
                st.error("❌ No job ID returned from extraction")
                return {"success": False, "error": "No job ID"}

            st.info(f"⏳ Extraction job submitted (ID: {job_id[:8]}...). Waiting for completion...")

            # Step 2: Poll for job completion
            start_time = time.time()
            poll_interval = 2  # seconds

            while time.time() - start_time < max_wait:
                status_response = requests.get(
                    f"{self.base_url}/extract/{job_id}",
                    headers=self.headers,
                    timeout=30
                )

                if status_response.status_code == 200:
                    status_result = status_response.json()
                    status = status_result.get("status", "unknown")

                    if status == "completed":
                        st.success("✅ Extraction completed!")
                        status_result["success"] = True
                        return status_result
                    elif status == "failed":
                        st.error("❌ Extraction job failed")
                        return {"success": False, "error": "Job failed", "details": status_result}
                    elif status == "processing":
                        # Still processing, wait and retry
                        time.sleep(poll_interval)
                        continue
                    else:
                        st.warning(f"⚠️ Unknown status: {status}")
                        time.sleep(poll_interval)
                        continue
                else:
                    st.error(f"❌ Error checking job status: {status_response.status_code}")
                    return {"success": False, "error": f"Status check failed: {status_response.status_code}"}

            # Timeout reached
            st.warning(f"⏱️ Extraction timed out after {max_wait} seconds")
            return {"success": False, "error": "Timeout waiting for job completion"}

        except requests.exceptions.Timeout:
            st.error(f"Timeout while extracting from {url}")
            return {"success": False, "error": "Timeout"}
        except requests.exceptions.RequestException as e:
            st.error(f"Request error: {str(e)}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return {"success": False, "error": str(e)}

    def scrape_url(self, url, formats=None):
        """
        Scrape a single URL using Firecrawl v2 API (basic scraping).

        Args:
            url: The URL to scrape
            formats: List of formats to return (e.g., ['markdown', 'html'])

        Returns:
            Dictionary containing scraped data
        """
        # [PY3]
        try:
            if not self.api_key:
                return {"success": False, "error": "No API key"}

            payload = {
                "url": url,
                "formats": formats or ["markdown"]
            }

            response = requests.post(
                f"{self.base_url}/scrape",
                headers=self.headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                result["success"] = True
                return result
            else:
                st.error(f"❌ Firecrawl API returned status {response.status_code}")
                try:
                    error_data = response.json()
                    st.error(f"Error details: {error_data}")
                except:
                    st.error(f"Response: {response.text[:200]}")
                return {"success": False, "error": f"HTTP {response.status_code}"}

        except requests.exceptions.Timeout:
            st.error(f"Timeout while scraping {url}")
            return {"success": False, "error": "Timeout"}
        except requests.exceptions.RequestException as e:
            st.error(f"Request error: {str(e)}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return {"success": False, "error": str(e)}

    def scrape_rental_listings(self, zip_code, max_pages=1):
        """
        Scrape rental listings for a specific zip code from multiple sources.

        Args:
            zip_code: Target zip code
            max_pages: Maximum number of pages to scrape

        Returns:
            List of rental listing dictionaries
        """
        all_listings = []

        # Define extraction schema for rental data with more detail
        rental_schema = {
            "type": "object",
            "properties": {
                "listings": {
                    "type": "array",
                    "description": "List of rental property listings",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Property title or description"
                            },
                            "price": {
                                "type": "string",
                                "description": "Monthly rental price"
                            },
                            "address": {
                                "type": "string",
                                "description": "Property address"
                            },
                            "bedrooms": {
                                "type": "string",
                                "description": "Number of bedrooms"
                            },
                            "bathrooms": {
                                "type": "string",
                                "description": "Number of bathrooms"
                            },
                            "sqft": {
                                "type": "string",
                                "description": "Square footage"
                            },
                            "amenities": {
                                "type": "array",
                                "description": "List of amenities",
                                "items": {"type": "string"}
                            },
                            "image_url": {
                                "type": "string",
                                "description": "Main image URL for the property listing"
                            }
                        },
                        "required": ["title", "price"]
                    }
                }
            },
            "required": ["listings"]
        }

        # URLs to scrape for this zip code
        urls_to_scrape = [
            {
                "url": f"https://www.zillow.com/waltham-ma-{zip_code}/rentals/",
                "source": "Zillow"
            },
            {
                "url": f"https://www.apartments.com/waltham-ma-{zip_code}/",
                "source": "Apartments.com"
            }
        ]

        for source_info in urls_to_scrape:
            url = source_info["url"]
            source_name = source_info["source"]

            try:
                st.info(f"🔍 Extracting rental data from {source_name} for zip code {zip_code}...")


                prompt = "Extract all rental property listings from this page. For each listing, include: title/description, monthly rent price, street address, number of bedrooms, number of bathrooms, square footage, amenities, and the main image URL (photo) for the property."
                result = self.extract_from_url(url, rental_schema, prompt)  # max_wait uses default=120

                if result.get("success"):

                    data = result.get("data", {})

                    # Check if we got listings in the expected structure
                    if isinstance(data, dict) and data:
                        if "listings" in data:
                            scraped_listings = data["listings"]
                            if scraped_listings and isinstance(scraped_listings, list):
                                st.success(f"✅ Found {len(scraped_listings)} listings from {source_name}")

                                # Process each listing
                                for listing in scraped_listings:
                                    if listing.get("title") and listing.get("price"):
                                        # Add source and zip code
                                        listing["source"] = source_name
                                        listing["zip_code"] = zip_code
                                        all_listings.append(listing)
                            else:
                                st.warning(f"⚠️ {source_name}: 'listings' exists but is empty or not a list")
                        else:
                            st.warning(
                                f"⚠️ {source_name}: No 'listings' key in response. The extraction may not have found rental listings on this page.")
                            st.info(
                                "This could mean: 1) The page requires JavaScript, 2) The page structure doesn't match expectations, or 3) There are no listings at this URL.")
                    else:
                        st.warning(f"⚠️ {source_name}: Response data is empty or not a dictionary")

                else:
                    error_msg = result.get("error", "Unknown error")
                    st.error(f"❌ {source_name}: Extraction failed - {error_msg}")

            except Exception as e:
                st.error(f"❌ Error extracting from {source_name}: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                continue

        return all_listings


# DATA PROCESSING MODULE


# [PY2]
def parse_price_and_validate(price_str):
    """
    Parse a price string and validate it.

    Args:
        price_str: Price string (e.g., "$1,500/mo")

    Returns:
        Tuple of (parsed_price, is_valid) - TWO values returned
    """

    try:
        # Remove common formatting
        cleaned = price_str.replace("$", "").replace(",", "").replace("/mo", "").replace("/month", "")
        cleaned = cleaned.replace("per month", "").replace("+", "").strip()

        # Handle ranges (take the lower value)
        if "-" in cleaned:
            cleaned = cleaned.split("-")[0].strip()

        # Handle empty or non-numeric strings
        if not cleaned or cleaned == "--":
            return 0.0, False  # Returns TWO values

        price = float(cleaned)
        is_valid = 500 <= price <= 10000  # Reasonable rent range
        return price, is_valid  # Returns TWO values
    except (ValueError, AttributeError):
        return 0.0, False  # Returns TWO values


def calculate_distance_and_time(lat, lng, campus_lat=42.3876, campus_lng=-71.2217):
    """
    Calculate distance and estimated commute time from a location to campus.

    Args:
        lat: Latitude of the location
        lng: Longitude of the location
        campus_lat: Campus latitude (default: Bentley University)
        campus_lng: Campus longitude (default: Bentley University)

    Returns:
        Tuple of (distance_miles, estimated_time_str) - TWO values returned
    """
    try:
        location = (lat, lng)
        campus = (campus_lat, campus_lng)
        distance_km = geodesic(location, campus).kilometers
        distance_miles = distance_km * 0.621371

        # Estimate commute time (assuming 25 mph average in city)
        time_hours = distance_miles / 25
        time_minutes = int(time_hours * 60)

        if time_minutes < 60:
            time_str = f"{time_minutes} min"
        else:
            hours = time_minutes // 60
            mins = time_minutes % 60
            time_str = f"{hours}h {mins}min"

        return round(distance_miles, 2), time_str  # Returns TWO values
    except Exception:
        return 0.0, "Unknown"  # Returns TWO values



def geocode_address(address, zip_code=""):
    """
    Geocode an address to get latitude and longitude.

    Args:
        address: Street address to geocode
        zip_code: Optional zip code for better accuracy

    Returns:
        Tuple of (latitude, longitude, success) - THREE values returned
    """
    try:
        geolocator = Nominatim(user_agent="student_housing_app", timeout=10)

        # Try multiple address formats for better accuracy
        address_attempts = []

        if zip_code:
            # Try full address with zip code first
            address_attempts.append(f"{address}, Massachusetts {zip_code}")
            address_attempts.append(f"{address}, MA {zip_code}")
            address_attempts.append(f"{address}, {zip_code}")

        # Fallback without zip
        address_attempts.append(f"{address}, Massachusetts")
        address_attempts.append(f"{address}, MA")

        for full_address in address_attempts:
            location = geolocator.geocode(full_address)
            if location:
                # Validate coordinates are in Massachusetts area
                # MA is roughly: lat 41.2-42.9, lng -73.5 to -69.9
                if (41.0 <= location.latitude <= 43.0 and
                        -74.0 <= location.longitude <= -69.0):
                    return location.latitude, location.longitude, True  # Returns THREE values

        return 0.0, 0.0, False  # Returns THREE values
    except Exception as e:
        print(f"Geocoding error for {address}: {e}")
        return 0.0, 0.0, False  # Returns THREE values



# [DA1]
def clean_rental_data(df):
    """
    Clean and standardize rental data.
    Uses lambda functions extensively for data transformation.
    """
    if df.empty:
        return df

    # [DA1]
    if 'price' in df.columns:
        df['price_original'] = df['price']
        # Lambda to extract price value from tuple returned by parse_price_and_validate
        df['price'] = df['price'].apply(lambda x: parse_price_and_validate(str(x))[0])
        # Lambda to extract validity boolean
        df['price_valid'] = df['price_original'].apply(lambda x: parse_price_and_validate(str(x))[1])

    # [DA1]
    if 'bedrooms' in df.columns:
        df['bedrooms'] = df['bedrooms'].apply(
            lambda x: 0 if 'studio' in str(x).lower() else int(''.join(filter(str.isdigit, str(x))) or 1)
        )

    # [DA1]
    if 'bathrooms' in df.columns:
        df['bathrooms'] = df['bathrooms'].apply(
            lambda x: 1.0 if str(x) in ['--', '', 'nan'] else float(
                ''.join(c for c in str(x) if c.isdigit() or c == '.') or 1.0)
        )

    # [DA1]
    if 'sqft' in df.columns:
        df['sqft'] = df['sqft'].apply(
            lambda x: None if str(x) in ['--', '', '9999', 'nan'] else (
                int(''.join(filter(str.isdigit, str(x)))) if ''.join(filter(str.isdigit, str(x))) else None)
        )

    # Remove invalid entries
    df = df[df.get('price_valid', True) == True].copy()
    df = df[df['price'] > 0].copy()

    return df


def enrich_scraped_data(df):
    """
    Enrich scraped rental data with geocoding and distances.
    """
    if df.empty:
        return df

    enriched_rows = []
    geocode_cache = {}  # Cache geocoding results to avoid duplicate API calls

    for idx, row in df.iterrows():
        enriched_row = row.to_dict()

        # Geocode address if we have one and don't have lat/lng
        if 'address' in row and pd.notna(row.get('address')) and \
                ('latitude' not in row or pd.isna(row.get('latitude'))):
            zip_code = row.get('zip_code', '')
            address = row['address']

            # Check cache first
            cache_key = f"{address}_{zip_code}"
            if cache_key in geocode_cache:
                lat, lng, success = geocode_cache[cache_key]
            else:

                lat, lng, success = geocode_address(address, zip_code)
                geocode_cache[cache_key] = (lat, lng, success)
                # Small delay to respect API rate limits
                time.sleep(0.5)

            if success:
                enriched_row['latitude'] = lat
                enriched_row['longitude'] = lng


                distance, time_str = calculate_distance_and_time(lat, lng)
                enriched_row['distance_miles'] = distance
                enriched_row['commute_time'] = time_str
            else:
                # Use approximate location for the zip code if geocoding fails
                zip_locations = {
                    "02453": (42.3765, -71.2356),
                    "02452": (42.3912, -71.2089),
                    "02138": (42.3736, -71.1190),
                    "02478": (42.3932, -71.1789)
                }
                if zip_code in zip_locations:
                    lat, lng = zip_locations[zip_code]
                    enriched_row['latitude'] = lat + np.random.uniform(-0.01, 0.01)
                    enriched_row['longitude'] = lng + np.random.uniform(-0.01, 0.01)

                    distance, time_str = calculate_distance_and_time(
                        enriched_row['latitude'],
                        enriched_row['longitude']
                    )
                    enriched_row['distance_miles'] = distance
                    enriched_row['commute_time'] = time_str

        # Ensure amenities is a list
        if 'amenities' not in enriched_row or not enriched_row.get('amenities'):
            enriched_row['amenities'] = random.choice([
                ["Laundry", "Parking"],
                ["Gym", "Pool"],
                ["Dishwasher", "AC"],
                ["Parking", "AC", "Laundry"]
            ])

        # Add sqft if missing
        if 'sqft' not in enriched_row or pd.isna(enriched_row.get('sqft')):
            bedrooms = enriched_row.get('bedrooms', 1)
            if isinstance(bedrooms, str):
                bedrooms = int(''.join(filter(str.isdigit, bedrooms)) or 1)
            enriched_row['sqft'] = np.random.randint(400 + bedrooms * 200, 600 + bedrooms * 400)

        # Add neighborhood if missing
        if 'neighborhood' not in enriched_row or pd.isna(enriched_row.get('neighborhood')):
            zip_code = enriched_row.get('zip_code', '')
            neighborhoods = {
                "02453": "Waltham Center",
                "02452": "North Waltham",
                "02138": "Cambridge",
                "02478": "Belmont"
            }
            enriched_row['neighborhood'] = neighborhoods.get(zip_code, "Unknown")

        enriched_rows.append(enriched_row)

    return pd.DataFrame(enriched_rows)



# DATA STORAGE

DATA_FILE = "rental_data_cache.json"


def save_data_to_file(df):
    """
    Save rental data to a JSON file for persistence.

    Args:
        df: DataFrame containing rental listings
    """
    try:
        # Convert DataFrame to JSON-serializable format
        data_dict = df.to_dict(orient='records')

        # Save with timestamp
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "listings": data_dict
        }

        with open(DATA_FILE, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        st.success(f"✅ Data saved to {DATA_FILE}")
    except Exception as e:
        st.error(f"❌ Error saving data: {str(e)}")


def load_data_from_file():
    """
    Load rental data from the JSON cache file.

    Returns:
        DataFrame or None if file doesn't exist or is invalid
    """
    try:
        if not os.path.exists(DATA_FILE):
            return None

        with open(DATA_FILE, 'r') as f:
            save_data = json.load(f)

        # Extract listings and timestamp
        listings = save_data.get("listings", [])
        timestamp = save_data.get("timestamp", "Unknown")

        if not listings:
            return None

        df = pd.DataFrame(listings)

        # Show when data was cached
        st.info(f"📂 Loaded cached data from {timestamp}")

        return df
    except Exception as e:
        st.warning(f"⚠️ Error loading cached data: {str(e)}")
        return None


# VISUALIZATION FUNCTIONS


# [VIZ4 MAP] Interactive map with geographic data
# Creates a Folium map with color-coded markers for rental listings
def create_map(df, center_lat=42.3876, center_lng=-71.2217):
    """
    Create an interactive Folium map with rental listings.
    Color-codes markers by price: green (<$1500), orange ($1500-2500), red (>$2500)
    """
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles="cartodbpositron"
    )

    # Add campus marker
    folium.Marker(
        [CONFIG["campus_location"]["lat"], CONFIG["campus_location"]["lng"]],
        popup="<b>Bentley University</b><br>Campus Location",
        icon=folium.Icon(color="red", icon="graduation-cap", prefix="fa"),
        tooltip="Bentley University"
    ).add_to(m)


    # [DA8]
    for idx, row in df.iterrows():
        # Color code by price
        if row['price'] < 1500:
            color = 'green'
        elif row['price'] < 2500:
            color = 'orange'
        else:
            color = 'red'

        # Create popup content with optional image
        image_html = ""
        if 'image_url' in row and pd.notna(row.get('image_url')) and row['image_url']:
            # Add onerror handler to hide broken images
            image_html = f'''
            <div style="width: 100%; max-height: 150px; overflow: hidden; margin-bottom: 8px; border-radius: 5px;">
                <img src="{row["image_url"]}"
                     loading="lazy"
                     style="width: 100%; height: 150px; object-fit: cover;"
                     onerror="this.parentElement.style.display='none';" />
            </div>
            '''

        popup_html = f"""
        <div style="width: 280px; max-width: 280px; font-size: 12px; overflow-x: hidden;">
            {image_html}
            <h4 style="margin: 0 0 8px 0; color: #333; font-size: 14px; line-height: 1.3;">{row['title'][:60]}</h4>
            <div style="border-bottom: 1px solid #ddd; margin-bottom: 8px;"></div>
            <p style="margin: 4px 0;"><b>💰 Price:</b> ${row['price']:,}/mo</p>
            <p style="margin: 4px 0;"><b>📍 Address:</b> {row['address'][:40]}</p>
            <p style="margin: 4px 0;"><b>🛏️ Beds:</b> {row['bedrooms']} | <b>🚿 Baths:</b> {row['bathrooms']}</p>
            <p style="margin: 4px 0;"><b>📐 Size:</b> {row['sqft']} sqft</p>
            <p style="margin: 4px 0;"><b>🚗 Distance:</b> {row['distance_miles']} mi ({row['commute_time']})</p>
            <p style="margin: 4px 0;"><b>✨ Amenities:</b> {', '.join(row['amenities'][:2])}</p>
        </div>
        """

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=folium.Popup(popup_html, max_width=350),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            tooltip=f"${row['price']:,}/mo - {row['bedrooms']}BR"
        ).add_to(m)

    return m



# [VIZ1] Chart with title, colors, labels, legend
def create_price_distribution_chart(df):
    """
    Create a price distribution histogram.
    Includes custom colors, labels, and median reference line.
    """
    fig = px.histogram(
        df,
        x='price',
        nbins=15,
        title='<b>Rental Price Distribution</b>',
        labels={'price': 'Monthly Rent ($)', 'count': 'Number of Listings'},
        color_discrete_sequence=['#3498db']
    )

    fig.update_layout(
        xaxis_title="Monthly Rent ($)",
        yaxis_title="Number of Listings",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=12)
    )

    fig.add_vline(
        x=df['price'].median(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Median: ${df['price'].median():,.0f}",
        annotation_position="top"
    )

    return fig



# [VIZ2] Chart with custom colors and labels
def create_price_by_bedroom_chart(df):
    """
    Create a box plot of prices by number of bedrooms.
    Uses color coding by bedroom count with custom palette.
    """
    fig = px.box(
        df,
        x='bedrooms',
        y='price',
        title='<b>Price Distribution by Number of Bedrooms</b>',
        labels={'bedrooms': 'Number of Bedrooms', 'price': 'Monthly Rent ($)'},
        color='bedrooms',
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_layout(
        xaxis_title="Number of Bedrooms",
        yaxis_title="Monthly Rent ($)",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig



# [VIZ3] Scatter plot with size and color encoding
def create_distance_vs_price_scatter(df):
    """
    Create a scatter plot of distance vs price.
    Uses bubble size for sqft and color for bedrooms.
    """
    fig = px.scatter(
        df,
        x='distance_miles',
        y='price',
        size='sqft',
        color='bedrooms',
        hover_name='title',
        hover_data=['address', 'bedrooms', 'commute_time'],
        title='<b>Distance from Campus vs. Monthly Rent</b>',
        labels={
            'distance_miles': 'Distance from Campus (miles)',
            'price': 'Monthly Rent ($)',
            'sqft': 'Square Footage',
            'bedrooms': 'Bedrooms'
        },
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        xaxis_title="Distance from Campus (miles)",
        yaxis_title="Monthly Rent ($)",
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig


def create_neighborhood_comparison(df):
    """
    Create a bar chart comparing neighborhoods.
    """
    # [DA6] Analyze data with pivot tables
    neighborhood_stats = df.pivot_table(
        values=['price', 'distance_miles'],
        index='neighborhood',
        aggfunc={'price': 'mean', 'distance_miles': 'mean'}
    ).round(2)

    neighborhood_stats = neighborhood_stats.reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Avg Price ($100s)',
        x=neighborhood_stats['neighborhood'],
        y=neighborhood_stats['price'] / 100,
        marker_color='#3498db'
    ))

    fig.add_trace(go.Bar(
        name='Distance (miles)',
        x=neighborhood_stats['neighborhood'],
        y=neighborhood_stats['distance_miles'],
        marker_color='#e74c3c'
    ))

    fig.update_layout(
        title='<b>Neighborhood Comparison</b>',
        barmode='group',
        xaxis_tickangle=-45,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


# MAIN APPLICATION

def main():
    """Main Streamlit application."""

    # [ST4] Customized page design (sidebar, fonts, colors, images, navigation)
    st.set_page_config(
        page_title="Student Housing Heatmap",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # [ST4] Custom CSS for styling - fonts, colors, backgrounds
    st.markdown("""
        <style>
        /* v2 - Updated header size */
        .main-header {
            font-size: 3rem !important;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            padding: 1rem 0;
        }
        .sub-header {
            font-size: 1.5rem !important;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<p class="main-header">🏠 Student Rental & Housing Heatmap</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Find affordable housing near Bentley University in Waltham/Boston</p>',
                unsafe_allow_html=True)

    # SIDEBAR CONFIGURATION

    with st.sidebar:
        # [ST4] Sidebar with image
        st.image("https://img.icons8.com/clouds/200/home.png", width=100)
        st.title("🔧 Configuration")

        # API Key input
        st.subheader("📡 Data Source")
        api_key = st.text_input(
            "Firecrawl API Key",
            type="password",
            help="Enter your Firecrawl API key to fetch new rental data"
        )

        st.divider()

        # [ST1] Streamlit widget - Slider
        st.subheader("💰 Price Filter")
        price_range = st.slider(
            "Monthly Rent Range ($)",
            min_value=500,
            max_value=5000,
            value=(1000, 3000),
            step=100,
            help="Filter listings by monthly rent"
        )

        # [ST2] Streamlit widget - Multiselect
        st.subheader("🛏️ Bedrooms")
        bedroom_options = [1, 2, 3, 4]
        selected_bedrooms = st.multiselect(
            "Number of Bedrooms",
            options=bedroom_options,
            default=bedroom_options,
            help="Select bedroom counts to display"
        )

        # [ST3] Streamlit widget - Selectbox
        st.subheader("📍 Zip Code")
        zip_options = ["All"] + CONFIG["target_zip_codes"]
        selected_zip = st.selectbox(
            "Filter by Zip Code",
            options=zip_options,
            help="Filter by specific zip code"
        )

        st.divider()

        # Additional slider for distance filter
        st.subheader("🚗 Distance to Campus")
        max_distance = st.slider(
            "Maximum Distance (miles)",
            min_value=0.5,
            max_value=10.0,
            value=5.0,
            step=0.5
        )

        # Amenities filter (another multiselect)
        st.subheader("✨ Amenities")
        all_amenities = ["Laundry", "Parking", "Pet Friendly", "Gym", "Pool",
                         "Doorman", "Dishwasher", "AC", "Balcony", "Storage"]
        selected_amenities = st.multiselect(
            "Must Have",
            options=all_amenities,
            default=[],
            help="Filter by required amenities"
        )


    # DATA LOADING

    # Initialize or load data
    refresh_data = st.sidebar.button("🔄 Fetch New Data")

    if 'rental_data' not in st.session_state:
        # On first load, try to load cached data
        with st.spinner("Loading rental data..."):
            df = load_data_from_file()

            if df is None:
                st.info("ℹ️ No cached data found. Click '🔄 Fetch New Data' to scrape listings with your API key.")
                # Create empty dataframe with required columns
                df = pd.DataFrame(columns=[
                    'title', 'price', 'address', 'zip_code', 'bedrooms', 'bathrooms',
                    'sqft', 'amenities', 'latitude', 'longitude', 'distance_miles',
                    'commute_time', 'neighborhood', 'source'
                ])

            st.session_state.rental_data = df

    if refresh_data:
        # Fetch new live data
        with st.spinner("Fetching live rental data..."):
            if api_key:
                scraper = FirecrawlScraper(api_key)
                all_listings = []

                progress_bar = st.progress(0)
                for i, zip_code in enumerate(CONFIG["target_zip_codes"]):
                    listings = scraper.scrape_rental_listings(zip_code)
                    all_listings.extend(listings)
                    progress_bar.progress((i + 1) / len(CONFIG["target_zip_codes"]))

                if all_listings:
                    st.success(f"✅ Successfully scraped {len(all_listings)} listings!")
                    df = pd.DataFrame(all_listings)
                    df = clean_rental_data(df)
                    df = enrich_scraped_data(df)

                    # Save to file
                    save_data_to_file(df)

                    # Update session state
                    st.session_state.rental_data = df

                    st.info(f"📊 Processed {len(df)} valid listings")
                else:
                    st.error("❌ No listings found. Please check the source URLs or try again later.")
            else:
                st.error("❌ Please enter a Firecrawl API key to fetch new data.")

    df = st.session_state.rental_data.copy()


    # DATA FILTERING

    # [DA4] Filter data by one condition

    df_filtered = df[df['price'].between(price_range[0], price_range[1])]


    # [DA5] Filter data by two or more conditions with AND
    df_filtered = df_filtered[
        (df_filtered['bedrooms'].isin(selected_bedrooms)) &
        (df_filtered['distance_miles'] <= max_distance)
        ]

    # Filter by zip code (single condition)
    if selected_zip != "All":
        df_filtered = df_filtered[df_filtered['zip_code'] == selected_zip]

    # Filter by amenities
    if selected_amenities:
        # [DA7]
        # Creates a new column 'has_amenities' based on a condition
        df_filtered['has_amenities'] = df_filtered['amenities'].apply(
            lambda x: all(amenity in x for amenity in selected_amenities)
        )
        df_filtered = df_filtered[df_filtered['has_amenities'] == True]

    # MAIN CONTENT

    # Key Metrics
    st.subheader("📊 Quick Stats")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Listings", len(df_filtered))
    with col2:
        avg_price = df_filtered['price'].mean() if len(df_filtered) > 0 else 0
        st.metric("Avg. Price", f"${avg_price:,.0f}/mo")
    with col3:
        # [DA3]
        # Finding the minimum (smallest) price value
        min_price = df_filtered['price'].min() if len(df_filtered) > 0 else 0
        st.metric("Lowest Price", f"${min_price:,.0f}/mo")
    with col4:
        avg_distance = df_filtered['distance_miles'].mean() if len(df_filtered) > 0 else 0
        st.metric("Avg. Distance", f"{avg_distance:.1f} mi")

    st.divider()

    # TABS FOR DIFFERENT VIEWS

    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map View", "📈 Analytics", "📋 Listings Table", "🔍 Search"])

    with tab1:
        st.subheader("Interactive Housing Map")
        st.caption("🔴 Campus Location | 🟢 < $1,500 | 🟠 $1,500-$2,500 | 🔴 > $2,500")

        if len(df_filtered) > 0:
            # [VIZ4 MAP] Create and display the interactive map
            housing_map = create_map(df_filtered)
            st_folium(housing_map, width=None, height=500, use_container_width=True)
        else:
            st.warning("No listings match your current filters. Try adjusting the filters.")

    with tab2:
        st.subheader("📈 Market Analytics")

        if len(df_filtered) > 0:
            # Charts row 1
            col1, col2 = st.columns(2)

            with col1:
                # [VIZ1] Price distribution histogram
                st.plotly_chart(create_price_distribution_chart(df_filtered), use_container_width=True)

            with col2:
                # [VIZ2] Price by bedroom box plot
                st.plotly_chart(create_price_by_bedroom_chart(df_filtered), use_container_width=True)

            # Charts row 2
            col3, col4 = st.columns(2)

            with col3:
                # [VIZ3] Distance vs Price scatter plot
                st.plotly_chart(create_distance_vs_price_scatter(df_filtered), use_container_width=True)

            with col4:
                # Neighborhood comparison (uses [DA6] pivot table internally)
                st.plotly_chart(create_neighborhood_comparison(df_filtered), use_container_width=True)

            # Summary statistics
            st.subheader("📊 Summary Statistics")

            # [DA6]
            # Creates a comprehensive summary by bedroom count
            summary_pivot = df_filtered.pivot_table(
                values=['price', 'sqft', 'distance_miles'],
                index='bedrooms',
                aggfunc={
                    'price': ['mean', 'min', 'max'],
                    'sqft': 'mean',
                    'distance_miles': 'mean'
                }
            ).round(2)

            st.dataframe(summary_pivot, use_container_width=True)
        else:
            st.warning("No data to display. Adjust your filters.")

    with tab3:
        st.subheader("📋 All Listings")

        if len(df_filtered) > 0:
            # [DA2]
            # User can select sort column and order
            sort_by = st.selectbox(
                "Sort by",
                options=['price', 'distance_miles', 'sqft', 'bedrooms'],
                format_func=lambda x: {
                    'price': '💰 Price',
                    'distance_miles': '📍 Distance',
                    'sqft': '📐 Square Footage',
                    'bedrooms': '🛏️ Bedrooms'
                }.get(x, x)
            )

            sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True)
            ascending = sort_order == "Ascending"

            # [DA2]
            df_sorted = df_filtered.sort_values(by=sort_by, ascending=ascending)

            # Display columns selection
            display_cols = ['title', 'address', 'zip_code', 'price', 'bedrooms',
                            'bathrooms', 'sqft', 'distance_miles',
                            'commute_time', 'neighborhood']

            # [DA9]
            # Calculate price per square foot
            df_sorted['price_per_sqft'] = (df_sorted['price'] / df_sorted['sqft']).round(2)

            st.dataframe(
                df_sorted[display_cols + ['price_per_sqft']].reset_index(drop=True),
                use_container_width=True,
                height=400
            )

            # Download button
            csv = df_sorted.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="housing_listings.csv",
                mime="text/csv"
            )
        else:
            st.warning("No listings to display.")

    with tab4:
        st.subheader("🔍 Advanced Search")

        col1, col2 = st.columns(2)

        with col1:
            search_query = st.text_input("Search by address or title", "")

            # [DA3]
            # Find the top 5 best deals based on value score
            st.subheader("🏆 Top 5 Best Deals")
            if len(df_filtered) > 0:
                # Calculate value score (size - price normalized)
                df_temp = df_filtered.copy()
                df_temp['value_score'] = (
                        df_temp['sqft'] / 10 -
                        df_temp['price'] / 100
                )
                # [DA3] Using nlargest to find top N values
                top_deals = df_temp.nlargest(5, 'value_score')[
                    ['title', 'address', 'price', 'bedrooms', 'sqft']
                ]
                st.dataframe(top_deals, use_container_width=True)

        with col2:
            st.subheader("💡 Quick Facts")
            if len(df_filtered) > 0:
                st.info(f"""
                **Market Overview:**
                - 📊 Total available listings: {len(df_filtered)}
                - 💰 Price range: ${df_filtered['price'].min():,} - ${df_filtered['price'].max():,}
                - 📍 Closest to campus: {df_filtered['distance_miles'].min():.1f} miles
                - 🏠 Most common: {df_filtered['bedrooms'].mode().values[0]} bedroom units
                """)

        # Search results
        if search_query:
            # [DA4] Filter by condition (text search)
            search_results = df_filtered[
                df_filtered['title'].str.contains(search_query, case=False, na=False) |
                df_filtered['address'].str.contains(search_query, case=False, na=False)
                ]

            st.subheader(f"Search Results ({len(search_results)} found)")
            if len(search_results) > 0:
                st.dataframe(search_results, use_container_width=True)
            else:
                st.warning("No results found for your search.")

    # FOOTER

    st.divider()
    st.markdown("""
        <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
            <p>🏠 Student Rental & Housing Heatmap | CS602 Final Project</p>
            <p>Data sources: Apartments.com, Zillow, JuneHomes, Craigslist</p>
            <p>Target Areas: Waltham (02453, 02452), Cambridge (02138), Belmont (02478)</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()