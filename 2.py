import nest_asyncio
import asyncio
from playwright.async_api import async_playwright
from dataclasses import dataclass, asdict, field
import pandas as pd
import os
import subprocess

# Ensure Playwright browsers are installed
try:
    subprocess.run(["playwright", "install"], check=True)
except Exception as e:
    print(f"Error installing browsers: {e}")
# Apply nest_asyncio to allow for running nested event loops (useful in environments like Jupyter)
nest_asyncio.apply()

@dataclass
class Business:
    """Holds business data"""
    name: str = None
    address: str = None
    latitude: float = None
    longitude: float = None

@dataclass
class BusinessList:
    """Holds list of Business objects and saves to both Excel and CSV"""
    business_list: list[Business] = field(default_factory=list)
    save_at = 'output'

    def dataframe(self):
        """Transform business_list to pandas DataFrame"""
        return pd.json_normalize(
            (asdict(business) for business in self.business_list), sep="_"
        )

    def save_to_excel(self, filename):
        """Saves pandas DataFrame to Excel (xlsx) file"""
        if not os.path.exists(self.save_at):
            os.makedirs(self.save_at)
        self.dataframe().to_excel(f"{self.save_at}/{filename}.xlsx", index=False)

    def save_to_csv(self, filename):
        """Saves pandas DataFrame to CSV file"""
        if not os.path.exists(self.save_at):
            os.makedirs(self.save_at)
        self.dataframe().to_csv(f"{self.save_at}/{filename}.csv", index=False)

def extract_coordinates_from_url(url: str) -> tuple[float, float]:
    """Helper function to extract coordinates from URL"""
    coordinates = url.split('/@')[-1].split('/')[0]
    return float(coordinates.split(',')[0]), float(coordinates.split(',')[1])

async def scrape_google_maps():
    ########
    # Input 
    ########
    search_query = "Police Stations in Bhilwara, Rajasthan, India"
    total_results = 10000  # Set limit to get up to 20,000 results

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto("https://www.google.com/maps", timeout=60000)
        await page.wait_for_timeout(5000)
        
        # Search for police stations in India
        await page.fill('//input[@id="searchboxinput"]', search_query)
        await page.wait_for_timeout(3000)
        await page.keyboard.press("Enter")
        await page.wait_for_timeout(5000)

        # Scrolling and collecting results
        previously_counted = 0
        while True:
            await page.mouse.wheel(0, 10000)
            await page.wait_for_timeout(3000)

            current_count = await page.locator('//a[contains(@href, "https://www.google.com/maps/place")]').count()
            print(f"Current number of listings found: {current_count}")

            if current_count >= total_results:
                listings = await page.locator('//a[contains(@href, "https://www.google.com/maps/place")]').all()[:total_results]
                listings = [listing.locator("xpath=..") for listing in listings]
                print(f"Total Scraped: {len(listings)}")
                break
            else:
                if current_count == previously_counted:
                    listings = await page.locator('//a[contains(@href, "https://www.google.com/maps/place")]').all()
                    print(f"Arrived at all available results\nTotal Scraped: {len(listings)}")
                    break
                else:
                    previously_counted = current_count
                    print("Currently Scraped:", previously_counted)

        business_list = BusinessList()

        # Scraping individual listing details
        for listing in listings:
            try:
                await listing.click()
                await page.wait_for_timeout(5000)

                name_attribute = 'aria-label'
                address_xpath = '//button[@data-item-id="address"]//div[contains(@class, "fontBodyMedium")]'

                business = Business()
                business.name = await listing.get_attribute(name_attribute) or ""
                print(f"Business Name: {business.name}")
                
                if await page.locator(address_xpath).count() > 0:
                    business.address = await page.locator(address_xpath).first.inner_text()
                else:
                    business.address = ""
                print(f"Business Address: {business.address}")

                # Extract latitude and longitude
                business.latitude, business.longitude = extract_coordinates_from_url(page.url)
                print(f"Coordinates: ({business.latitude}, {business.longitude})")

                business_list.business_list.append(business)
            except Exception as e:
                print(f'Error occurred while scraping listing: {e}')

        #########
        # Output
        #########
        filename = "police_stations_in_bhilwara"
        print("Saving data to CSV...")
        business_list.save_to_csv(filename)
        print("Data saved successfully.")

        await browser.close()

# Run the function
try:
    asyncio.run(scrape_google_maps())
except RuntimeError:
    asyncio.create_task(scrape_google_maps())
