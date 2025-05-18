# beatport_selenium_scraper.py
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Setup headless browser
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

SEARCH_URL = "https://www.beatport.com/search?q="

# Normalize enharmonic key equivalents
def normalize_key(raw_key):
    key_line = raw_key.strip().split("\n")[0]  # remove date & price
    enharmonics = {
        "A#": "Bb", "Bb": "Bb",
        "C#": "Db", "Db": "Db",
        "D#": "Eb", "Eb": "Eb",
        "F#": "Gb", "Gb": "Gb",
        "G#": "Ab", "Ab": "Ab"
    }
    parts = key_line.split()
    if len(parts) == 2:
        note, mode = parts[0], parts[1]
        note = enharmonics.get(note.replace("♯", "#"), note)
        return f"{note} {mode}"
    return key_line

# Fetch metadata for a single track
def get_single_track_metadata(track_name):
    from urllib.parse import quote
    driver.get(SEARCH_URL + quote(track_name))
    try:
        row = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='tracks-table-row']"))
        )

        text = row.text
        if "BPM" in text and "-" in text:
            parts = text.split(" - ")
            bpm_part = parts[0].split()[-2]  # example: '150 BPM'
            key_part = parts[1].strip()
            bpm = float(bpm_part)
            key = normalize_key(key_part)
            print(f"{track_name}: BPM={bpm}, Key={key}")
            return {"bpm": bpm, "key": key}

        print(f"BPM/Key not found in row for: {track_name}")
        return None

    except TimeoutException:
        print(f"No result found for: {track_name} (timed out)")
        return None
    except Exception as e:
        print(f"Error for {track_name}: {e}")
        print(driver.page_source[:1000])
        return None

# Batch mode for scraping multiple tracks
if __name__ == '__main__':
    input_file = "tracklist.txt"
    output_file = "beatport_metadata.json"
    metadata = {}

    with open(input_file, encoding="utf-8") as f:
        track_names = [line.strip() for line in f if line.strip()]

    for track in track_names:
        info = get_single_track_metadata(track)
        if info:
            metadata[track] = info
        time.sleep(2)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    driver.quit()
    print(f"\nSaved metadata for {len(metadata)} tracks to {output_file}")
