#!/usr/bin/env python3
"""
Download clean, high-resolution official reference images of Indian Rupee banknotes:
- Old (Mahatma Gandhi Series, valid in circulation): ₹10, ₹20, ₹50, ₹100
- New (Mahatma Gandhi New Series): ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000
Each denomination includes 1 Obverse (Front) and 1 Reverse (Back).
"""

import os
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

TARGET_DIR = Path("/home/jon/mytorch/currency_references")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "AntigravityResearchCurrencyBot/1.0 (contact@mytorch.org)"
}

BANKNOTES = [
    # ₹10
    {
        "id": "inr_10_old_front",
        "denom": 10,
        "series": "Mahatma Gandhi Series (Older, Valid)",
        "side": "front",
        "color": "Orange-Violet",
        "motif": "Mahatma Gandhi portrait, RBI seal",
        "source_file": "10_Rupee_(Obverse)_2016.jpg",
        "out_file": "inr_10_old_front.jpg"
    },
    {
        "id": "inr_10_old_back",
        "denom": 10,
        "series": "Mahatma Gandhi Series (Older, Valid)",
        "side": "back",
        "color": "Orange-Violet",
        "motif": "Rhinoceros, Elephant, Bengal Tiger",
        "source_file": "10_Rupee_(Reverse)_2016.jpg",
        "out_file": "inr_10_old_back.jpg"
    },
    {
        "id": "inr_10_new_front",
        "denom": 10,
        "series": "Mahatma Gandhi New Series",
        "side": "front",
        "color": "Chocolate Brown",
        "motif": "Mahatma Gandhi portrait, Ashoka Pillar",
        "source_file": "India_new_10_INR,_MG_series,_2018,_obverse.jpg",
        "out_file": "inr_10_new_front.jpg"
    },
    {
        "id": "inr_10_new_back",
        "denom": 10,
        "series": "Mahatma Gandhi New Series",
        "side": "back",
        "color": "Chocolate Brown",
        "motif": "Konark Sun Temple wheel",
        "source_file": "India_new_10_INR,_MG_series,_2018,_reverse.jpg",
        "out_file": "inr_10_new_back.jpg"
    },

    # ₹20
    {
        "id": "inr_20_old_front",
        "denom": 20,
        "series": "Mahatma Gandhi Series (Older, Valid)",
        "side": "front",
        "color": "Reddish-Orange",
        "motif": "Mahatma Gandhi portrait",
        "source_file": "India_P-089A_20_Rupees_Gandhi_2002,_obverse.jpg",
        "out_file": "inr_20_old_front.jpg"
    },
    {
        "id": "inr_20_old_back",
        "denom": 20,
        "series": "Mahatma Gandhi Series (Older, Valid)",
        "side": "back",
        "color": "Reddish-Orange",
        "motif": "Mount Harriet in Andaman and Nicobar Islands",
        "source_file": "India_P-089A_20_Rupees_Gandhi_2002,_reverse.jpg",
        "out_file": "inr_20_old_back.jpg"
    },
    {
        "id": "inr_20_new_front",
        "denom": 20,
        "series": "Mahatma Gandhi New Series",
        "side": "front",
        "color": "Greenish-Yellow",
        "motif": "Mahatma Gandhi portrait, Ashoka Pillar",
        "source_file": "India_new_20_INR,_MG_series,_2019,_obverse.jpg",
        "out_file": "inr_20_new_front.jpg"
    },
    {
        "id": "inr_20_new_back",
        "denom": 20,
        "series": "Mahatma Gandhi New Series",
        "side": "back",
        "color": "Greenish-Yellow",
        "motif": "Ellora Caves (Kailash Temple)",
        "source_file": "India_new_20_INR,_MG_series,_2019,_reverse.jpg",
        "out_file": "inr_20_new_back.jpg"
    },

    # ₹50
    {
        "id": "inr_50_old_front",
        "denom": 50,
        "series": "Mahatma Gandhi Series (Older, Valid)",
        "side": "front",
        "color": "Violet-Pink",
        "motif": "Mahatma Gandhi portrait",
        "source_file": "India_50_INR,_MG_series,_2011,_obverse.jpg",
        "out_file": "inr_50_old_front.jpg"
    },
    {
        "id": "inr_50_old_back",
        "denom": 50,
        "series": "Mahatma Gandhi Series (Older, Valid)",
        "side": "back",
        "color": "Violet-Pink",
        "motif": "Parliament of India",
        "source_file": "India_50_INR,_MG_series,_2011,_reverse.jpg",
        "out_file": "inr_50_old_back.jpg"
    },
    {
        "id": "inr_50_new_front",
        "denom": 50,
        "series": "Mahatma Gandhi New Series",
        "side": "front",
        "color": "Fluorescent Cyan-Blue",
        "motif": "Mahatma Gandhi portrait, Ashoka Pillar",
        "source_file": "India_new_50_INR,_MG_series,_2018,_obverse.jpg",
        "out_file": "inr_50_new_front.jpg"
    },
    {
        "id": "inr_50_new_back",
        "denom": 50,
        "series": "Mahatma Gandhi New Series",
        "side": "back",
        "color": "Fluorescent Cyan-Blue",
        "motif": "Hampi with Chariot",
        "source_file": "India_new_50_INR,_MG_series,_2018,_reverse.jpg",
        "out_file": "inr_50_new_back.jpg"
    },

    # ₹100
    {
        "id": "inr_100_old_front",
        "denom": 100,
        "series": "Mahatma Gandhi Series (Older, Valid)",
        "side": "front",
        "color": "Blue-Green",
        "motif": "Mahatma Gandhi portrait",
        "source_file": "India_P-091m_100_Rupees_1996_XF,_obverse.jpg",
        "out_file": "inr_100_old_front.jpg"
    },
    {
        "id": "inr_100_old_back",
        "denom": 100,
        "series": "Mahatma Gandhi Series (Older, Valid)",
        "side": "back",
        "color": "Blue-Green",
        "motif": "Mount Kangchenjunga",
        "source_file": "India_P-091m_100_Rupees_1996_XF,_reverse.jpg",
        "out_file": "inr_100_old_back.jpg"
    },
    {
        "id": "inr_100_new_front",
        "denom": 100,
        "series": "Mahatma Gandhi New Series",
        "side": "front",
        "color": "Lavender / Purple",
        "motif": "Mahatma Gandhi portrait, Ashoka Pillar",
        "source_file": "India_new_100_INR,_Mahatma_Gandhi_New_Series,_2018,_obverse.png",
        "out_file": "inr_100_new_front.png"
    },
    {
        "id": "inr_100_new_back",
        "denom": 100,
        "series": "Mahatma Gandhi New Series",
        "side": "back",
        "color": "Lavender / Purple",
        "motif": "Rani ki Vav (Queen's Stepwell)",
        "source_file": "India_new_100_INR,_Mahatma_Gandhi_New_Series,_2018,_reverse.png",
        "out_file": "inr_100_new_back.png"
    },

    # ₹200
    {
        "id": "inr_200_new_front",
        "denom": 200,
        "series": "Mahatma Gandhi New Series",
        "side": "front",
        "color": "Bright Orange-Yellow",
        "motif": "Mahatma Gandhi portrait, Ashoka Pillar",
        "source_file": "India,_200_INR,_2018,_obverse.jpg",
        "out_file": "inr_200_new_front.jpg"
    },
    {
        "id": "inr_200_new_back",
        "denom": 200,
        "series": "Mahatma Gandhi New Series",
        "side": "back",
        "color": "Bright Orange-Yellow",
        "motif": "Sanchi Stupa",
        "source_file": "India,_200_INR,_2018,_reverse.jpg",
        "out_file": "inr_200_new_back.jpg"
    },

    # ₹500
    {
        "id": "inr_500_new_front",
        "denom": 500,
        "series": "Mahatma Gandhi New Series",
        "side": "front",
        "color": "Stone Grey",
        "motif": "Mahatma Gandhi portrait, Ashoka Pillar, Swachh Bharat logo",
        "source_file": "India_new_500_INR,_MG_series,_2016,_obverse.jpg",
        "out_file": "inr_500_new_front.jpg"
    },
    {
        "id": "inr_500_new_back",
        "denom": 500,
        "series": "Mahatma Gandhi New Series",
        "side": "back",
        "color": "Stone Grey",
        "motif": "Red Fort with Indian flag",
        "source_file": "India_new_500_INR,_MG_series,_2016,_reverse.jpg",
        "out_file": "inr_500_new_back.jpg"
    },

    # ₹2000
    {
        "id": "inr_2000_new_front",
        "denom": 2000,
        "series": "Mahatma Gandhi New Series",
        "side": "front",
        "color": "Magenta / Pink",
        "motif": "Mahatma Gandhi portrait, Ashoka Pillar",
        "source_file": "India_new_2000_INR,_MG_series,_2016,_obverse.jpg",
        "out_file": "inr_2000_new_front.jpg"
    },
    {
        "id": "inr_2000_new_back",
        "denom": 2000,
        "series": "Mahatma Gandhi New Series",
        "side": "back",
        "color": "Magenta / Pink",
        "motif": "Mangalyaan (Mars Orbiter Mission)",
        "source_file": "India_new_2000_INR,_MG_series,_2016,_reverse.jpg",
        "out_file": "inr_2000_new_back.jpg"
    },
]

def resolve_wikimedia_url(filename: str) -> str:
    url = f"https://commons.wikimedia.org/w/api.php?action=query&titles=File:{urllib.parse.quote(filename)}&prop=imageinfo&iiprop=url&format=json"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        for p_id, p_info in pages.items():
            if "imageinfo" in p_info and p_info["imageinfo"]:
                return p_info["imageinfo"][0]["url"]
    return None

def main():
    print(f"Downloading {len(BANKNOTES)} reference banknotes to {TARGET_DIR}...")
    manifest = []

    for item in BANKNOTES:
        src = item["source_file"]
        dest = TARGET_DIR / item["out_file"]
        print(f"[{item['id']}] Resolving URL for {src}...")

        direct_url = resolve_wikimedia_url(src)
        if not direct_url:
            print(f"  FAILED to resolve {src}")
            continue

        print(f"  Downloading -> {dest.name}")
        req = urllib.request.Request(direct_url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
                f.write(resp.read())
            size_kb = dest.stat().st_size / 1024
            print(f"  Done ({size_kb:.1f} KB)")
            item_record = dict(item)
            item_record["path"] = str(dest.resolve())
            item_record["size_kb"] = round(size_kb, 1)
            manifest.append(item_record)
        except Exception as e:
            print(f"  Error downloading {direct_url}: {e}")

        # Polite delay to respect Wikimedia API
        time.sleep(0.5)

    metadata_file = TARGET_DIR / "banknotes_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Successfully downloaded {len(manifest)}/{len(BANKNOTES)} reference banknotes.")
    print(f"Directory: {TARGET_DIR}")
    print(f"Metadata : {metadata_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
