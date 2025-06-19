import csv
import html
import random
import re
import time
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

TARGET = 100
TARGET_WB = 100
OUTFILE = Path("/mnt/c/Users/Lilian/Desktop/scrapingRen-EM.csv")

BASE_URL = "https://parti-renaissance.fr"
PROJECT_URL = "https://ensemble-2024.fr/notre-projet"

WAYBACK_SITES = {
    "news": "https://web.archive.org/web/20211023175854/https://en-marche.fr/articles/",
    "press_releases": "https://web.archive.org/web/20210921112432/https://en-marche.fr/articles/communiques/",
    "media": "https://web.archive.org/web/20211023175854/https://en-marche.fr/articles/medias/",
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
]

session = requests.Session()
_whitespace = re.compile(r"\s+")

def normalize(text: str) -> str:
    return _whitespace.sub(" ", text.replace("\xa0", " ")).strip()

def get_soup(url: str, delay: float = 0) -> BeautifulSoup:
    time.sleep(delay)
    response = session.get(url, timeout=30, headers={"User-Agent": random.choice(USER_AGENTS)})
    response.raise_for_status()
    doc = BeautifulSoup(response.text, "lxml")
    doc.base_url = url
    return doc

def record(rows: list, content: str, category: str, source: str, archive_date: str = ""):
    rows.append({
        "text": content,
        "party": "Renaissance",
        "category": category,
        "source": source,
        "archive_date": archive_date,
    })

def extract_text(node: BeautifulSoup) -> str:
    segments = []
    for element in node.select("p, li"):
        if not element.get_text(strip=True):
            continue
        text = normalize(element.get_text(" ", strip=True))
        bullet = "• " + text if element.name == "li" else text
        segments.append(bullet)
    return "\n\n".join(segments)

rows = []
already_seen = set()

homepage = get_soup(BASE_URL, 1)
for item in homepage.select("div.valeurs-item"):
    heading = item.select_one("h1.heading, h2.heading, h3.heading")
    block = item.select_one("div.rich-text-block-2")
    if heading and block:
        text = extract_text(block)
        if len(text) >= 120:
            slug = normalize(heading.get_text())[:40]
            record(rows, text, "value", f"{BASE_URL}#{slug}")

project_page = get_soup(PROJECT_URL, 1)
for card in project_page.select("div.program_measures-card-content"):
    title = card.find_previous("h2", class_="program_measures-title")
    text = extract_text(card)
    if len(text) >= 120:
        slug = normalize(title.get_text() if title else "measure")[:40]
        record(rows, text, "program", f"{PROJECT_URL}#{slug}")

def extract_timestamp(url: str) -> str:
    match = re.search(r"/web/(\d{14})/", url)
    return match.group(1) if match else ""

def crawl_archive(listing_url: str, label: str, limit: int):
    ts = extract_timestamp(listing_url)
    queue = [listing_url]
    visited = set()
    count = 0
    while queue and count < limit:
        page_url = queue.pop(0)
        if page_url in visited:
            continue
        visited.add(page_url)

        archive_page = get_soup(page_url, 1.5)
        for link in archive_page.select("article h2 a[href], article h3 a[href]"):
            article_url = urljoin(page_url, link["href"])
            if article_url in already_seen:
                continue
            already_seen.add(article_url)

            article_page = get_soup(article_url, 0.5)
            body = article_page.find("article") or article_page
            content = extract_text(body)
            if len(content) >= 120:
                record(rows, content, f"archive-{label}", article_url, ts)
                count += 1
                if count >= limit:
                    break

        for nav in archive_page.select("div.listing__paginator a[href]"):
            next_url = urljoin(page_url, nav["href"])
            if next_url not in visited:
                queue.append(next_url)

for section, url in WAYBACK_SITES.items():
    crawl_archive(url, section, TARGET_WB)

OUTFILE.parent.mkdir(parents=True, exist_ok=True)
with OUTFILE.open("w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, ["text", "party", "category", "source", "archive_date"], delimiter="|", quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()
    writer.writerows(rows[:TARGET])

print("CSV saved:", OUTFILE)