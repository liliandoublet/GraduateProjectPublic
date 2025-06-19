import csv
import random
import re
import time
import sys
import html
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

TARGET = 271
OUTFILE = Path("/mnt/c/Users/Lilian/Desktop/communiques_reconq.csv")

SITES = {
    "National": "https://www.parti-reconquete.fr/communiques",
    "Fede75": "https://fede75.parti-reconquete.fr/articles",
    "Fede92": "https://fede92.parti-reconquete.fr/articles",
    "Fede93": "https://fede93.parti-reconquete.fr/articles",
    "Fede94": "https://fede94.parti-reconquete.fr/articles",
    "Fede35": "https://fede35.parti-reconquete.fr/articles",
    "Fede99": "https://fede99.parti-reconquete.fr/articles",
    "Fede68": "https://fede68.parti-reconquete.fr/articles",
    "Fede59": "https://fede59.parti-reconquete.fr/articles",
    "Fede45": "https://fede45.parti-reconquete.fr/articles",
    "Fede78": "https://fede78.parti-reconquete.fr/articles",
    "Fede23": "https://fede23.parti-reconquete.fr/articles",
    "Fede31": "https://fede31.parti-reconquete.fr/articles",
    "Fede26": "https://fede26.parti-reconquete.fr/articles",
}

UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
]

session = requests.Session()
_space_pattern = re.compile(r"\s+")

def get_soup(url, wait=0):
    time.sleep(wait)
    response = session.get(url, timeout=20, headers={"User-Agent": random.choice(UAS)})
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")

def clean_text(text):
    return _space_pattern.sub(" ", text.replace("\xa0", " ")).strip()

def extract_text(document):
    container = document.select_one("#content") or document.select_one("div.subcontainer") or document
    paragraphs = [
        clean_text(p.get_text(" ", strip=True))
        for p in container.select("p")
        if len(p.get_text(strip=True)) >= 20
    ]
    return html.unescape("\n\n".join(paragraphs))

def find_next_page(page, visited):
    for link in page.select("a.page-link[href*='?page='], a.btn.btn-secondary[href*='?page=']"):
        if "suivant" in link.get_text(" ", strip=True).lower():
            next_url = urljoin(page.base_url, link["href"])
            if next_url not in visited:
                return next_url
    return None

links = []
rows = []

for label, start_url in SITES.items():
    current_url = start_url
    visited = set()
    while current_url and len(links) < TARGET:
        if current_url in visited:
            break
        visited.add(current_url)

        page = get_soup(current_url, 2)
        for anchor in page.select("div.news-box > a[href]"):
            href = urljoin(current_url, anchor["href"])
            if href not in links:
                links.append(href)

        if len(links) >= TARGET:
            break
        current_url = find_next_page(page, visited)

    if len(links) >= TARGET:
        break

for link in tqdm(links, desc="Articles", unit="art"):
    if len(rows) >= TARGET:
        break
    try:
        article_page = get_soup(link, 1)
        text = extract_text(article_page)
        if len(text) < 200:
            continue
        rows.append({
            "text": text,
            "party": "Reconquête",
            "type": "official",
            "source": link
        })
        time.sleep(1.2)
    except Exception as e:
        print(link, e, file=sys.stderr)

OUTFILE.parent.mkdir(parents=True, exist_ok=True)
with OUTFILE.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, ["text", "party", "type", "source"], delimiter="|", quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()
    writer.writerows(rows[:TARGET])

print("CSV saved:", OUTFILE)