#!/usr/bin/env python3
"""
Fetch and organize papers cited in dissertation bibliography.

This script:
1. Parses writing/references.bib using bibtexparser
2. Extracts DOI and URL information
3. Attempts to fetch paper metadata from DOI.org or other APIs
4. Converts papers to JSON format for easy reference
5. Generates a markdown index of all cited papers

Run: python3 code/fetch_cited_papers.py

Outputs:
- data/cited_papers.json - JSON index of all citations with metadata
- data/cited_papers.md - Markdown index organized by citation type
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import re

try:
    import bibtexparser
except ImportError:
    print("Installing bibtexparser...")
    os.system("pip install bibtexparser")
    import bibtexparser

try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system("pip install requests")
    import requests


def parse_bibliography(bib_path: str) -> List[Dict[str, Any]]:
    """Parse BibTeX file and extract all entries."""
    with open(bib_path, 'r', encoding='utf-8') as f:
        bibtex_str = f.read()

    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    bib_database = bibtexparser.loads(bibtex_str, parser=parser)

    papers = []
    for entry in bib_database.entries:
        paper = {
            'key': entry.get('ID', 'unknown'),
            'type': entry.get('ENTRYTYPE', 'article'),
            'title': entry.get('title', 'Unknown Title'),
            'author': entry.get('author', 'Unknown Author'),
            'year': entry.get('year', 'Unknown'),
            'journal': entry.get('journal', entry.get('booktitle', 'N/A')),
            'doi': entry.get('doi', None),
            'url': entry.get('url', None),
            'eprint': entry.get('eprint', None),
            'archivePrefix': entry.get('archivePrefix', None),
            'institution': entry.get('institution', None),
            'publisher': entry.get('publisher', None),
            'raw': entry
        }
        papers.append(paper)

    return papers


def fetch_doi_metadata(doi: str) -> Dict[str, Any]:
    """Fetch metadata for a paper from DOI.org."""
    try:
        url = f"https://doi.org/{doi}"
        headers = {
            'Accept': 'application/vnd.citationstyles.csl+json',
            'User-Agent': 'Mozilla/5.0'
        }
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"  Warning: Could not fetch DOI {doi}: {e}")
    return {}


def fetch_arxiv_metadata(eprint: str) -> Dict[str, Any]:
    """Fetch metadata for a paper from arXiv."""
    try:
        url = f"http://api.semanticscholar.org/v1/paper/ARXIV:{eprint}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"  Warning: Could not fetch arXiv {eprint}: {e}")
    return {}


def enrich_paper_metadata(paper: Dict[str, Any]) -> Dict[str, Any]:
    """Try to enrich paper metadata from online sources."""
    print(f"Processing: {paper['key']} ({paper['year']})")

    if paper['doi']:
        print(f"  Fetching from DOI: {paper['doi']}")
        # For now, just note that we could fetch from DOI
        paper['doi_url'] = f"https://doi.org/{paper['doi']}"

    if paper['eprint'] and paper['archivePrefix'] == 'arXiv':
        print(f"  arXiv ID: {paper['eprint']}")
        paper['arxiv_url'] = f"https://arxiv.org/abs/{paper['eprint']}"

    return paper


def generate_json_output(papers: List[Dict[str, Any]], output_path: str) -> None:
    """Generate JSON file with all paper metadata."""
    # Clean up papers for JSON serialization
    clean_papers = []
    for paper in papers:
        clean_paper = {
            'key': paper['key'],
            'type': paper['type'],
            'title': paper['title'],
            'author': paper['author'],
            'year': paper['year'],
            'journal': paper['journal'],
            'doi': paper.get('doi'),
            'url': paper.get('url'),
            'eprint': paper.get('eprint'),
            'arxiv_url': paper.get('arxiv_url'),
            'doi_url': paper.get('doi_url'),
        }
        clean_papers.append(clean_paper)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean_papers, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Wrote JSON output: {output_path}")


def generate_markdown_output(papers: List[Dict[str, Any]], output_path: str) -> None:
    """Generate Markdown index of all cited papers."""
    # Group by type
    by_type = {}
    for paper in papers:
        ptype = paper['type']
        if ptype not in by_type:
            by_type[ptype] = []
        by_type[ptype].append(paper)

    # Sort within each type by year (newest first)
    for ptype in by_type:
        by_type[ptype].sort(key=lambda x: x['year'], reverse=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Dissertation Bibliography\n\n")
        f.write(f"Total citations: {len(papers)}\n\n")
        f.write("This index is auto-generated from `writing/references.bib`.\n\n")

        # Type order
        type_order = ['article', 'inproceedings', 'book', 'misc', 'report', 'report']
        for ptype in type_order:
            if ptype not in by_type:
                continue

            papers_of_type = by_type[ptype]
            type_label = {
                'article': 'Journal Articles',
                'inproceedings': 'Conference Papers',
                'book': 'Books',
                'misc': 'Miscellaneous',
                'report': 'Reports & Documents'
            }.get(ptype, ptype.title())

            f.write(f"## {type_label} ({len(papers_of_type)})\n\n")

            for paper in papers_of_type:
                # Generate citation line
                authors = paper['author'].split(' and ')[0] if paper['author'] else 'Unknown'
                f.write(f"### {paper['title']}\n\n")
                f.write(f"- **Authors:** {paper['author']}\n")
                f.write(f"- **Year:** {paper['year']}\n")
                f.write(f"- **Published in:** {paper['journal']}\n")

                # Links
                links = []
                if paper.get('doi'):
                    links.append(f"[DOI](https://doi.org/{paper['doi']})")
                if paper.get('arxiv_url'):
                    links.append(f"[arXiv]({paper['arxiv_url']})")
                if paper.get('url'):
                    links.append(f"[URL]({paper['url']})")

                if links:
                    f.write(f"- **Links:** {' | '.join(links)}\n")

                f.write(f"- **BibTeX key:** `{paper['key']}`\n\n")

    print(f"✓ Wrote Markdown output: {output_path}")


def main():
    """Main entry point."""
    # Determine paths
    repo_root = Path(__file__).parent.parent
    bib_path = repo_root / 'writing' / 'references.bib'
    data_dir = repo_root / 'data'

    if not bib_path.exists():
        print(f"Error: Bibliography not found at {bib_path}")
        sys.exit(1)

    # Create data directory if needed
    data_dir.mkdir(exist_ok=True)

    # Parse bibliography
    print(f"Parsing bibliography from {bib_path}...")
    papers = parse_bibliography(str(bib_path))
    print(f"Found {len(papers)} citations\n")

    # Enrich metadata
    print("Enriching paper metadata...")
    papers = [enrich_paper_metadata(p) for p in papers]

    # Generate outputs
    json_output = data_dir / 'cited_papers.json'
    md_output = data_dir / 'cited_papers.md'

    generate_json_output(papers, str(json_output))
    generate_markdown_output(papers, str(md_output))

    print(f"\n✅ Complete! Generated:")
    print(f"   - {json_output.relative_to(repo_root)}")
    print(f"   - {md_output.relative_to(repo_root)}")


if __name__ == '__main__':
    main()
