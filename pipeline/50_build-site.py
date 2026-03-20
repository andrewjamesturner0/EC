#!/usr/bin/env python3
"""
Build a static HTML site from the EC knowledge base.

Reads output/final_output.md, cluster_theme_map.json, extraction files, and
framing content. Generates site/ with index, per-theme pages, per-episode
pages, and supplementary pages.

Usage: python3 pipeline/50_build-site.py
"""

import html
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
SITE = ROOT / "site"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_theme_map():
    with open(OUTPUT / "cluster_theme_map.json") as f:
        return json.load(f)


def load_final_output():
    return (OUTPUT / "final_output.md").read_text(encoding="utf-8")


def load_framing(name):
    path = OUTPUT / "framing" / name
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def load_clusters_json():
    with open(OUTPUT / "clusters.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def slugify(text):
    """Convert display name to URL-safe slug."""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def filename_to_title(filename):
    """Convert transcript filename to human-readable title."""
    name = filename.replace(".txt", "")
    # Strip numeric prefix (with or without empiricalcyclingpodcast-)
    name = re.sub(r"^\d+-(?:empiricalcyclingpodcast-)?", "", name)
    # Replace hyphens with spaces, title-case
    name = name.replace("-", " ").strip()
    # Title case, but keep small words lowercase mid-title
    words = name.split()
    if not words:
        return filename
    result = [words[0].capitalize()]
    small = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
             "for", "of", "with", "by", "from", "is", "vs", "ii", "iii", "iv"}
    for w in words[1:]:
        if w.lower() in small and len(w) <= 4:
            result.append(w.lower())
        else:
            result.append(w.capitalize())
    return " ".join(result)


def numeric_id_from_filename(filename):
    m = re.match(r"(\d+)", filename)
    return m.group(1) if m else filename


def series_from_filename(filename):
    if "ten-minute-tips" in filename:
        return "Ten Minute Tips"
    elif "watts-doc" in filename:
        return "Watts Doc"
    elif "perspectives" in filename:
        return "Perspectives"
    return "Other"


def parse_themes(md_text, theme_map):
    """Parse final_output.md into a list of theme dicts."""
    theme_order = theme_map["theme_order"]
    themes_meta = theme_map["themes"]

    # Split on ## N. Theme Name
    theme_pattern = re.compile(r"^## (\d+)\. (.+)$", re.MULTILINE)
    splits = list(theme_pattern.finditer(md_text))

    themes = []
    for i, match in enumerate(splits):
        section_num = int(match.group(1))
        display_name = match.group(2).strip()
        start = match.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(md_text)
        body = md_text[start:end].strip()

        # Find the theme_key from theme_order
        theme_key = theme_order[section_num - 1] if section_num <= len(theme_order) else slugify(display_name)
        slug = slugify(display_name)

        # Count source clusters to get episode count
        source_clusters = themes_meta.get(theme_key, {}).get("source_clusters", [])

        # Parse subsections
        subsections = parse_subsections(body)

        themes.append({
            "section_num": section_num,
            "display_name": display_name,
            "theme_key": theme_key,
            "slug": slug,
            "subsections": subsections,
            "source_clusters": source_clusters,
        })

    return themes


def parse_subsections(body):
    """Parse ### subsections from a theme body."""
    pattern = re.compile(r"^### (.+)$", re.MULTILINE)
    splits = list(pattern.finditer(body))

    subsections = []
    for i, match in enumerate(splits):
        title = match.group(1).strip()
        start = match.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(body)
        content = body[start:end].strip()

        # Remove trailing --- separators
        content = re.sub(r"\n---\s*$", "", content).strip()

        # Extract sources
        sources = []
        sources_match = re.search(
            r"\*\*Sources?:?\*\*\s*Episodes?\s*(.+?)(?:\n|$)", content
        )
        if sources_match:
            raw = sources_match.group(1)
            sources = [s.strip().rstrip(",") for s in re.split(r",\s*", raw) if s.strip()]

        subsections.append({
            "title": title,
            "anchor": slugify(title),
            "content_md": content,
            "sources": sources,
        })

    return subsections


def parse_extractions():
    """Parse all extraction files into a dict keyed by episode filename."""
    extractions_dir = OUTPUT / "extractions"
    episodes = {}

    for path in sorted(extractions_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        # Split on ## Episode: {filename}
        pattern = re.compile(r"^## Episode:\s*(.+)$", re.MULTILINE)
        splits = list(pattern.finditer(text))

        for i, match in enumerate(splits):
            filename = match.group(1).strip()
            start = match.end()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
            block = text[start:end].strip()
            # Remove leading --- separators
            block = re.sub(r"^---\s*\n?", "", block).strip()
            # Remove trailing --- separators
            block = re.sub(r"\n---\s*$", "", block).strip()
            episodes[filename] = block

    return episodes


def build_episode_backlinks(themes):
    """Build reverse map: numeric_id → list of (theme_slug, theme_name, section_num)."""
    backlinks = {}
    for theme in themes:
        for sub in theme["subsections"]:
            for src in sub["sources"]:
                nid = numeric_id_from_filename(src)
                if nid not in backlinks:
                    backlinks[nid] = set()
                backlinks[nid].add((
                    theme["slug"],
                    theme["display_name"],
                    theme["section_num"],
                ))
    # Convert sets to sorted lists
    return {k: sorted(v, key=lambda x: x[2]) for k, v in backlinks.items()}


def count_theme_episodes(themes):
    """Count unique source episodes per theme."""
    counts = {}
    for theme in themes:
        eps = set()
        for sub in theme["subsections"]:
            for src in sub["sources"]:
                eps.add(numeric_id_from_filename(src))
        counts[theme["theme_key"]] = len(eps)
    return counts


# ---------------------------------------------------------------------------
# Markdown to HTML (lightweight)
# ---------------------------------------------------------------------------

def md_to_html(text, section_map=None):
    """Convert markdown text to HTML. section_map maps section numbers to theme slugs."""
    # Escape HTML first
    text = html.escape(text)

    # Convert (Section N) cross-references to links
    if section_map:
        def replace_section_ref(m):
            nums = re.findall(r"\d+", m.group(0))
            parts = []
            for n in nums:
                n_int = int(n)
                if n_int in section_map:
                    slug, name = section_map[n_int]
                    parts.append(f'<a href="../themes/{slug}.html" class="xref">Section {n}</a>')
                else:
                    parts.append(f"Section {n}")
            if len(parts) == 1:
                return f"({parts[0]})"
            return "(" + ", ".join(parts) + ")"

        text = re.sub(r"\(Sections?\s+[\d,\s]+(?:(?:and|Section)\s+[\d,\s]*)*\)", replace_section_ref, text)
        # Standalone: "Section 1" not already inside an <a> tag
        def replace_single_ref(m):
            n_int = int(m.group(1))
            if n_int in section_map:
                slug, name = section_map[n_int]
                return f'<a href="../themes/{slug}.html" class="xref">Section {m.group(1)}</a>'
            return m.group(0)
        text = re.sub(r'(?<!>)Section (\d+)(?![^<]*</a>)', replace_single_ref, text)

    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)

    # Convert source episode filenames to links (both full and short forms)
    # Full: 123456-empiricalcyclingpodcast-slug.txt
    # Short: 123456-slug (no empiricalcyclingpodcast-, no .txt)
    def _episode_link(m):
        nid = m.group(0).split("-")[0]
        title = filename_to_title(m.group(0))
        return f'<a href="../episodes/{nid}.html" class="episode-link">{title}</a>'

    text = re.sub(
        r"(\d{6,})-empiricalcyclingpodcast-[a-z0-9-]+\.txt",
        _episode_link, text,
    )
    text = re.sub(
        r"(\d{6,})-(?!empiricalcyclingpodcast)(?:ten-minute-tips|watts-doc|perspectives|instagram)[a-z0-9-]*",
        _episode_link, text,
    )

    # Process line by line for block elements
    lines = text.split("\n")
    result = []
    in_list = False
    in_paragraph = False

    for line in lines:
        stripped = line.strip()

        # Blank line
        if not stripped:
            if in_list:
                result.append("</ul>")
                in_list = False
            if in_paragraph:
                result.append("</p>")
                in_paragraph = False
            continue

        # List item
        if stripped.startswith("- "):
            if in_paragraph:
                result.append("</p>")
                in_paragraph = False
            if not in_list:
                result.append('<ul class="recommendations">')
                in_list = True
            result.append(f"<li>{stripped[2:]}</li>")
            continue

        # Heading (### within content — shouldn't appear, but handle)
        if stripped.startswith("### "):
            if in_list:
                result.append("</ul>")
                in_list = False
            if in_paragraph:
                result.append("</p>")
                in_paragraph = False
            result.append(f"<h4>{stripped[4:]}</h4>")
            continue

        # Regular text
        if not in_paragraph:
            result.append("<p>")
            in_paragraph = True
        else:
            result.append(" ")
        result.append(stripped)

    if in_list:
        result.append("</ul>")
    if in_paragraph:
        result.append("</p>")

    return "\n".join(result)


def md_simple(text, section_map=None):
    """Simpler markdown conversion for short texts (cross-refs, methodology)."""
    text = html.escape(text)

    # Section references
    if section_map:
        def replace_ref(m):
            nums = re.findall(r"\d+", m.group(0))
            parts = []
            for n in nums:
                n_int = int(n)
                if n_int in section_map:
                    slug, name = section_map[n_int]
                    parts.append(f'<a href="themes/{slug}.html" class="xref">Section {n} ({name})</a>')
                else:
                    parts.append(f"Section {n}")
            if len(parts) == 1:
                return f"({parts[0]})"
            return "(" + ", ".join(parts) + ")"

        # Parenthesized: (Section 1), (Section 3, Section 5), (Sections 2, 3, 4, and 5)
        text = re.sub(r"\(Sections?\s+[\d,\s]+(?:(?:and|Section)\s+[\d,\s]*)*\)", replace_ref, text)
        # Non-parenthesized: Sections 2, 3, 4, and 5
        def replace_ref_bare(m):
            nums = re.findall(r"\d+", m.group(0))
            parts = []
            for n in nums:
                n_int = int(n)
                if n_int in section_map:
                    slug, name = section_map[n_int]
                    parts.append(f'<a href="themes/{slug}.html" class="xref">Section {n} ({name})</a>')
                else:
                    parts.append(f"Section {n}")
            return ", ".join(parts)
        text = re.sub(r"Sections\s+\d+(?:,\s*\d+)*(?:,?\s+and\s+\d+)", replace_ref_bare, text)
        # Standalone: "Section 1" not already inside an <a> tag
        def replace_ref_single(m):
            n_int = int(m.group(1))
            if n_int in section_map:
                slug, name = section_map[n_int]
                return f'<a href="themes/{slug}.html" class="xref">Section {m.group(1)} ({name})</a>'
            return m.group(0)
        text = re.sub(r'(?<!>)Section (\d+)(?![^<]*</a>)', replace_ref_single, text)

    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)

    lines = text.split("\n")
    result = []
    in_list = False
    for line in lines:
        s = line.strip()
        if not s:
            if in_list:
                result.append("</ul>")
                in_list = False
            continue
        if s.startswith("- "):
            if not in_list:
                result.append("<ul>")
                in_list = True
            result.append(f"<li>{s[2:]}</li>")
        else:
            if in_list:
                result.append("</ul>")
                in_list = False
            result.append(f"<p>{s}</p>")
    if in_list:
        result.append("</ul>")
    return "\n".join(result)


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

def nav_html(themes, current_slug=None, depth=""):
    """Generate sidebar navigation HTML. depth is '../' for nested pages."""
    items = []
    for t in themes:
        cls = ' class="active"' if t["slug"] == current_slug else ""
        items.append(
            f'<li{cls}><a href="{depth}themes/{t["slug"]}.html">'
            f'{t["section_num"]}. {html.escape(t["display_name"])}</a></li>'
        )

    return f"""<nav class="sidebar" id="sidebar">
  <div class="sidebar-header">
    <a href="{depth}index.html" class="site-title">EC Knowledge Base</a>
  </div>
  <ul class="nav-list">
    {chr(10).join(items)}
  </ul>
  <div class="nav-footer">
    <a href="{depth}cross-references.html">Cross-References</a>
    <a href="{depth}methodology.html">Methodology</a>
  </div>
</nav>"""


def page_html(title, body, themes, current_slug=None, depth=""):
    nav = nav_html(themes, current_slug, depth)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title) + ' — ' if title != 'EC Knowledge Base' else ''}EC Knowledge Base</title>
  <link rel="stylesheet" href="{depth}style.css">
</head>
<body>
  <button class="sidebar-toggle" id="sidebar-toggle" aria-label="Toggle navigation">&#9776;</button>
  {nav}
  <main class="content">
    {body}
  </main>
  <script>
    document.getElementById('sidebar-toggle').addEventListener('click', function() {{
      document.getElementById('sidebar').classList.toggle('open');
    }});
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Page generators
# ---------------------------------------------------------------------------

def format_executive_summary(text):
    """Split the (1)...(7) executive summary into an ordered list."""
    # Split on (N) markers
    parts = re.split(r"\((\d+)\)\s*", text)
    # parts[0] is the preamble, then alternating: number, content
    preamble = parts[0].strip()
    items = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            content = parts[i + 1].strip().rstrip(".")
            items.append(f"<li>{html.escape(content)}.</li>")

    if not items:
        return f"<p>{html.escape(text)}</p>"

    return (
        f"<p>{html.escape(preamble)}</p>\n"
        f'<ol class="summary-list">{"".join(items)}</ol>'
    )


def generate_index(themes, executive_summary, episode_counts):
    cards = []
    for t in themes:
        ep_count = episode_counts.get(t["theme_key"], 0)
        sub_count = len(t["subsections"])
        cards.append(f"""<a href="themes/{t['slug']}.html" class="theme-card">
  <span class="card-num">{t['section_num']}</span>
  <h2>{html.escape(t['display_name'])}</h2>
  <p class="card-meta">{sub_count} topics &middot; {ep_count} episodes</p>
</a>""")

    # Split executive summary into numbered findings
    summary_html = format_executive_summary(executive_summary)

    body = f"""<div class="index-header">
  <h1>Empirical Cycling Podcast: Training Knowledge Base</h1>
  <div class="executive-summary">
    {summary_html}
  </div>
</div>
<div class="theme-grid">
  {chr(10).join(cards)}
</div>"""

    return page_html(
        "EC Knowledge Base",
        body, themes, current_slug=None, depth="",
    )


def generate_theme_page(theme, themes, section_map):
    # Table of contents for subsections
    toc_items = []
    for sub in theme["subsections"]:
        toc_items.append(
            f'<li><a href="#{sub["anchor"]}">{html.escape(sub["title"])}</a></li>'
        )

    toc = f"""<div class="page-toc">
  <h3>In this section</h3>
  <ol>{chr(10).join(toc_items)}</ol>
</div>"""

    # Render subsections
    sections_html = []
    for sub in theme["subsections"]:
        content = md_to_html(sub["content_md"], section_map)
        sections_html.append(
            f'<section class="subsection" id="{sub["anchor"]}">\n'
            f'  <h3>{html.escape(sub["title"])}</h3>\n'
            f'  {content}\n'
            f'</section>'
        )

    body = f"""<h1>{theme['section_num']}. {html.escape(theme['display_name'])}</h1>
{toc}
{chr(10).join(sections_html)}"""

    return page_html(
        f"{theme['section_num']}. {theme['display_name']}",
        body, themes, current_slug=theme["slug"], depth="../",
    )


def generate_episode_page(filename, extraction_content, backlinks, themes, clusters_meta):
    nid = numeric_id_from_filename(filename)
    title = filename_to_title(filename)
    series = series_from_filename(filename)

    # Get word count from clusters.json
    word_count = "—"
    for cluster_data in clusters_meta.values():
        for t in cluster_data.get("transcripts", []):
            if t["filename"] == filename:
                word_count = f'{t["word_count"]:,}'
                break

    # Render extraction content
    if extraction_content:
        content_html = render_extraction(extraction_content)
    else:
        content_html = '<p class="no-content">No extraction content available for this episode.</p>'

    # Backlinks
    bl_html = ""
    ep_backlinks = backlinks.get(nid, [])
    if ep_backlinks:
        links = []
        for slug, name, snum in ep_backlinks:
            links.append(f'<a href="../themes/{slug}.html">{snum}. {html.escape(name)}</a>')
        bl_html = f"""<div class="appears-in">
  <h3>Referenced in</h3>
  <div class="theme-tags">{chr(10).join(links)}</div>
</div>"""

    body = f"""<div class="episode-header">
  <h1>{html.escape(title)}</h1>
  <div class="episode-meta">
    <span class="series-badge">{html.escape(series)}</span>
    <span>{word_count} words</span>
    <a href="../transcripts/{html.escape(filename)}" class="transcript-link">View transcript</a>
  </div>
</div>
{bl_html}
<div class="extraction-content">
{content_html}
</div>"""

    return page_html(title, body, themes, depth="../")


def render_extraction(text):
    """Render extraction block markdown to HTML."""
    # Remove metadata lines at start (Numeric ID, Word count, Series)
    text = re.sub(r"^\*\*Numeric ID:\*\*.*\n?", "", text)
    text = re.sub(r"^\*\*Word count:\*\*.*\n?", "", text)
    text = re.sub(r"^\*\*Series:\*\*.*\n?", "", text)
    text = text.strip()

    result = []
    lines = text.split("\n")
    in_list = False
    in_p = False

    for line in lines:
        s = line.strip()

        if not s:
            if in_list:
                result.append("</ul>")
                in_list = False
            if in_p:
                result.append("</p>")
                in_p = False
            continue

        # ### heading
        if s.startswith("### "):
            if in_list:
                result.append("</ul>")
                in_list = False
            if in_p:
                result.append("</p>")
                in_p = False
            heading = html.escape(s[4:])
            result.append(f"<h3>{heading}</h3>")
            continue

        # List
        if s.startswith("- "):
            if in_p:
                result.append("</p>")
                in_p = False
            if not in_list:
                result.append("<ul>")
                in_list = True
            item = html.escape(s[2:])
            item = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", item)
            item = re.sub(r"\*(.+?)\*", r"<em>\1</em>", item)
            result.append(f"<li>{item}</li>")
            continue

        # Regular text
        escaped = html.escape(s)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(r"\*(.+?)\*", r"<em>\1</em>", escaped)
        if not in_p:
            result.append("<p>")
            in_p = True
        else:
            result.append(" ")
        result.append(escaped)

    if in_list:
        result.append("</ul>")
    if in_p:
        result.append("</p>")

    return "\n".join(result)


def generate_cross_references(text, themes, section_map):
    content = md_simple(text, section_map)
    body = f"""<h1>Cross-References</h1>
<p class="page-intro">Connections between themes — how findings in one area inform or constrain recommendations in another.</p>
{content}"""
    return page_html("Cross-References", body, themes, depth="")


def generate_methodology(text, themes):
    content = md_simple(text)
    body = f"""<h1>Methodology</h1>
{content}"""
    return page_html("Methodology", body, themes, depth="")


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """\
:root {
  --sidebar-w: 260px;
  --content-max: 760px;
  --bg: #fafaf9;
  --surface: #fff;
  --text: #1a1a1a;
  --text-secondary: #555;
  --accent: #2563eb;
  --accent-light: #dbeafe;
  --border: #e5e5e5;
  --card-hover: #f5f5f4;
  --radius: 6px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html { font-size: 17px; -webkit-text-size-adjust: 100%; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  color: var(--text);
  background: var(--bg);
  line-height: 1.65;
  display: flex;
  min-height: 100vh;
}

/* Sidebar */
.sidebar {
  position: fixed;
  top: 0; left: 0;
  width: var(--sidebar-w);
  height: 100vh;
  overflow-y: auto;
  background: var(--surface);
  border-right: 1px solid var(--border);
  padding: 1.25rem 0;
  display: flex;
  flex-direction: column;
  z-index: 100;
}
.sidebar-header { padding: 0 1.25rem 1rem; border-bottom: 1px solid var(--border); }
.site-title {
  font-weight: 700;
  font-size: 0.95rem;
  color: var(--text);
  text-decoration: none;
  letter-spacing: -0.01em;
}
.nav-list {
  list-style: none;
  padding: 0.75rem 0;
  flex: 1;
  overflow-y: auto;
}
.nav-list li a {
  display: block;
  padding: 0.35rem 1.25rem;
  color: var(--text-secondary);
  text-decoration: none;
  font-size: 0.85rem;
  line-height: 1.4;
  transition: background 0.15s;
}
.nav-list li a:hover { background: var(--card-hover); color: var(--text); }
.nav-list li.active a { color: var(--accent); font-weight: 600; background: var(--accent-light); }
.nav-footer {
  padding: 0.75rem 1.25rem;
  border-top: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}
.nav-footer a {
  font-size: 0.8rem;
  color: var(--text-secondary);
  text-decoration: none;
}
.nav-footer a:hover { color: var(--accent); }

/* Content */
.content {
  margin-left: var(--sidebar-w);
  max-width: var(--content-max);
  padding: 2rem 2.5rem 4rem;
  flex: 1;
}

/* Sidebar toggle (mobile) */
.sidebar-toggle {
  display: none;
  position: fixed;
  top: 0.75rem; left: 0.75rem;
  z-index: 200;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.4rem 0.6rem;
  font-size: 1.2rem;
  cursor: pointer;
  line-height: 1;
}

/* Typography */
h1 { font-size: 1.75rem; font-weight: 700; margin-bottom: 1rem; letter-spacing: -0.02em; line-height: 1.25; }
h2 { font-size: 1.3rem; font-weight: 600; margin: 2rem 0 0.75rem; letter-spacing: -0.01em; }
h3 { font-size: 1.1rem; font-weight: 600; margin: 1.75rem 0 0.5rem; color: var(--text); }
h4 { font-size: 1rem; font-weight: 600; margin: 1.25rem 0 0.4rem; }
p { margin-bottom: 0.75rem; }
a { color: var(--accent); }

/* Subsections */
.subsection { padding-top: 0.5rem; border-top: 1px solid var(--border); margin-top: 1.5rem; }
.subsection:first-child { border-top: none; margin-top: 0; }
.subsection h3 { margin-top: 0.5rem; }

/* Recommendations lists */
ul.recommendations { margin: 0.5rem 0 0.75rem 1.25rem; }
ul.recommendations li { margin-bottom: 0.3rem; }
ul { margin: 0.5rem 0 0.75rem 1.25rem; }
li { margin-bottom: 0.3rem; }

/* Episode links */
.episode-link {
  display: inline;
  font-size: 0.85rem;
  color: var(--accent);
  text-decoration: none;
}
.episode-link:hover { text-decoration: underline; }

/* Cross-reference links */
.xref { font-weight: 500; }

/* Page TOC */
.page-toc {
  background: var(--card-hover);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.25rem;
  margin-bottom: 1.5rem;
}
.page-toc h3 { margin: 0 0 0.5rem; font-size: 0.9rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }
.page-toc ol { margin: 0; padding-left: 1.25rem; }
.page-toc li { font-size: 0.88rem; margin-bottom: 0.2rem; }
.page-toc a { text-decoration: none; }
.page-toc a:hover { text-decoration: underline; }

/* Index */
.index-header { margin-bottom: 2rem; }
.executive-summary { font-size: 0.95rem; color: var(--text-secondary); line-height: 1.7; }
.summary-list { margin: 0.75rem 0 0 1.25rem; }
.summary-list li { margin-bottom: 0.5rem; }
.theme-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 1rem;
}
.theme-card {
  display: block;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem;
  text-decoration: none;
  color: var(--text);
  transition: border-color 0.15s, box-shadow 0.15s;
}
.theme-card:hover { border-color: var(--accent); box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.theme-card .card-num { font-size: 0.8rem; color: var(--text-secondary); }
.theme-card h2 { font-size: 1rem; margin: 0.25rem 0 0.5rem; line-height: 1.3; }
.card-meta { font-size: 0.8rem; color: var(--text-secondary); margin: 0; }

/* Episode page */
.episode-header { margin-bottom: 1.5rem; }
.episode-meta {
  display: flex;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
  font-size: 0.85rem;
  color: var(--text-secondary);
  margin-top: 0.5rem;
}
.series-badge {
  display: inline-block;
  background: var(--accent-light);
  color: var(--accent);
  padding: 0.15rem 0.6rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
}
.transcript-link {
  font-size: 0.85rem;
}
.appears-in {
  background: var(--card-hover);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.75rem 1rem;
  margin-bottom: 1.5rem;
}
.appears-in h3 { font-size: 0.85rem; margin: 0 0 0.5rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.04em; }
.theme-tags { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.theme-tags a {
  display: inline-block;
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 0.2rem 0.7rem;
  border-radius: var(--radius);
  font-size: 0.82rem;
  text-decoration: none;
  color: var(--text);
}
.theme-tags a:hover { border-color: var(--accent); color: var(--accent); }

.extraction-content h3 { color: var(--accent); font-size: 1rem; }
.no-content { color: var(--text-secondary); font-style: italic; }
.page-intro { color: var(--text-secondary); margin-bottom: 1.5rem; }

/* Responsive */
@media (max-width: 768px) {
  .sidebar { transform: translateX(-100%); transition: transform 0.25s; }
  .sidebar.open { transform: translateX(0); box-shadow: 2px 0 12px rgba(0,0,0,0.15); }
  .sidebar-toggle { display: block; }
  .content { margin-left: 0; padding: 3rem 1.25rem 3rem; }
  .theme-grid { grid-template-columns: 1fr; }
}

/* Print */
@media print {
  .sidebar, .sidebar-toggle { display: none; }
  .content { margin-left: 0; max-width: 100%; }
}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    theme_map = load_theme_map()
    final_md = load_final_output()
    exec_summary = load_framing("executive_summary.md")
    cross_refs = load_framing("cross_references.md")
    methodology = load_framing("methodology_note.md")
    clusters_meta = load_clusters_json()

    print("Parsing themes...")
    themes = parse_themes(final_md, theme_map)
    episode_counts = count_theme_episodes(themes)
    print(f"  {len(themes)} themes, {sum(len(t['subsections']) for t in themes)} subsections")

    print("Parsing extractions...")
    extractions = parse_extractions()
    print(f"  {len(extractions)} episodes")

    print("Building backlinks...")
    backlinks = build_episode_backlinks(themes)

    # Section map: section_num → (slug, display_name)
    section_map = {t["section_num"]: (t["slug"], t["display_name"]) for t in themes}

    # Create output dirs
    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "themes").mkdir(parents=True)
    (SITE / "episodes").mkdir(parents=True)

    # Copy transcripts into site so episode links work
    transcripts_src = ROOT / "data" / "transcripts"
    transcripts_dst = SITE / "transcripts"
    if transcripts_src.exists():
        print("Copying transcripts...")
        shutil.copytree(transcripts_src, transcripts_dst)
        print(f"  {len(list(transcripts_dst.iterdir()))} transcript files")

    # CSS
    print("Writing CSS...")
    (SITE / "style.css").write_text(CSS)

    # Index
    print("Generating index...")
    (SITE / "index.html").write_text(
        generate_index(themes, exec_summary, episode_counts)
    )

    # Theme pages
    print("Generating theme pages...")
    for t in themes:
        path = SITE / "themes" / f"{t['slug']}.html"
        path.write_text(generate_theme_page(t, themes, section_map))
        print(f"  {t['slug']}.html ({len(t['subsections'])} subsections)")

    # Episode pages — from extractions + any remaining episodes in clusters.json
    print("Generating episode pages...")
    ep_count = 0
    generated_nids = set()

    # First: episodes with extraction content
    for filename, content in sorted(extractions.items()):
        nid = numeric_id_from_filename(filename)
        generated_nids.add(nid)
        path = SITE / "episodes" / f"{nid}.html"
        path.write_text(
            generate_episode_page(filename, content, backlinks, themes, clusters_meta)
        )
        ep_count += 1

    # Second: episodes from clusters.json that lack extraction content
    for cluster_data in clusters_meta.values():
        for t in cluster_data.get("transcripts", []):
            nid = str(t["numeric_id"])
            if nid not in generated_nids:
                generated_nids.add(nid)
                path = SITE / "episodes" / f"{nid}.html"
                path.write_text(
                    generate_episode_page(t["filename"], None, backlinks, themes, clusters_meta)
                )
                ep_count += 1

    print(f"  {ep_count} episode pages")

    # Supplementary pages
    print("Generating supplementary pages...")
    (SITE / "cross-references.html").write_text(
        generate_cross_references(cross_refs, themes, section_map)
    )
    (SITE / "methodology.html").write_text(
        generate_methodology(methodology, themes)
    )

    print(f"\nDone! Site written to {SITE}/")
    print(f"  {len(themes)} theme pages")
    print(f"  {ep_count} episode pages")
    print(f"  2 supplementary pages")
    print(f"\nView: python3 -m http.server -d site")


if __name__ == "__main__":
    main()
