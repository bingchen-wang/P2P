"""Sphinx configuration for the P2P documentation."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "p2p-matplotlib"),
)
sys.path.insert(0, str(ROOT))

from p2p import __version__  # noqa: E402


project = "P2P"
author = "Bingchen Wang"
copyright = "2026, Bingchen Wang"
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The theme follows P2P's visual language: the Experiment Studio's dark frame
# in dark mode, warm paper in light mode, coral and teal accents and serif titles.
html_theme = "furo"
html_title = f"P2P {release}"
html_short_title = "P2P"
html_static_path = ["_static", "../assets"]
html_css_files = [
    # Newsreader for titles and Inter for text, shared with the P2P website.
    # Without network access the stacks fall back to Iowan/Georgia and system sans.
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
    "&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap",
    "p2p.css",
]
html_favicon = "../../studio/public/p2p-favicon.png"
pygments_style = "friendly"
pygments_dark_style = "github-dark"
html_theme_options = {
    "light_logo": "p2p-logo-light.png",
    "dark_logo": "p2p-logo-dark.png",
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/bingchen-wang/P2P/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/bingchen-wang/P2P",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 '
                "3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 "
                "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94"
                "-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 "
                "1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89"
                "-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 "
                ".67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 "
                "1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 "
                "1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 "
                "0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 "
                '16 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
    # Light mode: warm paper, green-black ink, coral labels, teal links and
    # leaf for what passed; text colours meet WCAG AA on the paper.
    "light_css_variables": {
        "color-brand-primary": "#b4502f",
        "color-brand-content": "#2b6b63",
        "color-brand-visited": "#2b6b63",
        "color-foreground-primary": "#1b1f1d",
        "color-foreground-secondary": "#4a524e",
        "color-foreground-muted": "#5c6560",
        "color-foreground-border": "#cfcbc0",
        "color-background-primary": "#faf9f5",
        "color-background-secondary": "#f3f1ea",
        "color-background-hover": "#ebe9e1",
        "color-background-border": "#e3e0d6",
        "color-sidebar-background": "#f3f1ea",
        "color-sidebar-background-border": "#e3e0d6",
        "color-sidebar-link-text--top-level": "#1b1f1d",
        "color-code-background": "#f3f1ea",
        "color-code-foreground": "#1b1f1d",
        "color-inline-code-background": "#f1efe8",
        "color-highlighted-background": "#f6e3d8",
        "color-api-name": "#2b6b63",
        "color-api-pre-name": "#4a524e",
        "p2p-card-background": "#ffffff",
        "p2p-verified": "#4a7434",
        "font-stack": 'Inter, "Avenir Next", Avenir, "Segoe UI", system-ui, sans-serif',
        "font-stack--monospace": '"SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace',
        "font-stack--headings": 'Newsreader, "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif',
    },
    "dark_css_variables": {
        "color-brand-primary": "#ec8062",
        "color-brand-content": "#77b7ae",
        "color-brand-visited": "#77b7ae",
        "color-foreground-primary": "#eef0e9",
        "color-foreground-secondary": "#c9cec7",
        "color-foreground-muted": "#969d98",
        "color-foreground-border": "#343d45",
        "color-background-primary": "#0d1013",
        "color-background-secondary": "#090b0e",
        "color-background-hover": "#171c21",
        "color-background-border": "#252c32",
        "color-sidebar-background": "#090b0e",
        "color-sidebar-background-border": "#252c32",
        "color-sidebar-link-text--top-level": "#eef0e9",
        "color-code-background": "#111417",
        "color-code-foreground": "#c9cec7",
        "color-inline-code-background": "#171c21",
        "color-highlighted-background": "#3a2a24",
        "color-api-name": "#77b7ae",
        "color-api-pre-name": "#969d98",
        "color-admonition-background": "#13181e",
        "color-card-background": "#13181e",
        "p2p-card-background": "#13181e",
        "p2p-verified": "#9dbe87",
        "color-card-border": "#293138",
    },
}
