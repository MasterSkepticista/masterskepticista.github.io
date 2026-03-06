#!/usr/bin/env python3
"""
Static blog generator inspired by Tufte's style, with support for markdown posts, sidenotes, math, and external collections.
Converts markdown posts to HTML using Tufte-inspired styling
"""

import argparse
import re
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlsplit, urlunsplit
import markdown
import frontmatter
import shutil
from jinja2 import Environment, FileSystemLoader

class BlogGenerator:
    def __init__(self, root_dir="."):
        self.root_dir = Path(root_dir)
        self.posts_dir = self.root_dir / "posts"
        self.code_dir = self.root_dir / "code"
        self.papers_dir = self.root_dir / "papers"
        self.talks_dir = self.root_dir / "talks"
        self.fonts_dir = self.root_dir / "fonts"
        self.output_dir = self.root_dir / "docs"
        self.templates_dir = self.root_dir / "templates"
        self.static_dir = self.root_dir / "static"
        self.config_path = self.root_dir / "site.json"
        self.config = {}
        self.base_url = ""
        self.refresh_config()
        
        # Setup Jinja2
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True,
        )
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)

    def refresh_config(self):
        self.config = self.load_config()
        self.base_url = self.normalize_base_url(self.config.get("base_url", ""))

    def load_config(self) -> dict:
        """Load site-wide configuration"""
        default_config = {
            "site_name": "blog",
            "author": "",
            "about_path": "about/",
            "base_url": "",
            "site_url": "",
            "subscribe_url": "",
            "twitter_url": "",
            "github_url": "",
            "scholar_url": "",
        }

        if not self.config_path.exists():
            return default_config

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            return {**default_config, **user_config}
        except Exception:
            return default_config

    @staticmethod
    def normalize_base_url(base_url: str) -> str:
        if not base_url:
            return ""
        normalized = base_url.strip()
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"
        return normalized.rstrip("/")

    def build_url(self, path: str = "") -> str:
        clean_path = path.lstrip("/")
        if self.base_url:
            return f"{self.base_url}/{clean_path}" if clean_path else f"{self.base_url}/"
        return f"/{clean_path}" if clean_path else "/"

    def build_site_links(self) -> list:
        explicit_links = self.config.get("links")
        if isinstance(explicit_links, list):
            parsed_links = []
            for item in explicit_links:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url", "")).strip()
                if not url:
                    continue
                icon = str(item.get("icon", item.get("name", "link"))).strip().lower().replace(" ", "_")
                name = str(item.get("name", icon.replace("_", " ").title())).strip()
                parsed_links.append({
                    "name": name,
                    "url": url,
                    "icon": icon,
                })
            if parsed_links:
                return parsed_links

        display_name_map = {
            "github": "GitHub",
            "twitter": "X",
            "x": "X",
            "scholar": "Google Scholar",
            "linkedin": "LinkedIn",
            "subscribe": "Subscribe",
            "youtube": "YouTube",
            "website": "Website",
        }

        links = []
        seen_keys = set()
        priority = [
            "github_url",
            "twitter_url",
            "x_url",
            "scholar_url",
            "linkedin_url",
            "subscribe_url",
            "youtube_url",
            "website_url",
        ]

        ordered_keys = priority + [k for k in self.config.keys() if k not in priority]
        for key in ordered_keys:
            if key in seen_keys:
                continue
            seen_keys.add(key)

            if not key.endswith("_url") or key == "site_url":
                continue

            value = self.config.get(key, "")
            if not isinstance(value, str) or not value.strip():
                continue

            icon_key = key[:-4].lower()
            links.append({
                "name": display_name_map.get(icon_key, icon_key.replace("_", " ").title()),
                "url": value.strip(),
                "icon": icon_key,
            })

        return links

    def base_context(self, asset_prefix: str) -> dict:
        return {
            "site": self.config,
            "home_url": self.build_url(),
            "code_url": self.build_url("code/"),
            "papers_url": self.build_url("papers/"),
            "slides_url": self.build_url("slides/"),
            "resume_url": self.config.get("resume_url", ""),
            "asset_prefix": asset_prefix,
            "site_links": self.build_site_links(),
        }

    @staticmethod
    def remove_duplicate_h1(content_html: str, post_title: str) -> str:
        match = re.match(r"^\s*<h1[^>]*>(.*?)</h1>\s*", content_html, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return content_html

        first_heading = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        if first_heading.lower() == post_title.strip().lower():
            return content_html[match.end():].lstrip()

        return content_html

    @staticmethod
    def _strip_single_paragraph_wrapper(html: str) -> str:
        stripped = html.strip()
        match = re.fullmatch(r"<p>(.*)</p>", stripped, flags=re.DOTALL)
        if match and stripped.count("<p>") == 1 and stripped.count("</p>") == 1:
            return match.group(1)
        return stripped

    def find_post_thumbnail(self, post_path: Path, slug: str) -> str:
        preferred_extensions = [".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif", ".svg"]
        search_dirs = [post_path / "images", post_path]

        for directory in search_dirs:
            if not directory.exists() or not directory.is_dir():
                continue

            thumbs_by_ext = {}
            for candidate in directory.iterdir():
                if not candidate.is_file() or candidate.stem.lower() != "thumb":
                    continue
                thumbs_by_ext[candidate.suffix.lower()] = candidate

            for extension in preferred_extensions:
                thumb_file = thumbs_by_ext.get(extension)
                if not thumb_file:
                    continue

                if directory.name == "images":
                    return self.build_url(f"posts/{slug}/images/{thumb_file.name}")
                return self.build_url(f"posts/{slug}/{thumb_file.name}")

        return ""

    def _render_sidenote_markdown(self, note_text: str) -> str:
        note_md = markdown.Markdown(extensions=['extra'])
        rendered = note_md.convert(note_text.strip())
        return self._strip_single_paragraph_wrapper(rendered)

    def _extract_sidenotes(self, content: str):
        result_parts = []
        replacements = []
        idx = 0

        while idx < len(content):
            is_sidenote_start = (
                content[idx] == '^'
                and idx + 1 < len(content)
                and content[idx + 1] == '['
                and (idx == 0 or content[idx - 1] != '\\')
            )

            if not is_sidenote_start:
                result_parts.append(content[idx])
                idx += 1
                continue

            start_idx = idx
            idx += 2
            depth = 1
            note_chars = []

            while idx < len(content):
                char = content[idx]

                if char == '\\' and idx + 1 < len(content):
                    note_chars.append(char)
                    idx += 1
                    note_chars.append(content[idx])
                elif char == '[':
                    depth += 1
                    note_chars.append(char)
                elif char == ']':
                    depth -= 1
                    if depth == 0:
                        idx += 1
                        break
                    note_chars.append(char)
                else:
                    note_chars.append(char)

                idx += 1

            if depth != 0:
                result_parts.append(content[start_idx])
                idx = start_idx + 1
                continue

            note_id = len(replacements) + 1
            placeholder = f"SIDENOTETOKEN{note_id}END"
            note_html = self._render_sidenote_markdown(''.join(note_chars))
            sidenote_markup = (
                f'<label for="sn-{note_id}" class="margin-toggle sidenote-number">{note_id}</label>'
                f'<input type="checkbox" id="sn-{note_id}" class="margin-toggle">'
                f'<span class="sidenote"><span class="sidenote-prefix">{note_id}. </span>{note_html}</span>'
            )

            result_parts.append(placeholder)
            replacements.append((placeholder, sidenote_markup))

        return ''.join(result_parts), replacements
        
    def process_markdown(self, content: str) -> str:
        """
        Process markdown with extensions:
        - Math equations ($$...$$ and $...$ syntax)
        - Sidenotes (^[note text])
        - Code highlighting
        - Tables
        """
        
        # Protect math blocks from markdown processing
        # Note: placeholders must avoid markdown markers like "__...__",
        # otherwise markdown can rewrite them before restore.
        math_blocks = []

        def add_math_placeholder(math_type: str, formula: str) -> str:
            placeholder = f"MATH{math_type.upper()}TOKEN{len(math_blocks)}END"
            math_blocks.append((placeholder, math_type, formula))
            return placeholder
        
        # Protect display math $$...$$
        def protect_display_math(match):
            return add_math_placeholder('display', match.group(1))
        
        content = re.sub(r'\$\$(.*?)\$\$', protect_display_math, content, flags=re.DOTALL)
        
        # Protect inline math $...$
        def protect_inline_math(match):
            return add_math_placeholder('inline', match.group(1))
        
        content = re.sub(r'(?<!\$)\$(?!\$)([^\$\n]+?)\$(?!\$)', protect_inline_math, content)

        content, sidenote_replacements = self._extract_sidenotes(content)
        
        # Parse markdown with extensions
        md = markdown.Markdown(
            extensions=[
                'extra',
                'tables',
                'fenced_code',
                'codehilite',
                'toc',
            ],
            extension_configs={
                'codehilite': {
                    'guess_lang': False,
                    'use_pygments': True,
                    'css_class': 'highlight',
                    'noclasses': False,
                },
                'toc': {
                    'permalink': False,
                },
            },
        )
        
        html_content = md.convert(content)

        for placeholder, sidenote_markup in sidenote_replacements:
            html_content = html_content.replace(placeholder, sidenote_markup)
        
        # Restore math blocks
        def restore_math(html):
            for placeholder, math_type, formula in math_blocks:
                if math_type == 'display':
                    html = html.replace(placeholder, f'$$\n{formula}\n$$')
                else:
                    html = html.replace(placeholder, f'${formula}$')
            return html
        
        html_content = restore_math(html_content)
        
        return html_content

    @staticmethod
    def format_date_fields(raw_date) -> tuple[str, str]:
        date_formatted = ''
        date_iso = ''
        if not raw_date:
            return date_formatted, date_iso

        try:
            if isinstance(raw_date, str):
                date_obj = datetime.fromisoformat(raw_date.replace('Z', '+00:00'))
            else:
                date_obj = raw_date
            date_formatted = date_obj.strftime('%B %Y')
            date_iso = date_obj.isoformat()
        except Exception:
            date_formatted = str(raw_date)
            date_iso = str(raw_date)

        return date_formatted, date_iso

    def render_summary_markdown(self, summary: str) -> str:
        if not summary:
            return ''

        summary_md = markdown.Markdown(extensions=['extra'])
        rendered = summary_md.convert(summary.strip())
        return self._strip_single_paragraph_wrapper(rendered)

    def read_external_collection(self, collection_dir: Path, fallback_title: str) -> dict:
        section = {
            'title': fallback_title,
            'entries': [],
        }

        if not collection_dir.exists() or not collection_dir.is_dir():
            return section

        entries = []
        for entry_file in sorted(collection_dir.glob('*.md')):
            if entry_file.name.startswith('_'):
                continue

            try:
                with open(entry_file, 'r', encoding='utf-8') as f:
                    entry = frontmatter.load(f)
            except Exception:
                continue

            external_url = str(entry.metadata.get('externalUrl', '')).strip()
            if not external_url:
                continue

            title = str(entry.metadata.get('title', entry_file.stem.replace('-', ' ').title())).strip()
            summary = str(entry.metadata.get('summary', '')).strip()
            date_formatted, date_iso = self.format_date_fields(entry.metadata.get('date', ''))

            entries.append({
                'title': title,
                'summary': summary,
                'summary_html': self.render_summary_markdown(summary),
                'external_url': external_url,
                'date_formatted': date_formatted,
                'date_iso': date_iso,
            })

        section['entries'] = sorted(entries, key=lambda item: item.get('date_iso', ''), reverse=True)
        return section

    def load_external_sections(self) -> list:
        section_specs = [
            (self.code_dir, 'Code', 'code'),
            (self.papers_dir, 'Papers', 'papers'),
            (self.talks_dir, 'Slides', 'slides'),
        ]

        sections = []
        for directory, title, slug in section_specs:
            section = self.read_external_collection(directory, title)
            section['slug'] = slug
            sections.append(section)

        return sections
    
    def read_post(self, post_path: Path) -> dict:
        """Read a post folder and extract metadata + content"""
        post_file = post_path / "post.md"
        
        if not post_file.exists():
            raise FileNotFoundError(f"post.md not found in {post_path}")
        
        with open(post_file, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)
        
        # Extract metadata with defaults
        metadata = {
            'title': post.metadata.get('title', 'Untitled'),
            'date': post.metadata.get('date', ''),
            'author': post.metadata.get('author', ''),
            'description': post.metadata.get('description', ''),
            'slug': post_path.name,
            'thumbnail_url': self.find_post_thumbnail(post_path, post_path.name),
            'external_url': post.metadata.get('externalUrl', ''),
        }
        
        # Parse date
        metadata['date_formatted'], metadata['date_iso'] = self.format_date_fields(metadata['date'])
        
        # Process markdown content
        html_content = self.process_markdown(post.content)
        html_content = self.remove_duplicate_h1(html_content, metadata['title'])

        site_url = self.config.get('site_url', '').rstrip('/')
        if site_url:
            metadata['canonical_url'] = f"{site_url}/posts/{metadata['slug']}/"
        else:
            metadata['canonical_url'] = ''
        
        return {
            **metadata,
            'content': html_content,
            'path': post_path,
        }
    
    def copy_post_assets(self, post_path: Path, output_path: Path):
        """Copy images and other assets from post folder"""
        assets_dirs = [
            post_path / 'images',
            post_path / 'assets',
            post_path / 'files',
        ]
        
        for assets_dir in assets_dirs:
            if assets_dir.exists():
                dest = output_path / assets_dir.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(assets_dir, dest)

    def copy_static_assets(self):
        """Copy static assets and fonts into output"""
        if self.static_dir.exists():
            static_dest = self.output_dir / 'static'
            if static_dest.exists():
                shutil.rmtree(static_dest)
            shutil.copytree(self.static_dir, static_dest)

        if self.fonts_dir.exists():
            fonts_dest = self.output_dir / 'fonts'
            if fonts_dest.exists():
                shutil.rmtree(fonts_dest)
            shutil.copytree(self.fonts_dir, fonts_dest)

        nojekyll = self.output_dir / '.nojekyll'
        nojekyll.write_text('', encoding='utf-8')
    
    def generate_post(self, post_path: Path, post_data: dict):
        """Generate HTML for a single post"""
        print(f"  Processing: {post_path.name}")
        
        # Create output directory for post
        post_output_dir = self.output_dir / 'posts' / post_data['slug']
        post_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy assets
        self.copy_post_assets(post_path, post_output_dir)
        
        # Render template
        template = self.env.get_template('post.html')
        context = self.base_context(asset_prefix='../../static')
        html = template.render(
            post=post_data,
            canonical_url=post_data.get('canonical_url', ''),
            **context,
        )
        
        # Write HTML
        output_file = post_output_dir / 'index.html'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
    
    def generate_index(self, posts: list):
        """Generate index/home page"""
        print("  Generating index page")
        
        # Sort posts by date (newest first)
        sorted_posts = sorted(
            posts,
            key=lambda p: p.get('date_iso', ''),
            reverse=True
        )
        
        template = self.env.get_template('index.html')
        context = self.base_context(asset_prefix='static')
        html = template.render(
            posts=sorted_posts,
            canonical_url=self.config.get('site_url', '').rstrip('/') + '/' if self.config.get('site_url') else '',
            **context,
        )
        
        output_file = self.output_dir / 'index.html'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)

    def generate_collection_page(self, section: dict):
        """Generate dedicated page for one external collection"""
        page_slug = section.get('slug', '').strip()
        if not page_slug:
            return

        print(f"  Generating {page_slug} page")

        page_output_dir = self.output_dir / page_slug
        page_output_dir.mkdir(exist_ok=True)

        site_url = self.config.get('site_url', '').rstrip('/')
        canonical_url = f"{site_url}/{page_slug}/" if site_url else ''

        template = self.env.get_template('collection.html')
        context = self.base_context(asset_prefix='../static')
        html = template.render(
            section=section,
            canonical_url=canonical_url,
            **context,
        )

        output_file = page_output_dir / 'index.html'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
    
    def generate(self):
        """Generate all posts and index"""
        self.refresh_config()

        if not self.posts_dir.exists():
            print("Warning: posts/ directory not found")
        
        print("Generating blog...")

        self.copy_static_assets()
        
        # Find all post directories
        post_dirs = []
        if self.posts_dir.exists():
            post_dirs = [
                d for d in self.posts_dir.iterdir()
                if d.is_dir() and (d / 'post.md').exists()
            ]

        if not post_dirs:
            print("No posts found in posts/ directory")
        
        # Generate each post
        posts = []
        for post_dir in sorted(post_dirs):
            try:
                post_data = self.read_post(post_dir)
                self.generate_post(post_dir, post_data)
                posts.append(post_data)
            except Exception as e:
                print(f"  Error processing {post_dir.name}: {e}")
        
        external_sections = self.load_external_sections()
        for section in external_sections:
            self.generate_collection_page(section)

        # Generate index
        self.generate_index(posts)
        
        print(f"✓ Generated {len(posts)} posts")
        print(f"✓ Output: {self.output_dir.resolve()}")

    def _iter_watch_files(self):
        watch_roots = [
            self.posts_dir,
            self.code_dir,
            self.papers_dir,
            self.talks_dir,
            self.fonts_dir,
            self.templates_dir,
            self.static_dir,
        ]
        for root in watch_roots:
            if not root.exists():
                continue
            for path in root.rglob('*'):
                if path.is_file():
                    yield path

        if self.config_path.exists() and self.config_path.is_file():
            yield self.config_path

    def _snapshot_watch_files(self) -> dict:
        snapshot = {}
        for path in self._iter_watch_files():
            stat = path.stat()
            snapshot[str(path)] = (stat.st_mtime_ns, stat.st_size)
        return snapshot

    def watch(self, interval: float = 0.7, initial_generate: bool = True):
        if interval <= 0:
            interval = 0.7

        print(f"Watching for changes (interval={interval}s)...")
        if initial_generate:
            self.generate()
        previous_snapshot = self._snapshot_watch_files()

        try:
            while True:
                time.sleep(interval)
                current_snapshot = self._snapshot_watch_files()
                if current_snapshot == previous_snapshot:
                    continue

                previous_paths = set(previous_snapshot.keys())
                current_paths = set(current_snapshot.keys())

                changed_paths = set()
                changed_paths.update(previous_paths.symmetric_difference(current_paths))
                for path in (previous_paths & current_paths):
                    if previous_snapshot[path] != current_snapshot[path]:
                        changed_paths.add(path)

                print(f"\nDetected {len(changed_paths)} change(s). Rebuilding...")
                for changed in sorted(changed_paths)[:8]:
                    try:
                        rel = Path(changed).relative_to(self.root_dir)
                        print(f"  - {rel}")
                    except ValueError:
                        print(f"  - {changed}")
                if len(changed_paths) > 8:
                    print(f"  - ... and {len(changed_paths) - 8} more")

                self.generate()
                previous_snapshot = current_snapshot
        except KeyboardInterrupt:
            print("\nWatch stopped.")

    def _create_request_handler(self):
        output_directory = str(self.output_dir)
        base_url = self.base_url

        class BlogRequestHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=output_directory, **kwargs)

            def do_GET(self):
                parsed = urlsplit(self.path)
                path = parsed.path

                if base_url:
                    if path == "/":
                        self.send_response(302)
                        self.send_header("Location", f"{base_url}/")
                        self.end_headers()
                        return
                    if path == base_url:
                        path = "/"
                    elif path.startswith(f"{base_url}/"):
                        path = path[len(base_url):]

                self.path = urlunsplit(("", "", path, parsed.query, parsed.fragment))
                return super().do_GET()

        return BlogRequestHandler

    def serve(self, host: str = "127.0.0.1", port: int = 4000, watch: bool = False, interval: float = 0.7):
        if port < 1 or port > 65535:
            raise ValueError("Port must be between 1 and 65535")

        self.generate()
        handler = self._create_request_handler()
        httpd = ThreadingHTTPServer((host, port), handler)

        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        preview_path = f"{self.base_url}/" if self.base_url else "/"
        print(f"Serving docs at http://{host}:{port}{preview_path}")

        try:
            if watch:
                self.watch(interval=interval, initial_generate=False)
            else:
                print("Press Ctrl+C to stop server.")
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            httpd.shutdown()
            httpd.server_close()
            print("Server stopped.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate static blog pages")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch files and regenerate automatically on save",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.7,
        help="Polling interval in seconds for --watch mode (default: 0.7)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve docs/ locally (can be combined with --watch)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for --serve mode (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4000,
        help="Port for --serve mode (default: 4000)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gen = BlogGenerator()
    if args.serve:
        gen.serve(host=args.host, port=args.port, watch=args.watch, interval=args.interval)
    elif args.watch:
        gen.watch(args.interval)
    else:
        gen.generate()
