# Blog.

A simple, elegant (vibecoded) static blog generator themed on [Tufte CSS](https://edwardtufte.github.io/tufte-css/).

## Features

- **Markdown posts** with YAML frontmatter for metadata
- **Math equations** using KaTeX (display: `$$...$$`, inline: `$...$`)
- **Code syntax highlighting** with Pygments (Rouge-like HTML/CSS output)
- **Sidenotes** with `^[note text]` syntax
- **Images & assets** - organize media in post folders
- **Static output** - ready for GitHub Pages

## Setup

```bash
# Install dependencies
python3 -m pip install -r requirements.txt

# Generate blog
python3 generator.py

# Watch + serve (Hugo/Jekyll style)
python3 generator.py --watch --serve
```

Output HTML goes to `docs/` directory, ready to push to GitHub Pages.

## Project Structure

```
.
├── posts/                    # Your blog posts
│   └── first-post/          # One post per folder
│       ├── post.md          # Main content
│       └── images/          # Optional: images referenced in post
├── templates/               # HTML templates
│   ├── base.html           # Base template
│   ├── post.html           # Post template
│   └── index.html          # Homepage
├── static/                  # CSS assets copied to docs/static/
│   ├── tufte.css           # Saved style from sample site
│   └── site.css            # Minimal local overrides
├── site.json                # Site configuration (name, links, base_url)
├── docs/                   # Generated output (push to GitHub Pages)
├── generator.py            # Build script
└── requirements.txt        # Python dependencies
```

## Creating a Post

1. Create a folder under `posts/`:
   ```bash
   mkdir posts/my-post-title
   ```

2. Add `post.md` with YAML frontmatter:
   ```markdown
   ---
   title: My Post Title
   date: 2024-03-01
   author: Your Name
   description: Brief description for listings
   ---

   # Post content in Markdown...
   ```

3. Add images to `posts/my-post-title/images/`

4. Generate:
   ```bash
   python3 generator.py
   ```

## Site Configuration

Edit `site.json`:

```json
{
  "site_name": "karan",
  "author": "Karan",
  "about_path": "about/",
  "base_url": "",
  "site_url": "",
  "subscribe_url": "",
  "twitter_url": "",
  "github_url": ""
}
```

For GitHub Pages project sites, set:

- `base_url` to `"/repo-name"`
- `site_url` to `"https://username.github.io/repo-name"`

### Sidenotes (my favorite feature)

```markdown
This is a claim.^[Here's a sidenote explaining it.]

You can have multiple sidenotes in one post.
```

Sidenotes support markdown content (links, images, emphasis):

```markdown
This benchmark is reproducible.^[See the [repo](https://github.com/example/repo) and
![Plot](images/perf-plot.png)]
```

## Deployment to GitHub Pages

1. Initialize git repo and GitHub Pages in `docs/` folder
2. Push to GitHub: `docs/` branch or `main` → `docs/` folder
3. Your blog is live!

## Fast Local Iteration (CLI Watch Mode)

Run this once and keep it open while writing:

```bash
python3 generator.py --watch
```

This watches changes in:
- `posts/`
- `templates/`
- `static/`
- `site.json`

## Local Preview Server (CLI)

Serve generated output from `docs/`:

```bash
python3 generator.py --serve
```

Serve and auto-regenerate on save:

```bash
python3 generator.py --watch --serve
```

Custom host/port:

```bash
python3 generator.py --watch --serve --host 0.0.0.0 --port 4000
```

## Customization

- Edit `site.json` for nav text and social links
- Edit `static/site.css` for small style adjustments

## Hosting

The `docs/` folder is ready to serve from:
- GitHub Pages (configure in repo settings)
- Any static host (Netlify, Vercel, etc.)
