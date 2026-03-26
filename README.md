# Patent Publication Microsite

This repository contains a plain HTML, CSS, and JavaScript static site intended for GitHub Pages deployment as a publication-grade patent or invention companion page.

The design is deliberately restrained and technical. It is positioned as a serious systems architecture page for a published patent or patent application, not as a startup landing page or marketing site.

## Included Files

```text
/
  index.html
  styles.css
  script.js
  README.md
  .nojekyll
```

The site also assumes that the following asset paths may exist later:

```text
/assets/patent.pdf
/assets/architecture-overview.png
/assets/gather-flow.png
/assets/figures/...
/assets/favicon.png
```

## Site Structure

The homepage is organized into the following sections:

1. Hero
2. Problem
3. Core Invention
4. Architecture Overview
5. Why It Matters
6. Differentiation
7. Method / Execution Flow
8. Artifacts
9. Inventor / Contact
10. Footer

## How to Deploy on GitHub Pages

This site is ready for direct static hosting. No build step is required.

### Option 1: Branch-based GitHub Pages

1. Push the files to a GitHub repository.
2. In the repository, open `Settings` -> `Pages`.
3. Under `Build and deployment`, choose:
   - `Source`: `Deploy from a branch`
   - `Branch`: `main` (or your preferred branch)
   - `Folder`: `/ (root)`
4. Save the settings.
5. GitHub Pages will publish the site using the root files directly.

### Option 2: GitHub Actions

If you prefer, you may deploy the same static files through a GitHub Actions workflow. This repository does not require Actions for normal deployment, but static Pages deployment through Actions is also compatible.

## Why `.nojekyll` Is Included

GitHub Pages runs Jekyll by default for many repositories. This site is pure static HTML and does not require Jekyll processing. The `.nojekyll` file ensures that GitHub Pages serves the files directly and does not interfere with asset paths or directory handling.

## How to Customize

### Title and Metadata

Edit the following in `index.html`:

- `<title>`
- hero heading
- subtitle
- thesis sentence
- metadata row
- publication status text

### Links

Update the following placeholders in `index.html`:

- `./assets/patent.pdf`
- `./assets/architecture-overview.png`
- `./assets/gather-flow.png`
- `./assets/figures/`
- repository links such as `https://github.com/`
- contact email such as `inventor@example.com`

### Contact Section

Update the contact section near the bottom of `index.html` with:

- inventor or organization email
- repository URL
- any professional profile or institution link if desired

### Styling

Edit theme variables at the top of `styles.css`:

- `--bg`
- `--bg-panel`
- `--text`
- `--text-muted`
- `--accent`
- `--line`

Typography is currently configured with:

- `IBM Plex Sans` for body text
- `Fraunces` for headings

If preferred, you may replace these with other system-safe or Google Fonts.

### JavaScript Behavior

`script.js` only adds:

- reveal-on-scroll effects
- active navigation highlighting
- current year in the footer

The site remains readable and usable without JavaScript.

## Asset Placement

Create an `assets` directory in the repository root when ready:

```text
/assets
  patent.pdf
  architecture-overview.png
  gather-flow.png
  favicon.png
  /figures
```

Recommended usage:

- `patent.pdf`: the published patent or application PDF
- `architecture-overview.png`: high-level architecture figure
- `gather-flow.png`: execution or gather flow figure
- `figures/`: optional supplemental figures

## GitHub Pages Path Notes

The site currently uses relative paths such as `./assets/patent.pdf`, which work well for most GitHub Pages deployments, including project sites and user sites.

If you later move the site into a deeper subdirectory or introduce templating, review the asset links in `index.html`.

## Recommended Next Edits Before Publishing

- Replace placeholder asset files with real publication-safe figures.
- Replace the repository placeholder URL.
- Replace the contact email placeholder.
- Confirm the publication status language matches the actual public record.
- Add a favicon in `assets/favicon.png`.

## Intended Use

This site is suitable as:

- a published patent companion page
- a technical invention overview page
- an engineer-facing architecture explainer
- a partner or acquirer-facing publication page

It is intentionally written to feel like a technical whitepaper microsite rather than a generic product site.
