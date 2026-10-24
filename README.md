# sipscout

An AI discovery engine for one research question:

> Why do first-time SIP investors on Groww continue, pause, reduce or stop their SIPs in the first 12 months, and which of those reasons is the biggest, most under-served opportunity?

Four stages: **Collect** → **Extract** → **Analyse** → **Serve**. This README covers what is built so far (Stage 1) and is extended as each stage lands.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e . --no-deps
copy .env.example .env   # then fill in keys
```

Credentials live only in `.env` (never committed). Play Store, App Store and TradingQ&A need no keys.

| Variable | Source |
|---|---|
| `AUTHOR_SALT` | any long random string; hashes usernames |
| `OPENROUTER_API_KEY` | openrouter.ai/keys (LLM calls go through OpenRouter) |
| `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` | reddit.com/prefs/apps (type: script) |
| `YOUTUBE_API_KEY` | Google Cloud Console, YouTube Data API v3 |

## Commands (Stage 1)

```bash
sipscout doctor
sipscout collect --source playstore --limit 200
sipscout collect --source all --yes
sipscout filter          # re-run keyword flags after editing config.yaml
sipscout export --format csv --relevant-only
```

`collect` is idempotent (upsert on `id`) and one failing source never stops `--source all`.

## Browser app (Stage 4)

The app is read-only. It reads the scrubbed snapshot in `app/data/` and never touches the pipeline database or any scraping credentials.

```bash
sipscout publish                       # writes app/data/ from the database
streamlit run app/streamlit_app.py     # opens in your browser
```

Pages: Overview, Themes, Evidence explorer, Ask the data, Methodology. Before `sipscout extract` has run, the app shows keyword-relevant records only and the Themes page explains what is missing.

**Ask the data** needs `OPENROUTER_API_KEY` in Streamlit secrets (locally: `.streamlit/secrets.toml`, which is git-ignored):

```toml
OPENROUTER_API_KEY = "..."
```

Answers pass an advice guardrail (`app/guardrail.py`): anything naming a fund, AMC or stock, recommending an action, or using urgency language is replaced by "I can only describe what users said, not give investment advice." Sessions are limited to 10 questions.

### Deploy to Streamlit Community Cloud

1. Create a **private** GitHub repo with a neutral name (for example `sip-discovery-engine`) and push this project. `.env`, `*.db` and `data/` exports are git-ignored; only `app/data/` (the scrubbed snapshot) is committed.
2. At share.streamlit.io, sign in with GitHub, choose **New app**, pick the repo, and set the main file to `app/streamlit_app.py`. Set the requirements file to `app/requirements.txt`.
3. Under **Advanced settings → Secrets**, add `OPENROUTER_API_KEY = "..."`.
4. After deploy, set the app's sharing to **public**.
5. Open the link in an incognito window and check every page, including a refused advice question such as "Should I stop my SIP?".

## Data ethics

Public data only. Usernames are replaced by a salted hash, never stored raw. Text is scrubbed for emails, UPI IDs, @mentions, PAN, phone numbers and long digit runs. Academic research; not investment advice; not affiliated with Groww.

## Limitations

- App Store RSS exposes only the most recent ~500 reviews per app.
- YouTube is limited by daily API quota (tracked in SQLite).
- Public posts over-represent people with complaints.
- Groww's internal lapse data is not available.
- Short texts (under `filters.min_text_chars`) are dropped and counted per run.

## Tests

```bash
pytest
```
