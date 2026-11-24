# AI Text Summarizer

A command-line tool that summarizes text files (or piped text) with Claude, using
the official [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python).

## Features

- Summarize a file or text from stdin
- Three styles: `bullets`, `paragraph` and `tldr`
- Optional word limit with `--max-words`
- Clear error messages for missing keys, rate limits and network problems

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-api-key
```

## Usage

```bash
# Bullet points (default)
python3 summarizer.py sample_article.txt

# One paragraph, at most 60 words
python3 summarizer.py sample_article.txt --style paragraph --max-words 60

# One-line TL;DR from piped text
cat sample_article.txt | python3 summarizer.py --style tldr
```

Example output:

```
- The city council voted to turn three downtown parking lots into parks over two years.
- About 400 parking spaces will be lost; a new 10-minute bus loop and discounted passes aim to offset this.
- Funding was approved for 200 bike parking spots and a study of a garage near the train station.
- Construction starts next spring, with resident workshops before designs are finalized.
```

Exit codes: `0` success, `1` input or API error, `2` Claude declined to summarize.

## Configuration

`MODEL` (`claude-opus-5-5`) and `MAX_TOKENS` are set at the top of `summarizer.py`.
Requests use `effort: low`, which is fast and inexpensive for summarization.

## Tests

The tests use a fake client, so they need neither an API key nor network access:

```bash
python3 -m unittest test_summarizer.py
```
