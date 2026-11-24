# AI Text Summarizer

A command-line tool that summarizes text files (or piped text) with Claude, using
the official [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python).

## Features

- Summarize a file or text from stdin
- Three styles: `bullets`, `paragraph` and `tldr`
- Optional word limit with `--max-words`
- Clear error messages for missing keys, rate limits and network problems
