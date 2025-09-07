# Notion Parsing and Context Extraction

## Overview

This project provides a scalable and extensible parser for Notion documents and databases, supporting advanced features like OCR for images and PDF extraction. It is designed for AI workflows, benchmarking, and document analysis.

## Features

- **NotionParser**: Parses Notion pages and databases, supporting many block types via the Strategy pattern for easy extensibility.
- **OCR & PDF Support**: Extracts text from images and PDFs embedded in Notion documents.
- **Benchmarking**: Analyze parsing performance, memory usage, and parsing efficacy.
- **AI Document Preparation**: Converts parsed content into LangChain documents and splits them for downstream AI tasks.
- **File Handlers**: Utilities for extracting text from images, files, and PDFs.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Rishikesh63/Notion-Parsing-and-Context-Extraction.git
   cd Notion-Parsing-and-Context-Extraction
   ```
2. Install Poetry (if not already installed):
   ```
   pip install poetry
   ```
3. Activate the Poetry shell for an isolated environment:
   ```
   poetry shell
   ```
4. Install dependencies with Poetry:
   ```
   poetry install
   ```

## Usage

1.  Set up your `.env` file with the following variables:
   ```
   NOTION_BASE_URL="https://www.notion.so/"
   NOTION_TOKEN = "your_notion_token"
   PAGE_ID = "your_page_or_database_id"
   ```

2. Run the parser:
   ```
   python run_parser.py
   ```

3. Outputs:
   - `parsed_output.json`: Structured Notion content.
   - `ai_documents.json`: AI-ready document chunks.

## Main Modules

- `parser.py`: Main NotionParser class, block parsing strategies.
- `file_handlers.py`: OCR and file extraction utilities.
- `benchmark.py`: Performance and quality analysis.
- `run_parser.py`: Entry point for parsing and document preparation.

## Requirements

All dependencies are managed via Poetry in `pyproject.toml`. Key dependencies include:
- notion-client
- langchain
- pandas
- pdfplumber
- Pillow
- pytesseract
- PyPDF2
- requests
- python-dotenv
- psutil

## Extending

To add support for new Notion block types, implement a new parsing method in `NotionParser` and register it in the `block_parsers` dictionary.


