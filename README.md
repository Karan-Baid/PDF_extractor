# PDF Extractor

This project extracts and analyzes headings and sections from multiple PDF documents using NLP and font-based heuristics. It is designed for tasks such as topic ranking, section extraction, and persona-driven document analysis.

## Features
- Extracts headings from PDFs using font size, boldness, and text patterns
- Ranks topics based on a persona, task, and context using sentence transformers
- Finds and analyzes sections between headings
- Outputs structured JSON with metadata, extracted sections, and refined text

## How It Works
1. Place your PDF files in the `input/` directory.
2. Edit `challenge1b_input.json` to specify the persona, task, description, and list of PDF filenames.
3. Run `app1b.py` to process the PDFs and generate `challenge1b_output.json`.

## Requirements
- Python 3.8+
- `pymupdf`
- `sentence-transformers`
- `torch`

Install dependencies:
```bash
pip install pymupdf sentence-transformers torch
```

## Usage
```bash
python app1b.py
```

## Input Format
- `challenge1b_input.json` should contain:
	- `persona`: role description
	- `job_to_be_done`: task
	- `challenge_info`: description
	- `documents`: list of PDF filenames

## Output
- `challenge1b_output.json` contains:
	- Metadata
	- Extracted sections with document name, section title, importance rank, and page number
	- Subsection analysis with refined text

## Example
See the provided sample PDFs and JSON files for reference.

## License
MIT