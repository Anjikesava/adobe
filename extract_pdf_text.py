import pdfplumber
import re
import json
import datetime
import os


def get_heading_level(text):
    pattern = r'^(\d+(\.\d+)*)'
    match = re.match(pattern, text)
    if match:
        return match.group(1).count('.') + 1
    return 1


def extract_headings_and_positions(pdf_path, font_size_threshold=15):
    headings = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            lines = {}
            for char in page.chars:
                y0 = round(char['top'])
                if y0 not in lines:
                    lines[y0] = []
                lines[y0].append(char)
            for y in sorted(lines.keys()):
                chars = lines[y]
                avg_size = sum(c['size'] for c in chars) / len(chars)
                line_text = ''.join(c['text'] for c in chars).strip()
                if avg_size >= font_size_threshold and line_text:
                    headings.append({
                        "page": page_num,
                        "title": line_text,
                        "font_size": avg_size,
                        "level": get_heading_level(line_text),
                        "y0": y
                    })
    return headings


def build_hierarchy(headings):
    outline = []
    stack = []
    for heading in headings:
        level = heading["level"]
        node = {
            "title": heading["title"],
            "page": heading["page"],
            "font_size": heading["font_size"],
            "y0": heading["y0"],
            "subheadings": [],
            "text": ""
        }
        while stack and stack[-1]["level"] >= level:
            stack.pop()
        if stack:
            stack[-1]["node"]["subheadings"].append(node)
        else:
            outline.append(node)
        stack.append({"level": level, "node": node})
    return outline


def clean_headings(headings):
    cleaned = []
    for h in headings:
        text = h['title'].strip()
        if len(text) < 3:
            continue
        if all(c.isdigit() or c in ' %.-,' for c in text):
            continue
        if "www." in text or text.lower().startswith("submitted"):
            continue
        cleaned.append(h)
    return cleaned


def flatten_outline(outline):
    flat = []

    def recurse(nodes):
        for node in nodes:
            flat.append(node)
            if node.get("subheadings"):
                recurse(node["subheadings"])

    recurse(outline)
    return flat


def extract_section_texts(pdf_path, outline):
    with pdfplumber.open(pdf_path) as pdf:
        flat_headings = flatten_outline(outline)
        for i, heading in enumerate(flat_headings):
            start_page = heading["page"] - 1
            start_y = heading["y0"]
            end_page = len(pdf.pages) - 1
            if i + 1 < len(flat_headings):
                next_h = flat_headings[i + 1]
                end_page = next_h["page"] - 1
            section_text = ""
            for page_num in range(start_page, end_page + 1):
                page = pdf.pages[page_num]
                page_text = page.extract_text() or ""
                if page_num == start_page:
                    lines_dict = {}
                    for char in page.chars:
                        y0 = round(char['top'])
                        lines_dict.setdefault(y0, []).append(char)
                    y_positions = sorted(lines_dict.keys())
                    lines = []
                    for y in y_positions:
                        if y > start_y:
                            line_chars = lines_dict[y]
                            line_text = ''.join(c['text'] for c in line_chars).strip()
                            if line_text:
                                lines.append(line_text)
                    section_text += "\n".join(lines) + "\n"
                else:
                    section_text += page_text + "\n"
            heading["text"] = section_text.strip()
    return outline


def detect_language(text):
    if any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9faf' for c in text):
        return 'jp'
    return 'en'


def add_language_to_outline(outline):
    for node in outline:
        node["language"] = detect_language(node.get("text", ""))
        if node.get("subheadings"):
            add_language_to_outline(node["subheadings"])


def rank_sections_by_keywords(sections, keywords):
    scored_sections = []
    for section in sections:
        score = 0
        text_lower = section.get("text", "").lower()
        for kw in keywords:
            score += text_lower.count(kw.lower())
        section_info = {
            "title": section["title"],
            "page": section["page"],
            "language": section.get("language", "unknown"),
            "font_size": section.get("font_size", 0),
            "score": score,
            "text_excerpt": section.get("text", "")[:150].replace("\n", " "),
            "document": section.get("document", "")
        }
        scored_sections.append((score, section_info))
    scored_sections.sort(key=lambda x: x[0], reverse=True)
    return [sec for score, sec in scored_sections if score > 0]


def process_document(doc_path):
    headings = extract_headings_and_positions(doc_path)
    outline = build_hierarchy(headings)
    cleaned = clean_headings(outline)
    outline_with_text = extract_section_texts(doc_path, cleaned)
    add_language_to_outline(outline_with_text)
    flat_sections = flatten_outline(outline_with_text)
    # Add document name to each section
    for sec in flat_sections:
        sec['document'] = os.path.basename(doc_path)
    return flat_sections


def main(input_data):
    all_sections = []
    persona_keywords = input_data.get('persona', {}).get('focus_areas', [])
    job_keywords = input_data.get('job_to_be_done', "").split()
    ranking_keywords = persona_keywords + job_keywords

    # Process each document and collect sections
    for doc_path in input_data.get('documents', []):
        sections = process_document(doc_path)
        all_sections.extend(sections)

    # Rank all sections across documents
    ranked_sections = rank_sections_by_keywords(all_sections, ranking_keywords)

    # Assign importance_rank
    for idx, section in enumerate(ranked_sections, 1):
        section["importance_rank"] = idx

    output_json = {
        "metadata": {
            "input_documents": [os.path.basename(d) for d in input_data.get('documents', [])],
            "persona": input_data.get("persona", {}),
            "job_to_be_done": input_data.get("job_to_be_done", ""),
            "processing_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        },
        "extracted_sections": [
            {
                "document": sec["document"],
                "page_number": sec["page"],
                "section_title": sec["title"],
                "importance_rank": sec["importance_rank"]
            } for sec in ranked_sections
        ],
        "sub_section_analysis": [
            {
                "document": sec["document"],
                "page_number": sec["page"],
                "section_title": sec["title"],
                "refined_text": sec["text_excerpt"]
            } for sec in ranked_sections
        ]
    }

    with open("challenge1b_output.json", "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    print("Processing complete.")
    print(f"Output saved to challenge1b_output.json")


if __name__ == "__main__":
    input_spec = {
        "persona": {
            "role": "PhD Researcher in Computational Biology",
            "expertise": "Computational Biology and Machine Learning",
            "focus_areas": ["graph neural networks", "drug discovery", "methodologies", "datasets", "performance"]
        },
        "job_to_be_done": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks",
        "documents": [
            r"C:\Users\nomul\OneDrive\Desktop\venv\MINI1 REPORT.pdf",
            r"C:\Users\nomul\OneDrive\Desktop\venv\519ML.pdf"
        ]
    }
    main(input_spec)

