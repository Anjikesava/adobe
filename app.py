import os
import re
import json
import datetime
import uuid
from flask import Flask, request, render_template_string, url_for, send_file
import tempfile
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB max upload

# Ensure static/outputs folder exists
OUTPUTS_DIR = os.path.join("static", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


### ==== Pipeline functions ==== ###

def extract_keywords_from_title_or_focus(focus_keywords_str, job_to_be_done_str):
    stopwords = set([
        'for', 'and', 'of', 'the', 'in', 'on', 'to', 'a', 'an',
        'with', 'by', 'using', 'from', 'at', 'is', 'as'
    ])
    focus_keywords = [k.strip().lower() for k in focus_keywords_str.split(",") if k.strip()]
    job_tokens = re.findall(r'\b[a-zA-Z]{2,}\b', job_to_be_done_str.lower())
    job_tokens_filtered = [t for t in job_tokens if t not in stopwords]
    combined_keywords = list(set(focus_keywords + job_tokens_filtered))
    return combined_keywords


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
                lines.setdefault(y0, []).append(char)
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


def rank_sections_tfidf(sections, keywords):
    query = " ".join(keywords).lower()
    docs = [sec.get("text", "").lower() for sec in sections]

    vectorizer = TfidfVectorizer(stop_words='english')
    if len(docs) == 0:
        return []
    tfidf_matrix = vectorizer.fit_transform(docs)
    query_vec = vectorizer.transform([query])
    scores = (tfidf_matrix * query_vec.T).toarray().flatten()

    scored_sections = []
    for i, score in enumerate(scores):
        if score > 0:
            sec = sections[i]
            section_info = {
                "title": sec["title"],
                "page": sec["page"],
                "language": sec.get("language", "unknown"),
                "font_size": sec.get("font_size", 0),
                "score": score,
                "text_excerpt": sec.get("text", "")[:150].replace("\n", " "),
                "document": sec.get("document", "")
            }
            scored_sections.append((score, section_info))

    scored_sections.sort(key=lambda x: x[0], reverse=True)
    return [sec for score, sec in scored_sections]


def process_document(doc_path):
    headings = extract_headings_and_positions(doc_path)
    outline = build_hierarchy(headings)
    cleaned = clean_headings(outline)
    outline_with_text = extract_section_texts(doc_path, cleaned)
    add_language_to_outline(outline_with_text)
    flat_sections = flatten_outline(outline_with_text)
    for sec in flat_sections:
        sec['document'] = os.path.basename(doc_path)
    return flat_sections


def run_full_pipeline(doc_paths, persona_role, focus_keywords_str, job_to_be_done_str):
    ranking_keywords = extract_keywords_from_title_or_focus(focus_keywords_str, job_to_be_done_str)

    all_sections = []
    for doc_path in doc_paths:
        all_sections.extend(process_document(doc_path))

    ranked_sections = rank_sections_tfidf(all_sections, ranking_keywords)

    for idx, sec in enumerate(ranked_sections, 1):
        sec['importance_rank'] = idx

    output_json = {
        "metadata": {
            "input_documents": [os.path.basename(p) for p in doc_paths],
            "persona_role": persona_role,
            "persona_focus_keywords": [k.strip() for k in focus_keywords_str.split(",") if k.strip()],
            "job_to_be_done": job_to_be_done_str,
            "processing_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        },
        "ranked_sections": [
            {
                "document": sec["document"],
                "page_number": sec["page"],
                "section_title": sec["title"],
                "importance_rank": sec["importance_rank"],
                "score": sec["score"],
                "text_excerpt": sec["text_excerpt"]
            }
            for sec in ranked_sections
        ]
    }

    # Save in static/outputs with unique filename
    unique_filename = f"output_{uuid.uuid4().hex}.json"
    output_path = os.path.join(OUTPUTS_DIR, unique_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    return ranked_sections, output_path


### ==== Flask routes and template ==== ###

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>PDF Persona-Driven Document Analyzer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css" rel="stylesheet" />
    <style>
        body { padding-top: 2rem; background-color: #f8f9fa; }
        .form-section { max-width: 800px; margin: auto; }
        .results-section { max-width: 900px; margin: 2rem auto; }
        .section-card { margin-bottom: 1rem; }
        .text-excerpt { font-style: italic; color: #555; }
        .score-badge { font-size: 0.8rem; }
    </style>
</head>
<body>
<div class="container">

    <h1 class="mb-4 text-center">PDF Persona-Driven Document Analyzer</h1>

    <div class="card form-section p-4 shadow-sm bg-white rounded">
        <form method="post" enctype="multipart/form-data" novalidate>

            <div class="mb-3">
                <label for="files" class="form-label">Upload PDF files (3-10):</label>
                <input type="file" class="form-control" id="files" name="files" multiple accept=".pdf" required>
                <div class="form-text">Select between 3 and 10 PDF documents.</div>
            </div>

            <div class="mb-3">
                <label for="persona_role" class="form-label">Persona Role:</label>
                <input type="text" class="form-control" id="persona_role" name="persona_role" required placeholder="e.g., PhD Researcher in Computational Biology" value="{{ persona_role|default('') }}">
            </div>

            <div class="mb-3">
                <label for="focus_keywords" class="form-label">Persona Focus Keywords (comma separated):</label>
                <input type="text" class="form-control" id="focus_keywords" name="focus_keywords" required placeholder="e.g., graph neural networks, drug discovery" value="{{ focus_keywords|join(', ') if focus_keywords else '' }}">
            </div>

            <div class="mb-3">
                <label for="job_to_be_done" class="form-label">Job to be Done:</label>
                <input type="text" class="form-control" id="job_to_be_done" name="job_to_be_done" required placeholder="e.g., Prepare literature review on methodologies and datasets" value="{{ job_to_be_done|default('') }}">
            </div>

            <button type="submit" class="btn btn-primary w-100">Analyze</button>
        </form>
    </div>

    {% if error %}
        <div class="alert alert-danger mt-4 text-center">{{ error }}</div>
    {% endif %}

    {% if results %}
        <div class="results-section mt-5">
            <h3>Ranked Sections (Total: {{ results|length }})</h3>
            <p><strong>Persona Role:</strong> {{ persona_role }}<br/>
            <strong>Focus Keywords:</strong> {{ focus_keywords|join(', ') }}<br/>
            <strong>Job to be Done:</strong> {{ job_to_be_done }}</p>

            {% for sec in results %}
            <div class="card section-card shadow-sm">
                <div class="card-body">
                    <h5 class="card-title">{{ sec.section_title }}</h5>
                    <h6 class="card-subtitle mb-2 text-muted">
                        {{ sec.document }} - Page {{ sec.page_number }} | Language: {{ sec.language or 'en' }}
                        <span class="badge bg-info text-dark score-badge float-end">Score: {{ '%.4f' % sec.score }}</span>
                    </h6>
                    <p class="card-text text-excerpt">{{ sec.text_excerpt }}...</p>
                    <small class="text-muted">Importance Rank: {{ sec.importance_rank }}</small>
                </div>
            </div>
            {% endfor %}

            <a href="{{ output_json_url }}" class="btn btn-success mt-3" download>Download JSON Output</a>
        </div>
    {% endif %}

</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''


@app.route("/", methods=["GET", "POST"])
def index():
    error_msg = None
    results = None
    output_json_path = None

    # Initialize for all code flows
    persona_role = ""
    focus_keywords_str = ""
    job_to_be_done = ""

    if request.method == "POST":
        files = request.files.getlist("files")
        persona_role = request.form.get("persona_role", "").strip()
        focus_keywords_str = request.form.get("focus_keywords", "").strip()
        job_to_be_done = request.form.get("job_to_be_done", "").strip()

        if not files or not (3 <= len(files) <= 10):
            error_msg = "Please upload between 3 and 10 PDF files."
        elif not all(f.filename.lower().endswith(".pdf") for f in files):
            error_msg = "All uploaded files must be PDFs."
        elif not (persona_role and focus_keywords_str and job_to_be_done):
            error_msg = "All form fields are required."
        else:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    doc_paths = []
                    for uploaded_file in files:
                        save_path = os.path.join(tmpdir, uploaded_file.filename)
                        uploaded_file.save(save_path)
                        doc_paths.append(save_path)

                    results, output_json_path = run_full_pipeline(
                        doc_paths, persona_role, focus_keywords_str, job_to_be_done
                    )
            except Exception as e:
                error_msg = f"An error occurred during processing: {e}"

    output_json_url = None
    if output_json_path and os.path.exists(output_json_path):
        # This builds the publicly accessible URL for the JSON file in static/outputs/
        filename = os.path.relpath(output_json_path, "static").replace("\\", "/")
        output_json_url = url_for('static', filename=filename)

    return render_template_string(
        HTML_TEMPLATE,
        results=results,
        output_json_url=output_json_url,
        error=error_msg,
        persona_role=persona_role,
        focus_keywords=focus_keywords_str.split(",") if focus_keywords_str else [],
        job_to_be_done=job_to_be_done
    )


if __name__ == "__main__":
    print("Starting Flask server at http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=True)

