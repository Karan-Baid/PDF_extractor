import re
from typing import List, Dict
from collections import defaultdict
import os
import pymupdf
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder, util


class SimplePDFHeadingExtractor:
    def __init__(self):
        self.font_size_threshold = 1.5  
        
    def extract_headings(self, pdf_path: str) -> List[Dict[str, any]]:
        doc = pymupdf.open(pdf_path)
        text_blocks = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:  
                                text_blocks.append({
                                    'text': text,
                                    'font_size': span["size"],
                                    'font_name': span["font"],
                                    'is_bold': bool(span["flags"] & 2**4),  
                                    'bbox': span["bbox"],
                                    'page_num': page_num + 1
                                })
        
        doc.close()

        headings = self._identify_headings(text_blocks)
        return headings
    
    def _identify_headings(self, text_blocks: List[dict]) -> List[Dict[str, any]]:
        if not text_blocks:
            return []
        
        font_sizes = [block['font_size'] for block in text_blocks]
        avg_font_size = sum(font_sizes) / len(font_sizes)
        
        text_blocks.sort(key=lambda b: (b['page_num'], b['bbox'][1]))
        
        potential_headings = []
        
        for block in text_blocks:
            text = block['text']
            font_size = block['font_size']
            is_bold = block['is_bold']
            bbox = block['bbox']

            if self._should_skip_text(text, bbox):
                continue
            
            is_heading = False

            if font_size > avg_font_size + self.font_size_threshold:
                is_heading = True

            elif is_bold and font_size >= avg_font_size - 0.5:
                is_heading = True
            
            elif (text.isupper() and 
                  5 < len(text) < 100 and
                  font_size >= avg_font_size - 0.5):
                is_heading = True

            elif re.match(r'^(\d+\.(\d+\.)*\s+|chapter\s+\d+|section\s+\d+)', text, re.IGNORECASE):
                is_heading = True

            elif (text.istitle() and 
                  5 < len(text) < 80 and 
                  not text.endswith('.') and
                  not text.endswith(',') and
                  font_size >= avg_font_size - 0.5):
                is_heading = True
            
            if is_heading:
                potential_headings.append(block)
        
        main_headings = self._filter_main_headings(potential_headings)
        
        return main_headings
    
    def _should_skip_text(self, text: str, bbox: tuple) -> bool:
        if len(text) < 3 or len(text) > 200:
            return True
        
        if re.match(r'^\d+$', text):
            return True
        
        if re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', text):
            return True
        
        if 'http' in text.lower() or '@' in text:
            return True

        if re.match(r'^\d+(\.\d+)?\s+(cup|tablespoon|teaspoon|tsp|tbsp|pound|lb|ounce|oz|gram|g|kg|ml|liter|l)s?(\s|$)', text.lower()):
            return True
        
  
        if re.match(r'^[•\-\*]\s+', text):
            return True

        page_height = 792 
        if bbox[1] < 50 or bbox[1] > page_height - 50:
            return True
        
        skip_starters = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        if any(text.lower().startswith(word + ' ') for word in skip_starters):
            return True
            
        return False 
        
    def _filter_main_headings(self, potential_headings: List[dict]) -> List[Dict[str, any]]:

        if not potential_headings:
            return []
        
        font_groups = defaultdict(list)
        for heading in potential_headings:
            key = (heading['font_size'], heading['is_bold'], heading['font_name'])
            font_groups[key].append(heading)

        text_frequency = defaultdict(int)
        for heading in potential_headings:
            text = heading['text'].lower()

            if text.endswith(':'):
                text_frequency[text] += 1

        main_heading_format = None
        max_font_size = 0
        
        for key, headings in font_groups.items():
            font_size, is_bold, font_name = key

            if self._is_likely_subheading_group(headings):
                continue

            if font_size > max_font_size:
                max_font_size = font_size
                main_heading_format = key

        if main_heading_format is None:
            for key, headings in font_groups.items():
                if not self._is_likely_subheading_group(headings):
                    main_heading_format = key
                    break
        
        main_headings = []
        
        for heading in potential_headings:
            text = heading['text']
            page_number = heading['page_num']  

            if text.lower() in text_frequency and text_frequency[text.lower()] > 2:
                continue
            
            if self._is_subheading_pattern(text):
                continue

            if main_heading_format:
                key = (heading['font_size'], heading['is_bold'], heading['font_name'])
                if key == main_heading_format:
                    main_headings.append({
                        "heading": text,
                        "page_num": page_number
                    })

                elif heading['font_size'] > main_heading_format[0]:
                    main_headings.append({
                        "heading": text,
                        "page_num": page_number
                    })
            else:
                main_headings.append({
                    "heading": text,
                    "page_num": page_number
                })
        
        return main_headings
    
    def _is_likely_subheading_group(self, headings: List[dict]) -> bool:
        if len(headings) < 3:
            return False
        

        colon_count = sum(1 for h in headings if h['text'].endswith(':'))
        if colon_count > len(headings) * 0.7:  
            return True

        texts = [h['text'].lower() for h in headings]
        short_repetitive = sum(1 for text in texts if len(text) < 15 and texts.count(text) > 1)
        if short_repetitive > len(headings) * 0.5:
            return True
        
        return False
    
    def _is_subheading_pattern(self, text: str) -> bool:
        text_lower = text.lower().strip()
        
        if text.endswith(':'):
            return True
        
        generic_subheadings = {
            'overview', 'summary', 'introduction', 'conclusion', 'notes', 
            'tips', 'warning', 'note', 'example', 'examples', 'details',
            'description', 'procedure', 'process', 'method', 'approach',
            'background', 'purpose', 'objective', 'goals', 'requirements'
        }
        
        if text_lower in generic_subheadings:
            return True
        
        # Pattern for numbered sub-items (a), (1), etc.
        if re.match(r'^\([a-z0-9]\)', text_lower):
            return True
        
        return False

def extract_pdf_headings(pdf_path: str) -> List[Dict[str, any]]:
    extractor = SimplePDFHeadingExtractor()
    return extractor.extract_headings(pdf_path)

def get_subsection_between_headings(text, start_heading, end_heading):

        start_pattern = re.escape(start_heading)
        end_pattern = re.escape(end_heading)

        pattern = rf"{start_pattern}(.*?){end_pattern}"

        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            return "No content found between the given headings."



def rank_topics(persona, task, description, input_documents, input_dir):
    bi_encoder = SentenceTransformer('multi-qa-mpnet-base-dot-v1', device='cpu')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')

    context = f"You are a {persona} for {description} and your task is to {task} "

    pdfs_embs = bi_encoder.encode(input_documents, convert_to_tensor=True)
    context_emb = bi_encoder.encode(context, convert_to_tensor=True)

    scores = util.cos_sim(context_emb, pdfs_embs)[0]
    ranked_indices_pdf = scores.argsort(descending=True)
    ranked_pdfs = [input_documents[i] for i in ranked_indices_pdf]

    final_topics = []
    for pdf_file in ranked_pdfs:
        pdf_path = input_dir / pdf_file   # <-- FIX HERE
        headings = extract_pdf_headings(str(pdf_path))
        headings_list = [item["heading"] for item in headings if "heading" in item]

        if not headings_list:
            continue

        topic_embs = bi_encoder.encode(headings_list, convert_to_tensor=True)
        heading_score = util.cos_sim(context_emb, topic_embs)[0]

        ranked_index = heading_score.argsort(descending=True)[0].item()
        best_heading = headings_list[ranked_index]

        if any(best_heading.lower() in t.lower() or t.lower() in best_heading.lower() for t in final_topics):
            continue

        final_topics.append(best_heading)

        if len(final_topics) == 5:
            break

    return final_topics



def extract_subsection_from_pdfs(pdf_dir, keyword, stop_heading):
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            with pymupdf.open(pdf_path) as doc:
                text = ""
                found = False

                for page in doc:
                    lines = page.get_text("text").splitlines()

                    for line in lines:
                        clean_line = line.strip()
                        if keyword.lower() in clean_line.lower() and not found:
                            found = True
                            continue
                        if found:
                            if stop_heading.lower() in clean_line.lower():
                                return pdf_path, text.strip()

                            text += clean_line + " "

                if found and text:
                    return pdf_path, text.strip()

    return None, None


def find_pdfs_with_keyword(pdf_dir, keyword):
    matching_pdf_filename = ""
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)

            try:
                with pymupdf.open(pdf_path) as doc:
                    for page in doc:
                        text = page.get_text("text")
                        if keyword.lower() in text.lower():
                            matching_pdf_filename = filename
                            return matching_pdf_filename 
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

    return matching_pdf_filename 

def get_page_no(heading,combined_data):
    for i in combined_data:
        if (i['heading']==heading):
            return i['page_num']
        
def clean_refined_text(text: str) -> str:
    text = re.sub(r"[\uf0b7\u2022•▪►]", " ", text)
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
def extract_text_from_pdfs(pdf_dir):
    all_text = ""
    for file_name in os.listdir(pdf_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, file_name)
            with pymupdf.open(pdf_path) as pdf:
                for page in pdf:
                    all_text += page.get_text("text") + "\n"
    return all_text


def process_pdfs():
    BASE_DIR = Path(__file__).parent
    input_dir = BASE_DIR/"input"
    all_headings=[]
    combined_data=[] 
    # pdf_files = list(input_dir.glob("*.pdf"))
    extracted_sections_list = [] 
    input_file = BASE_DIR/"challenge1b_input.json"
    with open(input_file, "r") as f:
        docs = json.load(f)

    task = docs["job_to_be_done"]["task"]
    descripton = docs["challenge_info"] ["description"]
    persona= docs["persona"]["role"]
    input_documents = [doc["filename"] for doc in docs["documents"]]

    for i in input_documents:
        pdf_path = input_dir /i
        headings = extract_pdf_headings(str(pdf_path))
        for j in headings:
            combined_data.append(j)
            all_headings.append(j['heading'])

    output = rank_topics(
        persona=persona,
        task=task,
        description=descripton,
        input_documents=input_documents,
        input_dir=input_dir   
    )

    
    for i, section_title in enumerate(output):
        document_name = find_pdfs_with_keyword(input_dir, section_title)

        section_data = {
            "document": document_name,
            "section_title": section_title,
            "importance_rank": i + 1,
            "page_number":get_page_no(section_title,combined_data)
        }
        extracted_sections_list.append(section_data)

    subsection_analysis = [] 

    for i, section_title in enumerate(output):
        document_name = find_pdfs_with_keyword(input_dir, section_title)
        pdf_path = document_name
        end=all_headings.index(section_title)+1
        end_heading=all_headings[end]
        text = extract_text_from_pdfs(input_dir)
        subsection = get_subsection_between_headings(text, section_title, end_heading)
        subsection=clean_refined_text(subsection)
        section_data = {
            "document": document_name,
            "refined_text": subsection,
            "page_number":get_page_no(section_title,combined_data)
        }
        subsection_analysis.append(section_data)
   
    metadata=[{"input_documents":input_documents},
              {"persona":persona},
              {"job_to_be_done":task}
    ]
    final_output={
    "metadata":metadata,
    "extracted_sections":extracted_sections_list,
    "subsection_analysis":subsection_analysis
    }
    output_file = BASE_DIR/"challenge1b_output.json"
    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=2)
        
import time
if __name__ == "__main__":
    start_time = time.time()   # Start timer
    
    print("Starting processing pdfs")
    process_pdfs()
    print("Completed processing pdfs")
    
    end_time = time.time()     # End timer
    print(f"Execution time: {end_time - start_time:.2f} seconds")