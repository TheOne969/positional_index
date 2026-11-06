import sys
import xml.etree.ElementTree as ET
import json

def process_stream(input_stream, output_file, max_docs=100000):
    # We open the output JSONL file in append mode. JSONL allows streaming writes.
    with open(output_file, 'a', encoding='utf-8') as f_out:
        docs_processed = 0
        
        # Determine the correct XML namespace by inspecting the Wikipedia dump structure
        # (Wikipedia XML tags are usually prefixed with this URL)
        ns = {'mw': 'http://www.mediawiki.org/xml/export-0.11/'}
        
        print(f"Starting to parse XML from stdin into {output_file}...")
        
        # Use iterparse to stream the XML file chunk-by-chunk without loading the massive file into RAM
        context = ET.iterparse(input_stream, events=('start', 'end'))
        
        # Define references to keep track of the current Wikipedia article being parsed
        current_page = None
        in_page = False
        
        # Iterate over XML parsing events generated lazily
        for event, elem in context:
            # Check if we've hit the opening tag of a new Wikipedia article
            if event == 'start' and elem.tag.endswith('page'):
                current_page = elem
                in_page = True
                
            # Check if we've hit the closing tag, meaning the article is fully parsed into memory
            elif event == 'end' and elem.tag.endswith('page'):
                if in_page and current_page is not None:
                    # Strip the namespace prefix to correctly search for child tags
                    tag_prefix = elem.tag.split('}')[0] + '}' if '}' in elem.tag else ''
                    
                    # Extract the Document ID
                    id_elem = current_page.find(f'{tag_prefix}id')
                    doc_id = id_elem.text if id_elem is not None else str(docs_processed)
                    
                    # Extract the Article Title
                    title_elem = current_page.find(f'{tag_prefix}title')
                    title = title_elem.text if title_elem is not None else ''
                    
                    # Extract the actual Wikipedia text content (nested inside revision -> text)
                    revision_elem = current_page.find(f'{tag_prefix}revision')
                    if revision_elem is not None:
                        text_elem = revision_elem.find(f'{tag_prefix}text')
                        text = text_elem.text if text_elem is not None else ''
                    else:
                        text = ''
                        
                    # Build the document dictionary
                    doc = {
                        "id": doc_id,
                        "title": title,
                        "text": text
                    }
                    
                    # Write the dictionary as a single minified JSON line
                    f_out.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    docs_processed += 1
                    
                    # Periodically flush status to the console
                    if docs_processed % 1000 == 0:
                        print(f"Processed {docs_processed} documents...")
                        
                    # Enforce the strict document limit so the extraction stops early
                    if docs_processed >= max_docs:
                        break
                
                in_page = False
                
                # CRITICAL: Clear the parsed XML element tree from RAM!
                # Without this `clear()`, iterparse would infinitely build the tree in memory and crash the server.
                if current_page is not None:
                    current_page.clear()
                
    print(f"Finished processing. Total {docs_processed} documents extracted.")

if __name__ == "__main__":
    output_file = sys.argv[2] if len(sys.argv) > 2 else "docs.jsonl"
    max_d = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    
    # Process from sys.stdin, allowing us to pipe curl and bzcat directly into the script
    # This prevents the need to download the massive compressed file to disk first!
    process_stream(sys.stdin, output_file, max_docs=max_d)
