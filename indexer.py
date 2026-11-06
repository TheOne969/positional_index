"""
Positional Inverted Indexer for Indian Languages (Hindi Wikipedia)

This module builds a high-performance positional inverted index:
  1. Tokenizes Hindi text (Devanagari script), preserving words and stripping punctuation.
  2. Computes Term Frequency (TF), Document Frequency (DF), and Collection Frequency (CF) for each term.
  3. Writes the entire postings list to secondary storage (postings.bin) packed as binary integers.
  4. Stores the vocabulary (lexicon) containing (byte_offset, DF, CF) in lexicon.json (for main memory).
"""

import json
import struct
import sys
import os
import string
import gzip

def tokenize(text):
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation + '।॥')
    return text.translate(translator).split()

def open_dataset_file(filepath):
    """
    Transparently opens either uncompressed (.jsonl) or gzipped (.jsonl.gz) files.
    If the requested file does not exist, automatically checks for its .gz counterpart.
    """
    actual_path = filepath
    if not os.path.exists(actual_path):
        if os.path.exists(actual_path + ".gz"):
            actual_path = actual_path + ".gz"
        elif actual_path.endswith(".gz") and os.path.exists(actual_path[:-3]):
            actual_path = actual_path[:-3]
        else:
            raise FileNotFoundError(f"Dataset file '{filepath}' (or '{filepath}.gz') not found.")
            
    print(f"Reading dataset from: {actual_path}")
    if actual_path.endswith(".gz"):
        return gzip.open(actual_path, "rt", encoding="utf-8")
    return open(actual_path, "r", encoding="utf-8")

def build_binary_index(input_filepath, bin_filepath, lex_filepath):
    print("Building in-memory inverted index...")
    
    # In-memory dictionary: term -> {doc_id: [positions]}
    index = {}
    
    docs_processed = 0
    with open_dataset_file(input_filepath) as f:
        for line in f:
            doc = json.loads(line)
            # Ensure doc_id is an integer so we can pack it as 4 bytes
            doc_id = int(doc["id"])
            text = doc["title"] + " " + doc["text"]
            
            tokens = tokenize(text)
            
            for pos, token in enumerate(tokens):
                if token not in index:
                    index[token] = {}
                if doc_id not in index[token]:
                    index[token][doc_id] = []
                index[token][doc_id].append(pos)
                
            docs_processed += 1
            if docs_processed % 1000 == 0:
                print(f"Processed {docs_processed} documents...")
                
    print("Writing binary postings and lexicon to disk...")
    # Vocabulary structure: term -> [byte_offset, Document Frequency (DF), Collection Frequency (CF)]
    lexicon = {}
    
    with open(bin_filepath, "wb") as f_bin:
        for term, postings in index.items():
            offset = f_bin.tell()
            num_docs = len(postings)  # Document Frequency (DF): count of distinct documents containing term
            
            cf = 0                    # Collection Frequency (CF): total occurrences across collection
            packed_list = []
            for doc_id, positions in postings.items():
                tf = len(positions)   # Term Frequency (TF): occurrences in this document
                cf += tf
                # Pack DocID and TF as two 4-byte unsigned integers (<II)
                packed_list.append(struct.pack('<II', doc_id, tf))
                # Pack the exact positional list as 4-byte unsigned integers
                packed_list.append(struct.pack(f'<{tf}I', *positions))
                
            f_bin.write(b''.join(packed_list))
            
            # Save vocabulary metadata:
            # [byte_offset in secondary storage postings.bin, DF (num_docs), CF (collection frequency)]
            lexicon[term] = [offset, num_docs, cf]
            
    with open(lex_filepath, "w", encoding="utf-8") as f_lex:
        json.dump(lexicon, f_lex, ensure_ascii=False)
        
    print(f"Index built! Saved {len(lexicon)} terms to secondary storage '{bin_filepath}' and vocabulary to '{lex_filepath}'.")

if __name__ == "__main__":
    input_file = "docs.jsonl" if len(sys.argv) <= 1 else sys.argv[1]
    bin_file = "postings.bin" if len(sys.argv) <= 2 else sys.argv[2]
    lex_file = "lexicon.json" if len(sys.argv) <= 3 else sys.argv[3]
    
    if not os.path.exists(input_file) and not os.path.exists(input_file + ".gz"):
        print(f"Error: Neither '{input_file}' nor '{input_file}.gz' found.")
        sys.exit(1)
        
    build_binary_index(input_file, bin_file, lex_file)
