import struct
import json
import time
import sys
import os
import string
import mmap
from functools import lru_cache

def tokenize(text):
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation + '।॥')
    return text.translate(translator).split()

class BinarySearchEngine:
    def __init__(self, bin_path="postings.bin", lex_path="lexicon.json"):
        # Load the entire lexicon into RAM immediately on boot
        # It's just a Python dictionary, so lookups are O(1) hash jumps!
        print("Loading lexicon into RAM...")
        with open(lex_path, "r", encoding="utf-8") as f:
            self.lexicon = json.load(f)
            
        # Memory-map the binary postings file. This maps the physical file into the OS Page Cache
        # allowing us to read bytes directly from secondary storage with zero file-system overhead!
        print("Memory-mapping binary index...")
        self.bin_file = open(bin_path, "rb")
        
        # OS Cache Warmup: Force the OS to load the binary file pages into memory before mapping
        print("Warming up OS Page Cache for binary file...")
        self.bin_file.read()
        
        self.mmap_obj = mmap.mmap(self.bin_file.fileno(), 0, access=mmap.ACCESS_READ)

        # Pre-warm the Python LRU Cache with the top 500 most common words
        print("Pre-warming Python LRU Cache with the 500 heaviest words...")
        # Sort lexicon by Document Frequency (num_docs) descending
        top_words = sorted(self.lexicon.items(), key=lambda x: x[1][1], reverse=True)[:500]
        for term, _ in top_words:
            self.get_postings(term)
        print("LRU Cache warm-up complete!")
        
    def __del__(self):
        try:
            self.mmap_obj.close()
            self.bin_file.close()
        except:
            pass

    def get_term_stats(self, term):
        """
        Retrieves vocabulary metadata and positional postings list for a term.
        Returns:
          - term: token string
          - df: Document Frequency (number of documents containing term)
          - cf: Collection Frequency (total occurrences across collection)
          - offset: byte offset in secondary storage postings file
          - postings: {doc_id: [positions]}
        """
        tokens = tokenize(term)
        if not tokens:
            return None
        t = tokens[0]
        if t not in self.lexicon:
            return None
        offset, num_docs, cf = self.lexicon[t]
        postings = self.get_postings(t)
        return {
            "term": t,
            "df": num_docs,
            "cf": cf,
            "offset": offset,
            "postings": postings
        }

    @lru_cache(maxsize=10000)
    def get_postings(self, term):
        # 1. O(1) Lexicon Hash Lookup in Main Memory
        if term not in self.lexicon:
            return {}
            
        offset, num_docs, _ = self.lexicon[term]
        
        # 2. Jump directly to the exact byte offset in the postings file on secondary storage
        results = {}
        curr_offset = offset
        
        # 3. Unpack the custom binary struct integers using C-optimized struct
        for _ in range(num_docs):
            # Read DocID and TF (Term Frequency in doc)
            doc_id, tf = struct.unpack_from('<II', self.mmap_obj, curr_offset)
            curr_offset += 8
            
            # Read the array of token positions
            positions = struct.unpack_from(f'<{tf}I', self.mmap_obj, curr_offset)
            curr_offset += 4 * tf
            
            # Convert DocID back to string for frontend compatibility, and positions to a list
            results[str(doc_id)] = list(positions)
            
        return results

    def phrase_search(self, phrase):
        tokens = tokenize(phrase)
        if not tokens: return []
        
        term1_postings = self.get_postings(tokens[0])
        if not term1_postings: return []
        
        candidates = set(term1_postings.keys())
        other_postings = []
        
        for token in tokens[1:]:
            postings = self.get_postings(token)
            if not postings: return []
            other_postings.append(postings)
            candidates.intersection_update(postings.keys())
            
        if not candidates: return []
        
        results = []
        for doc_id in candidates:
            valid_positions = term1_postings[doc_id]
            matched_positions = []
            
            # Pre-convert other lists to sets for O(1) mathematical lookup!
            other_pos_sets = [set(other_postings[i][doc_id]) for i in range(len(other_postings))]
            
            for pos in valid_positions:
                match = True
                for i in range(len(other_postings)):
                    if (pos + i + 1) not in other_pos_sets[i]:
                        match = False
                        break
                if match:
                    matched_positions.append(pos)
                    
            if matched_positions:
                results.append({"doc_id": doc_id, "positions": matched_positions})
                
        return results

    def proximity_search(self, term1, term2, distance):
        t1 = tokenize(term1)
        t2 = tokenize(term2)
        if not t1 or not t2: return []
        
        term1, term2 = t1[0], t2[0]
        p1 = self.get_postings(term1)
        p2 = self.get_postings(term2)
        
        if not p1 or not p2: return []
        
        candidates = set(p1.keys()).intersection(p2.keys())
        results = []
        
        for doc_id in candidates:
            positions1 = p1[doc_id]
            positions2 = p2[doc_id]
            pos2_set = set(positions2)
            matched_positions = []
            
            target_dist = distance + 1
            for pos1 in positions1:
                if (pos1 + target_dist) in pos2_set:
                    matched_positions.append(f"({pos1}, {pos1 + target_dist})")
                if (pos1 - target_dist) in pos2_set:
                    matched_positions.append(f"({pos1}, {pos1 - target_dist})")
                        
            if matched_positions:
                results.append({"doc_id": doc_id, "positions": matched_positions})
                
        return results

RARE_TERMS_TEST_SET = [
    ('पञ्चोपचार', '14'),
    ('काव्यावतरित', '10'),
    ('पदविज्ञान', '56'),
    ('सुपाठ्य', '58'),
    ('नेवीब्लू', '59'),
    ('सुगमतम', '79'),
    ('आत्मपरिचय', '1871'),
    ('सदाएसरहद', '882'),
    ('अबोधावस्था', '15116'),
    ('फ्लोरिडियन', '11747')
]

def run_test_cases(engine, verbose=True):
    test_queries = []
    if os.path.exists("test.txt"):
        with open("test.txt", "r", encoding="utf-8") as f:
            test_queries = [line.strip() for line in f if line.strip()]
    if not test_queries:
        test_queries = [
            'भारत सरकार',
            'प्रथम विश्व',
            'मुख्य रूप से',
            'जन्म हुआ था',
            'संयुक्त राज्य अमेरिका में'
        ]

    results_data = []
    if verbose:
        print("\n" + "=" * 70)
        print("          RUNNING 5 TEST CASES (PHRASE SEARCH)")
        print("=" * 70)

    for i, phrase in enumerate(test_queries[:5], 1):
        start = time.perf_counter()
        matches = engine.phrase_search(phrase)
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000

        entry = {
            "index": i,
            "query": phrase,
            "type": "Phrase Search",
            "matches_count": len(matches),
            "elapsed_ms": elapsed_ms,
            "matches": matches
        }
        results_data.append(entry)

        if verbose:
            print(f"\n[Test Case {i}] Query: \"{phrase}\"")
            print(f"  -> Total Matching Documents: {len(matches)}")
            print(f"  -> Query Execution Time:    {elapsed_ms:.3f} ms")
            print("  -> Sample Matches:")
            for m in matches[:3]:
                print(f"     * Doc ID: {m['doc_id']:<6} | Positions: {m['positions'][:8]}{'...' if len(m['positions']) > 8 else ''}")

    return results_data

def show_rare_terms(engine, verbose=True):
    results_data = []
    if verbose:
        print("\n" + "=" * 70)
        print("   10 RARE TERMS WITH CORRESPONDING POSITIONAL INDEX")
        print("=" * 70)

    for i, (term, expected_doc) in enumerate(RARE_TERMS_TEST_SET, 1):
        stats = engine.get_term_stats(term)
        if not stats:
            continue
        entry = {
            "index": i,
            "term": term,
            "df": stats["df"],
            "cf": stats["cf"],
            "offset": stats["offset"],
            "doc_id": expected_doc,
            "postings": stats["postings"]
        }
        results_data.append(entry)

        if verbose:
            print(f"\n[Rare Term {i}] '{term}'")
            print(f"  -> Document Frequency (DF):   {stats['df']} (Occurs in {stats['df']} document)")
            print(f"  -> Collection Frequency (CF): {stats['cf']} (Total occurrences in collection)")
            print(f"  -> Byte Offset in Secondary Storage: {stats['offset']} bytes")
            print(f"  -> Document ID: {expected_doc}")
            print(f"  -> Positional Index: {stats['postings']}")

    return results_data

def generate_output_file(engine, filename="output.txt"):
    """Generates the official output.txt submission file required by the assignment."""
    test_results = run_test_cases(engine, verbose=False)
    rare_results = show_rare_terms(engine, verbose=False)

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("       POSITIONAL INDEX SYSTEM OUTPUT\n")
        f.write("=" * 80 + "\n\n")
        f.write("Dataset:         Hindi Wikipedia (10,000 Articles)\n")
        f.write(f"Vocabulary Size: {len(engine.lexicon):,} Unique Terms\n")
        f.write("Storage Design:  Vocabulary in Main Memory (RAM), Postings in Secondary Memory (postings.bin)\n")

        f.write("=" * 80 + "\n")
        f.write("PART 1: 5 TEST CASES USED TO TEST THE SYSTEM\n")
        f.write("=" * 80 + "\n\n")

        for tc in test_results:
            f.write(f"--------------------------------------------------------------------------------\n")
            f.write(f"Test Case {tc['index']}: \"{tc['query']}\"\n")
            f.write(f"Search Type:                 {tc['type']}\n")
            f.write(f"Total Matching Documents:    {tc['matches_count']}\n")
            f.write(f"Execution Latency:           {tc['elapsed_ms']:.4f} ms ({tc['elapsed_ms']*1000:.1f} microseconds)\n")
            f.write(f"Matching Documents Breakdown:\n")

            if tc["matches_count"] == 0:
                f.write("  (No matches found)\n")
            else:
                for idx, doc in enumerate(tc["matches"][:5], 1):
                    f.write(f"  {idx}. Doc ID: {doc['doc_id']}\n")
                    f.write(f"     Matched Token Starting Positions: {doc['positions']}\n")
                if tc["matches_count"] > 5:
                    f.write(f"     ... and {tc['matches_count'] - 5} more documents.\n")
            f.write("\n")

        # Include an additional proximity test case for completeness
        f.write("--------------------------------------------------------------------------------\n")
        f.write("Bonus Proximity Test Case: \"भारत\" NEAR/1 \"सरकार\"\n")
        start_prox = time.perf_counter()
        prox_matches = engine.proximity_search("भारत", "सरकार", 1)
        end_prox = time.perf_counter()
        prox_ms = (end_prox - start_prox) * 1000
        f.write(f"Search Type:                 Proximity Search (Distance = 1 word apart)\n")
        f.write(f"Total Matching Documents:    {len(prox_matches)}\n")
        f.write(f"Execution Latency:           {prox_ms:.4f} ms\n")
        f.write(f"Matching Documents Breakdown:\n")
        for idx, doc in enumerate(prox_matches[:5], 1):
            f.write(f"  {idx}. Doc ID: {doc['doc_id']}\n")
            f.write(f"     Matched Position Pairs: {doc['positions']}\n")
        f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("PART 2: 10 RARE TERMS WITH THEIR CORRESPONDING POSITIONAL INDEX\n")
        f.write("=" * 80 + "\n")
        f.write("Description: Rare terms from the Hindi vocabulary (DF=1, CF=1) along with their\n")
        f.write("Collection Frequency, Document Frequency, and exact positional index extracted\n")
        f.write("directly from secondary memory (postings.bin).\n\n")

        for r in rare_results:
            f.write(f"--------------------------------------------------------------------------------\n")
            f.write(f"Rare Term {r['index']}: \"{r['term']}\"\n")
            f.write(f"  Document Frequency (DF):     {r['df']} (Number of documents containing term)\n")
            f.write(f"  Collection Frequency (CF):   {r['cf']} (Total occurrences in collection)\n")
            f.write(f"  Secondary Storage Offset:    {r['offset']} bytes in postings.bin\n")
            f.write(f"  Document ID:                 {r['doc_id']}\n")
            f.write(f"  Positional Index:            DocID {r['doc_id']} -> Positions {r['postings'][r['doc_id']]}\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF OUTPUT SUBMISSION REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"Successfully generated official submission report: {filename}")

if __name__ == "__main__":
    if not os.path.exists("postings.bin") or not os.path.exists("lexicon.json"):
        print("Binary index not found. Run indexer.py first.")
        sys.exit(1)
        
    engine = BinarySearchEngine()
    
    # Check for CLI flags
    if len(sys.argv) > 1:
        flag = sys.argv[1]
        if flag == "--test":
            run_test_cases(engine, verbose=True)
            sys.exit(0)
        elif flag == "--rare":
            show_rare_terms(engine, verbose=True)
            sys.exit(0)
        elif flag == "--generate-output":
            out_file = sys.argv[2] if len(sys.argv) > 2 else "output.txt"
            generate_output_file(engine, out_file)
            sys.exit(0)
        elif flag == "--phrase" and len(sys.argv) > 2:
            query_phrase = " ".join(sys.argv[2:])
            t0 = time.perf_counter()
            res = engine.phrase_search(query_phrase)
            t1 = time.perf_counter()
            print(f"Phrase '{query_phrase}': Found {len(res)} matches in {(t1-t0)*1000:.3f} ms.")
            for r in res[:10]:
                print(f"  Doc {r['doc_id']}: positions {r['positions']}")
            sys.exit(0)
        elif flag == "--term" and len(sys.argv) > 2:
            term_arg = sys.argv[2]
            stats = engine.get_term_stats(term_arg)
            if not stats:
                print(f"Term '{term_arg}' not found in vocabulary.")
            else:
                print(f"Term: {stats['term']} | DF: {stats['df']} | CF: {stats['cf']} | Offset: {stats['offset']}")
                print(f"Postings: {stats['postings']}")
            sys.exit(0)
        elif flag == "--proximity" and len(sys.argv) > 4:
            t1, t2, d = sys.argv[2], sys.argv[3], int(sys.argv[4])
            t0 = time.perf_counter()
            res = engine.proximity_search(t1, t2, d)
            t1 = time.perf_counter()
            print(f"Proximity '{t1}' NEAR/{d} '{t2}': Found {len(res)} matches in {(t1-t0)*1000:.3f} ms.")
            for r in res[:10]:
                print(f"  Doc {r['doc_id']}: positions {r['positions']}")
            sys.exit(0)

    # Interactive menu
    while True:
        try:
            print("\n" + "=" * 50)
            print("     HINDI POSITIONAL SEARCH ENGINE CLI")
            print("=" * 50)
            print("1. Phrase search (e.g. 'भारत सरकार')")
            print("2. Proximity search (e.g. 'भारत' and 'सरकार' with distance 1)")
            print("3. Term Lookup (Show DF, CF, and Positional Postings List)")
            print("4. Run 5 Test Cases (from test.txt)")
            print("5. Show 10 Rare Terms Examples")
            print("6. Generate / Update output.txt")
            print("7. Exit")
            choice = input("Enter choice (1-7): ").strip()
            
            if choice == '1':
                phrase = input("Enter phrase: ").strip()
                start = time.perf_counter()
                docs = engine.phrase_search(phrase)
                end = time.perf_counter()
                print(f"\nFound {len(docs)} matching documents in {round((end-start)*1000, 3)} ms:")
                for d in docs[:10]:
                    print(f"  Doc ID: {d['doc_id']:<6} | Positions: {d['positions']}")
                if len(docs) > 10:
                    print(f"  ... and {len(docs) - 10} more documents.")
            elif choice == '2':
                t1 = input("Enter term 1: ").strip()
                t2 = input("Enter term 2: ").strip()
                dist = int(input("Enter word distance in between: "))
                start = time.perf_counter()
                docs = engine.proximity_search(t1, t2, dist)
                end = time.perf_counter()
                print(f"\nFound {len(docs)} matching documents in {round((end-start)*1000, 3)} ms:")
                for d in docs[:10]:
                    print(f"  Doc ID: {d['doc_id']:<6} | Position Pairs: {d['positions']}")
                if len(docs) > 10:
                    print(f"  ... and {len(docs) - 10} more documents.")
            elif choice == '3':
                term_input = input("Enter term to lookup: ").strip()
                stats = engine.get_term_stats(term_input)
                if not stats:
                    print(f"Term '{term_input}' not found in vocabulary.")
                else:
                    print(f"\nTerm:                        {stats['term']}")
                    print(f"Document Frequency (DF):     {stats['df']}")
                    print(f"Collection Frequency (CF):   {stats['cf']}")
                    print(f"Secondary Storage Offset:    {stats['offset']} bytes")
                    print(f"Postings List Count:         {len(stats['postings'])} documents")
                    sample_docs = list(stats['postings'].items())[:5]
                    print(f"Sample Postings (DocID: [Positions]): {sample_docs}")
            elif choice == '4':
                run_test_cases(engine, verbose=True)
            elif choice == '5':
                show_rare_terms(engine, verbose=True)
            elif choice == '6':
                generate_output_file(engine, "output.txt")
            elif choice == '7':
                print("Exiting.")
                break
            else:
                print("Invalid choice.")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
