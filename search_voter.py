#!/usr/bin/env python3
"""
Voter search with intelligent matching and confidence scoring

Usage:
    python search_voter.py "Jefferson Escoto"
    python search_voter.py "Jeff Escoto"
    python search_voter.py "test" --run-tests
"""

import os
import sys
import argparse
from pinecone import Pinecone
from embedding_generator import EmbeddingGenerator
import json
from typing import List, Dict, Tuple
from fuzzywuzzy import fuzz
import pandas as pd
from tabulate import tabulate
import time
import numpy as np

class VoterSearcher:
    """Search with intelligent matching and confidence scoring"""
    
    def __init__(self):
        """Initialize Pinecone and embedding generator"""
        # Pinecone setup
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
            
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(host="https://quincy-voters-3d2397f.svc.aped-4627-b74a.pinecone.io")
        self.namespace = "voter-names"
        
        # Embedding setup
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        self.embedder = EmbeddingGenerator(api_key=openai_key, model="text-embedding-3-large")
    
    def calculate_string_similarity(self, query: str, result_name: str) -> Dict[str, float]:
        """Calculate string-based similarity metrics"""
        query_lower = query.lower().strip()
        result_lower = result_name.lower().strip()
        
        # Split names for component matching
        query_parts = set(query_lower.split())
        result_parts = set(result_lower.split())
        
        # Calculate various similarity metrics
        scores = {
            'exact_match': 1.0 if query_lower == result_lower else 0.0,
            'ratio': fuzz.ratio(query_lower, result_lower) / 100.0,
            'partial_ratio': fuzz.partial_ratio(query_lower, result_lower) / 100.0,
            'token_sort': fuzz.token_sort_ratio(query_lower, result_lower) / 100.0,
            'token_set': fuzz.token_set_ratio(query_lower, result_lower) / 100.0,
        }
        
        # Check for subset matches (e.g., "Jeff" in "Jefferson")
        subset_match = 0.0
        for q_part in query_parts:
            for r_part in result_parts:
                if len(q_part) >= 3:  # Only check meaningful substrings
                    if q_part in r_part or r_part in q_part:
                        subset_match = max(subset_match, 0.8)
                    elif q_part[:3] == r_part[:3]:  # First 3 letters match
                        subset_match = max(subset_match, 0.6)
        
        scores['subset_match'] = subset_match
        
        # Check if last names match (often most important)
        last_name_match = 0.0
        if len(query_parts) > 1 and len(result_parts) > 1:
            query_last = list(query_parts)[-1]
            result_last = list(result_parts)[-1]
            if query_last == result_last:
                last_name_match = 1.0
            elif fuzz.ratio(query_last, result_last) > 85:
                last_name_match = 0.8
        
        scores['last_name_match'] = last_name_match
        
        return scores
    
    def calculate_confidence(self, vector_score: float, string_scores: Dict[str, float]) -> Tuple[float, str]:
        """
        Calculate overall confidence and determine match quality
        
        The key insight: vector embeddings already handle semantic similarity
        (including nicknames), so we use string matching mainly for validation
        """
        
        # If exact match, highest confidence
        if string_scores['exact_match'] == 1.0:
            return 0.99, "EXACT_MATCH"
        
        # Trust high vector scores more - they understand nicknames!
        if vector_score > 0.96:
            # Very high vector score = trust it completely
            return vector_score, "VERY_HIGH_CONFIDENCE"
        
        # Strong vector match (0.92-0.96)
        if vector_score > 0.92:
            # If last name matches perfectly, boost confidence
            if string_scores['last_name_match'] == 1.0:
                return min(vector_score * 1.03, 0.99), "VERY_HIGH_CONFIDENCE"
            # Even without perfect string match, trust the embedding
            return vector_score, "HIGH_CONFIDENCE"
        
        # Good vector match (0.88-0.92)
        if vector_score > 0.88:
            if string_scores['last_name_match'] > 0.8:
                return vector_score, "HIGH_CONFIDENCE"
            else:
                return vector_score, "GOOD_MATCH"
        
        # Moderate vector match (0.83-0.88)
        if vector_score > 0.83:
            # Still trust the vector score primarily
            return vector_score, "GOOD_MATCH"
        
        # Lower confidence matches (0.78-0.83)
        if vector_score > 0.78:
            if string_scores['last_name_match'] > 0.9:
                return vector_score, "POSSIBLE_MATCH"
            else:
                return vector_score * 0.95, "POSSIBLE_MATCH"
        
        # Low confidence
        return vector_score, "LOW_CONFIDENCE"
    
    def search_enhanced(self, query: str, top_k: int = 10, confidence_threshold: float = 0.75) -> List[Dict]:
        """
        Search for voters with enhanced confidence scoring
        
        Args:
            query: Name to search for
            top_k: Number of results from vector search
            confidence_threshold: Minimum confidence to include
            
        Returns:
            List of matches with confidence scores
        """
        # Normalize query for consistency - lowercase!
        query_normalized = ' '.join(query.strip().lower().split())
        
        # Generate embedding for query
        query_embedding = self.embedder.get_embedding(query_normalized, dimensions=3072)
        
        # Search in Pinecone
        results = self.index.query(
            namespace=self.namespace,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        matches = []
        for match in results['matches']:
            metadata = match['metadata']
            vector_score = match['score']
            
            # Calculate string similarity
            full_name = metadata.get('full_name', '')
            string_scores = self.calculate_string_similarity(query, full_name)
            
            # Calculate final confidence
            confidence, match_quality = self.calculate_confidence(vector_score, string_scores)
            
            # Skip if below threshold (but show for debugging)
            if confidence < confidence_threshold and confidence < 0.7:
                continue
            
            # Build result
            result = {
                'voter_id': metadata.get('voter_id'),
                'ma_voter_id': metadata.get('ma_voter_id'),
                'full_name': full_name,
                'first_name': metadata.get('first_name'),
                'last_name': metadata.get('last_name'),
                'age': metadata.get('age'),
                'gender': metadata.get('gender'),
                'ward': metadata.get('ward'),
                'precinct': metadata.get('precinct'),
                'vector_score': round(vector_score, 4),
                'confidence': round(confidence, 4),
                'match_quality': match_quality,
                'string_similarity': {k: round(v, 3) for k, v in string_scores.items() if v > 0}
            }
            
            matches.append(result)
        
        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        return matches
    
    def analyze_match(self, query: str, result: Dict) -> Dict:
        """Analyze why a match was made"""
        analysis = {
            'query': query,
            'result': result['full_name'],
            'confidence': result['confidence'],
            'match_quality': result['match_quality'],
            'vector_score': result['vector_score'],
            'explanation': []
        }
        
                        # Explain the match
        if result['match_quality'] == 'EXACT_MATCH':
            analysis['explanation'].append("Names are identical")
        elif result['vector_score'] > 0.96:
            analysis['explanation'].append("Extremely high semantic match (nickname or same person)")
        elif result['vector_score'] > 0.92:
            analysis['explanation'].append("Very high semantic similarity (likely nickname variation)")
        elif result['vector_score'] > 0.88:
            analysis['explanation'].append("High semantic similarity")
        elif result['vector_score'] > 0.83:
            analysis['explanation'].append("Good semantic match")
        
        # String similarity insights
        string_sim = result.get('string_similarity', {})
        if string_sim.get('last_name_match', 0) > 0.8:
            analysis['explanation'].append("Last names match")
        if string_sim.get('subset_match', 0) > 0.6:
            analysis['explanation'].append("Name components overlap (possible nickname)")
        if string_sim.get('token_set', 0) > 0.85:
            analysis['explanation'].append("Same name components in different order")
        
        return analysis
    
    def test_search_system(self):
        """Test the search system with various queries"""
        test_cases = [
            # Real variations that should work
            ("Jefferson Escoto", "Testing full name match"),
            ("Jeff Escoto", "Testing nickname variation"),
            ("J Escoto", "Testing initial + last name"),
            ("Escoto", "Testing last name only"),
            ("JEFFERSON ESCOTO", "Testing case variation"),
            ("Rebecca White", "Testing different person"),
            ("Katherine Nguyen", "Testing another person"),
            
            # Edge cases
            ("John Smith", "Testing common name"),
            ("Smith", "Testing very common last name"),
            ("J", "Testing single letter"),
            ("XYZ ABC", "Testing non-existent person"),
        ]
        
        print("\n" + "="*80)
        print("SEARCH SYSTEM TEST")
        print("="*80)
        
        for query, description in test_cases:
            print(f"\n🔍 Test: {description}")
            print(f"   Query: '{query}'")
            print("-" * 60)
            
            matches = self.search_enhanced(query, top_k=3, confidence_threshold=0.0)
            
            if matches:
                # Show top match
                top_match = matches[0]
                print(f"✓ Top match: {top_match['full_name']}")
                print(f"  Confidence: {top_match['confidence']}")
                print(f"  Quality: {top_match['match_quality']}")
                print(f"  Vector Score: {top_match['vector_score']}")
                
                # Analyze the match
                analysis = self.analyze_match(query, top_match)
                if analysis['explanation']:
                    print(f"  Why: {'; '.join(analysis['explanation'])}")
                
                # Show other matches if significantly different
                if len(matches) > 1 and matches[1]['confidence'] > 0.7:
                    print(f"\n  Also found:")
                    for match in matches[1:3]:
                        print(f"  - {match['full_name']} (confidence: {match['confidence']})")
            else:
                print("✗ No matches found above threshold")
            
            time.sleep(0.5)  # Rate limiting
    
    def batch_search(self, queries: List[str], output_file: str = None) -> pd.DataFrame:
        """Process multiple searches and return results as DataFrame"""
        results = []
        
        print(f"\nProcessing {len(queries)} queries...")
        for query in queries:
            matches = self.search_enhanced(query, top_k=1)
            
            if matches:
                match = matches[0]
                results.append({
                    'query': query,
                    'found_name': match['full_name'],
                    'confidence': match['confidence'],
                    'quality': match['match_quality'],
                    'voter_id': match['voter_id'],
                    'ward': match.get('ward', ''),
                    'age': match.get('age', '')
                })
            else:
                results.append({
                    'query': query,
                    'found_name': 'NO MATCH',
                    'confidence': 0,
                    'quality': 'NO_MATCH',
                    'voter_id': '',
                    'ward': '',
                    'age': ''
                })
        
        df = pd.DataFrame(results)
        
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Results saved to: {output_file}")
        
        return df


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Voter search with intelligent matching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python search_voter.py "John Smith"              # Search for a voter
  python search_voter.py "Jeff Escoto"             # Handles variations automatically
  python search_voter.py "test" --run-tests        # Run test suite
  python search_voter.py --batch names.txt         # Batch search from file
        """
    )
    
    parser.add_argument('name', nargs='?', help='Name to search for')
    parser.add_argument('--run-tests', action='store_true',
                       help='Run test suite')
    parser.add_argument('--top', type=int, default=5,
                       help='Number of results to return (default: 5)')
    parser.add_argument('--threshold', type=float, default=0.75,
                       help='Minimum confidence threshold (default: 0.75)')
    parser.add_argument('--batch', type=str,
                       help='Batch search from file')
    parser.add_argument('--analyze', action='store_true',
                       help='Show detailed analysis of matches')
    parser.add_argument('--json', action='store_true',
                       help='Output as JSON')
    
    args = parser.parse_args()
    
    # Initialize searcher
    try:
        searcher = VoterSearcher()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Handle different modes
    if args.run_tests or (args.name and args.name.lower() == "test"):
        searcher.test_search_system()
    
    elif args.batch:
        # Batch search from file
        if not os.path.exists(args.batch):
            print(f"File not found: {args.batch}")
            sys.exit(1)
        
        with open(args.batch, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        df = searcher.batch_search(queries, output_file=args.batch.replace('.txt', '_results.csv'))
        print("\nResults Summary:")
        print(tabulate(df, headers='keys', tablefmt='grid'))
        
        # Stats
        matched = len(df[df['confidence'] > 0])
        print(f"\nMatched: {matched}/{len(queries)} ({matched/len(queries)*100:.1f}%)")
    
    elif args.name:
        # Single search
        print(f"\n🔍 Searching for: '{args.name}'")
        print("-" * 60)
        
        matches = searcher.search_enhanced(
            args.name, 
            top_k=args.top, 
            confidence_threshold=args.threshold
        )
        
        if matches:
            if args.json:
                # Clean output for JSON
                for match in matches:
                    match.pop('string_similarity', None)
                print(json.dumps(matches, indent=2))
            else:
                for i, match in enumerate(matches, 1):
                    print(f"\n{i}. {match['full_name']}")
                    print(f"   Confidence: {match['confidence']} ({match['match_quality']})")
                    print(f"   Voter ID: {match['voter_id']}")
                    print(f"   MA ID: {match['ma_voter_id']}")
                    
                    if match.get('age'):
                        print(f"   Age: {match['age']}")
                    if match.get('ward'):
                        print(f"   Location: Ward {match['ward']}, Precinct {match['precinct']}")
                    
                    if args.analyze:
                        analysis = searcher.analyze_match(args.name, match)
                        print(f"   Analysis: {'; '.join(analysis['explanation'])}")
                        print(f"   Vector Score: {match['vector_score']}")
        else:
            print(f"\nNo matches found with confidence >= {args.threshold}")
            print("Try lowering the threshold with --threshold 0.5")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()