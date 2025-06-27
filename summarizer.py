#!/usr/bin/env python3
"""
Text Summarization Tool using NLP
A simple yet powerful tool demonstrating extractive and abstractive summarization
Author: [Your Name]
"""

import os
import argparse
import sys
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    from nltk.cluster.util import cosine_distance
    import numpy as np
    import networkx as nx
    from collections import Counter
    import re
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

class TextSummarizer:
    """
    A comprehensive text summarization tool supporting both extractive and abstractive methods
    """
    
    def __init__(self):
        """Initialize the summarizer with pre-trained models"""
        print("Initializing Text Summarizer...")
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        # Initialize abstractive model (BART)
        try:
            print("Loading BART model for abstractive summarization...")
            self.abstractive_summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                tokenizer="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✓ BART model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load BART model: {e}")
            self.abstractive_summarizer = None
        
        # Initialize stop words for extractive summarization
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess input text"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', ' ', text)
        return text
    
    def extractive_summarization(self, text: str, num_sentences: int = 3) -> str:
        """
        Implement extractive summarization using TextRank algorithm
        """
        try:
            # Tokenize into sentences
            sentences = sent_tokenize(text)
            
            if len(sentences) <= num_sentences:
                return text
            
            # Create word frequency matrix
            word_freq = {}
            for sentence in sentences:
                words = sentence.lower().split()
                for word in words:
                    if word not in self.stop_words and len(word) > 2:
                        word_freq[word] = word_freq.get(word, 0) + 1
            
            # Calculate sentence scores
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                words = sentence.lower().split()
                score = 0
                word_count = 0
                for word in words:
                    if word in word_freq:
                        score += word_freq[word]
                        word_count += 1
                if word_count > 0:
                    sentence_scores[i] = score / word_count
            
            # Get top sentences
            top_sentences = sorted(sentence_scores.items(), 
                                 key=lambda x: x[1], reverse=True)[:num_sentences]
            top_sentences = sorted([idx for idx, score in top_sentences])
            
            # Return summary
            summary = ' '.join([sentences[i] for i in top_sentences])
            return summary
            
        except Exception as e:
            print(f"Error in extractive summarization: {e}")
            return text[:500] + "..." if len(text) > 500 else text
    
    def abstractive_summarization(self, text: str, max_length: int = 150, 
                                min_length: int = 50) -> str:
        """
        Implement abstractive summarization using BART model
        """
        if not self.abstractive_summarizer:
            return "Abstractive summarization not available. Using extractive method."
        
        try:
            # Split long texts into chunks
            max_input_length = 1024
            if len(text) > max_input_length:
                # Split into overlapping chunks
                chunks = []
                words = text.split()
                chunk_size = max_input_length // 4  # Approximate word count
                
                for i in range(0, len(words), chunk_size):
                    chunk = ' '.join(words[i:i + chunk_size])
                    chunks.append(chunk)
                
                # Summarize each chunk and combine
                summaries = []
                for chunk in chunks:
                    try:
                        result = self.abstractive_summarizer(
                            chunk,
                            max_length=max_length // len(chunks),
                            min_length=min_length // len(chunks),
                            do_sample=False
                        )
                        summaries.append(result[0]['summary_text'])
                    except:
                        continue
                
                return ' '.join(summaries)
            
            else:
                # Summarize directly
                result = self.abstractive_summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                return result[0]['summary_text']
                
        except Exception as e:
            print(f"Error in abstractive summarization: {e}")
            return self.extractive_summarization(text)
    
    def summarize(self, text: str, method: str = "both", 
                 extractive_sentences: int = 3, 
                 abstractive_max_length: int = 150) -> Dict[str, str]:
        """
        Main summarization function
        """
        if not text.strip():
            return {"error": "Empty text provided"}
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        results = {}
        
        if method in ["extractive", "both"]:
            print("Generating extractive summary...")
            results["extractive"] = self.extractive_summarization(
                processed_text, extractive_sentences
            )
        
        if method in ["abstractive", "both"]:
            print("Generating abstractive summary...")
            results["abstractive"] = self.abstractive_summarization(
                processed_text, abstractive_max_length
            )
        
        # Add statistics
        results["original_length"] = len(text.split())
        if "extractive" in results:
            results["extractive_length"] = len(results["extractive"].split())
        if "abstractive" in results:
            results["abstractive_length"] = len(results["abstractive"].split())
        
        return results

def read_file(filepath: str) -> str:
    """Read text from file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Text Summarization Tool using NLP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python summarizer.py --text "Your long text here..."
  python summarizer.py --file document.txt --method abstractive
  python summarizer.py --file article.txt --method both --sentences 5
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='Text to summarize')
    input_group.add_argument('--file', type=str, help='File path containing text')
    
    # Summarization options
    parser.add_argument('--method', choices=['extractive', 'abstractive', 'both'], 
                       default='both', help='Summarization method (default: both)')
    parser.add_argument('--sentences', type=int, default=3, 
                       help='Number of sentences for extractive summary (default: 3)')
    parser.add_argument('--max-length', type=int, default=150,
                       help='Max length for abstractive summary (default: 150)')
    parser.add_argument('--output', type=str, help='Output file path (optional)')
    
    args = parser.parse_args()
    
    # Get input text
    if args.text:
        input_text = args.text
    else:
        input_text = read_file(args.file)
        if not input_text:
            print("Failed to read input file")
            return
    
    # Initialize summarizer
    summarizer = TextSummarizer()
    
    # Generate summaries
    print(f"\nProcessing text ({len(input_text.split())} words)...")
    print("=" * 50)
    
    results = summarizer.summarize(
        input_text,
        method=args.method,
        extractive_sentences=args.sentences,
        abstractive_max_length=args.max_length
    )
    
    # Display results
    output_lines = []
    
    output_lines.append("TEXT SUMMARIZATION RESULTS")
    output_lines.append("=" * 50)
    output_lines.append(f"Original text length: {results.get('original_length', 0)} words")
    output_lines.append("")
    
    if "extractive" in results:
        output_lines.append("EXTRACTIVE SUMMARY (TextRank Algorithm):")
        output_lines.append("-" * 40)
        output_lines.append(results["extractive"])
        output_lines.append(f"\nLength: {results.get('extractive_length', 0)} words")
        output_lines.append("")
    
    if "abstractive" in results:
        output_lines.append("ABSTRACTIVE SUMMARY (BART Model):")
        output_lines.append("-" * 40)
        output_lines.append(results["abstractive"])
        output_lines.append(f"\nLength: {results.get('abstractive_length', 0)} words")
        output_lines.append("")
    
    # Calculate compression ratios
    if "extractive" in results:
        compression_ratio = (results.get('extractive_length', 0) / 
                           results.get('original_length', 1)) * 100
        output_lines.append(f"Extractive compression ratio: {compression_ratio:.1f}%")
    
    if "abstractive" in results:
        compression_ratio = (results.get('abstractive_length', 0) / 
                           results.get('original_length', 1)) * 100
        output_lines.append(f"Abstractive compression ratio: {compression_ratio:.1f}%")
    
    # Print results
    output_text = "\n".join(output_lines)
    print(output_text)
    
    # Save to file if specified
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_text)
            print(f"\nResults saved to: {args.output}")
        except Exception as e:
            print(f"Error saving output: {e}")

if __name__ == "__main__":
    main()
