#!/usr/bin/env python3
"""
Demo script for Text Summarization Tool
Shows practical examples of the summarizer in action
"""

from summarizer import TextSummarizer

def demo_news_article():
    """Demo with a sample news article"""
    
    sample_text = """
    Artificial Intelligence has revolutionized the way we process and understand human language. 
    Natural Language Processing, a subset of AI, focuses on the interaction between computers and humans through natural language. 
    The ultimate objective of NLP is to read, decipher, understand, and make sense of human languages in a manner that is valuable.
    
    Recent advances in transformer architecture have led to breakthrough models like BERT, GPT, and BART. 
    These models have achieved state-of-the-art performance on various NLP tasks including text summarization, 
    question answering, and language translation. The BART model, in particular, has shown exceptional performance 
    in abstractive text summarization tasks.
    
    Text summarization is the process of distilling the most important information from a source text. 
    There are two main approaches: extractive summarization selects sentences directly from the source text, 
    while abstractive summarization generates new sentences that capture the main ideas. 
    Modern applications of text summarization include news aggregation, research paper analysis, 
    and social media monitoring.
    
    The field continues to evolve with new architectures and training methodologies. 
    Recent research focuses on improving coherence, factual accuracy, and handling longer documents. 
    As computational resources become more accessible, we can expect even more sophisticated 
    summarization systems in the near future.
    """
    
    print("🤖 Text Summarization Tool Demo")
    print("=" * 50)
    print("Sample Text: AI and NLP News Article")
    print(f"Original length: {len(sample_text.split())} words\n")
    
    # Initialize summarizer
    summarizer = TextSummarizer()
    
    # Run both summarization methods
    results = summarizer.summarize(sample_text, method="both")
    
    print("📄 EXTRACTIVE SUMMARY:")
    print("-" * 30)
    print(results.get("extractive", "Not available"))
    print(f"Length: {results.get('extractive_length', 0)} words")
    
    print("\n🧠 ABSTRACTIVE SUMMARY:")
    print("-" * 30)
    print(results.get("abstractive", "Not available"))
    print(f"Length: {results.get('abstractive_length', 0)} words")
    
    # Calculate compression ratios
    original_length = results.get('original_length', 1)
    if 'extractive_length' in results:
        ext_ratio = (results['extractive_length'] / original_length) * 100
        print(f"\n📊 Extractive compression: {ext_ratio:.1f}%")
    
    if 'abstractive_length' in results:
        abs_ratio = (results['abstractive_length'] / original_length) * 100
        print(f"📊 Abstractive compression: {abs_ratio:.1f}%")

def demo_comparison():
    """Demo showing different summarization parameters"""
    
    text = """
    Machine learning is a method of data analysis that automates analytical model building. 
    It is a branch of artificial intelligence based on the idea that systems can learn from data, 
    identify patterns and make decisions with minimal human intervention. Machine learning algorithms 
    build a model based on training data in order to make predictions or decisions without being 
    explicitly programmed to do so. Machine learning algorithms are used in a wide variety of 
    applications, such as in medicine, email filtering, speech recognition, and computer vision, 
    where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.
    
    The field of machine learning is closely related to computational statistics, which focuses on 
    making predictions using computers. The study of mathematical optimization delivers methods, 
    theory and application domains to the field of machine learning. Data mining is a related field 
    of study, focusing on exploratory data analysis through unsupervised learning. Some implementations 
    of machine learning use data in a training phase to create a model, and then use that model in 
    an operational phase to evaluate new data. This is called supervised learning.
    """
    
    print("\n\n🔄 COMPARISON DEMO")
    print("=" * 50)
    
    summarizer = TextSummarizer()
    
    # Different extractive lengths
    print("📏 Different extractive summary lengths:")
    for sentences in [1, 2, 3]:
        result = summarizer.extractive_summarization(text, sentences)
        print(f"\n{sentences} sentences: {result}")
    
    # Different abstractive lengths
    print("\n📏 Different abstractive summary lengths:")
    for max_len in [50, 100, 150]:
        result = summarizer.abstractive_summarization(text, max_len)
        print(f"\nMax {max_len} words: {result}")

if __name__ == "__main__":
    demo_news_article()
    demo_comparison()
    
    print("\n\n✨ Demo completed!")
    print("Try running the CLI tool with: python summarizer.py --help")
