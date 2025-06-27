# Text Summarization Tool 📝

A powerful Natural Language Processing tool that implements both **extractive** and **abstractive** text summarization techniques using state-of-the-art AI models.

## 🚀 Features

- **Dual Summarization Methods**:
  - **Extractive**: Uses TextRank algorithm to select the most important sentences
  - **Abstractive**: Leverages Facebook's BART model to generate new summary text
- **Flexible Input**: Support for direct text input or file reading
- **Customizable Output**: Adjustable summary length and sentence count
- **Performance Metrics**: Compression ratios and word count statistics
- **CLI Interface**: Easy-to-use command-line interface

## 🛠️ Technology Stack

- **Python 3.7+**
- **Transformers** (Hugging Face) - BART model for abstractive summarization
- **NLTK** - Natural language processing utilities
- **PyTorch** - Deep learning framework
- **NetworkX** - Graph-based algorithms for TextRank
- **NumPy** - Numerical computations

## 📦 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/text-summarization-tool.git
cd text-summarization-tool
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (done automatically on first run):
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 🎯 Usage

### Command Line Interface

**Basic usage**:
```bash
# Summarize text directly
python summarizer.py --text "Your long text here..."

# Summarize from file
python summarizer.py --file document.txt

# Use specific method
python summarizer.py --file article.txt --method abstractive

# Customize parameters
python summarizer.py --file text.txt --method both --sentences 5 --max-length 200
```

**Advanced options**:
```bash
# Save output to file
python summarizer.py --file input.txt --output summary.txt

# Fine-tune extractive summarization
python summarizer.py --text "Long text..." --method extractive --sentences 3

# Adjust abstractive summary length
python summarizer.py --file doc.txt --method abstractive --max-length 100
```

### Python Integration

```python
from summarizer import TextSummarizer

# Initialize
summarizer = TextSummarizer()

# Summarize text
text = "Your long document text here..."
results = summarizer.summarize(text, method="both")

print("Extractive Summary:", results["extractive"])
print("Abstractive Summary:", results["abstractive"])
```

## 📊 Example Output

```
TEXT SUMMARIZATION RESULTS
==================================================
Original text length: 450 words

EXTRACTIVE SUMMARY (TextRank Algorithm):
----------------------------------------
The research demonstrates significant improvements in natural language processing. 
Machine learning models show 85% accuracy in text classification tasks. 
These findings contribute to advancing AI applications in real-world scenarios.

Length: 28 words

ABSTRACTIVE SUMMARY (BART Model):
----------------------------------------
Recent research shows major advances in NLP with ML models achieving 85% accuracy 
in text classification, contributing to practical AI applications.

Length: 22 words

Extractive compression ratio: 6.2%
Abstractive compression ratio: 4.9%
```

## 🧠 How It Works

### Extractive Summarization
1. **Tokenization**: Split text into sentences
2. **Preprocessing**: Remove stop words and normalize text  
3. **Scoring**: Calculate sentence importance using word frequency
4. **Selection**: Choose top-ranked sentences
5. **Assembly**: Combine sentences maintaining original order

### Abstractive Summarization
1. **Model Loading**: Initialize pre-trained BART model
2. **Encoding**: Convert text to numerical representations
3. **Generation**: Use transformer architecture to create new text
4. **Decoding**: Convert back to human-readable summary
5. **Post-processing**: Clean and format output

## 📈 Performance

- **Speed**: ~2-5 seconds for documents up to 1000 words
- **Accuracy**: Leverages state-of-the-art BART model (CNN/DailyMail dataset)
- **Memory**: Optimized for CPU usage, GPU acceleration available
- **Scalability**: Handles documents up to 10,000 words efficiently

## 🔧 Configuration

The tool supports various configuration options:

- `--method`: Choose between `extractive`, `abstractive`, or `both`
- `--sentences`: Number of sentences for extractive summary (default: 3)
- `--max-length`: Maximum length for abstractive summary (default: 150)
- `--output`: Save results to specified file

## 🎓 Educational Value

This project demonstrates:
- **NLP Fundamentals**: Tokenization, preprocessing, text analysis
- **Machine Learning**: Pre-trained models, transfer learning
- **Algorithm Implementation**: TextRank, graph-based ranking
- **Software Engineering**: Clean code, CLI design, error handling
- **AI Applications**: Practical use of transformer models

## 🚀 Future Enhancements

- [ ] Web interface using Flask/Streamlit
- [ ] Support for multiple languages
- [ ] Document format support (PDF, DOCX)
- [ ] Fine-tuning on domain-specific data
- [ ] Batch processing capabilities
- [ ] REST API development

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

Nikhil Sah
- GitHub: [@Nikhil-JUIT27](https://github.com/Nikhil-JUIT27)

## 🙏 Acknowledgments

- Hugging Face for the transformers library
- Facebook AI for the BART model
- NLTK team for natural language processing tools
- NetworkX developers for graph algorithms

---

⭐ **Star this repository if you found it helpful!**
