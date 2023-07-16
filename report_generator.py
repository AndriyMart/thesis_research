import difflib
from rouge import Rouge
from nltk.tokenize import word_tokenize
import string
import re

report_path = "./results/report.html"
class ReportGenerator:

    def __init__(self):
        pass

    def generate_report(self, summary, reference_summary_path):
        # Read the reference summary from the file
        reference_summary = self._read_reference_summary(reference_summary_path)

        # Calculate ROUGE scores
        rouge_scores = self._calculate_rouge_scores(summary, reference_summary)
        print(rouge_scores)

        # Generate HTML with report
        self._generate_html(summary, reference_summary, report_path)

    def _tokenize_text_with_alignment(self, text):
        # Tokenize the text and create raw and normalized tokens at the same time
        raw_tokens = word_tokenize(text)
        normalized_tokens = [self._normalize_token(token) for token in raw_tokens]
        return raw_tokens, normalized_tokens

    def _normalize_token(self, token):
        # Lowercase, remove punctuation, and remove extra whitespaces
        return re.sub(' +', ' ', token.lower().translate(str.maketrans('', '', string.punctuation)).strip())

    def _calculate_rouge_scores(self, text, reference):
        rouge = Rouge()
        scores = rouge.get_scores(text, reference, avg=True)
        return scores

    def _generate_html_for_summary(self, raw_tokens, norm_tokens, highlight_words):
        html = ""
        for raw_word, norm_word in zip(raw_tokens, norm_tokens):
            if norm_word in highlight_words:
                html += '<span style="background-color: #FFFF00">' + raw_word + '</span> '
            else:
                html += raw_word + ' '
        return html

    def _generate_html(self, summary, reference_summary, output_html):
        # Tokenize the texts with alignment
        reference_raw_tokens, reference_norm_tokens = self._tokenize_text_with_alignment(reference_summary)
        summary_raw_tokens, summary_norm_tokens = self._tokenize_text_with_alignment(summary)

        # Use difflib to find common words
        matcher = difflib.SequenceMatcher(None, reference_norm_tokens, summary_norm_tokens)
        common_words = set(
            [reference_norm_tokens[match.a] for match in matcher.get_matching_blocks() if match.size > 0])

        # Create HTML for reference and summary
        reference_html = self._generate_html_for_summary(reference_raw_tokens, reference_norm_tokens, common_words)
        summary_html = self._generate_html_for_summary(summary_raw_tokens, summary_norm_tokens, common_words)

        html = """
        <h1>Reference</h1>
        <p>{}</p>
        <h1>Summary</h1>
        <p>{}</p>
        """.format(reference_html, summary_html)

        # Save the HTML to a file
        with open(output_html, 'w') as f:
            f.write(html)

        print(f"HTML saved to {output_html}")

    def _read_reference_summary(self, file_path):
        with open(file_path, 'r') as file:
            data = file.read().replace('\n', '')
        return data
