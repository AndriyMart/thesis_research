import difflib
from nltk.tokenize import word_tokenize
import string
import re

report_path = "./results/report.html"
rouge_scores_comments = {
    'rouge-1-r': "Measures the overlap of <b>unigram (single word)</b> sequences between the generated summary and the reference summary, taking into account the recall (R) aspect. A higher recall score indicates that the generated summary contains a significant portion of the information present in the reference summary.",
    'rouge-1-p': "Takes into account the precision (P) aspect. A higher precision score indicates that the generated summary contains a higher proportion of information that is directly relevant to the reference summary.",
    'rouge-1-f': "Combines both the precision (P) and recall (R) aspects.",
    'rouge-2-r': "Focuses on the overlap of <b>bigram (two-word)</b> sequences. Considers the recall (R) aspect.",
    'rouge-2-p': "Focuses on the overlap of <b>bigram (two-word)</b> sequences. Takes into account the precision (P) aspect.",
    'rouge-2-f': "Combines both the precision (P) and recall (R) aspects.",
    'rouge-l-r': "Measures the overlap of <b>Longest Common Subsequences</b> (LCS) between the generated summary and the reference summary. Considers the recall (R) aspect.",
    'rouge-l-p': "Measures the overlap of <b>Longest Common Subsequences</b> (LCS). Considers the recall (R) aspect.",
    'rouge-l-f': "Combines both the precision (P) and recall (R) aspects.",
}


class ReportGenerator:

    def __init__(self):
        pass

    def generate_report(self, report_configs):
        html_report = self._get_rouge_scores_table(report_configs)

        for report_config in report_configs:
            # Generate HTML with report
            html_report += self._generate_html_for_report(report_config)

        html_report += "</div></div>"

        # Save the HTML to a file
        with open(report_path, 'w') as f:
            f.write(html_report)

        print(f"HTML saved to {report_path}")

    def _tokenize_text_with_alignment(self, text):
        # Tokenize the text and create raw and normalized tokens at the same time
        raw_tokens = word_tokenize(text)
        normalized_tokens = [self._normalize_token(token) for token in raw_tokens]
        return raw_tokens, normalized_tokens

    def _normalize_token(self, token):
        # Lowercase, remove punctuation, and remove extra whitespaces
        return re.sub(' +', ' ', token.lower().translate(str.maketrans('', '', string.punctuation)).strip())

    def _generate_html_for_summary(self, raw_tokens, norm_tokens, highlight_words):
        html = ""
        for raw_word, norm_word in zip(raw_tokens, norm_tokens):
            if norm_word in highlight_words:
                html += '<span style="background-color: #9cd59d6b">' + raw_word + '</span> '
            else:
                html += raw_word + ' '
        return html

    def _generate_html_for_report(self, report_config):
        text, text_rank_summary, bert_summary, reference_summary, text_rank_rouge_scores, bert_rouge_scores, index \
            = [report_config[k] for k in
               ('text',
                'text_rank_summary',
                'bert_summary',
                'reference_summary',
                'text_rank_rouge_scores',
                'bert_rouge_scores',
                'index')]

        # Tokenize the texts with alignment
        original_raw_tokens, original_norm_tokens = self._tokenize_text_with_alignment(text)
        text_rank_summary_raw_tokens, text_rank_summary_norm_tokens = self._tokenize_text_with_alignment(
            text_rank_summary)
        reference_raw_tokens, reference_norm_tokens = self._tokenize_text_with_alignment(reference_summary)

        print(text_rank_summary_raw_tokens)

        # Use difflib to find common words between summary and original
        original_matcher = difflib.SequenceMatcher(None, original_norm_tokens, text_rank_summary_norm_tokens)
        original_common_words = set([original_raw_tokens[match.a] for match in original_matcher.get_matching_blocks() if
                                     match.size > 0])

        # Use difflib to find common words between summary and reference
        reference_matcher = difflib.SequenceMatcher(None, reference_norm_tokens, text_rank_summary_norm_tokens)
        reference_common_words = set(
            [reference_raw_tokens[match.a] for match in reference_matcher.get_matching_blocks() if
             match.size > 0])

        # Combine common words from original and reference
        common_words = original_common_words.union(reference_common_words)

        # Create HTML for original, summary, and reference summary
        original_html = self._generate_html_for_summary(original_raw_tokens, original_norm_tokens,
                                                        text_rank_summary_norm_tokens)
        reference_html = ' '.join(reference_raw_tokens)

        # ROUGE scores table
        text_rank_rouge_table = self._build_rouge_table(text_rank_rouge_scores, bert_rouge_scores)

        return f""" 
            <div style="margin:20px 0; border-top: 1px dashed lightgrey;">
                <h1 style="color: #2d3748;">Report #{index}</h1>
                <h2 style="color: #2d3748;">Original</h2>
                <p style="color: #4a5568;">{original_html}</p>
                <h2 style="color: #2d3748; margin-top: 20px;">TextRank Summary</h2>
                <p style="color: #4a5568;">{text_rank_summary}</p>
                <h2 style="color: #2d3748; margin-top: 20px;">BERT Summary</h2>
                <p style="color: #4a5568;">{bert_summary}</p>
                <h2 style="color: #2d3748; margin-top: 20px;">Reference Summary</h2>
                <p style="color: #4a5568;">{reference_html}</p>
                <h2 style="color: #2d3748; margin-top: 20px;">ROUGE Scores</h2>
                {text_rank_rouge_table}
            </div>
        """

    def _read_reference_summary(self, file_path):
        with open(file_path, 'r') as file:
            data = file.read().replace('\n', '')
        return data

    def _get_rouge_scores_table(self, report_configs):
        def get_avg(scores):
            return round(sum(scores) / len(scores), 2)

        def get_avg_rouge_scores(rouge_scores):
            return {
                'rouge-1': {
                    'r': get_avg([score['rouge-1']['r'] for score in rouge_scores]),
                    'p': get_avg([score['rouge-1']['p'] for score in rouge_scores]),
                    'f': get_avg([score['rouge-1']['f'] for score in rouge_scores]),
                },
                'rouge-2': {
                    'r': get_avg([score['rouge-2']['r'] for score in rouge_scores]),
                    'p': get_avg([score['rouge-2']['p'] for score in rouge_scores]),
                    'f': get_avg([score['rouge-2']['f'] for score in rouge_scores]),
                },
                'rouge-l': {
                    'r': get_avg([score['rouge-l']['r'] for score in rouge_scores]),
                    'p': get_avg([score['rouge-l']['p'] for score in rouge_scores]),
                    'f': get_avg([score['rouge-l']['f'] for score in rouge_scores]),
                }
            }

        avg_bert_rouge_scores = get_avg_rouge_scores([i['bert_rouge_scores'] for i in report_configs])
        avg_text_rank_rouge_scores = get_avg_rouge_scores([i['text_rank_rouge_scores'] for i in report_configs])

        # ROUGE scores table
        html = """
            <div style='width: 100%; display: flex; flex-direction: column; align-items: center;'>
                <div style="font-family: Arial, sans-serif; margin: 0 auto; padding: 20px; max-width: 800px;">
                    <h1 style="color: #2d3748;">Average ROUGE Scores</h1>"""

        html += self._build_rouge_table(avg_text_rank_rouge_scores, avg_bert_rouge_scores)
        return html

    def _build_rouge_table(self, text_rank_config, bert_config):
        table = """<table style="width:100%; border-collapse: collapse; margin: 20px 0; margin-bottom: 40px;">
            <tr style="background-color: #edf2f7;">
                <th style="padding: 10px 0;">Score Type</th>
                <th style="padding: 10px 0;">TextRank Score</th> 
                <th style="padding: 10px 0;">BERT Score</th> 
                <th style="padding: 10px 0;">Comments</th>
            </tr>"""

        # Add scores
        for score_type in text_rank_config.keys():
            for k in text_rank_config[score_type].keys():
                text_rank_score = text_rank_config[score_type][k]
                bert_score = bert_config[score_type][k]
                table += f"""
                            <tr style="border-top: 1px solid #e2e8f0;">
                                <td style='text-align:center; padding: 10px 0;'>{score_type}-{k}</td>
                                <td style='text-align:center; padding: 10px 0;'>{text_rank_score:.2f}</td>
                                <td style='text-align:center; padding: 10px 0;'>{bert_score:.2f}</td>
                                <td style='text-align:center; padding: 10px 0; font-size: 12px; width: 50%'>
                                {rouge_scores_comments.get(f'{score_type}-{k}', '')}</td>
                            </tr>"""

        table += "</table>"
        return table
