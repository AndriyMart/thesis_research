from bert_summarizer import BERTSummarizer
from report_generator import ReportGenerator
from text_rank_summarizer import TextRankSummariser
from rouge import Rouge

# Path static fields
file_path = "dataset_old/business/001.txt"
output_base_path = "./results/"
text_base_path = "./dataset/"

def get_text_path(index):
    return f'{text_base_path}{index}.txt'

def get_reference_summary_path(index):
    return f'{text_base_path}{index}_summary.txt'

def generate_bert_summary(text):
    classifier = BERTSummarizer()
    return classifier.summarize(text)

def generate_text_rank_summary(text):
    text_rank_classifier = TextRankSummariser()
    return text_rank_classifier.summarize(text)


def save_summary(prefix, summary, index):
    with open(f'{output_base_path}{prefix}_{index}.txt', 'w') as file:
        file.write(summary)


def generate_report(report_configs):
    report_generator = ReportGenerator()
    report_generator.generate_report(report_configs)


def read_text_from_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read().replace('\n', '')
    return data


def calculate_rouge_scores(summary, reference):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference, avg=True)
    return scores


report_configs = []

for index in range(5):
    # Generate path to text and reference summary based on index of report
    text_path = get_text_path(index + 1)
    reference_summary_path = get_reference_summary_path(index + 1)

    # Read the text from the file
    text = read_text_from_file(text_path)

    # Read the reference summary from the file
    reference_summary = read_text_from_file(reference_summary_path)

    # Generate the summaries
    text_rank_summary = generate_text_rank_summary(text)
    bert_summary = generate_bert_summary(text)

    # Save the summary to the file
    save_summary("text_rank", text_rank_summary, index + 1)
    save_summary("bert", bert_summary, index + 1)

    # Generate ROUGE scores
    text_rank_rouge_scores = calculate_rouge_scores(text_rank_summary, reference_summary)
    bert_rouge_scores = calculate_rouge_scores(bert_summary, reference_summary)

    report_configs.append({
        'text': text,
        'text_rank_summary': text_rank_summary,
        'bert_summary': bert_summary,
        'reference_summary': reference_summary,
        'text_rank_rouge_scores': text_rank_rouge_scores,
        'bert_rouge_scores': bert_rouge_scores,
        'index': index + 1
    })

# Generate the report
generate_report(report_configs)


