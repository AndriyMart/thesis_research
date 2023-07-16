from report_generator import ReportGenerator
from text_rank_classifier import TextRankClassifier

# Path static fields
file_path = "./dataset/business/001.txt"
reference_summary_path = "results/reference_res.txt"
output_path = "results/res.txt"


def generate_text_rank_summary():
    text_rank_classifier = TextRankClassifier()
    return text_rank_classifier.text_rank(file_path, 2)


def save_summary(summary):
    with open(output_path, 'w') as file:
        file.write(summary)


def generate_report(summary):
    report_generator = ReportGenerator()
    report_generator.generate_report(summary, reference_summary_path)


# Generate the summary
text_rank_summary = generate_text_rank_summary()
print(text_rank_summary)

# Save the summary to the file
save_summary(text_rank_summary)

# Generate the report
generate_report(text_rank_summary)
