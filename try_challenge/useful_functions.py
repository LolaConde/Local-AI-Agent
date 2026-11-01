# Imports
import pandas as pd

# Access to the dataset of questions and answers
file = './questions/answers.json'
questions_df = pd.read_json(file)

def access_one_dataset_question(index: int) -> str:
    """Accesses a specific question from the dataset based on the provided index (starts in 1) and returns the question. If any attached file is associated with the question, it also includes the file name in the output."""
    index_df = index - 1
    question_str = questions_df.iloc[index_df]['question']
    attached_file = questions_df.iloc[index_df]['file_name']
    if attached_file != "":
        question_str += f"\n (Attached file name: {attached_file})"
    return question_str

