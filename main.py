# --------------------------------------------------------------------------------
# Copyright (c) 2023, Rishabh Dev
# All rights reserved.
#
# This main.py file is part of a Data Mining project for the university course
# at Laurentian University.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# --------------------------------------------------------------------------------

import os
import re
import nltk
import random
import numpy as np
import pandas as pd
import seaborn as sns
nltk.download('wordnet')
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.decomposition import PCA
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


def difference_in_length(answer1, answer2):
    return abs(len(answer1) - len(answer2))

def fraction_of_words_with_matching_base_forms(answer1, answer2):
    lemmatizer = WordNetLemmatizer()
    words1 = [lemmatizer.lemmatize(w) for w in re.findall(r'\w+', answer1.lower())]
    words2 = [lemmatizer.lemmatize(w) for w in re.findall(r'\w+', answer2.lower())]

    matching_words = sum((Counter(words1) & Counter(words2)).values())
    average_length = (len(words1) + len(words2)) / 2
    return matching_words / average_length

def max_idf_of_matching_base_form(answer1, answer2):
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform([answer1, answer2])
    feature_names = tfidf.get_feature_names_out()
    max_idf = max(tfidf.idf_)
    return max_idf

def tf_idf_vector_similarity(answer1, answer2):
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform([answer1, answer2])
    return cosine_similarity(matrix[0:1], matrix[1:])[0][0]

def tf_idf_vector_similarity_of_letters(answer1, answer2):
    tfidf = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}')
    matrix = tfidf.fit_transform([answer1, answer2])
    return cosine_similarity(matrix[0:1], matrix[1:])[0][0]

def lowercase_string_match(answer1, answer2):
    return answer1.lower() == answer2.lower()

def read_excel(file_path):
    return pd.read_excel(file_path, engine='openpyxl', header=0)

def preprocess_data(df):
    # Preprocessing steps such as lowercasing, removing special characters, etc. can be added here
    for column in df.columns:
        if df[column].dtype == 'object':  # Apply preprocessing only on string columns
            df[column] = df[column].apply(lambda x: x.lower() if isinstance(x, str) else x)  # Lowercase
            df[column] = df[column].apply(lambda x: re.sub(r'[^\w\s]', '', x) if isinstance(x, str) else x)  # Remove special characters
            df[column] = df[column].apply(lambda x: re.sub(r'\s+', ' ', x).strip() if isinstance(x, str) else x)  # Trim extra whitespace
    return df

def preprocess_answer_key(df):
    # Preprocessing steps specific to the answer key DataFrame
    df = preprocess_data(df)
    return preprocess_data(df)

def preprocess_student_answers_grades(df):
    # Preprocessing steps specific to the student answers and grades DataFrame
    df = preprocess_data(df)

    # Create a new column 'answer_processed' with preprocessed answers
    df['answer_processed'] = df['answer'].apply(lambda x: x.lower() if isinstance(x, str) else x)  # Lowercase
    df['answer_processed'] = df['answer_processed'].apply(lambda x: re.sub(r'[^\w\s]', '', x) if isinstance(x, str) else x)  # Remove special characters
    df['answer_processed'] = df['answer_processed'].apply(lambda x: re.sub(r'\s+', ' ', x).strip() if isinstance(x, str) else x)  # Trim extra whitespace

    return preprocess_data(df)

def preprocess_answer_groupings(df):
    # Preprocessing steps specific to the answer groupings DataFrame
    df = preprocess_data(df)

    # Preprocessing for the 'grouplabel' column
    df['grouplabel'] = df['grouplabel'].apply(lambda x: x.lower() if isinstance(x, str) else x)  # Lowercase
    df['grouplabel'] = df['grouplabel'].apply(lambda x: re.sub(r'[^\w\s]', '', x) if isinstance(x, str) else x)  # Remove special characters
    df['grouplabel'] = df['grouplabel'].apply(lambda x: re.sub(r'\s+', ' ', x).strip() if isinstance(x, str) else x)  # Trim extra whitespace

    return preprocess_data(df)

def get_answer_groupings(df):
    answer_groupings = defaultdict(list)
    for index, row in df.iterrows():
        group_label = row['grouplabel']
        answers = [answer for answer in row[1:] if not pd.isnull(answer)]
        answer_groupings[group_label] = answers
    return answer_groupings

def get_answer_key_mapping(answer_key_df, answer_groupings):
    mapping = {}
    for index, row in answer_key_df.iterrows():
        question_number = row['Q#']
        correct_answers = [row[col] for col in row.index[2:] if not pd.isnull(row[col])]
        for group_label, answers in answer_groupings.items():
            for answer in answers:
                if answer in correct_answers:
                    mapping[(question_number, answer)] = group_label
    return mapping

def calculate_accuracy(grades_df, answer_key_mapping):
    total = len(grades_df)
    correct = 0
    for index, row in grades_df.iterrows():
        question_number = row['Q#']
        answer = row['answer']
        correct_answers = [row['G1'], row['G2'], row['G3']]
        if answer_key_mapping.get((question_number, answer)) is not None:
            correct += sum([1 for grade in correct_answers if grade == 1])
        else:
            correct += sum([1 for grade in correct_answers if grade == -1])
    accuracy = correct / (total * 3)  # 3 grades per answer
    return accuracy

def calculate_cosine_similarity(student_answer, correct_answers):
    tfidf_vectorizer = TfidfVectorizer()
    all_answers = [student_answer] + correct_answers
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_answers)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return cosine_similarities

def compare_student_answers(question_numbers, student_answers, answer_key_df, answer_groupings, threshold=0.5):
    # Check if the lengths of question_numbers and student_answers match
    if len(question_numbers) != len(student_answers):
        raise ValueError("The lengths of question_numbers and student_answers must match.")

    # Display the header for the table-like output
    print(f'')
    print(f"{'Q#':<5}{'Student Answer':<40}{'Cosine Similarity':<40}{'Correctness'}")
    print("-" * 90)

    # Initialize a list to store the cosine similarity values for the heatmap
    heatmap_data = []

    max_correct_answers = 0

    # Iterate through the question numbers and student answers
    for question_number, student_answer in zip(question_numbers, student_answers):
        # Extract correct answers for the given question number
        correct_answers = []
        for _, row in answer_key_df[answer_key_df['Q#'] == question_number].iterrows():
            correct_answers.extend([row[col] for col in row.index[2:] if not pd.isnull(row[col])])

        # Calculate cosine similarity
        cosine_similarities = calculate_cosine_similarity(student_answer, correct_answers)

        # Determine correctness
        max_cosine_similarity = max(cosine_similarities.flatten())
        correctness = "Correct" if max_cosine_similarity >= threshold else "Incorrect"

        # Format cosine similarities
        formatted_cosine_similarities = ', '.join(['{:.2f}'.format(val) for val in cosine_similarities.flatten()])

        # Display the results in a table-like format
        print(f"{question_number:<5}{student_answer[:37]:<40}{formatted_cosine_similarities:<40}{correctness}")

        # Append the cosine similarity values to the heatmap_data list
        heatmap_data.append(cosine_similarities.flatten())

        # Update the maximum number of correct answers
        max_correct_answers = max(max_correct_answers, len(correct_answers))

    # Pad the heatmap_data with np.nan values
    padded_heatmap_data = [np.pad(row, (0, max_correct_answers - len(row)), constant_values=np.nan) for row in heatmap_data]

    # Plot the heatmap
    # Adjust figure height to show full y-axis text and rotate y-axis labels
    plt.subplots_adjust(left=0.2, bottom=0.1, top=0.9)
    sns.set(font_scale=0.8) # Adjust font size if necessary
    sns.heatmap(padded_heatmap_data, annot=True, xticklabels=range(1, max_correct_answers + 1), yticklabels=student_answers, cmap='Purples')
    plt.xlabel('Correct Answer Index')
    plt.ylabel('Student Answers')
    plt.title('Cosine Similarity between Student Answers and Correct Answers')
    plt.yticks(rotation=0)
    plt.show()

def student_answers_to_tfidf(student_answers_df):
    student_answers = student_answers_df['answer'].tolist()
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(student_answers)
    return tfidf_matrix

def plot_clusters(reduced_data, cluster_labels, answer_groupings, title):
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        label_indices = np.where(cluster_labels == label)
        plt.scatter(reduced_data[label_indices, 0], reduced_data[label_indices, 1], s=50, label=f'Group {label}')

    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def assign_cluster_labels(answers_grades_df, answer_groupings):
    cluster_labels = []
    for index, row in answers_grades_df.iterrows():
        question_number = row['Q#']
        answer = row['answer']
        for group_label, answers in answer_groupings.items():
            if answer in answers:
                cluster_labels.append(group_label)
                break
    return np.array(cluster_labels)

def plot_correctness_distribution(question_numbers, student_answers, answer_key_df, threshold=0.5):
    # Initialize a dictionary to store the correctness count for each question
    correctness_count = {qn: {"correct": 0, "incorrect": 0} for qn in question_numbers}

    # Iterate through the question numbers and student answers
    for question_number, student_answer in zip(question_numbers, student_answers):
        # Extract correct answers for the given question number
        correct_answers = []
        for _, row in answer_key_df[answer_key_df['Q#'] == question_number].iterrows():
            correct_answers.extend([row[col] for col in row.index[2:] if not pd.isnull(row[col])])

        # Calculate cosine similarity
        cosine_similarities = calculate_cosine_similarity(student_answer, correct_answers)

        # Determine correctness
        max_cosine_similarity = max(cosine_similarities.flatten())
        is_correct = max_cosine_similarity >= threshold

        # Update the correctness count for the current question number
        if is_correct:
            correctness_count[question_number]["correct"] += 1
        else:
            correctness_count[question_number]["incorrect"] += 1

    # Prepare data for the bar plot
    labels = []
    correct_counts = []
    incorrect_counts = []
    for qn, counts in correctness_count.items():
        labels.append(qn)
        correct_counts.append(counts["correct"])
        incorrect_counts.append(counts["incorrect"])

    # Create a bar plot with correct and incorrect counts for each question
    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, correct_counts, width, label='Correct')
    rects2 = ax.bar(x + width / 2, incorrect_counts, width, label='Incorrect')

    ax.set_xlabel('Question Number')
    ax.set_ylabel('Count')
    ax.set_title('Correctness Distribution for Each Question')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # plt.show()

def display_grader_table():
    table_data = [
        (1, 651, 652, 651, 0.992),
        (2, 609, 617, 613, 0.946),
        (3, 587, 587, 492, 0.574),
        (4, 567, 574, 541, 0.864),
        (5, 655, 668, 658, 0.831),
        (6, 568, 582, 548, 0.838),
        (7, 645, 649, 652, 0.854),
        (8, 416, 425, 409, 0.966),
        (13, 613, 535, 557, 0.659),
        (20, 643, 674, 678, 0.449),
    ]

    print(f"{'Q#':<5}{'Grader 1':<12}{'Grader 2':<12}{'Grader 3':<12}{'Kappa'}")
    print("-" * 50)

    for row in table_data:
        print(f"{row[0]:<5}{row[1]:<12}{row[2]:<12}{row[3]:<12}{row[4]:.3f}")

def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def get_true_labels_and_scores(question_numbers, student_answers, answer_key_df, threshold=0.5):
    y_true = []
    y_score = []

    for question_number, student_answer in zip(question_numbers, student_answers):
        correct_answers = []
        for _, row in answer_key_df[answer_key_df['Q#'] == question_number].iterrows():
            correct_answers.extend([row[col] for col in row.index[2:] if not pd.isnull(row[col])])

        cosine_similarities = calculate_cosine_similarity(student_answer, correct_answers)
        max_cosine_similarity = max(cosine_similarities.flatten())

        y_true.append(1 if max_cosine_similarity >= threshold else 0)
        y_score.append(max_cosine_similarity)

    return y_true, y_score

def lda_clustering(tfidf_matrix, n_clusters=10):
    lda = LDA(n_components=n_clusters, random_state=42)
    lda_matrix = lda.fit_transform(tfidf_matrix)
    return lda_matrix

def plot_lda_clusters(lda_matrix, cluster_labels, answer_groupings, title):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(lda_matrix)

    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        label_indices = np.where(cluster_labels == label)
        plt.scatter(reduced_data[label_indices, 0], reduced_data[label_indices, 1], s=50, label=f'Group {label}')

    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def train_logistic_regression(preprocessed_train_answers_grades_df, preprocessed_answer_key_df):
    # Convert student answers to a TF-IDF matrix
    vectorizer = TfidfVectorizer()
    student_answers_tfidf = vectorizer.fit_transform(preprocessed_train_answers_grades_df['answer'])

    # Map student answers to their corresponding correct answer labels
    answer_key_dict = preprocessed_answer_key_df.set_index('Q#')['Answers'].to_dict()
    preprocessed_train_answers_grades_df['correct_answer'] = preprocessed_train_answers_grades_df['Q#'].map(answer_key_dict)

    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(student_answers_tfidf, preprocessed_train_answers_grades_df['correct_answer'])

    # Evaluate the model on the training dataset
    train_predictions = model.predict(student_answers_tfidf)
    train_accuracy = accuracy_score(preprocessed_train_answers_grades_df['correct_answer'], train_predictions)

    return model, train_accuracy

def train_mdt(preprocessed_train_answers_grades_df, preprocessed_answer_key_df):
    # Convert student answers to a TF-IDF matrix
    vectorizer = TfidfVectorizer()
    student_answers_tfidf = vectorizer.fit_transform(preprocessed_train_answers_grades_df['answer'])

    # Map student answers to their corresponding correct answer labels
    answer_key_dict = preprocessed_answer_key_df.set_index('Q#')['Answers'].to_dict()
    preprocessed_train_answers_grades_df['correct_answer'] = preprocessed_train_answers_grades_df['Q#'].map(answer_key_dict)

    # Train the Mixture of Decision Trees model
    model = RandomForestClassifier(n_estimators=10)
    model.fit(student_answers_tfidf, preprocessed_train_answers_grades_df['correct_answer'])

    # Evaluate the model on the training dataset
    train_predictions = model.predict(student_answers_tfidf)
    train_accuracy = accuracy_score(preprocessed_train_answers_grades_df['correct_answer'], train_predictions)

    return model, train_accuracy

def train_lsa_and_classifier(preprocessed_train_answers_grades_df, preprocessed_answer_key_df, n_components=100):
    # Have some bugs with THIS
    lsa, student_answers_lsa = train_lsa(preprocessed_train_answers_grades_df, n_components)

    # Get the correct labels for each student answer
    y_true = get_true_labels(preprocessed_train_answers_grades_df, preprocessed_answer_key_df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(student_answers_lsa, y_true, test_size=0.2, random_state=42)

    # Train the logistic regression classifier
    lr_classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr_classifier.fit(X_train, y_train)

    # Calculate the training accuracy
    train_accuracy = lr_classifier.score(X_train, y_train)

    return lr_classifier, train_accuracy

class Powergrading:
    def __init__(self, answer_key_df, answer_groupings):
        self.answer_key_df = answer_key_df
        self.answer_groupings = answer_groupings
        self.answer_key_mapping = get_answer_key_mapping(answer_key_df, answer_groupings)

    def evaluate(self, answers_grades_df):
        return calculate_accuracy(answers_grades_df, self.answer_key_mapping)

    def similarity(self, answer1, answer2):
        return calculate_cosine_similarity(answer1, [answer2])[0][0]

    def k_medoids_clustering(self, student_answers, n_clusters=10, n_subclusters=5):
        # Dummy implementation, replace with actual k-medoids clustering
        clusters = [student_answers[i::n_clusters] for i in range(n_clusters)]
        subclusters = [[cluster[i::n_subclusters] for i in range(n_subclusters)] for cluster in clusters]
        return clusters, subclusters

    def automatic_labels(self, clusters, subclusters, answer_key):
        # Dummy implementation, replace with actual automatic labeling
        return {i: f'Group {i}' for i in range(len(clusters))}

    def train(self, student_answers):
        clusters, subclusters = self.k_medoids_clustering(student_answers)
        auto_labels = self.automatic_labels(clusters, subclusters, self.answer_key_df)
        return auto_labels

def main():
    data_dir = './dataset'
    train_answers_grades_file = 'studentanswers_grades_100.xlsx'
    test_answers_grades_file = 'studentanswers_grades_698.xlsx'
    answer_key_file = 'questions_answer_key.xlsx'
    answer_groupings_file = 'answer_groupings.xlsx'

    # Read the answer key, student answers and grades, and answer groupings
    answer_key_df = read_excel(os.path.join(data_dir, answer_key_file))
    train_answers_grades_df = read_excel(os.path.join(data_dir, train_answers_grades_file))
    test_answers_grades_df = read_excel(os.path.join(data_dir, test_answers_grades_file))
    answer_groupings_df = read_excel(os.path.join(data_dir, answer_groupings_file))

    # Fix column names in the answer_groupings DataFrame
    answer_groupings_df.columns = ['Q#', 'grouplabel'] + [f'answer_{i}' for i in range(1, len(answer_groupings_df.columns) - 1)]

    # Pre-process DataFrames
    preprocessed_answer_key_df = preprocess_answer_key(answer_key_df)
    preprocessed_train_answers_grades_df = preprocess_student_answers_grades(train_answers_grades_df)
    preprocessed_test_answers_grades_df = preprocess_student_answers_grades(test_answers_grades_df)
    preprocessed_answer_groupings_df = preprocess_answer_groupings(answer_groupings_df)

    question_number = random.choice(list(preprocessed_test_answers_grades_df['Q#'].unique()))
    filtered_df = preprocessed_test_answers_grades_df[preprocessed_test_answers_grades_df['Q#'] == question_number]
    answer1, answer2 = filtered_df.sample(2)['answer'].values

    print(f'')
    print(f"Question number: {question_number}")
    print(f"Answer 1: {answer1}")
    print(f"Answer 2: {answer2}")
    print(f'')
    print("Difference in length:", difference_in_length(answer1, answer2))
    print("Fraction of words with matching base forms:", fraction_of_words_with_matching_base_forms(answer1, answer2))
    print("Max idf of matching base form:", max_idf_of_matching_base_form(answer1, answer2))
    print("tf-idf vector similarity:", tf_idf_vector_similarity(answer1, answer2))
    print("tf-idf vector similarity of letters:", tf_idf_vector_similarity_of_letters(answer1, answer2))
    print("Lowercase string match:", lowercase_string_match(answer1, answer2))
    print(f'')

    # Get answer groupings
    answer_groupings = get_answer_groupings(preprocessed_answer_groupings_df)

    # Convert student answers to a TF-IDF matrix
    student_answers_tfidf = student_answers_to_tfidf(preprocessed_train_answers_grades_df)

    # Perform LDA clustering
    lda_matrix = lda_clustering(student_answers_tfidf, n_clusters=10)

    # Assign cluster labels based on answer groupings
    cluster_labels = assign_cluster_labels(preprocessed_train_answers_grades_df, answer_groupings)

    # Plot the LDA clusters for answer groupings
    plot_lda_clusters(lda_matrix, cluster_labels, answer_groupings, 'LDA Clusters for Answer Groupings')

    # Assign cluster labels based on answer groupings
    cluster_labels = assign_cluster_labels(preprocessed_train_answers_grades_df, answer_groupings)

    # Reduce the dimensionality of the data using PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(student_answers_tfidf.toarray())

    # Plot the clusters for answer groupings
    plot_clusters(reduced_data, cluster_labels, answer_groupings, 'Clusters for Answer Groupings')

    # Implement the powergrading model
    powergrading_model = Powergrading(preprocessed_answer_key_df, answer_groupings)

    # Train the model
    student_answers = preprocessed_train_answers_grades_df['answer'].tolist()
    auto_labels = powergrading_model.train(student_answers)

    # Evaluate the model
    train_accuracy = powergrading_model.evaluate(preprocessed_train_answers_grades_df)
    test_accuracy = powergrading_model.evaluate(preprocessed_test_answers_grades_df)

    model, log_accuracy = train_logistic_regression(preprocessed_train_answers_grades_df, preprocessed_answer_key_df)
    mdt_model, mdt_train_accuracy = train_mdt(preprocessed_train_answers_grades_df, preprocessed_answer_key_df)

    # Assuming preprocessed_train_answers_grades_df contains the preprocessed studentanswers_grades_100 DataFrame
    example_question_numbers = preprocessed_train_answers_grades_df.iloc[:19]['Q#'].tolist()
    example_student_answers = preprocessed_train_answers_grades_df.iloc[:19]['answer'].tolist()

    compare_student_answers(example_question_numbers, example_student_answers, preprocessed_answer_key_df, answer_groupings)
    plot_correctness_distribution(example_question_numbers, example_student_answers, preprocessed_answer_key_df)

    print(f'')
    print(f'')
    display_grader_table()
    print(f'')
    print(f'Powergrading Model Accuracy: {train_accuracy:.2%}')
    print(f'')
    print(f'Logistic Regression Model Accuracy: {log_accuracy:.2%}')
    print(f'')
    print(f'Mixture of Decision Trees Model Train Accuracy: {mdt_train_accuracy:.2%}')
    print(f'')

    y_true, y_score = get_true_labels_and_scores(example_question_numbers, example_student_answers, preprocessed_answer_key_df)
    # plot_roc_curve(y_true, y_score)

if __name__ == '__main__':
    main()

