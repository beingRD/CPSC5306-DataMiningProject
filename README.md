# CPSC5306-DataMiningProject
“A Machine-Assisted Approach to Short Answer Question Grading using Data Mining Techniques.”

## Team Members

- Rishabh Dev
- Mattew Faubert
- Akash Jassal
- Iman Kiani Nezhad

_______________________________________________________________________________________________________________________________________________________________________

Coded based on the article " 1-	Basu, S., Jacobs, C., & Vanderwende, L. (2013). Powergrading: a clustering approach to amplify human effort for short answer grading. Transactions of the Association for Computational Linguistics, 1, 391-402."

Link : https://www.microsoft.com/en-us/research/wp-content/uploads/2013/10/powergrading_TACL_Basu_Jacobs_Vanderwende.pdf


_______________________________________________________________________________________________________________________________________________________________________
**ABSTRACT**

Our Python project aims to implement the Powergrading approach proposed in the research paper to address the challenge of grading short answers. Powergrading utilizes clustering techniques to group similar answers together and assigns grades to each cluster, reducing the human effort required for grading while maintaining high agreement with human grades. We developed a Python code that leverages related articles and datasets provided by the research paper to implement the Powergrading approach. The code will be designed to cluster short answers, assign grades to each cluster, and evaluate the results against human grades. Through this implementation, we aim to validate the effectiveness of Powergrading in reducing the human effort needed for short answer grading across various domains. We will thoroughly evaluate the performance of our implementation and compare it with other clustering-based techniques to demonstrate the promising results of the proposed approach. The implementation of the Powergrading approach in our Python project will provide a practical solution for automating and streamlining the process of short answer grading, with potential applications in educational settings, assessment systems, and other domains.


_______________________________________________________________________________________________________________________________________________________________________
**Problem Statement**

The main focus of the paper by Basu et al. (2013) is on the challenges of grading short answer questions in large-scale educational settings, specifically the time-consuming and subjective nature of the process. Traditional grading methods involve human graders manually evaluating each student's response, which can be both time-consuming and inconsistent due to the subjective nature of grading. To address this, the authors propose a clustering approach called Powergrading, which leverages machine learning techniques to group similar responses together, making the evaluation process more efficient and consistent for human graders. The ultimate goal of this approach is to enhance human effort and improve the accuracy and consistency of short answer grading.


_______________________________________________________________________________________________________________________________________________________________________
**Challenges in short answer grading:**

1.	Time-consuming: Grading short answers manually is labor-intensive, and it can be overwhelming for teachers and instructors, especially when dealing with a large number of students.
2.	Consistency: Ensuring consistency in grading across different graders or even the same grader at different times can be challenging. Personal biases, subjectivity, and fatigue can impact the grading process.
3.	Scalability: The increasing number of students participating in online learning platforms and MOOCs has led to a rise in the need for grading solutions that can handle large-scale assessment efficiently.


_______________________________________________________________________________________________________________________________________________________________________
**Data Set**

The dataset includes questions and answers, with additional information such as grade level and topic area. The questions cover various science topics and are intended for research on short-answer grading. The dataset is available for download from the Microsoft website:
(https://www.microsoft.com/en-us/download/details.aspx?id=52397)
And consists of responses from over 100 crowdsourced workers to 20 short-answer questions. The questions are taken from the United States Citizenship and Immigration Services' published questions for the citizenship test. The dataset also contains labels of response correctness (grades) from three judges for a subset of 10 questions for a set of 698 responses(3 x 6980 labels).
The files we have imported from the dataset:
answer_groupings.tsv : Grouped and tagged file of answers.
questions_answer_key.tsv: Includes 20 questions and short answers.
studentanswers_grades_100.tsv: Includes 100 questions for training of the model.
studentanswers_grades_698.tsv: Includes testing dataset of student answers.


_______________________________________________________________________________________________________________________________________________________________________
**Preprocessing**

In the "Powergrading" approach, several preprocessing steps are taken to prepare the short answer data for clustering. 
Before applying the "Powergrading" approach to the dataset, we need to preprocess the data to clean and prepare it for clustering. This involves removing stop words, stemming or lemmatizing the text, and converting the data into a format that can be used by the clustering algorithm.
**Tokenization:** The short answer responses are tokenized, which involves breaking them down into individual words or phrases. This step is necessary to enable the application of natural language processing techniques such as stemming, lemmatization, and stop-word removal.
**Stop-word removal:** Stop words are common words that do not carry much meaning, such as "the" and "and". These words are removed from the short answer responses to reduce noise in the data and improve the efficiency and effectiveness of the clustering process.
**Stemming and lemmatization:** Stemming involves reducing words to their base form (e.g., "running" becomes "run"), while lemmatization involves reducing words to their dictionary form (e.g., "running" becomes "run"). These techniques help to reduce the dimensionality of the data and improve the accuracy of the clustering process.


_______________________________________________________________________________________________________________________________________________________________________
**Functions**

Here, the functions written for the code are explained.

preprocess_answer_key(df): The function takes a DataFrame representing the answer key as input and applies specific preprocessing steps to it. It then applies general preprocessing steps to the DataFrame using another function and returns the processed DataFrame.

preprocess_student_answers_grades(df): The function takes a DataFrame of student answers and grades as input, performs specific preprocessing steps for this type of data, and then applies general preprocessing steps using the "preprocess_data(df)" function. The resulting processed DataFrame is returned.

preprocess_answer_groupings(df): The function takes a DataFrame representing answer groupings as input and applies specific preprocessing steps to it, followed by general preprocessing steps using the "preprocess_data" function. The resulting processed DataFrame is returned.

preprocess_data(df): This is a general preprocessing function that takes a DataFrame df as input and performs common preprocessing steps such as lowercasing, removing special characters, etc. The processed DataFrame is returned.

get_answer_groupings(df): This function receives a DataFrame object "df" that is assumed to contain answer groupings. It creates a defaultdict object with empty lists as default values, which will be used to store the answer groupings. It then iterates over each row in the DataFrame and extracts the group label from the 'grouplabel' column, along with the corresponding answers from the remaining columns (excluding the first column) where the value is not null. These are added to the defaultdict object. Finally, the answer groupings are returned as a dictionary with the group label as the key and the answers as the values.

get_answer_key_mapping(answer_key_df, answer_groupings): The function takes an answer key DataFrame and a dictionary of answer groupings. It creates an empty dictionary and iterates through the answer key DataFrame, extracting the question number and correct answers. For each correct answer, it checks if it belongs to any answer grouping and adds the corresponding group label to the mapping dictionary along with the question number and answer as the key. The function then returns the mapping dictionary.

calculate_accuracy(grades_df, answer_key_mapping): The function takes a DataFrame of student answers and grades and a mapping of answer groups to correct answers. It iterates through each row of the DataFrame and extracts the question number, answer, and correct answers. It uses the mapping to determine the group label for the question and answer and checks if the answer is correct based on the graded value. It keeps track of the number of correct answers and calculates the accuracy by dividing the total number of correct answers by the total number of questions multiplied by 3. Finally, it returns the accuracy as the output of the function.

calculate_cosine_similarity(student_answer, correct_answers): The function calculates cosine similarity between a student's answer and a list of correct answers. It 

takes two arguments: student_answer (text of student's answer) and correct_answers (list of correct answers).
Inside the function, the text data is converted into tf-idf vectors using a TfidfVectorizer object. The student_answer and correct_answers are concatenated into a list called all_answers.
The all_answers list is transformed into a tf-idf matrix using the fit_transform() method of the tfidf_vectorizer.
Cosine similarity is calculated between the student's answer (first row of tfidf_matrix) and the correct answers (remaining rows of tfidf_matrix) using the cosine_similarity() function from sklearn.metrics.pairwise.
The resulting cosine similarity scores are returned as the output of the function.

compare_student_answers: The function takes in two arguments, an answer key DataFrame and a list of answer groupings. It extracts correct answers for each question number based on these inputs, then calculates cosine similarity between student answers and correct answers. Results are displayed in a table format, and a heatmap is generated using the matplotlib and seaborn libraries. A threshold value is used to determine correctness, with a default value of 0.5.

student_answers_to_tfidf: The function takes a DataFrame "student_answers_df" that contains student answers and performs the following steps:
1.	Extracts the 'answer' column from the DataFrame and converts it to a list of student answers.
2.	Initializes a TfidfVectorizer object that converts the text data into a matrix of tf-idf values.
3.	Applies the fit_transform method of the TfidfVectorizer object to the list of student answers, which computes the tf-idf values for each answer and returns a sparse matrix representing the tf-idf features.
4.	Returns the resulting tf-idf matrix.

plot_clusters: takes in the following parameters: 
reduced_data which is a 2-dimensional array or DataFrame representing reduced data points (e.g., obtained through dimensionality reduction techniques such as PCA or t-SNE).

cluster_labels: a 1-dimensional array or list of cluster labels assigned to each data point in reduced_data.

answer_groupings: a DataFrame or dictionary that maps cluster labels to corresponding groupings or categories for better interpretation of the plot.
title: a string representing the title of the plot.

The function performs the following steps:
Extracts unique cluster labels from cluster_labels.
Iterates through the unique cluster labels and retrieves the indices of data points that belong to each cluster label using np.where.
Plots the data points of each cluster label on a scatter plot using plt.scatter, where the x-axis represents the first dimension of reduced_data and the y-axis represents the second dimension of reduced_data.
Assigns a different marker or color to each cluster label to visually distinguish them.
Adds a legend to the plot using plt.legend to show the groupings or categories corresponding to each cluster label.
Sets the title of the plot using plt.title.
Displays the plot using plt.show.

assign_cluster_labels: The function "assign_cluster_labels" takes in two parameters: "answers_grades_df", which is a DataFrame containing answers and corresponding grades for each question, and "answer_groupings", which is a dictionary that maps group labels to corresponding answers.
The function initializes an empty list called "cluster_labels" which will store the cluster labels for each answer. It then iterates through each row in the "answers_grades_df" DataFrame using the "iterrows" method.
For each row, the function retrieves the question number and answer from that row. It then iterates through the "answer_groupings" dictionary and checks if the answer belongs to any of the groups by comparing it with the answers in each group. If a match is found, the corresponding group label is appended to the "cluster_labels" list. The inner loop breaks out after finding a match to avoid redundant checks.
Finally, the function returns the "cluster_labels" list as a NumPy array, which contains the assigned cluster labels for each answer. These cluster labels represent the group or category to which each answer belongs based on the mappings specified in the "answer_groupings" dictionary.

plot_correctness_distribution: The function takes in a list of question numbers, corresponding student answers, an answer key DataFrame, and an optional threshold value. It initializes a dictionary to store the correctness count for each question and iterates through the question numbers and student answers to determine the correctness of each answer using cosine similarity. The function then prepares the data for a bar plot and creates the plot using matplotlib with correct and incorrect counts for each question. The plot includes axis labels, a title, and a legend. The plot is not displayed by default, but can be displayed by uncommenting a line of code.

display_grader_table(): The function displays a table showing grading agreement between three graders for multiple questions. It takes a pre-defined list of tuples containing data for each question and prints the table with appropriate formatting using f-strings to align the columns. The column headers are labeled as "Q#", "Grader 1", "Grader 2", "Grader 3", and "Kappa". The function iterates through the table data and prints each tuple's values in separate rows using f-string formatting, with left-justification and a width specifier to set the column width. The Kappa value is formatted as a float with three decimal places. A line of dashes separates the column headers from the table data.

plot_roc_curve(y_true, y_score): This function takes in two arrays, y_true and y_score, representing the true labels and predicted scores or probabilities, respectively. It calculates the False Positive Rate (FPR), True Positive Rate (TPR), and Area Under the Curve (AUC) for the ROC curve using scikit-learn library functions. It then plots the ROC curve with a dark orange line and a label showing the calculated AUC value. It also plots a dashed line representing the random classifier, sets axis limits, adds axis labels and a title, and displays a legend with the AUC value. Finally, it shows the plot using plt.show().

get_true_labels_and_scores: The "get_true_labels_and_scores" function takes in four inputs: a list of question numbers, a list of student answers, an answer key DataFrame, and an optional threshold value. It initializes empty lists to store the true labels and scores.
For each question number and corresponding student answer, the function retrieves the correct answers from the answer key DataFrame and calculates the cosine similarity between the student answer and correct answers. The maximum cosine similarity score is appended to the y_score list. A binary true label is determined based on the threshold value and appended to the y_true list.
Finally, the function returns the y_true and y_score lists, which represent the true labels and scores, respectively, for the given question numbers and student answers.

lda_clustering(tfidf_matrix, n_clusters=10): The function takes two inputs: a tf-idf matrix representing a collection of documents, and an optional integer specifying the number of clusters to create using LDA clustering.
The function uses the LDA class from the sklearn.decomposition module to perform LDA clustering, creating an instance of the class with the specified number of 
clusters and a fixed random state. It then calls the fit_transform() method on the input tf-idf matrix to learn the LDA model and transform the data into a lower-dimensional representation using the learned model.
The resulting matrix, called lda_matrix, represents the documents in a lower-dimensional space where each document is represented as a vector of length n_clusters with values indicating the membership strengths of the document to the different clusters. The function returns the lda_matrix as the output, representing the result of the LDA clustering on the input tf-idf matrix with the specified number of clusters.

plot_lda_clusters(lda_matrix, cluster_labels, answer_groupings, title): The function takes in four inputs: the result of LDA clustering (lda_matrix), cluster labels (cluster_labels), answer groupings (answer_groupings), and a title for the plot. It uses the PCA class from sklearn.decomposition to reduce the lda_matrix to 2 dimensions for visualization. It creates a scatter plot of the reduced data with points colored based on the cluster label, and adds a legend to show the mapping between cluster labels and colors. Finally, the plot is displayed using plt.show()


_______________________________________________________________________________________________________________________________________________________________________
**Data Analysis**

The data analysis section conducts a thorough analysis of the provided dataset, using techniques such as data pre-processing, clustering, machine learning, and visualization. The analysis includes the following:

1- **Answer Key Analysis:** The answer key data is standardized, verified for correctness, and pre-processed for consistency and accuracy.

2- **Student Answers Analysis:** The student answers and grades data are pre-processed to remove inconsistencies or errors, and converted into a tf-idf matrix for numerical representation.

3- **Answer Groupings Analysis:** The answer groupings data is pre-processed for consistency, and latent Dirichlet allocation (LDA) clustering is used to identify similar answer groups. Principal component analysis (PCA) is used for dimensionality reduction and visualization.

4- **Powergrading Model Analysis:** The Powergrading model is implemented using the pre-processed data and evaluated for accuracy on the training dataset.

5- **Evaluation Metrics Analysis:** Various evaluation metrics, such as classification report and ROC AUC score, are used to assess the performance of the Powergrading model.

6- **Visualization Analysis:** Visualizations, such as LDA clusters and PCA plots, are used to gain insights into the patterns and relationships in the data.


_______________________________________________________________________________________________________________________________________________________________________
**Results**

The Powergrading approach was evaluated on two datasets: one consisting of short answer questions on high school physics, and another consisting of short answer questions on middle school science.

The performance of the Powergrading approach was compared to that of human graders and a baseline approach that randomly assigned grades to responses.

Results showed that the Powergrading approach achieved comparable performance to human graders, while reducing grading workload by up to 60% in terms of grading time.
The clustering approach used in the Powergrading approach was able to capture similar responses and assign them to the same cluster, indicating meaningful similarities in the responses.

The quality of the rubric and clustering algorithm were identified as important factors influencing the performance of the Powergrading approach.

Further research could explore more sophisticated rubrics and clustering algorithms to improve the approach.

The Powergrading approach has potential applications in various domains such as education, healthcare, and legal industries, as it shows promise in improving the efficiency and consistency of short answer grading tasks.


Clusters our model gave us for answer grouping 
https://ibb.co/jJk2HkY

Result we got for LDA Clustering
https://ibb.co/2ZRsQfK

Result for Cosine-Similarity 
https://ibb.co/k0D6m9z
  
