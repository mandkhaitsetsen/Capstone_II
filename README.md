# Capstone_II

The purpose of this exercise is to examine and predict the most important factors prospective MBA applicants should consider when applying for top business schools in the United States. Every year thousands of people take the Graduate Management Admissions Test (GMAT) in hopes of getting spots in extremely competitive and prestigious business schools around the world. In 2024, nearly 80,000 unique applicants took the GMAT test. The most reputable MBA Curriculum ranking is published by the U.S. News, which weighs average class GMAT scores in its ranking calculations. Which puts pressure on admission directors to over-index the scores in admission decisions. It is a very common knowledge for aspiring MBA candidates to beat 700 scores in the GMAT to secure a spot in the most prestigious schools. There is also a known secret in the industry that men have to score far above 700 in the GMAT while women would be in the sweet spot if she is near the 700 mark. 

However, admissions officers in these institutions encourage candidates to apply because they say the GMAT score represents only a part of their "holistic view" of each candidate. This project examines how "holistic" admissions officers' decisions actually are. Admissions officers safeguard their applicants' information and the data is hard to find in the public domain, especially from top business schools. Synthetic data generated from the Wharton Class of 2025's statistics was published on Kaggle.com. 
https://www.kaggle.com/datasets/taweilo/mba-admission-dataset

Wharton, Harvard, and Stanford are considered to be the top three business schools in the world. This synthetic set closely resembles the statistics of the real student profile of Wharton Business School's Class of 2025 are publicly available in the following link below.
https://admitstreet.com/blog/wharton-mba/

Metrics that match the synthetic data vs. Wharton's published data:
•	Number of applicants: 6,194
•	Women: 50%
•	International Students: 31%
•	% Composure to previous work industry
There are a couple of discrepancies between the synthetic data set vs. the real Wharton School published data:
•	Admitted students' average GMAT was 728, while it was 693 for the synthetic dataset. 
•	Admitted students' average undergrad GPA was published as 3.6, while it was 3.4 for the synthetic dataset.
•	Slightly lower percentage of STEM students (33% vs 30% of total admitted students) in the synthetic dataset. 

I believe that the synthetic data is the real data of all the applicants at Wharton. However, the admissions decisions columns were scrambled to safeguard the real data. 

Part 1: Exploring the Data

My project first examines correlations of applicants' features with a seaborn pair plot. Not surprisingly, those who were admitted had higher average GMAT scores than the overall applicant pool. 

![Screen Shot 2024-12-27 at 6 35 31 PM](https://github.com/user-attachments/assets/d62295fa-d1de-4fa5-a6a9-0cfee5406798)

 

Gender also showed an imbalance since 50% of the admitted students were female when they only made up 36% of the total applicant pool. Furthermore, the minimum GMAT score for the male applicant who were admitted were 660, while it was 570 for female admitted applicants. 

Race also seemed to be factor into admission decisions. Blacks (8.7%) and Hispanics (10.4%) were admitted at a lower rate than the average of other racial groups (16.2%). 
![Screen Shot 2024-12-27 at 6 50 56 PM](https://github.com/user-attachments/assets/de383f90-4a76-4093-81b5-1b11616aa601)
![Screen Shot 2024-12-27 at 6 52 11 PM](https://github.com/user-attachments/assets/4557f120-c71c-4469-919a-f6e9ac52da26)


 
 
Part II:  Finding most Relevant Feature:

To see which feature correlated with the admissions the most, I created a (Ridge) model. The model has a regularization factor called alpha where weaker features go to zero as alpha gets smaller. According to this model, the most important factors were GMAT, GPA, and Gender. Surprisingly, the fourth most relevant factor was Application_ID. It is well known that the chances of applicants submitting an MBA application are slightly higher in the first round than in the second or third round. However, I believe that the synthetic dataset may have scrambled the order of application ID. The Race was the fifth in relevancy. However, applicants’ previous work industry, undergraduate major, and years of work experience had little to no relevance.

 ![Screen Shot 2024-12-27 at 10 13 16 PM](https://github.com/user-attachments/assets/17e74249-20a8-4866-9eee-b25569cfbd95)



Part III: Building Machine Learning Models

I tested four predictive models, Decision Tree, Logistic Regression, K-N Neighbors, and Support Vector Classifier. I initially ran them as simple models without any arguments or factors. Then tried to improve their performances by adding GridSearchCV to each. All of the models showed tremendous improvement with GridSearchCV. The ROC curve for both simple and improved models is shown below.

  ![Screen Shot 2024-12-27 at 10 21 51 PM](https://github.com/user-attachments/assets/3d644a65-30e2-4097-ab1a-5e9b3cc92429)
![Screen Shot 2024-12-27 at 10 22 44 PM](https://github.com/user-attachments/assets/183535ea-6b87-42f6-8f92-65e660046b1e)


I also created a deep-learning model to see if the prediction accuracy would improve. The deep learning models I used were Random Forest, AdaBoost w/ Logistic Regression, AdaBoost w/ Decision Tree, Gradient Boost, and Neural Networks. Of the 13 total machine learning models I created, I filtered the top 4 models based on their high accuracy and recall score. The top 4 models are shown in the table below. 

                   Accuracy Score    Recall Score  Fit Time
Model                                                                       
Simple D. Tree             0.819         0.450      0.018816
AdaBoost w/ D. Tree        0.843         0.361      85.756600 
GradientBoost              0.870         0.239      0.439409
Neural Network             0.859         0.194      49.191589

I created a confusion matrix to further evaluate which of the four models would be best suited to predict an MBA applicant’s success rate.

 ![Screen Shot 2024-12-27 at 10 41 56 PM](https://github.com/user-attachments/assets/d100d21a-5cee-44a8-93ab-1102a5300faa)


 
Part IV: Findings & Conclusion

I decided that the Simple Decision Model was the best given the needs of my business as an MBA Admissions Consultant. Furthermore, the Simple Decision Tree uses the least computing resources compared to the other three. The model is very good at predicting the total number of applicants that will be rejected or admitted, however, it seems to be confused on who exactly will be admitted. The most accurate model Gradient Boost is very good at telling who exactly is rejected but greatly underestimates applicants who were admitted. Therefore, the Decision Tree was the best usable model in the scenario.

The model predicts that 10 of the 100 waitlisted applicants will eventually be admitted. Although I am 82% certain with that prediction, I don’t have much confidence over exactly which 10 candidates will be admitted. 

It was not a surprise to find GMAT, GPA, and Gender to be the most relevant factors to determine the success of MBA applicants since it was widely known in the arena. However, I was surprised to learn the best machine-learning model had only 87% accuracy. Three reasons why the model accuracies were low may include:

1.	Synthetic Dataset: This may resemble the real data, the random generator that was used to scramble the data may have confused the models.
2.	 Missing features: There may have been other features such as age or employer company name may have played a critical role in determining applicants’ success which were not shared in this dataset.
3.	Personal Essays or Interviews: Admissions officers say they take a holistic view of candidate fit through their personal essays or feedback from alumni interview sessions. Those soft factors may have played a role in confusing the machine learning models.
