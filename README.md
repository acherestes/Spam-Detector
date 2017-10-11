# Spam-Detector
Spam detector using Naive Bayes

# 1. Approach

I implemented a Naive Bayes classifier for spam recognition. One of the strong assumptions of NB is that the features are independent of each other. In other words, the algorithm does not take into account the order of words in the e-mail, and instead just looks for whether the word occurs in the e-mail. However, it does perform well for text recognition tasks even if some of the features are strongly correlated. Moreover, one of the reasons I chose NB for this project was due to the large size of the vocabulary (cca. 100,000 words). Most other algorithms will struggle with sizes this big - decision trees will likely overfit if given 100,000 attributes, while something like KNN will struggle in finding closeness between points in 100,000 dimensions etc. The two other viable options I found are SVMs and N-Grams. I will talk further about N-Grams in section 4.

The data was obtained from the Enron e-mail dataset.

I initially considered implementing the Gaussian NB algorithm through the sklearn library. The advantages to that are that the program would run much faster, as well as the fact that it is significantly easier to implement. You can find a sample implementation of GNB using sklearn in the file test.py. I decided to write the algorithm from scratch in order to understand it in more detail. If I would have to impliment a similar algorithm again though, I would probably use the sklearn version because it is a lot more optimal. Regarding the accuracy metrics, reasonable choices are F1 score, area under the curve and accuracy. Given that the classification task is simply straight forward and the data was evenly distributed (60% spam, 40% ham), I decided to use a simple accuracy metric because I believe it to be a good representation of the algorithm's performance.

# 2. How to set it up

2.1 Clone the GitHub repository
2.2 Download the e-mail data from http://www2.aueb.gr/users/ion/data/enron-spam/. Place it inside a folder named "Data" within the folder with the GitHub repository. After extracting the files enron1 - enron6, you should check to see that each of them has a ham and spam folder within them.
2.3 Run Spam_Classifier.py. For each e-mail in the test set, the program prints out the number of correctly classified e-mails, the total number of e-mails observed so far and the total number of e-mails in the folder.
2.4 If you'd like the program to run on your own set of e-mails, you first have to save them as .txt files. Then create training and test folders and ham/spam folders for both the training and test sets. Change ham_path and spam_path in Spam_Classifier.py to be the paths to the folders corresponding to your training data, and test_spam and test_ham to be the paths to the folders for your test data.
2.5 The file test.py is an example of how Gaussian NB could be implemented using the sklearn library, as well as a few possible score metrics. The file is there as a reference and was not otherwise used for this problem.
  
# 3. Results and Discussion

I initially trained the algorithm on enron1 and tested it on enron6. This correctly classified spam e-mails with 76.9% accuracy (3,464 / 4,500), and correctly classified ham e-mails with 70.5% accuracy (1,058 / 1500). The running time of the program was 113 minutes. I then trained the algorithm on enron1-enron5 (5x more training data) and tested it on enron6. This lead to a similar accuracy in spam classification of 74.8%, but an abrupt decline in ham accuracy down to 46.3%. The running time was 300 minutes.

This result was disappointing, because intutively one expects that if the algorithm is provided with more data it will perform better. However, in hindsight this is somewhat expected. After analysing the data found in enron1-enron5, I found that there are over 180,000 words in the vocabulary. This is unusual, since the English vocabulary for usual conversation is made up of only about 30,000 words. I believe the significant drop in accuracy is due to typos and other inconsistencies in writing. The low accuracy on ham e-mails was a big issue, since it basically meant that the program would send half of ham-emails to the spam folder.

As a solution to this, I chose to disregard rare words (words whose number of occurrences is less than 10 on the training set) in order to account for typos and inconsistencies with punctuation. This simple modification led to a correct classification of spam e-mails of 75.1%, and an accuracy of 78.6% for ham e-mails. This means that around one in five e-mails is incorrectly sent to spam. The running time was of 57 minutes.

# 4. Further Improvements

4.1 I believe there is still a lot of room for improvement in data preprocessing. The data manipulation I implemented improved performance on the test set from 46% to 78%. However, the algorithm still cannot optimally utilize the entire dataset. I experimented with only training on enron1 and testing on enron6 with this new "exclude rare words" feature, and the accuracy on ham e-mails exceeded 93%. This tells me that the algorithm cannot interpret large amounts of data in a meaningful way. I think this is the most important potential improvement to the program.

4.2 The algorithm is currently only able to interpret mails in a .txt format. A lot of e-mails come with a picture as an attachment in which the actual body of the e-mail shows. To this program, this would register as a blank mail. It would be useful to implement a tool that converts the words in an image to .txt files.

4.3 NB famously implies independence between features, which often leads to wrong predictions for other sets of problems, yet works well in practicefor text classification. However, N-Grams takes into account phrases as well as words. For example, finding the words "get", "rich" and "now" in the same e-mail is not necessarily indicative of spam if the e-mail is long enough. However, finding the phrase "Get Rich Now!" does. N-Grams takes this into account and I think it would be a useful tool in boosting the performance of the algorithm.

# Sources:
1. Introduction to Machine Learning: https://www.udacity.com/course/intro-to-machine-learning--ud120, chapter 2, slides 22-43
2. Enron Spam/Ham e-mail dataset: http://www2.aueb.gr/users/ion/data/enron-spam/
3. V. Metsis, I. Androutsopoulos and G. Paliouras, "Spam Filtering with 
Naive Bayes - Which Naive Bayes?": http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.5542&rep=rep1&type=pdf
4. IOANNIS KANARIS, KONSTANTINOS KANARIS, IOANNIS HOUVARDAS, and EFSTATHIOS STAMATATOS, "WORDS VS. CHARACTER N-GRAMS FOR ANTI-SPAM FILTERING" http://www.icsd.aegean.gr/lecturers/stamatatos/papers/ijait-spam.pdf
