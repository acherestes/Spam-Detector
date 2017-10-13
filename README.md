# Spam-Detector
Spam detector using Naive Bayes

# 1. Approach

I implemented a Naive Bayes classifier for spam/ham recognition (ham = non-spam). One of the strong assumptions of NB is that the features are independent of each other. In other words, the algorithm does not take into account the order of words in the e-mail, and instead just looks for whether the word occurs in the e-mail. However, it does perform well for text recognition tasks even if some of the features are strongly correlated. Moreover, one of the reasons I chose NB for this project was due to the large size of the vocabulary (cca. 100,000 words). Most other algorithms will struggle with sizes this big - decision trees will likely overfit if given 100,000 attributes, while something like KNN will struggle in finding closeness between points in 100,000 dimensions etc. The two other viable options I found are SVMs and n-Grams. I will talk further about n-Grams in section 4.

The algorithm computes:
<a href="https://www.codecogs.com/eqnedit.php?latex=P(spam|mail)&space;=&space;\frac{P(mail|spam)&space;*&space;P(spam)}{P(mail|spam)&space;*&space;P(spam)&space;&plus;&space;P(mail|ham)&space;*&space;P(ham)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(spam|mail)&space;=&space;\frac{P(mail|spam)&space;*&space;P(spam)}{P(mail|spam)&space;*&space;P(spam)&space;&plus;&space;P(mail|ham)&space;*&space;P(ham)}" title="P(spam|mail) = \frac{P(mail|spam) * P(spam)}{P(mail|spam) * P(spam) + P(mail|ham) * P(ham)}" /></a>

Here, P(ham) and P(spam) are the prior probabilities of the e-mail being spam/ham. These can be estimated by either computing the fraction of spam e-mails in the dataset or by using online statistics on the frequency of spam. P(ham) = 1 - P(spam).

P(mail|spam) and P(mail|ham) represent the probability of the occurrence of the words in the e-mail given that the e-mail's true label is ham or spam. E.g. Given that the e-mail body is "Get Rich Now!" and we know that the e-mail is spam, what is the probability for the occurrence for the words "get", "rich" and "now", as well as the probability for all other words in the spam vocabulary to not occur in the e-mail. 

In turn, P(mail|spam) and P(mail|ham) are computed using the following:

<a href="https://www.codecogs.com/eqnedit.php?latex=P(D|C_{i})&space;=&space;\prod_{w_{j}&space;\in&space;D}^{n}&space;P(w_{j}|C_{i})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(D|C_{i})&space;=&space;\prod_{w_{j}&space;\in&space;D}^{n}&space;P(w_{j}|C_{i})" title="P(D|C_{i}) = \prod_{w_{j} \in D}^{n} P(w_{j}|C_{i})" /></a>

Here D stands for the content of the mail w.r.t the vocabulary, whereas C_{i} represents the classification labels (ham/spam). Then, the product is applied over all tokens w_{j} in D given the classification label C_{i}. The issue with using this formula is that the number of multiplications of probabilities increases linearly with the size of the vocabulary D. In other words, given the content of the e-mail, for each word in the vocabulary, the algorithm has to compute the probability of each of the words in the e-mail to occur, as well as the probability of all words that do not occur in the e-mail but are in the classification label vocabulary not to occur.

E.g. Suppose the example is, once again, "Get Rich Now !", and the spam vocabulary contains the following words: v = ['money', 'get', 'rich', 'profit', 'fast']. Notice the word "now" is not present in the vocabulary so therefore it will be disregarded. The way the algorithm interprets this classification task is to compute P(0,1,1,0,0|spam), where a 0 on the 1st position indicates that the first word in the vocabulary does not appear in the e-mail, whereas a 1 indicates that it does.

Multiplying sub-unitary numbers quickly leads to underflow as the size of the vocabulary increases, so instead of using the formula above, I used sum of logs as opposed to multiplication of probabilities, as follows:

<a href="https://www.codecogs.com/eqnedit.php?latex=log(P(D|C_{i}))&space;=&space;\sum_{w_{j}&space;\in&space;D}^{n}&space;log(P(w_{j}|C_{i}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?log(P(D|C_{i}))&space;=&space;\sum_{w_{j}&space;\in&space;D}^{n}&space;log(P(w_{j}|C_{i}))" title="log(P(D|C_{i})) = \sum_{w_{j} \in D}^{n} log(P(w_{j}|C_{i}))" /></a>




The data was obtained from the Enron e-mail dataset.

I initially considered implementing the Gaussian NB algorithm through the sklearn library. The advantages to that are that the program would run much faster, as well as the fact that it is significantly easier to implement. You can find a sample implementation of GNB using sklearn in the file test.py. I decided to write the algorithm from scratch in order to understand it in more detail. If I would have to impliment a similar algorithm again though, I would probably use the sklearn version because it is a lot more optimal. Regarding the accuracy metrics, reasonable choices are F1 score, area under the curve and accuracy. Given that the classification task is reasonably straight forward and the data was evenly distributed (60% spam, 40% ham), I decided to use a simple accuracy metric because I believe it to be a good representation of the algorithm's performance.

# 2. How to set it up

1. Clone the GitHub repository
2. Download the e-mail data from http://www2.aueb.gr/users/ion/data/enron-spam/. Place it inside a folder named "Data" within the folder with the GitHub repository. After extracting the files enron1 - enron6, you should check to see that each of them has a ham and spam folder within them.
3. If you'd like to classify a single e-mail, run GUI.py. The interface takes as input the text in your e-mail and classifies it.
4. If you'd like to classify multiple e-mails, uncomment line 33 in Spam_Classifier.py and then run it. Provided that you downloaded the Enron dataset and saved it at the locations indicated by test_ham and test_ham, the program should run by itself. For each e-mail in the test set, the program prints out the number of correctly classified e-mails, the total number of e-mails observed so far and the total number of e-mails in the folder.
5. If you'd like the program to run on your own set of e-mails, you first have to save them as .txt files. Then create training and test folders and ham/spam folders for both the training and test sets. Both for training and testing purposes, you will have to manually classify the e-mails into ham and spam. Change ham_path and spam_path in Spam_Classifier.py to be the paths to the folders corresponding to your training data, and test_spam and test_ham to be the paths to the folders for your test data. It is important to create different ham/spam folders even for the test data in order to easily be able to quantify the algorithm's accuracy.
6. The file test.py is an example of how Gaussian NB could be implemented using the sklearn library, as well as a few possible score metrics. The file is there as a reference and was not otherwise used for this problem.
  
# 3. Results and Discussion

I initially trained the algorithm on enron1 and tested it on enron6. This correctly classified spam e-mails with 76.9% accuracy (3,464 / 4,500), and correctly classified ham e-mails with 70.5% accuracy (1,058 / 1,500). The running time of the program was 113 minutes. I then trained the algorithm on enron1-enron5 (5x more training data) and tested it on enron6. This lead to a similar accuracy in spam classification of 74.8%, but an abrupt decline in ham accuracy down to 46.3%. The running time was 300 minutes.

This result was disappointing, because intutively one expects that if the algorithm is provided with more data it will perform better. However, in hindsight this is somewhat expected. After analysing the data found in enron1-enron5, I found that there are over 180,000 words in the vocabulary. This is unusual, since the English vocabulary for usual conversation is made up of only about 30,000 words. I believe the significant drop in accuracy is due to typos and other inconsistencies in writing. The low accuracy on ham e-mails was a big issue, since it basically meant that the program would send half of ham-emails to the spam folder.

As a solution to this, I chose to disregard rare words (words whose number of occurrences is less than 10 on the training set) in order to account for typos and inconsistencies with punctuation. This simple modification led to a correct classification of spam e-mails of 75.1%, and an accuracy of 78.6% for ham e-mails. This means that around one in five e-mails is incorrectly sent to spam. The running time was of 57 minutes.

# 4. Further Improvements

4.1 I believe there is still a lot of room for improvement in data preprocessing. The data manipulation I implemented improved performance on the test set from 46% to 78%. However, the algorithm still cannot optimally utilize the entire dataset. I experimented with only training on enron1 and testing on enron6 with this new "exclude rare words" feature, and the accuracy on ham e-mails exceeded 93%. This tells me that the algorithm cannot interpret large amounts of data in a meaningful way. I think this is the most important potential improvement to the program. An approach that would be fairly easy to implement and would deal well with the issue of typos and inconsistencies in writing would be to import a list of the 30,000 words most commonly used in the English language and only compute word frequencies for these words. In other words, instead of checking whether a word appears at least ten times in the training data, if would instead check if the word exists in the list of commonly used words. If it does, it computes probabilities and frequencies for the word, and if it does not it discards it. This would eliminate obscure words and numbers from the training set which would help with both improving the algorithm’s performance and its running time due to the fact that the data is cleaner.

4.2 The algorithm is currently only able to interpret mails in a .txt format. A lot of e-mails come with a picture as an attachment in which the actual body of the e-mail shows. To this program, this would register as a blank mail. It would be useful to implement a tool that converts the words in an image to .txt files.

4.3 NB famously implies independence between features, which often leads to wrong predictions for other sets of problems, yet works well in practice for text classification. However, n-Grams takes into account phrases as well as words. For example, finding the words "get", "rich" and "now" in the same e-mail is not necessarily indicative of spam if the e-mail is long enough. However, finding the phrase "Get Rich Now!" is. n-Grams takes this into account and I think it would be a useful tool in boosting the performance of the algorithm. In a nutshell, the idea behind n-grams is to look for sequences of objects (words, letters etc.) of length n and predict the n+1th term in the sequence. Similarly to Hidden Markov Models, n-grams computes P(x_{n}|x_{n-1}...x_{1}).

# Sources:
1. Introduction to Machine Learning: https://www.udacity.com/course/intro-to-machine-learning--ud120, chapter 2, slides 22-43
2. Enron Spam/Ham e-mail dataset: http://www2.aueb.gr/users/ion/data/enron-spam/
3. V. Metsis, I. Androutsopoulos and G. Paliouras, "Spam Filtering with 
Naive Bayes - Which Naive Bayes?": http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.5542&rep=rep1&type=pdf
4. IOANNIS KANARIS, KONSTANTINOS KANARIS, IOANNIS HOUVARDAS, and EFSTATHIOS STAMATATOS, "WORDS VS. CHARACTER N-GRAMS FOR ANTI-SPAM FILTERING" http://www.icsd.aegean.gr/lecturers/stamatatos/papers/ijait-spam.pdf
