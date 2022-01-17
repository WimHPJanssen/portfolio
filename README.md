

## [Churn Prediction](https://github.com/WimHPJanssen/Churn-Prediction/blob/main/Churn%20Prediction.ipynb)

***Description:*** In this project I use predictive analytics to see whether two proposed email strategies are effective regarding customer retainment for Telco (a provider of telecommunication services). In order to obtain the best model for both of the business problems, I perform thresholding by exploiting the precision-recall trade-off. It appears that for the two problems the selection of the algorithm does not seem to matter much. However, the selection of the classification thresholds is very important. <BR>
***Language:*** Python <BR>
***Main libraries:*** Scikit-Learn, Matplotlib, Seaborn, Pandas

**This project includes:**
* Exploratory data analysis
* Data visualizations
* Logistic regression, Random Forest Classifier, Support Vector Machines
* Hyperparameter tuning and grid search
* Thresholding
* Precision-recall plots


## [Classifying Fake News Articles](https://github.com/WimHPJanssen/Classifying-Fake-News-Articles/blob/main/Classifying%20Fake%20News%20Articles.ipynb)

***Description:*** In this project I classify news articles into fake news articles and real news articles, based on the linguistic features of the text. In order to classify these articles, I will use different feature engineering techniques. Traditional feature engineering methods work very well on the dataset. Document embeddings generated from several word embedding models are also tested, but do not perform that well. The best results are obtained using document embeddings obtained from BERT sentence embeddings.<BR>
***Language:*** Python <BR>
***Main libraries:*** Spacy, Gensim, Scikit-Learn, Pandas 

**This project includes:**
* Natural Language Processing
* Logistic regression
* Pre-processing of text
* Bag of Words Model
* TF-IDF model
* Word2Vec and FastText (pre-trained)
* Word2Vec and FastText (self-trained): word embedding size treated as hyperparameter through gridsearch
* BERT sentence embeddings


## [Soccer Player Position Prediction](https://github.com/WimHPJanssen/Soccer-Player-Position-Prediction/blob/main/Soccer%20Player%20Position%20Prediction.ipynb)

***Description:*** In this project I use FIFA 2019 player attribute ratings to predict the best position of players. In order to achieve a better prediction model, I explore the deletion of anomalies before training the model. I use several anomaly detection methods, such as DBSCAN, PCA and its reconstruction errors, Isolation Forest, Minimum Covariance Determinant, and Local Outlier Factor. However, outlier deletion doesn't increase model performance that much. <BR>
***Language:*** Python <BR>
***Main libraries:*** Scikit-Learn, Seaborn, Matplotlib, Pandas

**This project includes:**
* Logistic regression
* Data visualizations
* Anomaly detection


## [Building Kaggle Database and Queries](https://github.com/WimHPJanssen/Building-Kaggle-Database-and-Queries/blob/main/Building%20Kaggle%20Database%20and%20Queries.ipynb)

***Description:*** I build a relational database and some queries on Kaggle metadata (data on users, competitions, datasets, forums etc.). <BR>
***Language:*** SQL, Python <BR>
***Main libraries:*** SQLAlchemy, Pandas

**This project includes:**
* Building a database with SQLAlchemy
* Using SQLAlchemy for select queries
* Using SQL for select queries


## [Political Spending by Scientists](https://github.com/WimHPJanssen/Political-Spending-by-Scientists/blob/main/Political-Spending-by-Scientists.md)

***Description:*** In this project I explore the donations that scientists give to US politicians in the 2016 election cycle, based on a dataset provided by FiveThirtyEight. Spending behavior differs a lot from the other election cycles, as there are more but smaller donations, especially for Democratic committees. The main finding is that scientists increase the frequency of their contributions, but spend those mainly on the same committee, sometimes even on the same date. <BR>
***Language:*** R <BR>
***Main libraries:*** Tidyverse

**This project includes:**
* Descriptive analytics
* Data visualizations

    
## [Traffic Sign Classification](https://github.com/WimHPJanssen/Traffic-Sign-Classification/blob/main/Traffic%20Sign%20Classification.ipynb)
    
***Description:*** In this project I build a convolutional neural network in order to predict the traffic sign category based on an image of a traffic sign. The accuracy of the final model on the test set is 96.4 percent. The model overfits a bit, hence increased regularization could be used to get better results. Part of this computer vision project is also to create saliency maps, so that we can see whether the model focusses on the right part of the image when predicting the traffic sign category. <BR>
***Language:*** Python <BR>
***Main Libraries:*** Keras, TensorFlow, Matplotlib, Numpy, Pandas
    
**This project includes:**
* Computer Vision
* Convolutional neural network
* Learning curves
* Saliency maps
