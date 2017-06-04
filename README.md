ANN

An Artificial Neural Network has been built to predict religion of countries using different attributes of the country flags like the number of bars, stripes, colors among others.

The data was publicly available on the UCI Machine Learning Repository. It contains details of various nations and their flags. There were some nominal attributes like language and religion, and some binary variables like presence or each color. The numerical attributes included area, population, colors, stripes and circles. Only 16 out of all the available attributes were chosen to predict the religion of a country. Standard Scaler from the scikit-learn package was used to normalize the data since all attributes were of a different kind. 

The data was also split in a 80:20 ratio into training and testing data. Manipulation using pandas dataframes was done to extract the religion from the main data and create a separate dataframe for all the other attributes. 

The MLPClassifier was used to fit a neural network to the data. Two hidden layers of 10 and 5 units respectively were chosen along with the logistic activation function with the initial learning rate kept as 0.3. Different values of these parameters were tried to get a model with the highest accuracy. 

The test data was then fit to the model to obtain a test accuracy. The output of the model consisted of the training and testing accuracy, the confusion matrix showing the actual and predicted value for each religion class and the classification report depicting the values of precision, recall and other performance measures. Training Accuracy was found to be about 87.77% and the testing accuracy was about 51.3%. Since the dataset was very small, consisting of only about 194 observations, the test data was only 20% of that. Since the accuracy is always very poor with a small data, the testing accuracy was found to be very low.

Hierarchical Clustering

Hierarchical Clustering has been implemented to group the countries having similar flags together. The scipy package was used to implement hierarchical clustering for this dataset. The 'average linkage' method was used to calculate the distances at every iteration. 
Cophenet correlation coefficient was used to measure the efficiency of the clustering algorithm. Since it was found to be closer to 1(0.83), we could trust that the algorithm is efficient.

A dendogram was plotted using matplotlib library. It effectively showed the formation of clusters with the distance on the y-axis. A dendogram is useful to visualize different clusters by drawing a horizontal line at a specific distance. Since the data consisted of 194 countries, we can see that the dendogram became very cluttered due to too many points. A nice way to overcome this would be by drawing a truncated dendogram for the last 'p' merged clusters. Here, I have shown a truncated dendogram for the last 12(p = 12) merged clusters. It's a neater version and gives a great overview of the clustering algorithm. 

It should be kept in mind that the single linkage, complete linkage, centroid method or the Ward's method could easily have been used instead of average linkage. However, for the flag data, the average linkage method gave the best results. 

The output could be interpreted as following:
1st line says that the 24th and the 33rd points were combined first at a distance of 0 and that only 2 points were combined to form this cluster. 
The second line gives the respective distances for each iteration where the clusters were formed. Hence, the first two points were combined at a distance of 0 and all the 194 countries were combined at a distance of 11.31. 

In this way, I've shown that two different algorithms could be applied to the same dataset to obtain very different kinds of insights. 
