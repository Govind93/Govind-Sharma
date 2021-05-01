## Credit Card Approval Assessment
Question 
Data preprocessing is a critical component in machine learning and its importance cannot be overstated. If you do not prepare your data correctly, you can use the fanciest machine learning algorithm in the world and your results will still be incorrect.

For this question, you will perform any and all data preprocessing steps on a dataset on the UCI ML Datasets Repository so that the clean dataset you end up with can be directly fed into any classification algorithm within the Scikit-Learn Python module without any further changes.

This dataset is the Credit Approval data at the following address:

https://archive.ics.uci.edu/ml/datasets/Credit+Approval

The UCI Repository provides four datasets, but only two of them will be relevant:

crx.names: Some basic info on the dataset together with the feature names & values
crx.data: The actual data in comma-separated format
Instructions:

If you are having issues with reading in the dataset directly (which is most likely due to UCI's or your web browser's SSL settings), you can download the file on your computer manually and then upload it to your Azure project, which you can then read in as a local file.
This is a very small dataset. So please do not perform any sampling.
Make sure you follow the best practices outlined in the Data Prep lecture presentation (on Chapters 2 and 3) on Canvas and the Data Prep tutorial on our website.
As a general rule, all categorical features need to be assumed to be nominal unless you have evidence to the contrary.
As for potential outliers in numerical descriptive features, this is an anonymised dataset, so please do not flag any numerical values as outliers regardless of their value for this question.
For this question, you are to set all unusual values (and all outliers, if there are any) to missing values. Also, you are to impute any missing values with the mode for categorical features and with the median for numerical features. If there are multiple modes for a categorical feature, use the mode that comes first alphabetically.
For the A2 numerical descriptive feature, you are to discretize it via equal-frequency binning with 3 bins named "low", "medium", and "high", and then use integer encoding for it.
For normalization, you are to use standard scaling. You are allowed to use Scikit-Learn's preprocessing submodule for this purpose.
The target feature needs be the last column in the clean data and its name needs to be target.
You must perform all your preprocessing steps using Python. For any cleaning steps that you perform via Excel or simple find-and-replace in a text editor or any other language or in any other way, you will receive zero points.
It's critical that the final clean data does not need any further processing so that it will work without any issues with any classifier within Scikit-Learn.
Once you are done, name your final clean dataset as df_clean (if it's not already named as such).
At the end, run each one of the following three lines in three separate code cells for a summary:

df_clean.shape

df_clean.describe(include='all').round(3) 

df_clean.head(5)

Save your final clean dataset exactly as "df_clean.csv". Make sure your file has the correct column names (including the target column). Next, you will upload this CSV file on to Canvas as part of your assignment solutions. That is, in addition to an HTML file (that contains your solutions), you also need to upload your clean data in CSV format on Canvas with this name.
