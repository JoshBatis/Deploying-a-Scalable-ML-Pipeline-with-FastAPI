# Model Card

## Model Details
A binary classification model for census income predictions by Joshua Batis

## Intended Use
To generate predictons for general decision making regarding households whose income exceeds $50,000.  Grouped by various demographic and employment details.

## Training Data
The data comes from census data gathered by UC Irvine Machine Learning Repository. It includes various features such as workclass, education, marital-status, occupation, relationship, race, sex, and native-country. 19% of the data was used for random testing and the random state was set to 72 for reproducibility.

## Evaluation Data
19% of the data was used for our model testing and it was preprocessed at the same time as our training data.

## Metrics
The precision, recall, and F1 scores were used on our model. These measures were chosen to measure our pipeline's performance.
Precision: 0.7461 | Recall: 0.6257 | F1: 0.6806

## Ethical Considerations
Bias may be found in the data depending on the demographic of the data and other factors that were not considered in the questioning.  This data does not reflect the population as a whole.

## Caveats and Recommendations
This data is not up to date and inflation has increased significantly.  These should be taken into account and the binary variable should be adjusted with current inflation.  The more up to date the data is the better.
