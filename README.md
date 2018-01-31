# Machine Learning with Animal Shelters

This ongoing project was originally a Kaggle competition in which participants were given training data containing information about the characterstics and outcomes of various shelter animals. They were instructed to generate a model that could accurately predict the outcome of animals based on their characterstics. This code represents a first attempt at machine learning is still a work in progress. The training data is provided.

Animal Train.csv: This is a training dataset that contains information about various shelter animals, including whether they are a cat or dog, their breed, color, and name. It also incudes their outcome: reunited with owner, adopted, transferred, died or euthanized.

Animal Shelters.r: This code takes in the Animal Train dataset and splits it 80/20 to provide both training and testing data. It splits the anlaysis by cats and dogs. XGBoost is used to generate models that predict animal outcomes based on relevant traits. The accuracy of the predictions are evaluated using the testing data.

Cat& Dog Outcomes.pdf: This figure was produced in the data mining phase of the project. It shows the percentage of cats and dogs that experienced each outcome.

Names and Outcomes.pdf: This figure was produced in the data mining phase of the project. It shows the percentage of animals with and without names in each outcome. It suggests that animals with names are far more likely to have a happy ending than those that don't.
