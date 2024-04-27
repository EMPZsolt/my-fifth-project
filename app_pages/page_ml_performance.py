import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    version = 'v1'

    st.warning(
        f"The ML performance page offers an easily understandable overview of "
        f"the dataset's division, the model's performance on the data, and a "
        f"concise explanation of each outcome."
    )
    st.write("### Images distribution")

    st.warning(
        f"The training set, comprising 70% of the entire dataset, serves as the "
        f"foundational data for training the model. During this phase, the model "
        f"learns to generalize and make predictions on new, unseen data.\n\n"
        f"The validation set, consisting of 10% of the dataset, aids in refining "
        f"the model's performance by fine-tuning it after each epoch, which represents "
        f"one complete pass of the training set through the model.\n\n"
        f"The test set, making up 20% of the dataset, provides insight into the ultimate "
        f" accuracy of the model after the training phase. It comprises data that the "
        f"model has not been exposed to previously."
    )

    labels_distribution = plt.imread(f"outputs/{version}/labels_distribution.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')

    labels_distribution = plt.imread(f"outputs/{version}/sets_distribution_pie.png")
    st.image(labels_distribution, caption='Sets distribution')

    st.write("---")

    st.write("### Model Performance")

    st.warning(
        f"**Classification Report**\n\n"
        f"Accuracy: The proportion of correctly classified instances among "
        f"all instances. It is calculated as the sum of true positives and "
        f"true negatives divided by the total number of instances.\n\n"
        f"Macro Average: The average of the precision, recall, and F1 "
        f"score calculated for each class separately. It treats all classes "
        f"equally regardless of their size.\n\n"
        f"Weighted Average: Similar to the macro average, but it takes "
        f"into account the number of instances in each class. It gives more "
        f"weight to the classes with more instances, thus providing a more "
        f"balanced evaluation.\n\n"
        f"Precision: Percentage of correct predictions. The ratio of true "
        f"positives to the sum of a true positive and false positive.\n\n "
        f"Recall: Percentage of positive cases detected. The ratio of true "
        f"positives to the sum of true positives and false negatives.\n\n"
        f"F1 Score: Percentage of correct positive predictions. Weighted "
        f"harmonic mean of precision and recall such that the best score "
        f"is 1.0 and the worst is 0.0.\n\n"
    )

    model_clf = plt.imread(f"outputs/{version}/classification_report.png")
    st.image(model_clf, caption='Classification Report')

    st.warning(
        f"**Confusion Matrix**\n\n"
        f"The Confusion Matrix serves as a critical performance evaluation "
        f"tool for classifiers. It provides a comprehensive breakdown of the "
        f"predictions made by the model in comparison to the actual values. "
        f"It is a table with 4 different combinations of predicted and actual values.\n\n"
        f"True Positive / True Negative: These represent instances where the prediction "
        f"aligns with the actual observation. In the context of leaf analysis, a true "
        f"positive would indicate accurately identifying an infected leaf as infected, "
        f"while a true negative would signify correctly identifying a healthy leaf as healthy.\n\n"
        f"False Positive / False Negative: These occur when the prediction contradicts "
        f"the actual observation. In leaf analysis, a false positive would indicate "
        f"misclassifying a healthy leaf as infected, whereas a false negative would "
        f"mean misclassifying an infected leaf as healthy.\n\n"
        f"A reliable model demonstrates a high rate of true positives and true negatives, "
        f"indicating accurate predictions. Conversely, it should strive to minimize false "
        f"positives and false negatives, as these indicate instances of misclassification, "
        f"which can have significant implications in practical applications.\n\n"
        )

    model_cm = plt.imread(f"outputs/{version}/confusion_matrix.png")
    st.image(model_cm, caption='Confusion Matrix')

    st.warning(
        f"**Model History - Accuracy and Losses of LSTM Model**\n\n"
        f"The Loss metric represents the cumulative errors incurred for each example in the "
        f"training (loss) or validation (val_loss) sets.\n\n"
        f"A lower loss value indicates better performance, reflecting how effectively the model "
        f"optimizes its predictions during each iteration.\n\n"
        f"Accuracy measures the agreement between the model's predictions (accuracy) and the true "
        f"labels (val_acc).\n\n"
        f"A high accuracy suggests that the model's predictions align well with the actual data, "
        f"demonstrating its ability to generalize beyond the training set."
    )

    col1, col2 = st.beta_columns(2)
    with col1: 
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')
    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.warning(
        f"The obtained accuracy exceeds the expected value (97%)."
    )
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))