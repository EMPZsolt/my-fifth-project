import streamlit as st
import matplotlib.pyplot as plt

version = "v1"
def page_project_hypothesis_body():
    st.write("### Project Hypothesis 1 and Validation")

    st.success(
       f"We theorized that cherry leaves affected by powdery mildew would display "
       f"distinguishable marks from healthy leaves."
    )

    st.info(
        f"Typically, the initial symptom is a light-green, circular lesion on either "
        f"leaf surface, followed by a subtle white cotton-like growth in the infected area."
    )

    st.warning(
        f"The model successfully learned to discern these features and generalize them for "
        f"accurate predictions. By avoiding overfitting to the training data, the model can "
        f"reliably predict future observations based on the learned patterns rather than "
        f"memorizing specific relationships."
        )

    st.write("### Project Hypotesis 2 and validation")

    st.success(
       f"The model test results show that softmax outperforms sigmoid as the "
       f"activation function of the CNN output layer."
    )

    st.info(
        f"Assessing the effectiveness of an activation function in a model "
        f"involves examining its predictive capability through plotting. "
        f"The learning curve, which tracks accuracy and error rates across "
        f"training and validation datasets during model training, serves as "
        f"a key indicator."
    )

    # Read images
    model_perf_softmax_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
    model_perf_softmax_losses = plt.imread(f"outputs/{version}/model_training_losses.png")
    model_perf_sigmoid_acc = plt.imread(f"outputs/{version}/sigmoid_model_training_acc.png")
    model_perf_sigmoid_losses = plt.imread(f"outputs/{version}/sigmoid_model_training_losses.png")

    # Create columns
    col1, col2 = st.beta_columns(2)

    # Display images in columns
    with col1:
        st.image(model_perf_softmax_acc, caption='Softmax accuracy')
        st.image(model_perf_softmax_losses, caption='Softmax losses')

    with col2:
        st.image(model_perf_sigmoid_acc, caption='Sigmoid accuracy')
        st.image(model_perf_sigmoid_losses, caption='Sigmoid losses')
    