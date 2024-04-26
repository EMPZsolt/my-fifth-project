![Cherry leaves logo](https://res.cloudinary.com/dwnzsvuln/image/upload/v1713994582/Cherry_leaves_pwuzop.png)

# Cherry Leaves Powdery Mildew Detector


## Table of Contents
- [Dataset Content](#dataset-content)
- [Business Requirements](#business-requirements)
- [Rationale to map the business requirements to the Data Visualizations and ML tasks](#rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
- [Hypothesis and validation](#hypothesis-and-validation)
- [Rationale for the model](#the-rationale-for-the-model)
- [Trial and error](#trial-and-error)
- [Implementation of the Business Requirements](#the-rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
- [ML Business case](#ml-business-case)
- [Conclusion and Potential Course of Actions](#conclusion-and-potential-course-of-actions)
- [Dashboard design](#dashboard-design-streamlit-app-user-interface)
    - [Page 1: Quick Project Summary](#page-1-quick-project-summary)
    - [Page 2: Leaf Visualizer](#page-2-leaf-visualizer)
    - [Page 3: Powdery Mildew Detector](#page-3-powdery-mildew-detector)
    - [Page 4: Project Hypothesis and Validation](#page-4-project-hypothesis-and-validation)
    - [Page 5: ML Performance Metrics](#page-5-ml-performance-metrics)
- [CRISP DM Process](#the-process-of-cross-industry-standard-process-for-data-mining)
- [Bugs](#bugs)
- [Deployment](#deployment)
    - [Heroku](#heroku)
    - [Fork Repository](#fork-repository)
    - [Clone Repository](#clone-repository)
- [Technologies used](#technologies-used)
    - [Languages](#languages)
    - [Main Data Analysis and Machine Learning Libraries](#main-data-analysis-and-machine-learning-libraries)
    - [Platforms](#platforms)
- [Credits](#credits)
    - [Content](#content)
    - [Media](#media)
    - [Code](#code)
- [Acknowledgements](#acknowledgements)


## Dataset Content

The dataset, acquired from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves), encompasses more than 4,000 images originating from the client's crop fields. These images present individual cherry leaves set against a neutral backdrop, showcasing both healthy specimens and those affected by powdery mildew, a fungal disease known to impact numerous plant species.

The cherry plantation crop holds substantial significance within the client's portfolio, particularly as bitter cherries represent their flagship product. Consequently, the company is deeply invested in preserving the quality of their yield. The dataset's emphasis on powdery mildew-infected cherry leaves underscores the urgent necessity for predictive analytics solutions to mitigate crop damage and ensure consistent product quality.

Through a meticulously crafted user story, we delve into the application of predictive analytics in real-world scenarios within the workplace. By harnessing the insights derived from this dataset, stakeholders can make well-informed decisions to safeguard crop health and maintain market competitiveness.


## Business Requirements

The cherry plantation crop at Farmy & Foods faces a pressing challenge: powdery mildew infestation. Currently, the process involves manual verification to determine if a cherry tree is affected. An employee spends roughly 30 minutes per tree, inspecting leaf samples visually for signs of powdery mildew. Upon detection, a specific compound is applied to eliminate the fungus, taking an additional minute. With thousands of cherry trees spread across multiple farms, this manual inspection process is severely hindered by scalability issues.

In response to this challenge, the IT team proposed implementing a machine learning (ML) system capable of instant detection using leaf images. This system aims to swiftly differentiate between healthy and powdery mildew-infected cherry trees, thereby streamlining the inspection process. If successful, this ML solution could be extended to other crops facing similar pest detection challenges, offering scalability and efficiency benefits.

The dataset provided by Farmy & Foods comprises cherry leaf images, forming the basis for training the ML model. The client seeks to achieve the following objectives:
- Conduct a study to visually distinguish between healthy cherry leaves and those infected by powdery mildew.
- Develop a predictive model to determine whether a cherry leaf is healthy or contaminated with powdery mildew.

By addressing these objectives, the client aims to optimize the inspection process, enhance crop management practices, and ultimately improve the quality and yield of their cherry plantation crop.

## Rationale to map the business requirements to the Data Visualizations and ML tasks
**Business Requirement 1**: Conduct an analysis to visually distinguish between healthy cherry leaves and those affected by powdery mildew.

- Examine average and variability images for both healthy and powdery mildew-infected cherry leaves.
- Highlight disparities between average healthy and powdery mildew-infected cherry leaves to facilitate visual differentiation.
- Generate image montages for each category to vividly illustrate the characteristics of healthy and powdery mildew-infected cherry leaves.

**Business Requirement 2**: Develop a predictive model for determining the health status of cherry leaves regarding powdery mildew infection.

- Utilize Neural Networks to establish correlations between cherry leaf features (images) and their corresponding health labels (healthy or powdery mildew-infected).
- When loading images into memory for model training, consider their shape to ensure compatibility with performance criteria.
- Explore various image shape options to strike an optimal balance between model size and performance.

## Hypothesis and validation
* List here your project hypothesis(es) and how you envision validating it (them).


## The rationale to map the business requirements to the Data Visualisations and ML tasks
* List your business requirements and a rationale to map them to the Data Visualisations and ML tasks.


## ML Business Case
* In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.

## Conclusion and Potential Course of Actions

In this project, our focus was on detecting powdery mildew in cherry leaves. We began by formulating hypotheses and validating them through research, analysis, and machine learning model development. Here are the key findings:

**Hypothesis 1** (Do infected leaves have distinguishable marks from healthy ones?):

Our hypothesis suggested that infected cherry leaves would display unique visual characteristics that differentiate them from healthy leaves. To verify this, we analyzed a large dataset of infected and healthy cherry leaf images. Our analysis confirmed the hypothesis, revealing consistent visual cues such as white powdery patches and leaf discoloration that distinguish infected leaves.

**Hypothesis 2** (Does SoftMax outperform sigmoid as the CNN output layer activation function?):

This hypothesis aimed to compare SoftMax and sigmoid activation functions for the CNN output layer. After multiple iterations and evaluations, we found that the model consistently performed better with the Softmax activation function. So this is the best choice for our powdery mildew detection model.

Some potential actions can be considered:

- **Quality Data Improvement**: Enhancing the quality of data, particularly images, could significantly improve the model's predictive capabilities, thereby increasing accuracy. Moreover, increasing the quantity of data could further boost accuracy by enabling deeper learning.

- **Further Research**: Explore more factors or variables contributing to powdery mildew detection. This could involve looking into different image processing techniques, exploring advanced machine learning algorithms, or incorporating additional data sources.

- **Implementation Strategies**: Investigate strategies for implementing validated hypotheses into practical applications.

- **Decision-Making**:Use validated hypotheses and model predictions to guide decision-making in agriculture.

## Dashboard Design

### Page 1: Quick Project Summary

#### Summary Page:
This page offers a rapid overview of the project and its goals. It encompasses general details regarding powdery mildew in cherry trees and the visual indicators utilized for identifying infected leaves. Additionally, it provides a link to the project README file.

#### General Information:
Powdery mildew is a parasitic fungal disease caused by Podosphaera clandestina in cherry trees. As the fungus proliferates, it forms a layer of mildew consisting of numerous spores atop the leaves. This disease poses a significant threat to new growth, hindering plant development and potentially affecting fruit, thereby leading to direct crop losses. Visual cues for detecting infected leaves include light-green circular lesions on leaf surfaces, progressing to the development of a subtle white cotton-like growth on both leaf surfaces and fruits, ultimately reducing yield and quality.

#### Project Dataset:
The dataset provided by Farmy & Foody comprises 4208 featured photos of single cherry leaves set against a neutral background. These leaves are either healthy or afflicted by cherry powdery mildew.

#### Business Requirements:
- The client seeks to conduct a study aimed at visually distinguishing between leaves containing parasites and those that are uninfected.
- The client desires to determine whether a given leaf contains a powdery mildew parasite or not.

### Page 2: Leaf Visualizer

#### Answers business requirements 1

This page facilitates the visualization of cherry leaves, emphasizing the disparities between average healthy leaves and those infected with powdery mildew. It addresses the following:

#### Checkbox 1 - Difference between average and variability image:
Allows users to compare the average appearance of cherry leaves with the variability across multiple images.

#### Checkbox 2 - Differences between average powdery mildew-infected and average healthy leaves:
Enables users to observe the distinctions between the average characteristics of cherry leaves affected by powdery mildew and those that are healthy.

#### Checkbox 3 - Image Montage:
Provides clients with an image montage feature, offering an array of randomly selected images showcasing either healthy or powdery mildew-infected cherry leaves.

### Page 3: Powdery Mildew Detector

#### Answers business requirements 2

This page is dedicated to detecting powdery mildew in cherry leaves. It fulfills the following objectives:
- User Interface with File Uploader Widget:
    - Allows users to upload multiple cherry leaf images for live prediction.
- Powdery Mildew Detection:
    - Processes the uploaded images using a machine learning model to predict whether the leaves are healthy or infected with powdery mildew.
- Prediction Statement and Probability:
    - Displays the prediction statement indicating if the leaf is infected or not with powdery mildew, along with the associated probability.

### Page 4: Project Hypothesis and Validation
Bloack for each project hypothesis, describe the conclusion and how you validated.
### Page 5: ML Performance Metrics
Label Frequencies for Train, Validation and Test Sets
Model History - Accuracy and Losses
Model evaluation result


## CRISP DM Process


## Unfixed Bugs
There is no known unfixed bugs in the project.

## Deployment

### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file. 

### Fork Repository 
To fork the repository, perform the following steps:

1. Go to the [GitHub repository](https://github.com/EMPZsolt/my-fifth-project/tree/main)
2. Click the fork button in the upper right hand corner


### Clone Repository
To clone the repository, perform the following steps:

1. Go to the [GitHub repository](https://github.com/EMPZsolt/my-fifth-project/tree/main)
2. Locate the Code button on the upper right hand corner
3. Select preference of cloning using HTTPS, SSH, or Github CLI, then click the copy button to copy the URL to the clipboard
4. Open Git bash
5. Change the working directory to where you wish to clone the repo to
6. Type `git clone` followed by pasting the URL from your clipboard
7. Hit enter to run the command and create the local clone

## Technologies used

### Languages
- **Python**

### Main Data Analysis and Machine Learning Libraries

- **tensorflow-cpu 2.6.0**: Used for training and creating the machine learning model.
- **numpy 1.19.2**: Utilized for converting data to arrays for efficient manipulation and computation.
- **scikit-learn 0.24.2**: Employed for evaluating the performance of the machine learning model through various metrics and techniques.
- **streamlit 0.85.0**: Utilized for developing the interactive dashboard and user interface for this project.
- **pandas 1.1.2**: Utilized for data manipulation and storage, particularly for creating and saving datasets as dataframes.
- **matplotlib 3.3.1**: Used for visualizing the distribution of datasets and generating plots for data analysis.
- **keras 2.6.0**: Utilized for configuring and setting hyperparameters for the machine learning model.
- **plotly 4.12.0**: Employed for plotting the learning curve of the model's performance over epochs.
- **seaborn 0.11.0**: Utilized for generating visualizations such as confusion matrices to analyze the model's performance.
- **streamlit 0.85.0**: Used for creating and sharing the user interface and dashboard for this project.
- **protobuf 3.20**: Used for serialization and deserialization of structured data, particularly for efficient communication between different components of the project.
- **altair**: Utilized for creating interactive visualizations and statistical graphics, enhancing the data exploration and presentation capabilities of the project.

### Platforms
- **GitHub**: Serves as a repository for storing the project code, allowing for version control and collaboration among team members.
- **Gitpod**: Utilized for project development, with its dashboard providing an interface for writing code and its terminal enabling actions such as committing changes to GitHub and pushing updates to GitHub Pages.
- **Jupyter Notebook**: Utilized for code editing and experimentation, providing an interactive environment for developing project components.
- **Heroku**: Employed for project deployment, facilitating the hosting and accessibility of the application online.
- **Kaggle**: Leveraged for acquiring datasets relevant to the project, offering a platform for seamless data exploration and acquisition.
- **Cloudinary**: Containing and providing the intro image for the Readme.


## Credits 

### Content 
- The text for the Home page was taken from [BC Tree Fruit Production Guide](https://www.bctfpg.ca/pest_guide/info/101/).
- The dataset is from [Kagle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).

### Media
- The header image in the Readme has been sourced from [Adobe Stock](https://stock.adobe.com/search?k=%22cherry+leaf%22&asset_id=497344868).

### Code
- The layout I used for this project is provided by [CodeInstitute](https://github.com/Code-Institute-Solutions/milestone-project-mildew-detection-in-cherry-leaves).
- The Streamlit dashboard pages, and the Jupyter Notebooks are based on Code Institute's Malaria Detector Walkthrough Project, which I used as a starting point for this project.


## Acknowledgements
I would like to acknowledge the following people who helped me along the way in completing my fifth milestone project:

My wife, who supported me through the project.
My Mentor, Mo Shami, who showed the direction, helped and encouraged me.
Thank you to entire Code Isntitute for making my development possible.
