![Cherry leaves logo](https://res.cloudinary.com/dwnzsvuln/image/upload/v1713994582/Cherry_leaves_pwuzop.png)

# Cherry Leaves Powdery Mildew Detector


## Table of Contents
- [Dataset Content](#dataset-content)
- [Business Requirements](#business-requirements)
- [Hypothesis and validation](#hypothesis-and-validation)
- [Rationale for the model](#the-rationale-for-the-model)
- [Trial and error](#trial-and-error)
- [Implementation of the Business Requirements](#the-rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
- [ML Business case](#ml-business-case)
- [Dashboard design](#dashboard-design-streamlit-app-user-interface)
- [CRISP DM Process](#the-process-of-cross-industry-standard-process-for-data-mining)
- [Bugs](#bugs)
- [Deployment](#deployment)
- [Technologies used](#technologies-used)
- [Credits](#credits)
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


## Hypothesis and how to validate?
* List here your project hypothesis(es) and how you envision validating it (them).


## The rationale to map the business requirements to the Data Visualisations and ML tasks
* List your business requirements and a rationale to map them to the Data Visualisations and ML tasks.


## ML Business Case
* In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.


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


## Unfixed Bugs
* You will need to mention unfixed bugs and why they were unfixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable for consideration, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

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


## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries used in the project and provide an example(s) of how you used these libraries.


## Credits 

* In this section, you need to reference where you got your content, media and from where you got extra help. It is common practice to use code from other repositories and tutorials. However, it is necessary to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/).

### Media

- The photos used on the home and sign-up page are from This Open-Source site.
- The images used for the gallery page were taken from this other open-source site.



## Acknowledgements
I would like to acknowledge the following people who helped me along the way in completing my fifth milestone project:

My wife, who supported me through the project.
My Mentor, Mo Shami, who showed the direction, helped and encouraged me.
Thank you to entire Code Isntitute for making my development possible.
