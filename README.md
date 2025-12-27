# Eye-Movement-Computer-Vision-Project
The project describes an AI-powered, camera driven screening system utilizes eye movements and saccades to detect early signs of Parkinson's disease. Using computer vision, the system captures eye movements via web interface. The data is collected and processed for disease prediction in a ML model. This could be used as preliminary screening tool.

## About Project:
This study makes use of video-based eye movement data recorded using a regular webcam with better resolution during controlled visual stimulus assessments. The data would be collected as short video recordings of eye movements captured during three neurological tests: saccade, smooth pursuit, and fixation. Computer vision algorithms are used to extract eye-tracking paramaters such as pupil movement speed, smoothness, and spatial jitter from these recordings. These generated numerical features make up the input dataset for the machine learning model utilized in parkinson's screening.

## Model Selection
Model name: Random forest Classifier
Description: The random forest classifier was chosen as the primary machine learning model for this project. The decision was influenced by its capacity to handle non-linear connections, resistance to noise, and applicability for small to medium-sized feature sets. Random forest outperformes linear models in terms of generalisation because eye movement features would vary significantly between individuals and do not follow linear patterns. Furthermore, it provides probability-based predictions, which are important for confidence estimate in medical screening applications.

## Model Training
The model was trained on manually simulated and feature engineered examples of eye movement behavior. Haar cascade based eye detection and motion analysis were used to extract key properties from video frames, including saccadic velocity, pursuit smoothness, and fixation jitter. These features were classified as healthy or Parkinson's signs, and the random forest model was trained with supervised learning. The training technique focuses on pattern recognition rather than raw picture learning, making it computationally efficient and appropriate for real-time screening situations.

## Metrics used for evaluation
1) Classification results: Healthy vs Abnormal value
2) Class probability: If 2 out of 3 tests show abnormal condition, prediction result as abnormal

## Strength and Weakenss of the model
Strength: No contact of any device to the patient. Only required good or high quality web camera or a good camera smartphone. No clinical environment is required. User friendly and reduced complexity for eye screening test.
Weakness: Data is very limited. Performance could impact due to lighting conditions and camera quality. Only intended for preliminary screening test.

## Possible improvements / real-world applications:
Improvements: Utilise clinically verified eye movement datasets to train the model. For increased accuracy, incorporate deep learning based eye tracking. Replace binary output with severity level classification. Use hyperparameter tweaking and cross-validation with professional neurologist. Boost resilience to changes in lighting and occlusion
Real-world applications: Early detection of parkinson's disease post verified by neurology specialist. Remote pre-clinical assessment via telemedicine platforms. Utilized as an assisstive tool for neurologists during routine checkups. Integration into FDA complaint, Indian BIS standards clinical decision support systems.

## Conclusion of the project
The viability of employing eye movement analysis bolstered by machine learning model for parkinson's disease could be represented by this project. Due to environmental conditions as well as low quality camera, the detection of pupil found to be be challenging. Otherwise, the technology makes neurological screening accessible without the need for specialised equipment by fusing computer vision, machine learning, and an intuitive web interface. Although the current model is a proof-of-concept, it could become a dependable helpful tool for early neurological assessment and preventive healthcare with future improvements that include clinically validated data and regulatory alignment.
