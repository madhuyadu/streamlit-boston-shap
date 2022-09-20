# streamlit-boston-shap
Streamlit app to predict median value of houses @ Boston &amp; also plot SHAP values to explain how each feature is contributing to model's predictions


# App features
- App takes user input in the form of sliding selectors on the side bar
- Input values can be fed by dragging the sliders under the section "Specify Input Parameters"
- Selectors are provided for each of the features used for prediction
- Model is trained on Boston housing data set (full set) & then used for prediction on user input fed
- The median value of housing is displayed in the main page under "Prediction of MEDV --> Median value of houses @ Boston"
- Feature importance is plotted via SHAP explainers/SHAP plots 
- 2 plots are displayed, one for overall model output and another one which explains the predictions for the first row of input


# Screenshots of the App
![image](https://user-images.githubusercontent.com/56335301/191017463-abc2ea2e-e44a-417b-86e0-d678f5206666.png)

![image](https://user-images.githubusercontent.com/56335301/191017520-2e17578c-5a76-4f63-89ee-46fc7de335b4.png)

![image](https://user-images.githubusercontent.com/56335301/191017573-b9d48665-9afb-4ef7-9dee-166aa34dfb9b.png)

![image](https://user-images.githubusercontent.com/56335301/191017624-29fe4672-45e0-41af-b890-78584d8b6761.png)

![image](https://user-images.githubusercontent.com/56335301/191017679-7b284564-cd01-4a5d-9ca9-2e340eac58a2.png)

# Credits
This web app is created based on tutorial on youtube channel "dataprofessor"
