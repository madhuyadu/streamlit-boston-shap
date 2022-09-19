import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Streamlit App for prediction of Median price of Housing @ Boston

This app predicts the **Boston House Price **!
""")
st.write('---')

# Loads the Boston House Price Dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

st.write("FEATURES")
st.info(X.head(5))

st.write("TARGET")
st.info(Y.head(5))

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')


def user_input_features():
    CRIM = st.sidebar.slider('CRIM', float(X.CRIM.min()), float(X.CRIM.max()), float(X.CRIM.mean()))
    ZN = st.sidebar.slider('ZN', float(X.ZN.min()), float(X.ZN.max()), float(X.ZN.mean()))
    INDUS = st.sidebar.slider('INDUS', float(X.INDUS.min()), float(X.INDUS.max()), float(X.INDUS.mean()))
    CHAS = st.sidebar.slider('CHAS', float(X.CHAS.min()), float(X.CHAS.max()), float(X.CHAS.mean()))
    NOX = st.sidebar.slider('NOX', float(X.NOX.min()), float(X.NOX.max()), float(X.NOX.mean()))
    RM = st.sidebar.slider('RM', float(X.RM.min()), float(X.RM.max()), float(X.RM.mean()))
    AGE = st.sidebar.slider('AGE', float(X.AGE.min()), float(X.AGE.max()), float(X.AGE.mean()))
    DIS = st.sidebar.slider('DIS', float(X.DIS.min()), float(X.DIS.max()), float(X.DIS.mean()))
    RAD = st.sidebar.slider('RAD', float(X.RAD.min()), float(X.RAD.max()), float(X.RAD.mean()))
    TAX = st.sidebar.slider('TAX', float(X.TAX.min()), float(X.TAX.max()), float(X.TAX.mean()))
    PTRATIO = st.sidebar.slider('PTRATIO', float(X.PTRATIO.min()), float(X.PTRATIO.max()), float(X.PTRATIO.mean()))
    B = st.sidebar.slider('B', float(X.B.min()), float(X.B.max()), float(X.B.mean()))
    LSTAT = st.sidebar.slider('LSTAT', float(X.LSTAT.min()), float(X.LSTAT.max()), float(X.LSTAT.mean()))
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)  # Model is trained on Boston housing dataset full data
# Apply Model to Make Prediction
prediction = model.predict(df) # Trained model is used to make prediction on user input

st.header('Prediction of MEDV --> Median value of houses @ Boston')
st.write(prediction)
st.write('---')


# Function to plot SHAP plots in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Explaining the model's predictions using SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
st.write('Feature importance based on SHAP values')

# ---NOT WORKING ----#
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight',)
# plt.model()

st_shap(shap.force_plot(explainer.expected_value, shap_values, X), 400)
st.write('---')

st.header('Feature Importance')
st.write("First prediction's explanation")
# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:]))
st.write('---')

# ---NOT WORKING ----#
#plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')
# plt.model()
# st_shap(shap.force_plot(explainer.expected_value, shap_values, X, plot_type= ''), 400)
# st.write('---')
