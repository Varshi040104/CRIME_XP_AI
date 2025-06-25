import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay
from sklearn.metrics import roc_auc_score,precision_score,recall_score
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
# Use custom CSS to set background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://as2.ftcdn.net/jpg/02/84/01/17/1000_F_284011744_BKgVHlpl8iSq05pVlTeZgRNe2UMHt2dv.jpg");
        background-attachment: fixed;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("""<h1 style='color:black;text-shadow: 
            -1px -1px 0 red,  
             1px -1px 0 red,
            -1px  1px 0 red,
             1px  1px 0 red;'>Exploring Crime Rate Patterns </h1>""", unsafe_allow_html=True)
new_data = pd.read_csv("new_crime_dataset2.csv")
final_data = pd.read_csv("final_crime_data.csv")

# App title
st.markdown("""
<h2 style='
    color: black;
    text-shadow: 
        -1px -1px 0 red,  
         1px -1px 0 red,
        -1px  1px 0 red,
         1px  1px 0 red;
'>Crime Dataset Viewer</h2>
""", unsafe_allow_html=True)






# Create buttons to view datasets

# --- Columns layout: buttons left, data middle ---
left_col, center_col, right_col = st.columns([1, 3, 1])

with left_col:
    show_new = st.button("📄 Show New Crime Dataset")
    show_final = st.button("✅ Show Final Processed Dataset")

with center_col:
    if show_new:
        st.markdown("""<h3 style='color: black;
    text-shadow: 
        -1px -1px 0 red,  
         1px -1px 0 red,
        -1px  1px 0 red,
         1px  1px 0 red;'>📋 New Crime Dataset</h3>""", unsafe_allow_html=True)
        df_new = pd.read_csv("new_crime_dataset2.csv")
        st.dataframe(df_new)

    if show_final:
        st.markdown("""<h3 style='color: black;
    text-shadow: 
        -1px -1px 0 red,  
         1px -1px 0 red,
        -1px  1px 0 red;'>🧾 Final Processed Dataset</h3>""", unsafe_allow_html=True)
        df_final = pd.read_csv("final_crime_data.csv")
        st.dataframe(df_final)



# Load your dataset
final_data = pd.read_csv("final_crime_data.csv")  # Or wherever your file is

# Define custom region colors
region_colors = {
    "North": "red",
    "South": "blue",
    "East": "green",
    "West": "orange",
    "Central": "purple",
    "Unknown": "gray"
}

# Title Section


# Button to Show Map
if st.button("🗺️ Show Crime Map"):
    fig = px.scatter_mapbox(
        final_data,
        lat="lat",
        lon="lon",
        color="Region",
        color_discrete_map=region_colors,
        hover_name="City",
        hover_data=["Region", "Crime Description"],
        zoom=4,
        height=600
    )
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    
    st.plotly_chart(fig, use_container_width=True)

region_columns = ['North_India', 'South_India', 'East_India', 'West_India', 'Central_India']
region_data = final_data[region_columns]

# Helper to label regions
def get_region(row):
    for region in region_columns:
        if row[region] == 1:
            return region.replace('_India', '')
    return 'Unknown'

final_data['Region_Label'] = final_data.apply(get_region, axis=1)

# Selection for type of plot
# Styled header before the selectbox
st.markdown("""<h3 style='color:black;text-shadow: 
            -1px -1px 0 red,  
             1px -1px 0 red,
            -1px  1px 0 red,
             1px  1px 0 red;'>Choose Type of Plots</h3>""",unsafe_allow_html=True)

# Selectbox with empty label (so we use only the styled one above)
plot_type = st.selectbox("", ["Matplotlib Plots", "Plotly Plots"])


# ------------------ Matplotlib Section ------------------ #
if plot_type == "Matplotlib Plots":
    st.markdown("""<h3 style='color:black;text-shadow: 
            -1px -1px 0 red,  
             1px -1px 0 red,
            -1px  1px 0 red,
             1px  1px 0 red;'>📊 Matplotlib - Total Crimes per Region</h3>""",unsafe_allow_html=True)
    region_counts = final_data[region_columns].sum()

    fig, ax = plt.subplots()
    region_counts.plot(kind='bar', color=['red', 'blue', 'green', 'orange', 'purple'], ax=ax)
    plt.title("Total Crime Count per Region")
    plt.xlabel("Region")
    plt.ylabel("Crime Count")
    st.pyplot(fig)

     #-------------------------------------------pieplot par----------
    region_counts = final_data['Region'].value_counts()
    labels = region_counts.index
    sizes = region_counts.values
    plt.title("Total Crime Count per Region")
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
  #-----------------------------------------------Hist Plot-------------------------
    City_counts = final_data['City'].value_counts().head(10)
    labels = City_counts.index
    sizes = City_counts.values
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title("Number of Cities")
    ax.axis('equal')
    st.pyplot(fig)    
#-------------------------------------------------Scatter plot-----------------
    
    # Group by Victim Age and count how many crimes occurred for each age
    age_crime_counts = final_data['Victim Age'].value_counts().sort_index().head(100)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(age_crime_counts.index, age_crime_counts.values, color='blue', alpha=0.7)

    # Labels and Title
    ax.set_xlabel("Victim Age")
    ax.set_ylabel("Crime Count")
    ax.set_title("Top 100 Victim Ages by Crime Count")
    ax.grid(True)
    # Show in Streamlit
    st.pyplot(fig)

# ------------------ Plotly Section ------------------ #
elif plot_type == "Plotly Plots":
    st.markdown("""<h3 style='color:black;text-shadow: 
            -1px -1px 0 red,  
             1px -1px 0 red,
            -1px  1px 0 red,
             1px  1px 0 red;'>📈 Plotly - Crime Domain by Region (Stacked Bar)</h3>""",unsafe_allow_html=True)

    # Group by Crime Domain and Region_Label
    crime_region = final_data.groupby(['Crime Domain', 'Region_Label']).size().reset_index(name='Count')

    fig = px.bar(
        crime_region,
        x="Crime Domain",
        y="Count",
        color="Region_Label",
        title="Crime Domain by Region",
        barmode='stack'
    )
    st.plotly_chart(fig, use_container_width=True)
    #--------------------bar chart -----------------
    trace = go.Histogram(
         x=final_data['Crime Description'],
         nbinsx=15,
         marker={'color': 'white'}
    )

    layout = go.Layout(
        title='Victims Ages during Crime',
        xaxis={'title': 'Crime Evidence'}
    )

    # Combine trace and layout
    figure = go.Figure(data=[trace], layout=layout)

    # Display it in Streamlit
    st.plotly_chart(figure, use_container_width=True)

    #---------------------------------------------------HIst Plot-----------------
   
    st.markdown("""<h3 style='color:black;text-shadow: 
            -1px -1px 0 red,  
             1px -1px 0 red,
            -1px  1px 0 red,
             1px  1px 0 red;'>🔍 Victim Age Distribution (Plotly Histogram)</h3>""", unsafe_allow_html=True)

    filtered_data = final_data[['Victim Age']].dropna().head(100)

    fig = px.histogram(
        filtered_data,
        x='Victim Age',
        nbins=20,
        title='Histogram of Victim Age (Top 100)',
        color_discrete_sequence=['orange']
    )

    # 👉 Add border lines between bins
    fig.update_traces(marker=dict(
        line=dict(width=1.5, color='black')  # Border width & color
    ))

    # Additional layout settings
    fig.update_layout(
        xaxis_title='Victim Age',
        yaxis_title='Count',
        bargap=0.1,
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)


# Encode target column 'Crime Domain'
#-----------------------------------------MOdelTRAINING TESTING PART--------------------------------------------
final_data = final_data.drop(columns=["Weapon Used", "Crime Description", "City", "Region"], errors="ignore")

# Encode target
le = LabelEncoder()
final_data["Crime Domain"] = le.fit_transform(final_data["Crime Domain"])

# Encode remaining non-numeric columns
for col in final_data.select_dtypes(include='object').columns:
    final_data[col] = LabelEncoder().fit_transform(final_data[col])

# Split data
X = final_data.drop(columns=["Crime Domain"])
y = final_data["Crime Domain"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Streamlit UI
st.markdown("""<h3 style='color:black;text-shadow: 
            -1px -1px 0 red,  
             1px -1px 0 red,
            -1px  1px 0 red,
             1px  1px 0 red;'>Crime Domain Model Evaluation</h3>""", unsafe_allow_html=True)

# Metric Selection
metric = st.selectbox("📊 Choose a metric to visualize:", ["Confusion Matrix", "ROC Curve"])

# Confusion Matrix
if metric == "Confusion Matrix":
    st.markdown("""<h3 style='color:black;text-shadow: 
            -1px -1px 0 red,  
             1px -1px 0 red,
            -1px  1px 0 red,
             1px  1px 0 red;'>📌 Classification Report</h3>""",unsafe_allow_html=True)
    report = classification_report(y_test, y_pred)
    st.code(report)

    st.subheader("📉 Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    st.pyplot(fig)

# ROC Curve (multi-class)
elif metric == "ROC Curve":
    st.markdown("""<h3 style='color:black;text-shadow: 
            -1px -1px 0 red,  
             1px -1px 0 red,
            -1px  1px 0 red,
             1px  1px 0 red;'>📈 ROC Curve (Multi-Class)</h3>""",unsafe_allow_html=True)

    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    ovr_model = OneVsRestClassifier(RandomForestClassifier(class_weight='balanced', random_state=42))
    ovr_model.fit(X_train, label_binarize(y_train, classes=np.unique(y)))
    y_score = ovr_model.predict_proba(X_test)

    fig, ax = plt.subplots()
    for i in range(y_test_bin.shape[1]):
        RocCurveDisplay.from_predictions(y_test_bin[:, i], y_score[:, i], ax=ax, name=f"Class {i}")
    st.pyplot(fig)


