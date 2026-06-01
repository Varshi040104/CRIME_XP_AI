📊 Exploring Crime Rate Patterns: A Machine Learning‑Driven Analysis
📌 Overview
This project analyzes crime rate patterns using machine learning and interactive visualizations. It leverages historical crime datasets to identify hotspots, trends, and risk factors, providing insights that can support law enforcement, policymakers, and researchers in crime prevention and resource allocation.

The project is implemented as a Streamlit web application, combining data preprocessing, visualization, and predictive modeling.
👉 Live Demo: Streamlit Cloud Deployment

🚀 Features
Dataset Viewer: Toggle between raw and processed crime datasets.

Interactive Crime Map: Visualize crime distribution across regions using Mapbox.

Matplotlib & Plotly Visualizations:

Bar charts, pie charts, scatter plots, and histograms.

Crime counts per region, city distribution, and victim age analysis.

Machine Learning Models:

Random Forest Classifier for predicting crime domains.

Multi-class ROC Curve and Confusion Matrix evaluation.

Custom Styling:

Background image and styled headers for a polished UI.

Cloud Deployment:

Fully deployed on Streamlit Cloud for easy access and sharing.

🛠️ Tech Stack
Programming Language: Python 3.7+

Libraries:

Data Handling: pandas, numpy

Visualization: matplotlib, plotly, streamlit

Machine Learning: scikit-learn

Framework: Streamlit (for interactive web app)

Deployment: Streamlit Cloud

📂 Project Structure
Code
├── new_crime_dataset2.csv        # Raw dataset
├── final_crime_data.csv          # Processed dataset
├── Proj_on_Crime.py              # Streamlit application source code
├── Report.pdf                    # Project documentation/report
└── README.md                     # Project overview (this file)
⚙️ Setup & Installation
Clone the repository:

bash
git clone <repo-url>
cd crime-rate-analysis
Install dependencies:

bash
pip install -r requirements.txt
(Ensure Python 3.7+ is installed)

Run the Streamlit app locally:

bash
streamlit run Proj_on_Crime.py
Or access the deployed version directly:
👉 Streamlit Cloud App : https://crimexpai-yhtxbhx63qxamr3snwkaav.streamlit.app/

📊 Usage
Launch the app and explore:

View datasets with one click.

Visualize crime maps by region.

Switch between Matplotlib and Plotly plots for deeper insights.

Run ML models to evaluate classification performance.

🔍 Machine Learning Workflow
Data Cleaning: Remove target-leaking features (e.g., Crime Code, Weapon Used).

Encoding: Label encode categorical features.

Model Training: Train Random Forest Classifier with balanced class weights.

Evaluation:

Confusion Matrix

Classification Report

Multi-class ROC Curve

📖 References
Dataset: Kaggle Crime Data

Libraries: Scikit-learn, Streamlit, Plotly

✅ Conclusion
This project demonstrates how data science and machine learning can uncover hidden crime patterns and provide actionable insights. By combining interactive visualizations with predictive modeling, it offers a practical tool for crime analysis and prevention.
With deployment on Streamlit Cloud, the project is accessible to anyone, anywhere, without requiring local setup.
