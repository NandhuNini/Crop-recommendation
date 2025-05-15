import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
import numpy as np
from matplotlib.ticker import MaxNLocator
import firebase_admin
from firebase_admin import credentials, db
import json
import os
import shap

# Initialize Firebase (will only run once)
if not firebase_admin._apps:
    # Load your service account JSON file
    service_account_path = ""  # change to your filename

    if os.path.exists(service_account_path):
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': ''  # replace with your URL
        })
    else:
        print("Service account JSON file not found. Firebase initialization skipped.")

# Function to upload CSV to Firebase
def upload_csv_to_firebase(csv_file):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file.name)

        # Convert DataFrame to dictionary
        data = df.to_dict(orient='records')

        # Get a reference to the database service
        ref = db.reference('soil_data')

        # Upload each record to Firebase
        for idx, record in enumerate(data):
            ref.child(f"record_{idx}").set(record)

        return f"Successfully uploaded {len(data)} records to Firebase!"
    except Exception as e:
        return f"Error uploading to Firebase: {str(e)}"

# Load the dataset (either from CSV or Firebase)
try:
    # Try to load from local CSV first
    df = pd.read_csv("soil_dataset (3).csv")
except FileNotFoundError:
    # If CSV not found, try loading from Firebase
    try:
        print("Local CSV not found. Attempting to load from Firebase...")
        ref = db.reference('soil_data')
        firebase_data = ref.get()
        if firebase_data:
            df = pd.DataFrame.from_dict(firebase_data, orient='index')
            print("Data loaded from Firebase successfully")
        else:
            raise Exception("No data found in Firebase")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        df = pd.DataFrame()  # empty dataframe as fallback

# Data Preprocessing
# Encode categorical columns
label_encoders = {}
for column in ['Soil Type', 'Crop Recommendation', 'Fertilizer Recommendation']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Features and targets
X = df.drop(columns=['Crop Recommendation', 'Fertilizer Recommendation', 'Water Retention Days'])
y_crop = df['Crop Recommendation']
y_fert = df['Fertilizer Recommendation']
y_days = df['Water Retention Days']

# Split data into training and testing sets
X_train, X_test, y_crop_train, y_crop_test = train_test_split(X, y_crop, test_size=0.2, random_state=42)
X_train, X_test, y_fert_train, y_fert_test = train_test_split(X, y_fert, test_size=0.2, random_state=42)
X_train, X_test, y_days_train, y_days_test = train_test_split(X, y_days, test_size=0.2, random_state=42)

# Train models
crop_model = RandomForestClassifier(random_state=42)
fert_model = RandomForestClassifier(random_state=42)
days_model = RandomForestRegressor(random_state=42)

crop_model.fit(X_train, y_crop_train)
fert_model.fit(X_train, y_fert_train)
days_model.fit(X_train, y_days_train)

# Evaluate models
def evaluate_models():
    # Make predictions
    crop_pred = crop_model.predict(X_test)
    fert_pred = fert_model.predict(X_test)
    days_pred = days_model.predict(X_test)

    # Calculate metrics
    metrics = {
        'crop_accuracy': accuracy_score(y_crop_test, crop_pred),
        'fert_accuracy': accuracy_score(y_fert_test, fert_pred),
        'days_rmse': np.sqrt(mean_squared_error(y_days_test, days_pred)),
        'crop_report': classification_report(y_crop_test, crop_pred, target_names=label_encoders['Crop Recommendation'].classes_, output_dict=True),
        'fert_report': classification_report(y_fert_test, fert_pred, target_names=label_encoders['Fertilizer Recommendation'].classes_, output_dict=True)
    }

    return metrics

# Get model evaluation metrics
model_metrics = evaluate_models()

# Initialize SHAP explainer
explainer_crop = shap.TreeExplainer(crop_model)
explainer_fert = shap.TreeExplainer(fert_model)
explainer_days = shap.TreeExplainer(days_model)

def find_closest_records(soil_type_encoded, pH, nitrogen, phosphorus, potassium, moisture, temperature, n=3):
    # Calculate Euclidean distance from all records
    distances = np.sqrt(
        (df['Soil Type'] - soil_type_encoded)**2 +
        (df['pH Level'] - pH)**2 +
        (df['Nitrogen Level (ppm)'] - nitrogen)**2 +
        (df['Phosphorus Level (ppm)'] - phosphorus)**2 +
        (df['Potassium Level (ppm)'] - potassium)**2 +
        (df['Moisture Level (%)'] - moisture)**2 +
        (df['Temperature (°C)'] - temperature)**2
    )

    # Get indices of closest records
    closest_indices = distances.argsort()[:n]
    return df.iloc[closest_indices]

def create_input_charts(soil_type, pH, nitrogen, phosphorus, potassium, moisture, temperature):
    # Create figure with 2 subplots for input parameters
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Chart 1: Soil parameters radar chart
    categories = ['pH', 'Nitrogen', 'Phosphorus', 'Potassium']
    values = [pH, nitrogen, phosphorus, potassium]

    # Normalize values for radar chart (0-1 scale)
    max_values = [9, 300, 300, 300]  # Max possible values for each parameter
    normalized_values = [v/max_v for v, max_v in zip(values, max_values)]

    # Repeat first value to close the radar chart
    normalized_values += normalized_values[:1]
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax1 = plt.subplot(121, polar=True)
    ax1.plot(angles, normalized_values, color='blue', linewidth=2, linestyle='solid')
    ax1.fill(angles, normalized_values, color='blue', alpha=0.25)
    ax1.set_yticklabels([])
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories)
    ax1.set_title('Soil Nutrient Radar Chart', pad=20)

    # Add actual values as annotations
    for angle, value, actual in zip(angles[:-1], normalized_values[:-1], values):
        ax1.text(angle, value + 0.05, f"{actual}", ha='center')

    # Chart 2: Environmental factors gauge charts
    ax2 = plt.subplot(122)
    ax2.axis('off')

    # Create moisture gauge
    moisture_ax = fig.add_axes([0.55, 0.5, 0.2, 0.4])
    moisture_ax.set_xlim(0, 100)
    moisture_ax.set_ylim(0, 1)
    moisture_ax.barh(0.5, moisture, height=0.4, color='skyblue')
    moisture_ax.axvline(x=moisture, color='red', linestyle='--')
    moisture_ax.text(moisture + 5, 0.5, f"{moisture}%", va='center')
    moisture_ax.set_title('Moisture Level (%)')
    moisture_ax.set_yticks([])

    # Create temperature gauge
    temp_ax = fig.add_axes([0.8, 0.5, 0.2, 0.4])
    temp_ax.set_xlim(0, 50)
    temp_ax.set_ylim(0, 1)
    temp_ax.barh(0.5, temperature, height=0.4, color='salmon')
    temp_ax.axvline(x=temperature, color='blue', linestyle='--')
    temp_ax.text(temperature + 2, 0.5, f"{temperature}°C", va='center')
    temp_ax.set_title('Temperature (°C)')
    temp_ax.set_yticks([])

    plt.tight_layout()
    return fig

def create_output_charts(soil_type, pH, nitrogen, phosphorus, potassium, moisture, temperature,
                       pred_crop_label, pred_fert_label, pred_days, input_data):
    # Get feature importances
    crop_importances = crop_model.feature_importances_
    fert_importances = fert_model.feature_importances_
    days_importances = days_model.feature_importances_

    feature_names = ['Soil Type', 'pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Moisture', 'Temperature']

    # Create figure with subplots for output predictions
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2)

    # Chart 3: Feature importance for recommendations
    ax1 = fig.add_subplot(gs[0, 0])
    width = 0.35
    x = np.arange(len(feature_names))
    ax1.bar(x - width/2, crop_importances, width, label='Crop Importance', color='green')
    ax1.bar(x + width/2, fert_importances, width, label='Fertilizer Importance', color='orange')
    ax1.set_xticks(x)
    ax1.set_xticklabels(feature_names, rotation=45)
    ax1.set_title('Why These Recommendations? (Feature Importance)')
    ax1.set_ylabel('Importance Score')
    ax1.legend()

    # Add recommendation text
    ax1.text(0.02, 0.95, f"Recommended Crop: {pred_crop_label}",
             transform=ax1.transAxes, fontsize=12, color='green')
    ax1.text(0.02, 0.90, f"Recommended Fertilizer: {pred_fert_label}",
             transform=ax1.transAxes, fontsize=12, color='orange')

    # Chart 4: Water retention timeline
    ax2 = fig.add_subplot(gs[0, 1])
    days = int(np.ceil(pred_days))
    x_days = np.arange(0, days + 1)
    y_water = np.sin(np.linspace(0, 2*np.pi, days + 1)) * 0.5 + 0.5  # Wave pattern

    ax2.plot(x_days, y_water, color='blue', marker='o', markersize=5)
    ax2.fill_between(x_days, y_water, color='blue', alpha=0.1)
    ax2.set_title('Water Retention Timeline')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Water Availability')
    ax2.set_xticks(np.arange(0, days + 1, max(1, days//5)))
    ax2.set_yticks([])
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add annotations for key points
    ax2.text(days/2, 0.9, f"Optimal watering period: {days} days", ha='center')
    ax2.text(0, 0.1, "Initial watering", ha='left')
    ax2.text(days, 0.1, "Re-water needed", ha='right')

    # Chart 5: SHAP values for crop prediction
    ax3 = fig.add_subplot(gs[1, 0])
    shap_values_crop = explainer_crop.shap_values(input_data)
    shap.summary_plot(shap_values_crop, input_data, feature_names=feature_names, plot_type="bar", show=False)
    ax3.set_title('SHAP Values for Crop Recommendation')
    plt.tight_layout()

    # Chart 6: SHAP values for fertilizer prediction
    ax4 = fig.add_subplot(gs[1, 1])
    shap_values_fert = explainer_fert.shap_values(input_data)
    shap.summary_plot(shap_values_fert, input_data, feature_names=feature_names, plot_type="bar", show=False)
    ax4.set_title('SHAP Values for Fertilizer Recommendation')
    plt.tight_layout()

    return fig

def predict_crop_fertilizer_water(soil_type, pH, nitrogen, phosphorus, potassium, moisture, temperature):
    # Encode soil type
    soil_type_encoded = label_encoders['Soil Type'].transform([soil_type])[0]

    # Prepare input data
    input_data = np.array([[soil_type_encoded, pH, nitrogen, phosphorus, potassium, moisture, temperature]])
    new_input = pd.DataFrame(input_data, columns=X.columns)

    # Predictions
    pred_crop = crop_model.predict(new_input)[0]
    pred_fert = fert_model.predict(new_input)[0]
    pred_days = days_model.predict(new_input)[0]

    # Get prediction probabilities
    crop_probs = crop_model.predict_proba(new_input)[0]
    fert_probs = fert_model.predict_proba(new_input)[0]

    # Get top 3 crops and fertilizers with their probabilities
    crop_classes = label_encoders['Crop Recommendation'].classes_
    top_crop_indices = crop_probs.argsort()[-3:][::-1]
    top_crops = [(crop_classes[i], f"{crop_probs[i]:.1%}") for i in top_crop_indices]

    fert_classes = label_encoders['Fertilizer Recommendation'].classes_
    top_fert_indices = fert_probs.argsort()[-3:][::-1]
    top_ferts = [(fert_classes[i], f"{fert_probs[i]:.1%}") for i in top_fert_indices]

    # Decode predictions
    pred_crop_label = label_encoders['Crop Recommendation'].inverse_transform([pred_crop])[0]
    pred_fert_label = label_encoders['Fertilizer Recommendation'].inverse_transform([pred_fert])[0]

    # Find closest records from dataset
    closest_records = find_closest_records(soil_type_encoded, pH, nitrogen, phosphorus, potassium, moisture, temperature)

    # Decode the closest records
    closest_records['Soil Type'] = label_encoders['Soil Type'].inverse_transform(closest_records['Soil Type'])
    closest_records['Crop Recommendation'] = label_encoders['Crop Recommendation'].inverse_transform(closest_records['Crop Recommendation'])
    closest_records['Fertilizer Recommendation'] = label_encoders['Fertilizer Recommendation'].inverse_transform(closest_records['Fertilizer Recommendation'])

    # Create comparison table
    comparison_table = pd.DataFrame({
        'Feature': ['Soil Type', 'pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Moisture', 'Temperature'],
        'Your Input': [soil_type, pH, nitrogen, phosphorus, potassium, moisture, temperature],
        'Closest Match 1': closest_records.iloc[0][['Soil Type', 'pH Level', 'Nitrogen Level (ppm)',
                                                  'Phosphorus Level (ppm)', 'Potassium Level (ppm)',
                                                  'Moisture Level (%)', 'Temperature (°C)']].values,
        'Closest Match 2': closest_records.iloc[1][['Soil Type', 'pH Level', 'Nitrogen Level (ppm)',
                                                  'Phosphorus Level (ppm)', 'Potassium Level (ppm)',
                                                  'Moisture Level (%)', 'Temperature (°C)']].values
    })

    # Create charts
    input_fig = create_input_charts(soil_type, pH, nitrogen, phosphorus, potassium, moisture, temperature)
    output_fig = create_output_charts(soil_type, pH, nitrogen, phosphorus, potassium, moisture, temperature,
                                    pred_crop_label, pred_fert_label, pred_days, input_data)

    # Return results with verification
    return (f"Recommended Crop: {pred_crop_label} (Confidence: {crop_probs.max():.1%})",
            f"Other possible crops: {', '.join([f'{crop} ({prob})' for crop, prob in top_crops])}",
            f"Recommended Fertilizer: {pred_fert_label} (Confidence: {fert_probs.max():.1%})",
            f"Other possible fertilizers: {', '.join([f'{fert} ({prob})' for fert, prob in top_ferts])}",
            f"Water Retention Days: {pred_days:.1f} days",
            input_fig,
            output_fig,
            comparison_table,
            f"Actual crops in similar conditions: {', '.join(closest_records['Crop Recommendation'].unique())}",
            f"Actual fertilizers in similar conditions: {', '.join(closest_records['Fertilizer Recommendation'].unique())}")

# Create Gradio interface
with gr.Blocks(title="Enhanced Soil Analysis System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌱 Enhanced Soil Analysis and Crop Recommendation System
    *Validated predictions with explainable AI and dataset verification*
    """)

    with gr.Tab("🌾 Main Application"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Soil Parameters")
                soil_type = gr.Dropdown(choices=["Loamy", "Sandy", "Clay", "Silty", "Peaty", "Chalky"],
                                      label="Soil Type", value="Loamy")
                pH = gr.Slider(minimum=4.0, maximum=9.0, value=6.5, step=0.1, label="pH Level")
                nitrogen = gr.Slider(minimum=0, maximum=300, value=150, step=1, label="Nitrogen Level (ppm)")
                phosphorus = gr.Slider(minimum=0, maximum=300, value=100, step=1, label="Phosphorus Level (ppm)")

            with gr.Column():
                gr.Markdown("### Environmental Factors")
                potassium = gr.Slider(minimum=0, maximum=300, value=220, step=1, label="Potassium Level (ppm)")
                moisture = gr.Slider(minimum=0, maximum=100, value=35, step=1, label="Moisture Level (%)")
                temperature = gr.Slider(minimum=0, maximum=50, value=28, step=0.1, label="Temperature (°C)")
                predict_btn = gr.Button("Analyze Soil & Get Recommendations", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Crop Recommendations")
                crop_output = gr.Textbox(label="Primary Recommendation")
                crop_alternatives = gr.Textbox(label="Alternative Options")

            with gr.Column():
                gr.Markdown("### Fertilizer Recommendations")
                fert_output = gr.Textbox(label="Primary Recommendation")
                fert_alternatives = gr.Textbox(label="Alternative Options")

        water_output = gr.Textbox(label="💧 Water Retention Estimate")

        with gr.Accordion("📊 Detailed Analysis Charts", open=False):
            input_plot = gr.Plot(label="Input Parameters Analysis")
            output_plot = gr.Plot(label="Recommendation Analysis with Explanations")

        with gr.Accordion("🔍 Verification Against Dataset", open=False):
            comparison_table = gr.Dataframe(label="Comparison with Similar Records in Dataset")
            similar_crops = gr.Textbox(label="Actual Crops Grown in Similar Conditions")
            similar_ferts = gr.Textbox(label="Actual Fertilizers Used in Similar Conditions")

        predict_btn.click(
            predict_crop_fertilizer_water,
            inputs=[soil_type, pH, nitrogen, phosphorus, potassium, moisture, temperature],
            outputs=[crop_output, crop_alternatives, fert_output, fert_alternatives,
                    water_output, input_plot, output_plot, comparison_table,
                    similar_crops, similar_ferts]
        )

    with gr.Tab("📊 Model Performance"):
        gr.Markdown("### Model Evaluation Metrics")
        gr.Markdown(f"""
        #### Crop Recommendation Model
        - **Accuracy**: {model_metrics['crop_accuracy']:.1%}
        - **Precision**: {model_metrics['crop_report']['weighted avg']['precision']:.1%}
        - **Recall**: {model_metrics['crop_report']['weighted avg']['recall']:.1%}
        - **F1-Score**: {model_metrics['crop_report']['weighted avg']['f1-score']:.1%}

        #### Fertilizer Recommendation Model
        - **Accuracy**: {model_metrics['fert_accuracy']:.1%}
        - **Precision**: {model_metrics['fert_report']['weighted avg']['precision']:.1%}
        - **Recall**: {model_metrics['fert_report']['weighted avg']['recall']:.1%}
        - **F1-Score**: {model_metrics['fert_report']['weighted avg']['f1-score']:.1%}

        #### Water Retention Model
        - **RMSE**: {model_metrics['days_rmse']:.2f} days
        """)

        with gr.Accordion("Detailed Classification Reports"):
            gr.Markdown("### Crop Recommendation Report")
            gr.Dataframe(pd.DataFrame(model_metrics['crop_report']).transpose())

            gr.Markdown("### Fertilizer Recommendation Report")
            gr.Dataframe(pd.DataFrame(model_metrics['fert_report']).transpose())

    with gr.Tab("📂 Data Management"):
        gr.Markdown("### Upload Data to Firebase")
        csv_upload = gr.File(label="Upload CSV File", file_types=[".csv"])
        upload_btn = gr.Button("Upload to Firebase")
        upload_output = gr.Textbox(label="Upload Status")

        upload_btn.click(
            upload_csv_to_firebase,
            inputs=csv_upload,
            outputs=upload_output
        )

# Launch the interface
demo.launch(share=True)