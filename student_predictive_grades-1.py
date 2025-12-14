import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

df = None##
model = None

#this function loads the excel file the dataset is in
def load_dataset():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")])
    
    if not file_path:
        return
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path, engine='openpyxl')
        clean_data()
        messagebox.showinfo("Success", "Dataset loaded successfully!")
           
    except Exception as e:
        messagebox.showerror("Error", f"Couldn't load file: {str(e)}")

#this function cleans the dataset by encoding the categorical data and removing invalid data
def clean_data():
    global df
    copy = df.copy()
    
    mean = copy[copy['exam_score'] != 200]['exam_score'].mean()
    copy['e xam_score'] = copy['exam_score'].replace(200, mean)

    copy = copy.dropna(subset=['student_id'])
    copy = copy.replace(['unknown', 'varies', ''], np.nan)

    for field in ['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours', 'attendance_percentage', 'sleep_hours', 'exercise_frequency', 'mental_health_rating', 'exam_score']:
        if field in copy.columns:
            copy[field] = pd.to_numeric(copy[field], errors='coerce')
            copy[field] = copy[field].fillna(copy[field].mean())

    for field in ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 'internet_quality', 'extracurricular_participation']:
        if field in copy.columns:
            copy[field] = copy[field].fillna('unknown')
            le = LabelEncoder()
            copy[field] = le.fit_transform(copy[field].astype(str))
    
    df = copy

#this function trains the model on the selected dataset
def train_model(features, target):
    global model, df

    target = target.strip()

    if target not in df.columns:
        messagebox.showerror("Error", f"'{target}' not found.")
        return

    try:
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        messagebox.showinfo("Done", f"Training complete with r2 score: {r2:.3f} and average error: {rmse:.3f}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train model: {e}")

#this function uses the trained model to predict exam scores on the 
def make_predictions(features):
    global model, df

    if model is None:
        messagebox.showerror("Error", "Train a model first!")
        return
    elif df is None:
        messagebox.showerror("Error", "No data loaded!")
        return
    
    try:
        X = df[features]
        predictions = model.predict(X)
        
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "exam_score predictions:\n\n")

        for i in range(len(predictions)):
            score = predictions[i]
            student_id = df.iloc[i]['student_id']
            result_text.insert(tk.END, f"Student ID {student_id}: {score:.0f}\n")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to make predictions: {e}")

#defines the GUI
root = tk.Tk()
root.title("Student Predictive Grades")

#button to load to call the load_datasset() function
load_button = tk.Button(root, text="Load Dataset", command=lambda: load_dataset())
load_button.pack(pady=10)

#feature selection
tk.Label(root, text="Features (Click 'Select Features' after checking boxes or features will not be registered):").pack()

#variables to store boolean values to check if field was selected
age = tk.BooleanVar()
study_hours_per_day = tk.BooleanVar()
social_media_hours = tk.BooleanVar()
netflix_hours  = tk.BooleanVar()
attendance_percentage = tk.BooleanVar()
sleep_hours = tk.BooleanVar()
exercise_frequency = tk.BooleanVar()
mental_health_rating = tk.BooleanVar()
gender = tk.BooleanVar()
part_time_job = tk.BooleanVar()
diet_quality = tk.BooleanVar()
parental_education_level = tk.BooleanVar()
internet_quality = tk.BooleanVar()
extracurricular_participation = tk.BooleanVar()

#necessary to format checkboxes in columns
checkbox_frame = tk.Frame(root)
checkbox_frame.pack(pady=10)

#buttons that select fields to include as features
age_bttn = tk.Checkbutton(checkbox_frame, text="Age", variable=age)
age_bttn.grid(row=0, column=0, sticky='w', padx=10, pady=2)

study_hours_per_day_bttn = tk.Checkbutton(checkbox_frame, text="Study Hours Per Day", variable=study_hours_per_day)
study_hours_per_day_bttn.grid(row=1, column=0, sticky='w', padx=10, pady=2)

social_media_hours_bttn = tk.Checkbutton(checkbox_frame, text="Social Media Hours", variable=social_media_hours)
social_media_hours_bttn.grid(row=2, column=0, sticky='w', padx=10, pady=2)

netflix_hours_bttn = tk.Checkbutton(checkbox_frame, text="Netflix Hours", variable=netflix_hours)
netflix_hours_bttn.grid(row=3, column=0, sticky='w', padx=10, pady=2)

attendance_percentage_bttn = tk.Checkbutton(checkbox_frame, text="Attendance Percentage", variable=attendance_percentage)
attendance_percentage_bttn.grid(row=4, column=0, sticky='w', padx=10, pady=2)

sleep_hours_bttn = tk.Checkbutton(checkbox_frame, text="Sleep Hours", variable=sleep_hours)
sleep_hours_bttn.grid(row=5, column=0, sticky='w', padx=10, pady=2)

exercise_frequency_bttn = tk.Checkbutton(checkbox_frame, text="Exercise Frequency", variable=exercise_frequency)
exercise_frequency_bttn.grid(row=6, column=0, sticky='w', padx=10, pady=2)

mental_health_rating_bttn = tk.Checkbutton(checkbox_frame, text="Mental Health Rating", variable=mental_health_rating)
mental_health_rating_bttn.grid(row=0, column=1, sticky='w', padx=10, pady=2)

gender_bttn = tk.Checkbutton(checkbox_frame, text="Gender", variable=gender)
gender_bttn.grid(row=1, column=1, sticky='w', padx=10, pady=2)

part_time_job_bttn = tk.Checkbutton(checkbox_frame, text="Part Time Job", variable=part_time_job)
part_time_job_bttn.grid(row=2, column=1, sticky='w', padx=10, pady=2)

diet_quality_bttn = tk.Checkbutton(checkbox_frame, text="Diet Quality", variable=diet_quality)
diet_quality_bttn.grid(row=3, column=1, sticky='w', padx=10, pady=2)

parental_education_level_bttn = tk.Checkbutton(checkbox_frame, text="Parental Education Level", variable=parental_education_level)
parental_education_level_bttn.grid(row=4, column=1, sticky='w', padx=10, pady=2)

internet_quality_bttn = tk.Checkbutton(checkbox_frame, text="Internet Quality", variable=internet_quality)
internet_quality_bttn.grid(row=5, column=1, sticky='w', padx=10, pady=2)

extracurricular_participation_bttn = tk.Checkbutton(checkbox_frame, text="Extracurricular Participation", variable=extracurricular_participation)
extracurricular_participation_bttn.grid(row=6, column=1, sticky='w', padx=10, pady=2)

#checks if each field variable is true and if so, adds it to the feature list
features_entry = []
def fill_features():
    try:    
        if age.get():
            features_entry.append("age")
        if study_hours_per_day.get():
            features_entry.append("study_hours_per_day")
        if social_media_hours.get():
            features_entry.append("social_media_hours")
        if netflix_hours.get():
            features_entry.append("netflix_hours")
        if attendance_percentage.get():
            features_entry.append("attendance_percentage")
        if sleep_hours.get():
            features_entry.append("sleep_hours")
        if exercise_frequency.get():
            features_entry.append("exercise_frequency")
        if mental_health_rating.get():
            features_entry.append("mental_health_rating")
        if gender.get():
            features_entry.append("gender")
        if part_time_job.get():
            features_entry.append("part_time_job")
        if diet_quality.get():
            features_entry.append("diet_quality")
        if parental_education_level.get():
            features_entry.append("parental_education_level")
        if internet_quality.get():
            features_entry.append("internet_quality")
        if extracurricular_participation.get():
            features_entry.append("extracurricular_participation")
        messagebox.showinfo("Success", "Features Selected.")
    except Exception as e:
        messagebox.showerror("Error", f"Couldn't input features: {str(e)}")

#button to call fill_features()
use_selected_button = tk.Button(root, text="Select Features", command=fill_features)
use_selected_button.pack(pady=5)

#allows the user to enter the target for the model
tk.Label(root, text="Target:").pack()
target_entry = tk.Entry(root)
target_entry.pack(pady=5)

#button to call train_model()
train_button = tk.Button(root, text="Train Model", command=lambda: train_model(features_entry, target_entry.get()))
train_button.pack(pady=10)

#button to call make_predictions()
predict_button = tk.Button(root, text="Make Predictions", command=lambda: make_predictions(features_entry))
predict_button.pack(pady=10)

#this is a text box where the predictions are shown
result_text = tk.Text(root, height=20, width=80)
result_text.pack(pady=10)

#starts the main loop for the GUI
root.mainloop()