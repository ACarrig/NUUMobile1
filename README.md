# Customer Churn Prediction Platform
Problem: Analyze customer behavior & predict the churn probability of records with missing data

## Authors
- Aidan Carrig
- Ik Teng Liaw
- Noah Epelbaum

# How to Set up?
1. Clone this repository: 
    ```bash
    git clone https://github.com/ACarrig/NUUMobile1.git
    ```
2. Install Node.js: 
    Download and install from https://nodejs.org/en/download
3. Install Docker Desktop: 
    Download and install from https://docs.docker.com/desktop/
4. Run the application: 
    Open a terminal (use Git Bash if you're on Windows) and execute: `./run.sh`

# Overview of Code
## Frontend
The frontend is built with React and is responsible for providing an interactive user interface.
It consists of three main pages:
- Upload Page: Users can upload their data files.
- Dashboard Page: Displays visualizations, summaries, and key metrics derived from the uploaded data.
- Predictions Page: Shows churn predictions and insights generated from the uploaded dataset.

Navigation between pages is simple and intuitive via a navigation bar at the top of the application.

### 📂 File Upload Page
The FileUpload page allows users to upload customer data files into the system.
Key features include:
- Drag & Drop File Upload:
    Users can drag and drop files into a designated area or manually select files from their system.
- Multiple File Support:
    Users can select and upload multiple files at once.
- File Preview and Removal:
    Before uploading, users can see a preview of selected files and remove any unwanted files.
- Upload Confirmation:
    After selecting files, users must confirm the upload by clicking a button. Upon success, users are automatically redirected to the Dashboard page.
- Uploaded Files Management:
    The page also displays a list of files already uploaded to the server.
    Users can:
    - View all previously uploaded files
    - Delete any file from the server (with confirmation modal to prevent accidental deletion)

### 📊 Dashboard Page
The Dashboard.js component is the central hub of the Churn Predictor Tool, providing an interactive dashboard that lets users explore uploaded datasets, view predictions, analyze summaries, and investigate return feedback.

Key Features:
- File and Sheet Selection:
    Users can select an uploaded Excel file and choose which sheet to work with using the FileSelector and SheetSelector components.
- Tabbed Navigation:
    After selecting a file and sheet, users can explore the data through four main tabs:
    - Predictions: View the model's feature importance chart.
    - Summary: View analysis and insights about different feature such as app usage distribution, device model frequency, SIM information, monthly sales trends, and feature correlations. Each summary card includes a button that lets you dive deeper by redirecting to a detailed analysis page for that specific feature.
    - Returns: Analyze return-related charts like defect types, feedback issues, responsible parties, and verification problems, with an AI-generated summary for quick insights.
    - Column Plotter: Generate custom graphs by selecting columns of interest.
- Dynamic Content Loading:
    The charts and summaries shown in each tab dynamically adapt based on the selected file, sheet, and available columns.
- Responsive and Modular Design:
    The dashboard is cleanly divided into modular components (such as charts for specific insights), making it easy to extend or customize.

### 🔮 Predictions Page
The Prediction Page provides a user-friendly interface to upload, select, and analyze churn prediction data from Excel files. It allows users to choose the file, the sheet within that file, and the prediction model to generate detailed churn insights.

Key Features:
- File Upload:
    Users can upload new Excel files directly through an upload modal. The list of available files updates dynamically upon successful uploads.
- File and Sheet Selection:
    - After uploading or selecting a file, the app fetches and displays the available sheets within that file for the user to select.
    - Selection is reflected in the URL query parameters (file and sheet) for easy sharing and bookmarking.
- Model Selection:
    Users can select which churn prediction model to use from three options:
    - XGBoost Model
    - Ensemble Model (XGBoost Classifier + Random Forest Classifier + Logistic Regression + Gradient Boosting Classifier)
    - Neural Network (MLPC)
- Summary Panel:
    Once a file, sheet, and model are selected, a summary panel presents aggregated churn prediction insights, such as overall churn rates and distributions, giving users a high-level view of the data.
- Prediction Table:
    A detailed table displays row-level churn predictions, enabling users to analyze individual records and prediction probabilities.
- Model Information:
    Additional model-specific information is displayed to help users understand the model used for predictions, such as performance metrics or feature importance.


## Backend
The backend is built with Python and Flask.
It handles:
- File upload and parsing
- Data preprocessing
- Running the churn prediction models
- Returning model predictions and analytics to the frontend

### Model
The model available for the predictions page include:
- XGBoost Classifier model
- Ensemble model (XGBoost Classifier + Random Forest Classifier + Logistic Regression + Gradient Boosting Classifier)
- MLP Classifier model (Neural Network)

# What would you work on next?
- Interactive AI Chat
- Improve the model
- add new datasets to be added to the model if it sees that the new dataset has a Churn column and retrain the model with the new added new dataset
