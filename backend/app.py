import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import app_usage_data
import dashboard

USERFILES_FOLDER = './backend/userfiles'

class NuuAPI:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/')
        def home():
            return "Flask home for NUU Project"

        # Load in file from frontend and save to folder for now
        @self.app.route('/upload', methods=['POST'])
        def upload_file():
            if 'files' not in request.files:
                return jsonify({'message': 'No files uploaded!'}), 400
            
            files = request.files.getlist('files')  # Get all files from the request
            if not files:
                return jsonify({'message': 'No files selected'}), 400

            saved_files = []
            for file in files:
                if file.filename:  # Ensure it's not an empty filename
                    file.save(f'{USERFILES_FOLDER}/{file.filename}')
                    saved_files.append(file.filename)

            return jsonify({'message': 'Files saved!', 'files': saved_files}), 200

        # Get all uploaded files
        @self.app.route('/get_files', methods=['GET'])
        def get_files():
            if not os.path.exists(USERFILES_FOLDER):  # Check if the folder exists
                return jsonify({'message': 'Folder not found!'}), 400

            files = os.listdir(USERFILES_FOLDER)
            if not files:
                return jsonify({'files': []}), 200

            file_data = [{'name': f} for f in files]
            return jsonify({'files': file_data}), 200

        # Route to select sheets of a file
        @self.app.route('/get_sheets/<file_name>', methods=['GET'])
        def get_sheets(file_name):
            try:
                # If file_name is 'All', fetch sheets from all files
                if file_name == "All":
                    # Handle the case where "All" is selected
                    sheet_names = dashboard.get_all_sheet_names()
                else:
                    sheet_names = dashboard.get_sheet_names(file_name)  # Get sheets for a specific file

                return jsonify({'sheets': sheet_names}), 200

            except Exception as e:
                # Ensure the error is returned as a JSON object
                return jsonify({'error': str(e)}), 500

        # Route to delete a file from the server
        @self.app.route('/delete_file/<filename>', methods=['DELETE'])
        def delete_file(filename):
            # print(f"Trying to delete: {filename}")
            file_path = os.path.join(USERFILES_FOLDER, filename)
            if not os.path.exists(file_path):
                return jsonify({'message': 'File not found!'}), 400
            
            os.remove(file_path)
            return jsonify({'message': 'File deleted!'}), 200
            
        # Route to app_usage_data.py and call method there to get analytics
        @self.app.route('/app_usage', methods=['GET'])
        def app_usage_analysis():
            return(app_usage_data.app_usage_info())

        @self.app.route('/top5apps', methods=['GET'])
        def top5apps():
            return app_usage_data.get_top_5_apps()
        
        @self.app.route('/app_usage_summary', methods=['GET'])
        def app_usage_summary():
            return app_usage_data.ai_summary()
        
        @self.app.route('/get_all_columns/<file>/<sheet>', methods=['GET'])
        def get_all_columns(file, sheet):
            try:
                columns = dashboard.get_all_columns(file, sheet)
                # print("Columns: ", columns)
                return jsonify({'columns': columns}), 200
            except Exception as e:
                return jsonify({'error': str}), 500
        
    # Method to run the Flask app
    def run(self):
        self.app.run(host='0.0.0.0', port=5001)

if __name__ == '__main__':
    api = NuuAPI()
    api.run()
