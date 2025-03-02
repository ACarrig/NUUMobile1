import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import app_usage_data

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
                    file.save(f'./backend/userfiles/{file.filename}')
                    saved_files.append(file.filename)

            return jsonify({'message': 'Files saved!', 'files': saved_files}), 200

        # Get all uploaded files
        @self.app.route('/get_files', methods=['GET'])
        def get_files():
            userfiles_folder = './backend/userfiles'  # Define the folder path
            if not os.path.exists(userfiles_folder):  # Check if the folder exists
                return jsonify({'message': 'Folder not found!'}), 400

            files = os.listdir(userfiles_folder)
            if not files:
                return jsonify({'files': []}), 200

            file_data = [{'name': f} for f in files]
            return jsonify({'files': file_data}), 200

        # Route to app_usage_data.py and call method there to get analytics
        @self.app.route('/app_usage', methods=['GET'])
        def app_usage_analysis():
            return(app_usage_data.app_usage_info())

    # Method to run the Flask app
    def run(self):
        self.app.run(host='0.0.0.0', port=5001)

if __name__ == '__main__':
    api = NuuAPI()
    api.run()
