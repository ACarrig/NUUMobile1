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
                    return jsonify({'message': 'Files saved!'}), 200
                
            return jsonify({'message': 'No more files'}), 400

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
