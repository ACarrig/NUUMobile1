import os, io
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import app_usage_data, dashboard, sim_info, return_info, churn_correlation, em_predictions, monthly_data, NetPred, xgb_predictions

import matplotlib
matplotlib.use('Agg')

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
                file_path = os.path.join(USERFILES_FOLDER, file_name)
                if not os.path.exists(file_path):
                    return jsonify({'error': f"File {file_name} not found!"}), 400

                # Get the sheet names
                sheet_names = dashboard.get_sheet_names(file_name)
                return jsonify({'sheets': sheet_names}), 200
            except Exception as e:
                print(f"Error in /get_sheets: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/get_all_columns/<file>/<sheet>', methods=['GET'])
        def get_all_columns(file, sheet):
            try:
                columns = dashboard.get_all_columns(file, sheet)
                print("Columns: ", columns)
                return jsonify({'columns': columns}), 200
            except Exception as e:
                return jsonify({'error': str}), 500
        
        @self.app.route("/get_column_data")
        def get_column_data():
            file = request.args.get('file')
            sheet = request.args.get('sheet')
            column = request.args.get('column')
            print(f"Received request for file: {file} and sheet: {sheet} and column: {column}")
            return jsonify(dashboard.get_column_data(file, sheet, column)), 200

        @self.app.route('/get_age_range/<file>/<sheet>', methods=['GET'])
        def get_age_range(file, sheet):
            try:
                # print(f"Received request for file: {file} and sheet: {sheet}")
                age_range = dashboard.get_age_range(file, sheet)
                # print("Age Range: ", age_range)
                return age_range, 200
            except Exception as e:
                print(f"Error: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/get_model_type/<file>/<sheet>', methods=['GET'])
        def get_model_type(file, sheet):
            try:
                # print(f"Received request for file: {file} and sheet: {sheet}")
                model_type = dashboard.get_model_type(file, sheet)
                # print("Model Type: ", model_type)
                return model_type, 200
            except Exception as e:
                print(f"Error: {str(e)}")
                return jsonify({'error': str(e)}), 500
            
        @self.app.route('/get_model_performance_by_channel/<file>/<sheet>', methods=['GET'])
        def get_model_performance_by_channel(file, sheet):
            try:
                # print(f"Received request for file: {file} and sheet: {sheet}")
                model_performance = dashboard.get_model_performance_by_channel(file, sheet)
                # print("Model Performance: ", model_performance)
                return model_performance, 200
            except Exception as e:
                print(f"Error: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/get_carrier_name/<file>/<sheet>', methods=['GET'])
        def get_carrier_name(file, sheet):
            try:
                # print(f"Received request for file: {file} and sheet: {sheet}")
                carrier_name = sim_info.get_carrier_name(file, sheet)
                # print("Carriers: ", carrier_name)
                return carrier_name, 200
            except Exception as e:
                print(f"Error: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/get_carrier_name_from_1slot/<file>/<sheet>/<slot>', methods=['GET'])
        def get_carrier_name_from_1slot(file, sheet, slot):
            try:
                carrier_name = sim_info.get_carrier_name_from_1slot(file, sheet, slot)
                # print(f"Carriers for {slot}: ", carrier_name)
                return carrier_name, 200
            except Exception as e:
                print(f"Error: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/get_carrier_name_from_slot/<file>/<sheet>', methods=['GET'])
        def get_carrier_name_from_slot(file, sheet):
            try:
                carrier_name = sim_info.get_carrier_name_from_slot(file, sheet)
                # print("Carriers: ", carrier_name)
                return carrier_name, 200
            except Exception as e:
                print(f"Error: {str(e)}")
                return jsonify({'error': str(e)}), 500
            
        @self.app.route('/get_carrier_country/<file>/<sheet>', methods=['GET'])
        def get_carrier_country(file, sheet):
            try:
                country = sim_info.get_carrier_country(file, sheet)
                print("Country: ", country)
                return country, 200
            except Exception as e:
                print(f"Error: {str(e)}")
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
            
        # # Route to app_usage_data.py and call method there to get analytics
        # @self.app.route('/app_usage', methods=['GET'])
        # def app_usage_analysis():
        #     return(app_usage_data.app_usage_info())

        @self.app.route('/get_app_usage/<file>/<sheet>', methods=['GET'])
        def get_app_usage(file,sheet):
            return app_usage_data.get_app_usage(file,sheet)
        
        @self.app.route('/get_most_used_app_counts/<file>/<sheet>')
        def most_used_app_counts(file, sheet):
            return jsonify(app_usage_data.get_most_used_app_counts(file, sheet))

        # @self.app.route('/app_usage_summary/<file>/<sheet>', methods=['GET'])
        # def app_usage_summary(file, sheet):
        #     return app_usage_data.app_ai_summary(file, sheet)
        
        @self.app.route('/ai_summary', methods=['GET'])
        def ai_summary():
            file = request.args.get('file')
            sheet = request.args.get('sheet')
            column = request.args.get('column')
            # print(f"Received request for summary for {column} in {sheet} of {file}")
            return dashboard.ai_summary(file,sheet,column)
        
        @self.app.route('/ai_summary2', methods=['GET'])
        def ai_summary2():
            file = request.args.get('file')
            sheet = request.args.get('sheet')
            column1 = request.args.get('column1')
            column2 = request.args.get('column2')
            
            # print(f"Received request for summary for {column1} and {column2} in {sheet} of {file}")
            
            return dashboard.ai_summary2(file, sheet, column1, column2)
        
        # @self.app.route('/comparison_summary', methods=['GET'])
        # def comparison_summary():
        #     # Retrieve the file, sheet, and columns from the request arguments
        #     file = request.args.get('file')
        #     sheet = request.args.get('sheet')
        #     columns = request.args.getlist('column')  # Get a list of columns
            
        #     # print(f"Received request for comparison summary for columns {columns} in {sheet} of {file}")
            
        #     # Ensure there are columns provided
        #     if not columns or len(columns) < 2:
        #         return jsonify({"error": "At least two columns must be provided"}), 400

        #     # Call the comparison_summary function with the selected columns
        #     return dashboard.comparison_summary(file, sheet, columns)

        @self.app.route('/num_returns/<file>/<sheet>', methods=['GET'])
        def num_returns(file, sheet):
            return return_info.returns_count(file, sheet)

        @self.app.route('/device_return_info/<file>/<sheet>', methods=['GET'])
        def device_return_info(file, sheet):
            return return_info.returns_info(file, sheet)
        
        @self.app.route('/feedback_info/<file>/<sheet>', methods=['GET'])
        def feedback_info(file, sheet):
            return return_info.feedback_info(file, sheet)
        
        @self.app.route('/verification_info/<file>/<sheet>', methods=['GET'])
        def verification_info(file, sheet):
            return return_info.verification_info(file, sheet)
        
        @self.app.route('/resparty_info/<file>/<sheet>', methods=['GET'])
        def resparty_info(file, sheet):
            return return_info.resparty_info(file, sheet)
        
        @self.app.route('/returns_comparison_summary', methods=['GET'])
        def returns_summary():
            file = request.args.get('file')
            sheet = request.args.get('sheet')
            print(f"Received request for summary for {sheet} of {file}")
            return return_info.returns_summary(file, sheet)
        
        @self.app.route('/param_churn_correlation/<file>/<sheet>', methods=['GET'])
        def param_churn_correlation(file, sheet):
            return churn_correlation.churn_relation(file, sheet)
        
        @self.app.route('/churn_corr_heatmap/<file>/<sheet>', methods=['GET'])
        def churn_corr_heatmap(file, sheet):
            heatmap = churn_correlation.churn_corr_heatmap(file, sheet)
            return jsonify(heatmap)
        
        @self.app.route('/churn_corr_summary/<file>/<sheet>', methods=['GET'])
        def param_churn_corr_summary(file, sheet):
            return churn_correlation.churn_corr_summary(file, sheet)
        
        @self.app.route('/<model_type>_predict_data/<file>/<sheet>', methods=['GET'])
        def predict_data(model_type, file, sheet):
            if model_type == 'em':
                predictor = em_predictions  # Ensemble predictor
            elif model_type == 'xgb':
                predictor = xgb_predictions
            elif model_type == 'nn':
                predictor = NetPred  # Neural Network predictor

            else:
                return jsonify({"error": "Invalid model type"}), 400

            result = predictor.predict_churn(file, sheet)
            return jsonify(result)

        @self.app.route('/<model_type>_get_features/<file>/<sheet>', methods=['GET'])
        def get_features(model_type, file, sheet):
            print(f"Received request for feature importances for {model_type}")
            if model_type == 'em':
                predictor = em_predictions
            elif model_type == 'xgb':
                predictor = xgb_predictions
            elif model_type == 'nn':
                predictor = NetPred

            else:
                return jsonify({"error": "Invalid model type"}), 400

            features = predictor.get_features(file, sheet)
            return jsonify(features)

        @self.app.route('/<model_type>_get_eval/<file>/<sheet>', methods=['GET'])
        def get_eval(model_type, file, sheet):
            if model_type == 'em':
                predictor = em_predictions
            elif model_type == 'xgb':
                predictor = xgb_predictions
            elif model_type == 'nn':
                predictor = NetPred

            else:
                return jsonify({"error": "Invalid model type"}), 400

            evaluation = predictor.evaluate_model(file, sheet)
            return jsonify(evaluation)

        @self.app.route('/<model_type>_download_data/<file>/<sheet>', methods=['GET'])
        def download_data(model_type, file, sheet):
            if model_type == 'em':
                predictor = em_predictions
            elif model_type == 'xgb':
                predictor = xgb_predictions
            elif model_type == 'nn':
                predictor = NetPred
            else:
                return jsonify({"error": "Invalid model type"}), 400

            result = predictor.download_churn(file, sheet)
            df = pd.DataFrame(result['predictions'])

            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            return send_file(
                io.BytesIO(csv_buffer.getvalue().encode()),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'{file}_{sheet}_{model_type}_predictions.csv'  # model_type added here
            )

        @self.app.route('/get_monthly_sales/<file>/<sheet>', methods=['GET'])
        def get_monthly_sales(file, sheet):
            return monthly_data.monthly_sales(file, sheet)
        
        @self.app.route('/get_monthly_retainment/<file>/<sheet>', methods=['GET'])
        def get_monthly_retainment(file, sheet):
            return monthly_data.device_retainment(file, sheet)

        @self.app.route('/get_monthly_model_sales/<file>/<sheet>', methods=['GET'])
        def get_monthly_model_sales(file, sheet):
            return monthly_data.monthly_model_sales(file, sheet)
    
    # Method to run the Flask app
    def run(self):
        self.app.run(host='0.0.0.0', port=5001, threaded=True, use_reloader=True)

if __name__ == '__main__':
    api = NuuAPI()
    api.run()
