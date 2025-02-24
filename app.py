from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        # Process the file (e.g., save it to the server or database)
        return jsonify({"message": "File uploaded successfully!"}), 200
    else:
        return jsonify({"error": "No file provided!"}), 400

if __name__ == '__main__':
    app.run(debug=True)
