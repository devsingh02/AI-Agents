from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import shutil
import io
import zipfile
import base64
from pipeline import classify  # Ensure this is your classification logic

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER_SINGLE = 'static/uploads/single'
UPLOAD_FOLDER_MULTIPLE = 'static/uploads/multiple'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER_SINGLE'] = UPLOAD_FOLDER_SINGLE
app.config['UPLOAD_FOLDER_MULTIPLE'] = UPLOAD_FOLDER_MULTIPLE
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directories exist
os.makedirs(UPLOAD_FOLDER_SINGLE, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_MULTIPLE, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_uploads(folder):
    upload_dir = Path(folder)
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/single')
def single():
    return render_template('single.html')

@app.route('/multiple')
def multiple():
    return render_template('multiple.html')

@app.route('/clear_uploads', methods=['POST'])
def clear_uploads_route():
    try:
        clear_uploads(app.config['UPLOAD_FOLDER_SINGLE'])
        clear_uploads(app.config['UPLOAD_FOLDER_MULTIPLE'])
        return jsonify({'message': 'Upload folders cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Single Image Classification Routes
@app.route('/upload_single', methods=['POST'])
def upload_single():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Clear old files
        clear_uploads(app.config['UPLOAD_FOLDER_SINGLE'])
        
        # Save new file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER_SINGLE'], filename)
        file.save(filepath)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/classify_single', methods=['POST'])
def classify_single():
    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER_SINGLE'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        classification_result, result_table, failure_labels = classify(filepath)
        return jsonify({
            'classification': classification_result,
            'result_table': result_table
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Multiple Image Classification Routes
@app.route('/upload_multiple', methods=['POST'])
def upload_multiple():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Save to temp directory
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER_MULTIPLE'], 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)
        
        # Read the file and convert to base64
        with open(filepath, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
        
        return jsonify({
            'filename': filename,
            'image': img_data,
            'status': 'Queued'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classify_multiple', methods=['POST'])
def classify_multiple():
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER_MULTIPLE'], 'temp')
    
    # Get list of files to process
    files = [f for f in os.listdir(temp_dir) 
             if os.path.isfile(os.path.join(temp_dir, f)) and allowed_file(f)]
    
    results = []
    for filename in files:
        filepath = os.path.join(temp_dir, filename)
        
        # Process single image
        classification_result, result_table, failure_labels = classify(filepath)
        
        # Move file to appropriate directory
        category_dir = os.path.join(app.config['UPLOAD_FOLDER_MULTIPLE'], classification_result.lower())
        os.makedirs(category_dir, exist_ok=True)
        dest_path = os.path.join(category_dir, filename)
        shutil.move(filepath, dest_path)
        
        # Read the processed image
        with open(dest_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
        
        # Return single result immediately
        result = {
            'filename': filename,
            'status': classification_result,
            'labels': failure_labels,
            'image': img_data
        }
        
        # Use Flask's Response streaming to send results one by one
        return jsonify(result)

    return jsonify({'message': 'All files processed'})

@app.route('/download_multiple/<category>')
def download_multiple(category):
    if category not in ['pass', 'fail']:
        return jsonify({'error': 'Invalid category'}), 400
    
    category_dir = os.path.join(app.config['UPLOAD_FOLDER_MULTIPLE'], category)
    if not os.path.exists(category_dir):
        return jsonify({'error': 'No images found'}), 404
    
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for filename in os.listdir(category_dir):
            filepath = os.path.join(category_dir, filename)
            if os.path.isfile(filepath) and allowed_file(filename):
                zf.write(filepath, filename)
    
    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'{category}_images.zip'
    )

@app.route('/save_report', methods=['POST'])
def save_report():
    data = request.get_json()
    html_content = data.get('htmlContent')
    filename = data.get('filename')
    date = data.get('date')

    if not html_content or not filename or not date:
        return jsonify({'error': 'Invalid data provided'}), 400

    try:
        # Create the Reports directory if it doesn't exist
        reports_dir = Path('Reports')
        reports_dir.mkdir(exist_ok=True)

        # Create the date-based subdirectory if it doesn't exist
        date_dir = reports_dir / date
        date_dir.mkdir(exist_ok=True)

        # Save the HTML report in the date-based subdirectory
        report_path = date_dir / filename
        with open(report_path, 'w', encoding='utf-8') as report_file:
            report_file.write(html_content)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Clear uploads on startup
    clear_uploads(app.config['UPLOAD_FOLDER_SINGLE'])
    clear_uploads(app.config['UPLOAD_FOLDER_MULTIPLE'])
    app.run(debug=True)