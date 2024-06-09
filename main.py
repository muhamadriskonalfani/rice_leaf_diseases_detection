from flask import Flask, request, render_template, url_for
from klasifikasi import process_file
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def index():
    file_url = ''
    custom_message = ''
    if request.method == 'POST':
        if 'submit' in request.form:
            file = request.files.get('file')
            if file:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                file_url = url_for('static', filename=f'uploads/{file.filename}')
                custom_message = process_file(file_path)  # Panggil fungsi di bebas.py
                # Save the file path to the hidden input for later deletion
                return render_template('index.html', file_url=file_url, file_path=file_path, custom_message=custom_message)
        elif 'clear' in request.form:
            file_path = request.form.get('file_path')
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            file_url = ''
            custom_message = ''
    return render_template('index.html', file_url=file_url, file_path='', custom_message=custom_message)

if __name__ == '__main__':
    app.run(debug=True)
