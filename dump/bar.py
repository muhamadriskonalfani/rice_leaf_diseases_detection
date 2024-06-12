from flask import Flask, render_template, jsonify
import time
import progressbar

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('bar.html')

@app.route('/long_task')
def long_task():
    total_steps = 100
    bar = progressbar.ProgressBar(max_value=total_steps)
    
    for i in range(total_steps):
        time.sleep(0.1)  # Simulasi tugas yang memakan waktu
        bar.update(i + 1)
    
    bar.finish()
    return jsonify({'status': 'Task completed!'})

if __name__ == '__main__':
    app.run(debug=True)
