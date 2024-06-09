from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    hasil = ''
    if request.method == 'POST':
        if 'submit' in request.form:
            angka1 = request.form.get('angka1')
            angka2 = request.form.get('angka2')
            try:
                hasil = int(angka1) + int(angka2)
            except ValueError:
                hasil = "Input harus berupa angka"
        elif 'clear' in request.form:
            hasil = ''
    return render_template('index.html', hasil=hasil)

if __name__ == '__main__':
    app.run(debug=True)
