from flask import Flask, request, redirect, url_for
import subprocess

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        number_of_days = request.form['number_of_days']
        # Execute the Python script with the provided input
        # Adjust the path to your script if needed
        result = subprocess.run(['python', 'C:/xampp/htdocs/templates/static/python/predict1.py', number_of_days], capture_output=True, text=True)
        # Optionally: Handle the result or return it to the user
        return f"Script output: {result.stdout}"
    return '''
        <form method="post">
            <label for="number_of_days">Number of Days</label>
            <input name="number_of_days" id="number_of_days" type="number" placeholder="Enter number of days">
            <label for="checkbox">Are you sure to continue?</label>
            <input id="checkbox" type="checkbox">
            <button type="submit">Submit</button>
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
