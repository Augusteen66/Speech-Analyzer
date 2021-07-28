from flask import Flask, render_template, request, jsonify
from func import summ, text2emo, sia, keywords, data_structure
from flask import after_this_request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

speech = ''

@app.route('/speech', methods = ['POST', 'GET'])
def speech():

    global speech

    @after_this_request
    def add_header(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

    if request.method == 'POST':
        speech = request.json['analyzeData']
        return 'Success', 200

    if request.method == 'GET':
        #speech = request.form.get('analyzeData')
        speech = request.json['analyzeData']
        #print(speech)
        summary = summ(speech)
        t2e = text2emo(speech)
        sent = sia(speech)
        kw = keywords(speech)
        new_obj = data_structure(summary,t2e,kw,sent)
        return jsonify(new_obj), 201


@app.route('/input', methods = ['POST', 'GET'])
def input():
    if request.method == "POST":
        speech = request.form["speech"]
        summary = summ(speech)
        return render_template('analysis.html', sp = summary)
    else:
        return render_template('input.html')

if(__name__) == '__main__':
    app.run(5000,debug = True)