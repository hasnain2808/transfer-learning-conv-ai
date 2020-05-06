import flask
from flask import request, jsonify
from flask import Flask, render_template, request
import requests
app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    requests.get('http://127.0.0.1:5000/gen_personality' )

    return render_template('widgets.html')



@app.route('/new_message', methods=['GET', 'POST'])
def new_message():
    # if the input is present Nothing received is used
    raw_text = "Nothing received"
    # from the user
    if 'input_dialogue' in request.args:
        raw_text = str(request.args['input_dialogue'])

       #test factual or non-factual
        x = requests.get('http://[put your local ip here]:5002/', params={ "input_dialogue" : raw_text} )
        con = int(x.content)
        
        #get the response from factual
        if con == 0:
            x = requests.get('http://[put your local ip here]:5000/', params={ "query" : raw_text} )
            print(x.content)
        #get the response from non-factual chit-chat
        else
            x = requests.get('http://[put your local ip here]/nextDialogue:5001/', params={ "input_dialogue" : raw_text} )
    return x.content
    # //to the user
    # return raw_text
app.run(host=0.0.0.0, port=5001)
