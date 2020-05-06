import requests
x = requests.get('http://[put your local ip here]/gen_personality:5001/', params={ "input_dialogue" : raw_text} )
print('the bot personality is')
print(x.content)

raw_text = input(">>> ")
while not raw_text:
    print('Prompt should not be empty!')
    raw_text = input(">>> ")
x = requests.get('http://[put your local ip here]/nextDialogue:5001/', params={ "input_dialogue" : raw_text} )
print(x.content)

while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        #test factual or non-factual
        x = requests.get('http://[put your local ip here]:5002/', params={ "input_dialogue" : raw_text} )
        con = int(x.content)
        
        #get the response from factual
        if con == 0:
            x = requests.get('http://[put your local ip here]:5003/', params={ "input_dialogue" : raw_text} )
            print(x.content)
        #get the response from non-factual chit-chat
        else
            x = requests.get('http://[put your local ip here]/nextDialogue:5001/', params={ "input_dialogue" : raw_text} )
            print(x.content)

