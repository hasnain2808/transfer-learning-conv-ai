# transfer-learning-conv-ai
A companion that could establish long term relationships with the user.<br>
Graph is used as the memory to store context for the chit-chat model and graph embeddings are used for the factual question answering<br>
A lstm classifier is built to classify intent to factual and chit-chat<br>
The input is processed using Stanford Core NLP for triplet extraction and coreference analysis and Spacy pipelines for other tasks such as tokenization, lemmatization, stemming, etc. The chit-chat model is trained using OpenAI gpt2 and using persona chat data. Our model ranks second on one metric and third on two other metrics.<br>
The final product can be used at various locations such as customer service, sales and healthcare. The notable benefit being long term relationships<br>


This projects is made up of three actual projects<br>
The classifier(completely implemented by us)<br>
The non factual question answering (90% implemented by us)<br>
The factual question answering (used as a sub module and converted by flask api's by us)<br>

This project provides two innovation one is the non factual question answering and the other is the classifier<br>

The installation of the project should be done as follows<br>
Clone the repository<br>
Install and run each of the submodule as per the instructions in the readme in their folders<br>
Run the chat.py file in the root<br>


Or run the docker file it will do everything for you

and then go to the [your ip]:5001 to see the gui chatbot

Example conversation

USER: hi

BOT: hello how are you doing?

USER: I am great how are you

BOT: I’m doing well. I’m listening to music.

USER: which music do you like

BOT: I love all music. I love all music.

USER: I like countryside

BOT: I am in the united states

USER: in Texas?

BOT: I’m in Texas

USER: that's great

BOT: I’m also in Canada

USER: how can you be in two places?

BOT: I’m in Canada, how about you?

USER: I am in japan

BOT: that is awesome! my parents are from japan

USER: ohh where in japan?

BOT: I’m from the united states.

USER: your parents?

BOT: my mom and dad are from japan


