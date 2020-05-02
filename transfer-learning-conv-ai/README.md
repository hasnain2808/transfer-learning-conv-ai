###Graph based Non-factual(Chit-Chat) question answering 

To install and use the training and inference scripts install the requirements:<br>

```bash

pip install -r requirements.txt
python -m spacy download en
```
Now install stanfordcorenlp 2015 from the official website and update the path accordingly in the train script <br>

Download the pretrained model from https://personabaseddeprdete.s3.ap-south-1.amazonaws.com/trained_model.zip <br>
unzip and put the contents in runs/ <br>
Verify  the path of the model in the interact_api.py file and run the interact_api.py file to start the server



Some parts of the code(10%) here were adapted from the hugging face transfer learning convai although the topic of the project as well as the submodule is completely new <br>
