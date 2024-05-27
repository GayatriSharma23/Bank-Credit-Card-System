from flask import Flask, request
from flask_cors import CORS
import logging
from response_generate import *
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

memory = ConversationBufferMemory(memory_key="chat_history",
                                  input_key='query', output_key='result',
                                  return_messages=True)

class SQLError(Exception):
    pass

class LLMError(Exception):
    pass

def generate_response(query):
    db = connection(connection_string)
    llm = Ollama(model='PIA_2')
    chain = SQLDatabaseChain.from_llm(llm, db, return_intermediate_steps=True, verbose=True, memory=memory)
    try:
        response = chain(query)
        intermediate_steps = response["intermediate_steps"]
        if len(intermediate_steps) > 3:
            sql_result = intermediate_steps[3]
        else:
            sql_result = None
        if not sql_result:
            return 'This information is not in my knowledge base.'
        else:
            answer = response['result']
            return answer
    except SQLError:
        return "There seems to be an issue with the database."
    except LLMError:
        return " Please try rephrasing or contact BI Team for further assistance."
    except Exception as e:
        return "We are having trouble understanding your request. Please try rephrasing or contact BI Team for further assistance."

@app.route("/api/chat", methods=["POST"])
def chat():
    app.logger.debug('Request Headers: %s', request.headers)
    app.logger.debug('Request Body: %s', request.get_data())

    if request.form:
        msg = request.form.get("msg")
        response = get_chat_response(msg)
        if response:
            return response
        else:
            return 'Please try rephrasing or contact BI Team for further assistance.'
    else:
        return "Unsupported Media Type: Use 'application/x-www-form-urlencoded'", 415

def get_chat_response(input):
    try:
        response = generate_response(input)
        return response
    except Exception as e:
        return 'An error occurred. Please try again.'

if __name__ == '__main__':
    app.run(debug=True)
