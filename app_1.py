from flask import Flask, request, jsonify
from flask_cors import CORS
from response_generate import *
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory

app = Flask(__name__)
CORS(app)

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
        print("There seems to be an issue with the database.")
    except LLMError:
        print(" Please try rephrasing or contact BI Team for further assistance.")
    except Exception as e:
        print(e)
        print("We are having trouble understanding your request. Please try rephrasing or contact BI Team for further assistance.")

@app.route("/api/chat", methods=["POST"])
def chat():
    if request.is_json:
        data = request.get_json()
        msg = data.get("msg")
        response = get_chat_response(msg)
        if response:
            return jsonify(response=response)
        else:
            return jsonify(response='Please try rephrasing or contact BI Team for further assistance.')
    else:
        return jsonify(error="Unsupported Media Type: Use 'Content-Type: application/json'"), 415

def get_chat_response(input):
    try:
        response = generate_response(input)
        print(response)
        return response
    except Exception as e:
        print(e)
        return 'An error occurred. Please try again.'

if __name__ == '__main__':
    app.run(debug=True)
