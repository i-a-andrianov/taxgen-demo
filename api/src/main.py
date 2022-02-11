#!flask/bin/python
from flask import Flask, jsonify, request, abort, send_file
import uuid
from nltk.corpus import wordnet as wn
from helpers import get_graph_with_node, generate_new_node, check_node_name

all_lemmas = list(wn.all_lemma_names('n'))
app = Flask(__name__)

opened_sessions = {}
root = "entity.n.01"


@app.post('/token')
def get_new_session():
    uid = str(uuid.uuid4())
    return jsonify(uid)


@app.get('/current')
def get_current_graph():
    uid = request.args['uid']
    if not uid:
        abort(400)
        return

    graph = opened_sessions.get(uid, None)
    if graph is None:
        graph = get_graph_with_node(root)
        opened_sessions[uid] = graph
    return jsonify(graph)


@app.get('/images/<node_id>')
def get_image(node_id):
    offset = wn.synset(node_id).offset()
    filename = f'images/n{offset}.JPEG'
    return send_file(filename, mimetype='image/jpeg')
    
    
@app.get('/search_node')
def search_node():
    node_name = request.args['node_name']
    node_name = check_node_name(node_name)
    return jsonify(node_name)


@app.post('/centered')
def center_graph_to():
    json = request.json
    uid = json['uid']
    start_node = json['start_node']
    if not uid:
        abort(400)
        return
    if not start_node:
        start_node = root

    graph = get_graph_with_node(start_node)
    opened_sessions[uid] = graph
    return jsonify(graph)


@app.post('/generate/words')
def generate_words():
    json = request.json
    uid = json['uid']
    start_node = json['start_node']
    if not uid or not start_node:
        abort(400)
        return

    graph = opened_sessions[uid]
    graph = generate_new_node(graph, start_node)
    opened_sessions[uid] = graph
    return jsonify(graph)


@app.post('/generate/relations')
def generate_relations():
    json = request.json
    uid = json['uid']
    start_node = json['start_node']
    end_node = json['end_node']
    if not uid or not start_node or not end_node:
        abort(400)
        return

    graph = opened_sessions[uid]
    graph = generate_new_node(graph, start_node, end_node)
    opened_sessions[uid] = graph
    return jsonify(graph)


if __name__ == '__main__':
    app.run(port=8888, host='0.0.0.0', debug=True, use_reloader=False)
