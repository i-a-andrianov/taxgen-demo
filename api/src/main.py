#!flask/bin/python
from flask import Flask, jsonify, request, abort, send_file
import uuid
import os
from nltk.corpus import wordnet as wn
from diffusers import StableDiffusionPipeline
import torch
from helpers import get_graph_with_node, check_node_name, generate_new_node
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


all_lemmas = list(wn.all_lemma_names('n'))
app = Flask(__name__)
model_id = "prompthero/openjourney"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
#pipe = pipe.to("cuda")

opened_sessions = {}
root = "entity.n.01"
candidates = {}
cur_index = 0


@app.post('/token')
def get_new_session():
    uid = str(uuid.uuid4())
    global cur_index
    cur_index = 0
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
    if '.n.' in node_id:
        synset = wn.synset(node_id)
        offset = str(wn.synset(node_id).offset())
        if len(offset) < 8:
            offset = "0"*(8-len(offset)) + offset
        filename = f'images/n{offset}.JPEG'
        if os.path.exists(os.path.join(dir_path,filename)):
            return send_file(os.path.join(dir_path,filename), mimetype='image/jpeg')
        else:
            prompt = f"an image of {synset.name()} ({synset.definition()})"
            image = pipe(prompt).images[0]
            image.save(os.path.join(dir_path,f"images/n{node_id}.jpeg"))
            return send_file(os.path.join(dir_path,f"images/n{node_id}.jpeg"), mimetype='image/jpeg')
    else:
        if not os.path.exists(f"images/{node_id}.jpeg"):
            prompt = f"an image of {node_id}"
            image = pipe(prompt).images[0]
            image.save(os.path.join(dir_path,f"images/{node_id}.jpeg"))
        return send_file(os.path.join(dir_path,f"images/{node_id}.jpeg"), mimetype='image/jpeg')


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
        global cur_index
        cur_index = 0
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

    global cur_index
    graph = opened_sessions[uid]
    if start_node not in candidates:
        cur_index = 0
    graph = generate_new_node(graph, start_node, candidates, cur_index)
    cur_index += 1
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

    global cur_index
    graph = opened_sessions[uid]
    if start_node not in candidates:
        cur_index = 0
    graph = generate_new_node(graph, start_node, candidates, cur_index, end_node)
    cur_index += 1
    opened_sessions[uid] = graph
    return jsonify(graph)


if __name__ == '__main__':
    app.run(port=8888, host='0.0.0.0', debug=True, use_reloader=False)
