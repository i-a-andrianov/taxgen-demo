import uuid
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
from gbert import predict_node_from_bert


def get_graph_with_node(start_node):
    hyponyms = [(i.name(), 7) for i in wn.synset(start_node).hyponyms()]
    second_hyponyms = [(j.name(), 9) for i, _ in hyponyms for j in wn.synset(i).hyponyms()]
    co_hypernyms = [(j.name(), 3) for i, _ in hyponyms for j in wn.synset(i).hypernyms()]

    hypernyms = [(i.name(), 3) for i in wn.synset(start_node).hypernyms()]
    second_hypernyms = [(j.name(), 1) for i, _ in hypernyms for j in wn.synset(i).hypernyms()]
    co_hyponyms = [(j.name(), 5) for i, _ in hypernyms for j in wn.synset(i).hyponyms()]

    all_nodes = hyponyms + second_hyponyms + co_hypernyms + hypernyms + second_hypernyms + co_hyponyms + [(wn.synset(start_node).name(), 5)]

    nodes, relations = _get_relations(all_nodes)

    return {
        'currentWord': start_node,
        'words': [
            {
            'id': node, 
            'word': node, 
            'level': level, 
            'definition': wn.synset(node).definition(), 
            'lemmas': [i.name().replace("_", " ") for i in wn.synset(node).lemmas()],
            'generated': False
            }
            for (node, level) in nodes.items()
        ],
        'relations': relations,
        
    }


def get_image_name(node_id):
    offset = str(wn.synset(node_id).offset())
    if len(offset) < 8:
        offset = "0"*(8-len(offset)) + offset
    filename = f'images/n{offset}.JPEG'
    return filename


def _get_relations(all_nodes):
    all_nodes = sorted(all_nodes, key=lambda x: x[1])
    graph = set()
    nodes = defaultdict(lambda: -1)

    for node1, level1 in all_nodes:
        level1 = max(nodes[node1], level1)
        for node2, level2 in all_nodes:
            level2 = max(nodes[node2], level2)
            if wn.synset(node1) in wn.synset(node2).hypernyms():
                graph.add((node1, node2))
                if level1 == level2:
                    level2 += 2

            elif wn.synset(node1) in wn.synset(node2).hyponyms():
                graph.add((node2, node1))
                if level1 == level2:
                    level1 += 2

            nodes[node1] = max(nodes[node1], level1)
            nodes[node2] = max(nodes[node2], level2)

    return nodes, [{'parent': parent, 'child': child} for (parent, child) in graph]


def generate_new_node(graph, start_node, candidates, cur_index, end_node=None):
    level, start_name = get_level_and_start_name(graph, start_node)
    node_x = predict_node_from_bert(start_name, candidates, cur_index)
    new_word = {
            'id': str(uuid.uuid4()),
            'word': node_x,
            'level': level,
            'definition': f"a member of {start_name} class",
            'lemmas': [],
            'generated': True
            }
    graph["words"].append(new_word)
    graph["relations"].append({'parent': start_node, 'child': new_word['id']})
    if end_node is not None:
        graph["relations"].append({'child': end_node, 'parent': new_word['id']})
    return graph


def get_level_and_start_name(graph, start_node):
    for word in graph['words']:
        if word['id'] == start_node:
            level = word['level'] + 1
            start_name = word['word']
            return level, start_name
    return None, None


def check_node_name(name):
    if '.n.' in name:
        try:
            return wn.synset(name).name()
        except WordNetError as e:
            print(f"Synset not found: {name}")
    name = name.replace(" ", "_")
    possible_synsets = wn.synsets(name, pos='n')
    if len(possible_synsets) == 1 and possible_synsets[0].pos() == 'n':
        return possible_synsets[0].name()
    elif len(possible_synsets) > 1:
        return [{"word": s.name(), "definition": s.definition()} for s in possible_synsets if s.pos() == 'n']
    else:
        return ""
