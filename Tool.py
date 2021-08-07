import matplotlib.pyplot as plt
from itertools import chain
import networkx as nx
import pandas as pd
import spacy
from spacy.matcher import Matcher

NLP = spacy.load("es_core_news_md")

# ruler = NLP.add_pipe("entity_ruler")
# ruler.add_patterns([{'label': 'ORG', 'pattern': 'Hoy duermo afuera'}])
# ruler.add_patterns([{'label': 'ORG', 'pattern': 'travesías en kayak'}])


MATCHER = Matcher(NLP.vocab)

MATCHER.add("Nucleo", [[{"DEP": "nsubj"}]])

# Caso 2 - "Los kayakistas expertos contratan travesías en kayak."
MATCHER.add(
    "NucleoMD",
    [
        [
            {"DEP": "nsubj"},
            {"DEP": "amod"},
        ]
    ],
)

# Caso 3.1 - "Los kayakistas de Córdoba contratan travesías en kayak."
MATCHER.add(
    "NucleoMI",
    [
        [
            {"DEP": "nsubj"},
            {"POS": "ADP"},
            {"POS": "NOUN"},
        ]
    ],
)

# MATCHER.add("xd", [verboSimple])
# MATCHER.add("xdd", [verboCompuesto])

# ---------------------- NEW VERSION ------------
sentWithOI = list()
sentWithoutOI = list()


def categorizeSentence(sentence):
    if len(getObjectsFromSentence(sentence)) == 1:
        # one object means that it has only do
        sentWithoutOI.append(sentence)
    elif len(getObjectsFromSentence(sentence)) == 2:
        # two objects mean that it has io and do
        sentWithOI.append(sentence)


# ----------- used ----------------
def getVerbPosition(sentence):
    pos = 0
    for token in sentence:
        relation = getRelation(sentence)
        if relation is not None and " " in relation:
            # este es el caso para cuando son dos palabras
            if token.text == relation.split(" ")[0]:
                return pos - 1
        elif token.text == relation:
            return pos
        pos += 1


def getRelation(sentence):
    for token in sentence:
        if (token.pos_ == "AUX" and token.nbor().pos_ == "VERB") or (
            token.dep_ == "ROOT" and token.nbor().pos_ == "ADP"
        ):
            return token.text + " " + token.nbor().text.capitalize()

        elif token.pos_ == "VERB" or token.lemma_ == "ser":
            return token.text
    return None


def getObjectsFromSentence(sentence):
    for token in sentence:
        if token.dep_ == "obj":
            return token.text
    return ""


def getSentenceEnts(sentence):
    return [ent.text for ent in sentence.ents]


def getEntityFromSubject(sentence, pair, pos_verb):
    # sujeto | entidades reconocidas - matcher
    recognizedEntities = getSentenceEnts(sentence[0:pos_verb])
    if len(recognizedEntities) >= 1:  # if there are recognized entities
        pair.append(recognizedEntities[0])
    else:
        _, start, end = MATCHER(sentence[0:pos_verb])[-1]
        pair.append(sentence[start:end].text)  # add the last span


def getEntityFromPredicate(sentence, pair, pos_verb):
    # predicado | entidades reconocidas - objeto - casos extras
    recognizedEntities = getSentenceEnts(sentence[pos_verb : len(sentence)])
    if len(recognizedEntities) >= 1:
        pair.append(recognizedEntities[0])
    elif getObjectsFromSentence(sentence) != "":
        pair.append(getObjectsFromSentence(sentence))
    else:
        pair.append(
            [
                token.text
                for token in sentence
                if token.dep_ == "ROOT" and token.pos_ == "NOUN"
            ][0]
        )


def getEntities(sentence):
    pos_verb = getVerbPosition(sentence)
    pair = list()  # tiene que tener 2 elementos
    # print(sentence[0:pos_verb], " xd", sentence[pos_verb:len(sentence)])
    getEntityFromSubject(sentence, pair, pos_verb)
    getEntityFromPredicate(sentence, pair, pos_verb)
    return pair


def buildTriples(sentence_list, dataset):
    for sentence in sentence_list:
        # sujeto-predicado-objeto
        triples = (
            getEntities(sentence)[0],
            getRelation(sentence),
            getEntities(sentence)[1],
        )
        dataset.append(triples)


def sentences_parser(paragraph):
    candidate_sent = list()
    paragraph = filter(None, paragraph.split("."))
    for each in paragraph:
        candidate_sent.append(NLP(each.lstrip()))
    return candidate_sent


def printGraph(dataset):

    relation = [triplet[1] for triplet in dataset]
    source = [triplet[0] for triplet in dataset]
    target = [triplet[2] for triplet in dataset]

    dataset = pd.DataFrame(
        {"Entidad1": source, "relacion": relation, "Entidad2": target}
    )

    plt.figure(figsize=(12, 12))
    G = nx.from_pandas_edgelist(
        df=dataset,
        source="Entidad1",
        target="Entidad2",
        edge_attr="relacion",
        create_using=nx.DiGraph(),
    )

    val_map = {"Class": 0.0}
    values = [val_map.get(node, 0.25) for node in G.nodes()]

    pos = nx.spring_layout(G, k=5)
    nx.draw(G, pos, with_labels=True, node_color=values, node_size=2000)
    labels = {e: G.edges[e]["relacion"] for e in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.savefig("data/output.png", bbox_inches="tight")


# if there is propriety, there is property
def does_it_imply_propriety(relation):
    return NLP(relation)[0].lemma_ == "tener"


def does_it_imply_subclass(relation):
    return NLP(relation)[0].lemma_ == "ser"


def get_all_entities(sentence_list):
    return set(chain.from_iterable(map(getEntities, sentence_list)))


def generate_nodes(sentence_list):
    result = []
    for sentence in sentence_list:

        relation = getRelation(sentence)
        if relation is None:
            continue

        entities = getEntities(sentence)

        # este es para setear a mano la relacion de las que son propiedades
        if does_it_imply_propriety(relation):
            result.append((entities[0], "hasProperty", entities[1]))
            result.append((entities[1], "propertyOf", entities[0]))

            # si tiene propiedad, entonces va a ser clase
            result.append((entities[0], "typeOf", "Class"))

        elif does_it_imply_subclass(relation):  # este es para subclases
            result.append((entities[0], "subclassOf", entities[1]))
            result.append((entities[1], "typeOf", "Class"))

        else:  # este es para las normales
            triples = (entities[0], relation.replace(" ", ""), entities[1])
            result.append(triples)

    nodes_with_properties = set(map(lambda triple: triple[0], result))
    literals = filter(
        lambda entity: entity not in nodes_with_properties,
        get_all_entities(sentence_list),
    )
    for literal in literals:
        result.append((literal, "typeOf", "Literal"))

    return result


def cosasParaHacerElGrafo(sentence_list):
    dataset = generate_nodes(sentence_list)
    printGraph(dataset)
    return dataset


# ----------------------
doc = "Los kayakistas inexpertos son kayakistas. Las travesías en kayak son travesías. La empresa ofrece travesías en kayak. Las travesías en kayak tienen duración. Los kayakistas contratan travesías en kayak. La empresa informa el arancel. Los kayakistas solicitan arancel. La empresa está ubicada en Buenos Aires. Los kayakistas expertos son kayakistas."
doc = sentences_parser(doc)

# buildTriples(doc)
dataset = cosasParaHacerElGrafo(doc)

print(dataset)
