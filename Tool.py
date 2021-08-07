import spacy
from spacy.matcher import Matcher
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from collections import Counter

NLP = spacy.load("es_core_news_md")

# ruler = NLP.add_pipe("entity_ruler")
# ruler.add_patterns([{'label': 'ORG', 'pattern': 'Hoy duermo afuera'}])
# ruler.add_patterns([{'label': 'ORG', 'pattern': 'travesías en kayak'}])


MATCHER = Matcher(NLP.vocab)
NUCLEO = [{"DEP": "nsubj"}]
NUCLEO_MD = [
    {"DEP": "nsubj"},
    {"DEP": "amod"},
]  # Caso 2 - "Los kayakistas expertos contratan travesías en kayak."
NUCLEO_MI = [
    {"DEP": "nsubj"},
    {"POS": "ADP"},
    {"POS": "NOUN"},
]  # Caso 3.1 - "Los kayakistas de Córdoba contratan travesías en kayak."

# verboSimple = [{"POS": "VERB"}]
# verboCompuesto = [ {"POS": "AUX"}, {"POS": "VERB"}]


MATCHER.add("Nucleo", [NUCLEO])
MATCHER.add("NucleoMD", [NUCLEO_MD])
MATCHER.add("NucleoMI", [NUCLEO_MI])

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
        if token.pos_ == "AUX" and token.nbor().pos_ == "VERB":
            return token.text + " " + token.nbor().text.capitalize()
        elif token.dep_ == "ROOT" and token.nbor().pos_ == "ADP":
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


def buildTriples(sentenceList, dataset):
    for sentence in sentenceList:
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


def printGraph(relations, source, target):
    dataset = pd.DataFrame(
        {"Entidad1": source, "relacion": relations, "Entidad2": target}
    )

    plt.figure(figsize=(12, 12))
    G = nx.from_pandas_edgelist(
        df=dataset,
        source="Entidad1",
        target="Entidad2",
        edge_attr="relacion",
        create_using=nx.DiGraph(),
    )
    pos = nx.spring_layout(G, k=5)
    nx.draw(G, pos, with_labels=True, node_color="pink", node_size=2000)
    labels = {e: G.edges[e]["relacion"] for e in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


def cosasParaHacerElGrafo(sentenceList, dataset):
    entities = list()

    entidadesQueTienenPropiedades = list()

    relations = []
    source = []
    target = []
    for sentence in sentenceList:

        relation = getRelation(sentence)
        if relation is None:
            continue

        # estos son para el Counter de las entidades
        entities.append(getEntities(sentence)[0])
        entities.append(getEntities(sentence)[1])

        # este es para setear a mano la relacion de las que son propiedades
        if NLP(relation)[0].lemma_ == "tener":
            dataset.append(
                (
                    getEntities(sentence)[0],
                    "hasProperty",
                    getEntities(sentence)[1],
                )
            )
            dataset.append(
                (
                    getEntities(sentence)[1],
                    "propertyOf",
                    getEntities(sentence)[0],
                )
            )

            entidadesQueTienenPropiedades.append(getEntities(sentence)[0])

            source.append(getEntities(sentence)[0])
            relations.append("hasProperty")
            target.append(getEntities(sentence)[1])

        # este es para subclases
        elif NLP(relation)[0].lemma_ == "ser":
            dataset.append(
                (
                    getEntities(sentence)[0],
                    "subclassOf",
                    getEntities(sentence)[1],
                )
            )

            source.append(getEntities(sentence)[0])
            relations.append("subclassOf")
            target.append(getEntities(sentence)[1])

            dataset.append((getEntities(sentence)[1], "typeoOf", "Class"))
            source.append(getEntities(sentence)[1])
            relations.append("typeOf")
            target.append("Class")

        else:  # este es para las normales
            source.append(getEntities(sentence)[0])
            relations.append(relation.replace(" ", ""))
            target.append(getEntities(sentence)[1])

            triples = (
                getEntities(sentence)[0],
                relation.replace(" ", ""),
                getEntities(sentence)[1],
            )
            dataset.append(triples)

        # este es para encontrar literales
        for token in getEntities(sentence):
            if NLP(token)[0].ent_type_ != "":
                dataset.append((token, "typeoOf", "Literal"))
                source.append(token)
                relations.append("typeOf")
                target.append("Literal")

    # estos de abajo son para definir clases

    # ocurrencesOfEntity= Counter(entities)
    # for i in ocurrencesOfEntity:
    #     if ocurrencesOfEntity[i] >= 3:
    #         dataset.append((i, "typeoOf", "Class"))
    #         #------ sacar
    #         source.append(i)
    #         relations.append("typeOf")
    #         target.append("Class")
    #         #------ sacar

    ocurrencesOfEntity = Counter(entidadesQueTienenPropiedades)
    for i in ocurrencesOfEntity:
        dataset.append((i, "typeoOf", "Class"))
        source.append(i)
        relations.append("typeOf")
        target.append("Class")

    printGraph(relations, source, target)


# ----------------------
doc = "Las travesías en kayak son travesías. La empresa ofrece travesías en kayak. Las travesías en kayak tienen duración. Los kayakistas contratan travesías en kayak. La empresa informa el arancel. Los kayakistas solicitan arancel. La empresa está ubicada en Buenos Aires. Los kayakistas expertos son kayakistas."
doc = sentences_parser(doc)

dataset = list()
# buildTriples(doc, dataset)
cosasParaHacerElGrafo(doc, dataset)

print(dataset)
