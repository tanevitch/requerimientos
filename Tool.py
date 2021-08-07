#!/usr/bin/env python
# coding: utf-8
import spacy
from spacy.matcher import Matcher
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg 

from collections import Counter

nlp = spacy.load('es_core_news_md')

# ruler = nlp.add_pipe("entity_ruler")
# ruler.add_patterns([{'label': 'ORG', 'pattern': 'Hoy duermo afuera'}])
# ruler.add_patterns([{'label': 'ORG', 'pattern': 'travesías en kayak'}])


matcher = Matcher(nlp.vocab)
nucleo = [{"DEP": "nsubj"}]
nucleoMD = [ {"DEP": "nsubj"}, {"DEP":"amod"}] #Caso 2 - "Los kayakistas expertos contratan travesías en kayak."
nucleoMI = [ {"DEP" : "nsubj"}, {"POS":"ADP"}, {"POS":"NOUN"}] #Caso 3.1 - "Los kayakistas de Córdoba contratan travesías en kayak." 

# verboSimple = [{"POS": "VERB"}]
# verboCompuesto = [ {"POS": "AUX"}, {"POS": "VERB"}]


matcher.add("Nucleo",  [nucleo])
matcher.add("NucleoMD",  [nucleoMD])
matcher.add("NucleoMI",  [nucleoMI])

# matcher.add("xd", [verboSimple])
# matcher.add("xdd", [verboCompuesto])

#---------------------- NEW VERSION ------------
sentWithOI= list()
sentWithoutOI = list()

def categorizeSentence(sentence):
    if len(getObjectsFromSentence(sentence)) == 1: #one object means that it has only do
        sentWithoutOI.append(sentence)
    elif len(getObjectsFromSentence(sentence)) == 2: #two objects mean that it has io and do
        sentWithOI.append(sentence)

#----------- used ----------------
def getVerbPosition(sentence):
    pos= 0
    for token in sentence:
        if (" " in getRelation(sentence)): # este es el caso para cuando son dos palabras
            if (token.text == getRelation(sentence).split(" ")[0]): 
                return pos-1
        elif (token.text == getRelation(sentence)):
            return pos
        pos+=1

def getRelation(sentence):  
    for token in sentence:
        if (token.pos_ =="AUX" and token.nbor().pos_ == "VERB"):
            return token.text + " " + token.nbor().text.capitalize()
        elif (token.dep_ == "ROOT" and token.nbor().pos_ == "ADP"):
            return token.text + " " + token.nbor().text.capitalize()
        elif (token.pos_ == "VERB" or token.lemma_ == "ser"):
            return token.text

def getObjectsFromSentence(sentence):
    for token in sentence:
        if token.dep_ =="obj":
            return token.text

def getSentenceEnts(sentence):
    return [ent.text for ent in sentence.ents]

def getEntityFromSubject(sentence, pair, pos_verb):
    #sujeto | entidades reconocidas - matcher
    recognizedEntities= getSentenceEnts(sentence[0:pos_verb])
    if len(recognizedEntities) >= 1: #if there are recognized entities
        pair.append(recognizedEntities[0])   
    else:
        matches = matcher(sentence[0:pos_verb])
        for match_id, start, end in matches:
            span = sentence[start:end]  
        pair.append(span.text) # add the lastest

def getEntityFromPredicate(sentence, pair, pos_verb):
     #predicado | entidades reconocidas - objeto - casos extras
    recognizedEntities= getSentenceEnts(sentence[pos_verb:len(sentence)])
    if len(recognizedEntities) >= 1: 
        pair.append(recognizedEntities[0])
    elif getObjectsFromSentence(sentence) != None:
        pair.append(getObjectsFromSentence(sentence))
    else:
        pair.append([token.text for token in sentence if token.dep_ =="ROOT" and token.pos_ =="NOUN"][0])

def getEntities(sentence):
    pos_verb= getVerbPosition(sentence)
    pair=list() #tiene que tener 2 elementos
    # print(sentence[0:pos_verb], " xd", sentence[pos_verb:len(sentence)])
    getEntityFromSubject(sentence, pair, pos_verb)
    getEntityFromPredicate(sentence, pair, pos_verb)      
    return pair    
        
def buildTriples(sentenceList, dataset):
    for sentence in sentenceList:    
        #sujeto-predicado-objeto
        triples = (getEntities(sentence)[0], getRelation(sentence), getEntities(sentence)[1])
        dataset.append(triples)

def sentences_parser(paragraph):
    candidate_sent= list()
    paragraph= filter(None,paragraph.split("."))
    for each in paragraph:
        candidate_sent.append(nlp(each.lstrip()))
    return candidate_sent

def printGraph(dataset):

    relation = [triplet[1] for triplet in dataset]
    source = [triplet[0] for triplet in dataset]
    target = [triplet[2] for triplet in dataset]

    dataset = pd.DataFrame({'Entidad1': source, 'relacion': relation, 'Entidad2':target})

    plt.figure(figsize=(12,12))
    G = nx.from_pandas_edgelist(df=dataset, source='Entidad1', target='Entidad2', edge_attr='relacion',
                                create_using=nx.DiGraph())
    pos = nx.spring_layout(G, k=5) 
    nx.draw(G, pos, with_labels=True, node_color='pink', node_size=2000)
    labels = {e: G.edges[e]['relacion'] for e in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show() 

def cosasParaHacerElGrafo(sentenceList, dataset):
    for sentence in sentenceList:    
        #este es para setear a mano la relacion de las que son propiedades
        if (nlp(getRelation(sentence))[0].lemma_ == "tener"):
            dataset.append((getEntities(sentence)[0], "hasProperty", getEntities(sentence)[1]))
            dataset.append((getEntities(sentence)[1], "propertyOf", getEntities(sentence)[0]))

            #si tiene propiedad, entonces va a ser clase
            dataset.append((getEntities(sentence)[0], "typeOf", "Class"))

        elif (nlp(getRelation(sentence))[0].lemma_ == "ser"): # este es para subclases
            dataset.append((getEntities(sentence)[0], "subclassOf", getEntities(sentence)[1]))
            dataset.append((getEntities(sentence)[1], "typeOf", "Class"))

        else: #este es para las normales
            triples = (getEntities(sentence)[0], getRelation(sentence).replace(" ", ""), getEntities(sentence)[1])
            dataset.append(triples)        

    # estos de abajo son para definir clases
        
    # ocurrencesOfEntity= Counter(entities)
    # for i in ocurrencesOfEntity:
    #     if ocurrencesOfEntity[i] >= 3:
    #         dataset.append((i, "typeoOf", "Class"))
    #         #------ sacar
    #         source.append(i)
    #         relations.append("typeOf")
    #         target.append("Class")
    #         #------ sacar)
            
    printGraph(dataset)
#----------------------  
doc = "Los kayakistas inexpertos son kayakistas. Las travesías en kayak son travesías. La empresa ofrece travesías en kayak. Las travesías en kayak tienen duración. Los kayakistas contratan travesías en kayak. La empresa informa el arancel. Los kayakistas solicitan arancel. La empresa está ubicada en Buenos Aires. Los kayakistas expertos son kayakistas."
doc= sentences_parser(doc)

dataset= list()
# buildTriples(doc, dataset)
cosasParaHacerElGrafo(doc, dataset)

print(dataset)



