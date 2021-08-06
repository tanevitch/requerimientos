#!/usr/bin/env python
# coding: utf-8
import spacy
from spacy.matcher import Matcher

from collections import Counter

nlp = spacy.load('es_core_news_md')

# ruler = nlp.add_pipe("entity_ruler")
# ruler.add_patterns([{'label': 'ORG', 'pattern': 'Hoy duermo afuera'}])
# ruler.add_patterns([{'label': 'ORG', 'pattern': 'travesías en kayak'}])


matcher = Matcher(nlp.vocab)
nucleo = [{"DEP": "nsubj"}]
nucleoMD = [ {"DEP": "nsubj"}, {"DEP":"amod"}] #Caso 2 - "Los kayakistas expertos contratan travesías en kayak."
nucleoMI = [ {"DEP" : "nsubj"}, {"POS":"ADP"}, {"POS":"NOUN"}] #Caso 3.1 - "Los kayakistas de Córdoba contratan travesías en kayak." 

matcher.add("Nucleo",  [nucleo])
matcher.add("NucleoMD",  [nucleoMD])
matcher.add("NucleoMI",  [nucleoMI])

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
        pos+=1
        if (token.pos_ == "VERB" or token.lemma_ == "ser"):
            return pos

def getRelation(sentence):  
    for token in sentence:
        if (token.pos_ =="AUX" and token.nbor().pos_ == "VERB"):
            return token.text + " "+ token.nbor().text
        elif (token.pos_ == "VERB" or token.lemma_ == "ser"):
            return token.text

def getObjectsFromSentence(sentence):
    for token in sentence:
        if token.dep_ =="obj":
            return token.text


def get_ent(sentence):
    return list(ent.text for ent in sentence.ents)

def getEntities(sentence):
    pos_verb= getVerbPosition(sentence)

    pair=list() #tiene que tener 2 elementos

    #sujeto | entidades reconocidas - matcher
    recognizedEntities= get_ent(sentence[0:pos_verb])
    if len(recognizedEntities) >= 1: #if there are recognized entities
        pair.append(recognizedEntities[0])   
    else:
        matches = matcher(sentence[0:pos_verb])
        for match_id, start, end in matches:
            span = sentence[start:end]  
        pair.append(span.text) # add the lastest

    #predicado | entidades reconocidas - objeto | casos extras
    recognizedEntities= get_ent(sentence[pos_verb:len(sentence)])
    if len(recognizedEntities) >= 1: 
        pair.append(recognizedEntities[0])
    elif getObjectsFromSentence(sentence) != None:
        pair.append(getObjectsFromSentence(sentence))
    else:
        pair.append([token.text for token in sentence if token.dep_ =="ROOT" and token.pos_ =="NOUN"][0])
    
    return pair    
        
           
def sentences_parser(paragraph):
    candidate_sent= list()
    paragraph= filter(None,paragraph.split("."))
    for each in paragraph:
        candidate_sent.append(nlp(each.lstrip()))
    return candidate_sent

#----------------------  
doc = "La empresa se llama Dublin. La empresa es conocida por sus travesías en kayak. Las travesías en kayak tienen duración."
doc= sentences_parser(doc)

entities= list()


dataset= list()
for sentence in doc:    
    #sujeto-predicado-objeto
    triplestore = (getEntities(sentence)[0], getRelation(sentence), getEntities(sentence)[1])

    if (nlp(getRelation(sentence))[0].lemma_ == "tener"):
        dataset.append((getEntities(sentence)[0], "hasProperty", getEntities(sentence)[1]))

    dataset.append(triplestore)

    entities.append(getEntities(sentence)[0])
    entities.append(getEntities(sentence)[1])
    
ocurrencesOfEntity= Counter(entities)
for i in ocurrencesOfEntity:
    if ocurrencesOfEntity[i] >= 2:
        triplestore= (i, "typeoOf", "class")
        dataset.append(triplestore)

print(dataset)
               