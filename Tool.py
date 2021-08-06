#!/usr/bin/env python
# coding: utf-8
import spacy
import os

nlp = spacy.load('es_core_news_md')

ruler = nlp.add_pipe("entity_ruler")
ruler.add_patterns([{'label': 'ORG', 'pattern': 'Hoy duermo afuera'}])
ruler.add_patterns([{'label': 'ORG', 'pattern': 'travesías en kayak'}])

#---------------------- NEW VERSION ------------
sentWithOI= list()
sentWithoutOI = list()

def getObjectsFromSentence(sentence):
    return [token.text for token in sentence if token.dep_ =="obj"]

def categorizeSentence(sentence):
    if len(getObjectsFromSentence(sentence)) == 1: #one object means that it has only do
        sentWithoutOI.append(sentence)
    elif len(getObjectsFromSentence(sentence)) == 2: #two objects mean that it has io and do
        sentWithOI.append(sentence)

def get_ent(sentence):
    return list(ent.text for ent in sentence.ents)

def get_entities(sentencesList):
    entity_pairs= list()
    for each in sentencesList:
        recognizedEntities= get_ent(each)
        print(each, recognizedEntities)
        if len(recognizedEntities) >= 1:
            entity_pairs.append(recognizedEntities)
        
def sentences_parser(paragraph):
    candidate_sent= list()
    paragraph= filter(None,paragraph.split("."))
    for each in paragraph:
        candidate_sent.append(nlp(each.lstrip()))
    return candidate_sent

#----------------------  
doc = "Hoy duermo afuera es una empresa. La empresa ofrece travesías en kayak a los kayakistas. Las travesías en kayak tienen duración."
doc= sentences_parser(doc)
for i in doc:
    categorizeSentence(i)
    # print("---------")
    # for x in i:
    #     print(x.dep_, end = ' ')
#get_entities(doc)