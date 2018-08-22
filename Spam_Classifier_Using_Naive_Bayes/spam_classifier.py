from bayes import createVocabList, setOfWords2Vec, trainNBO, classifyNB
import random


def textParse(bigString):
    import re
    listOfTokens = re.split('//W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainingMat = []
    trainingClasses = []
    for docIndex in trainingSet:
        trainingMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainingClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNBO(trainingMat, trainingClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(wordVector, p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print ("the error rate is: ", float(errorCount)/len(testSet))


spamTest()
