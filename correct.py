import nltk
from nltk import bigrams, FreqDist
import re
import os
from math import log



# 函数，创建confusion Matrix
confusionMatrix_ins = {}
confusionMatrix_del = {}
confusionMatrix_sub = {}
confusionMatrix_trans = {}
unigramDict = {} #noisy channel model分母
bigramDict = {}



# bi-gram模型
def LM():
    dataset = open('./ans.txt','r') #语料
    train_data = []
    unigram = {}
    bigram = {}
    bigram_freq = {}
    voc = []
    vocSize = 0
    # 导入词典
    with open('vocab.txt', 'r') as f:
        for line in f:
            voc.append(line.strip())
            voc.append(line.lower().strip())
    vocSize = len(voc)
    
    # 获取训练语料
    for i in range(1000):
        line = dataset.readline().split('\t')[1]
        train_data.append(line)

    for sentence in train_data:
        senlist = nltk.word_tokenize(sentence)
        senlist = ['sos'] + senlist #添加开始
        word_freq = nltk.FreqDist(senlist)
        for word in word_freq:
            if word not in unigram:
                unigram[word] = word_freq[word]
            else:
                unigram[word] += word_freq[word]
    # bi-gram
    for sentence in train_data:
        senlist = nltk.word_tokenize(sentence)
        # 添加开始
        senlist = ['sos'] + senlist
        word_freq = nltk.FreqDist(nltk.bigrams(senlist))
        for biword in word_freq:
            if biword not in bigram:
                bigram[biword] = word_freq[biword]
            else:
                bigram[biword] += word_freq[biword]
    # 概率 add-1 smooth
    for biword in bigram:
        bigram_freq[biword] = (bigram[biword]+1)/(unigram[biword[0]]+vocSize) 
    return train_data,voc,bigram_freq

# 计算PPL
def PPL(data,bigram_freq):
    pp = []
    for sentence in data:
        logprob = 0
        wt = 0
        senlist = nltk.word_tokenize(sentence)
        senlist = ['sos'] + senlist
        
        for word in nltk.bigrams(senlist):
            if word in bigram_freq:
                logprob += log(bigram_freq[word],2)
                wt += 1
        if wt > 0:
            pp.append([sentence,pow(2,-(logprob/wt))])
    temp = 0
    for i in pp:
        temp += i[1]
    print("二元语法模型的困惑度:", temp/len(pp))


# 获取候选，所有1或2 edit-distance
def getCandidates(word): 
    L1 = known([word])
    L2_tmp = editOneDistance(word)
    L2 = known(L2_tmp)
    L3 = known(editsTwoDistance(word))
    L4 = set([word])
    L = L1|L2|L3|L4
    return L
    #return set(known([word]) or known(editOneDistance(word)) or known(editsTwoDistance(word)) or [word])
# 是否在词典中
def known(words): 
    L = []
    for i in words:
        if i in voc:
            #print(i)
            L.append(i)
    return set(L)
# 获取所有1-edit-distance候选
def editOneDistance(word):
    letters    = 'abcdefghijklmnopqrstuvwxyz-'
    splits     = [(word[:i], word[i:])  for i in range(len(word) + 1)]
    deletes    = [L + R[1:]   for L, R in splits if R]
    trans1 = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    trans2 = [L + R[2] + R[1] + R[0] + R[3:] for L, R in splits if len(R)>2]
    trans = trans1 + trans2
    subs   = [L + c + R[1:]  for L, R in splits if R for c in letters]
    inserts    = [L + c + R  for L, R in splits for c in letters]
    return set(deletes + trans + subs + inserts)

def editsTwoDistance(word): 
    return set(e2 for e1 in editOneDistance(word) for e2 in editOneDistance(e1))


# 创建四个confusion Matrix
def createConfusionMatrix(realword,misword):
    """
        N：未编辑;I：插入;D: 删除;T: trans;S：sub
        realword：正确单词
        misword：错误单词
        计算Confusion Matrix和uni/bigramDict
    """
    lenreal = len(realword)
    lenmis = len(misword)
    dp = [[0 for i in range(lenmis+1)] for j in range(lenreal+1)] 
    #编辑距离，dp[i][j]表示realword到i之前，misword到j之前的minimal edit-distance
    pre = [[' ' for i in range(lenmis + 1)] for j in range(lenreal+1)] #追溯矩阵，记录
    pre[0][0] = 'N'
    
    for i in range(lenreal+1):
        dp[i][0] = i
        pre[i][0] = 'D'
        
    for i in range(lenmis+1):
        dp[0][i] = i
        pre[0][i] = 'I'
        
    for i in range(1,lenreal+1):
        for j in range(1,lenmis+1):
            if realword[i-1] == misword[j-1]: #未编辑
                dp[i][j] = dp[i-1][j-1]
                pre[i][j] = 'N'
            else:
                delCost = dp[i-1][j] + 1
                insCost = dp[i][j-1] + 1
                if delCost <= insCost:
                    dp[i][j] = delCost
                    pre[i][j] = 'D'
                else:
                    dp[i][j] = insCost
                    pre[i][j] = 'I'
                subCost = dp[i-1][j-1] + 1 #count
                if subCost < dp[i][j]:
                    dp[i][j] = subCost
                    pre[i][j] = 'S'
            # 计算反转，如果有trans
            if (i>1) and (j>1) and (realword[i-1] == misword[j-2]) and (realword[i-2] == misword[j-1]):
                transCost = dp[i-2][j-2] + 1 
                if transCost < dp[i][j]:
                    dp[i][j] = transCost
                    pre[i][j] = 'T'
        
    i = lenreal
    j = lenmis
    firstChar = ' '
    secondChar = ' '
    # 确定编辑类型，根据回溯矩阵，计算confusion Matrix
    while i>0 and (j+1>0):
        if pre[i][j] == 'I': # Ins
            if i > 0:
                firstChar = realword[i-1]
            else:
                firstChar = '#'
            secondChar = misword[j-1]
            bigram = firstChar + secondChar
            if bigram not in confusionMatrix_ins:
                confusionMatrix_ins[bigram] = 1
            else:
                confusionMatrix_ins[bigram] += 1
            j = j - 1
        elif pre[i][j] == 'D': #del
            if i > 1:
                firstChar = realword[i-2]
            else:
                firstChar = '#'
            secondChar = realword[i-1]
            bigram = firstChar+secondChar
            if bigram not in confusionMatrix_del:
                confusionMatrix_del[bigram] = 1
            else:
                confusionMatrix_del[bigram] += 1
            i = i - 1
        elif pre[i][j] == 'T': # trans
            firstChar = realword[i-2]
            secondChar = realword[i-1]
            bigram = firstChar+secondChar
            if bigram not in confusionMatrix_trans:
                confusionMatrix_trans[bigram] = 1
            else:
                confusionMatrix_trans[bigram] += 1
            i = i-2
            j = j-2
        else:  # N，未编辑
            if pre[i][j] == 'S': #sub
                firstChar = realword[i-1] 
                secondChar = misword[j-1]
                bigram = firstChar + secondChar
                if bigram not in confusionMatrix_sub:
                    confusionMatrix_sub[bigram] = 1
                else:
                    confusionMatrix_sub[bigram] += 1
            i = i-1
            j = j-1
                
                
# 创建分母
def createBigrams(word):
    if "#" not in unigramDict: # adds @ for empty 
        unigramDict["#"]=1
    else:   
        unigramDict["#"]=unigramDict["#"] + 1
    prevChar = "#"
    for currChar in word:
        if currChar not in unigramDict:
            unigramDict[currChar] = 1
        else:
            unigramDict[currChar] = unigramDict[currChar] + 1
        bigram = prevChar + currChar    
        if bigram not in bigramDict:
            bigramDict[bigram] = 1
        else:   
            bigramDict[bigram] = bigramDict[bigram] + 1
        prevChar = currChar

# 计算misword同realword的最小edit-distance，对齐，计算概率P(x|w)
def calProbility(misword,realword):
    
    lenreal = len(realword)
    lenmis = len(misword)
    dp = [[0 for i in range(lenmis+1)] for j in range(lenreal+1)] 
    #编辑距离，dp[i][j]表示realword到i之前，misword到j之前的min-edit-distance
    pre = [[' ' for i in range(lenmis + 1)] for j in range(lenreal+1)] #追溯矩阵，记录
    pre[0][0] = 'N'

    for i in range(lenreal+1):
        dp[i][0] = i
        pre[i][0] = 'D'

    for i in range(lenmis+1):
        dp[0][i] = i
        pre[0][i] = 'I'


    for i in range(1,lenreal+1):
        for j in range(1,lenmis+1):
            if realword[i-1] == misword[j-1]: #未编辑
                dp[i][j] = dp[i-1][j-1]
                pre[i][j] = 'N'
            else:
                delCost = dp[i-1][j] + 1
                insCost = dp[i][j-1] + 1
                if delCost <= insCost:
                    dp[i][j] = delCost
                    pre[i][j] = 'D'
                else:
                    dp[i][j] = insCost
                    pre[i][j] = 'I'
                subCost = dp[i-1][j-1] + 1 #加1还是加2比较好
                if subCost < dp[i][j]:
                    dp[i][j] = subCost
                    pre[i][j] = 'S'
            # 计算反转，如果有trans
            if (i>1) and (j>1) and (realword[i-1] == misword[j-2]) and (realword[i-2] == misword[j-1]):
                transCost = dp[i-2][j-2] + 1 #注意，加1还是2
                if transCost < dp[i][j]:
                    dp[i][j] = transCost
                    pre[i][j] = 'T'
        
    prob = 1.0
    # 回溯判断类型，找概率
    i = lenreal
    j = lenmis
    firstChar = ' '
    secondChar = ' '
    
    # 确定编辑类型，根据回溯矩阵，计算confusion Matrix
    while i>0 and (j+1>0):
        if pre[i][j] == 'I': # Ins
            if i > 0:
                firstChar = realword[i-1]
            else:
                firstChar = '#'
            secondChar = misword[j-1]
            bigram = firstChar + secondChar

            prob = prob * (0 if bigram not in confusionMatrix_ins else confusionMatrix_ins[bigram]/unigramDict[firstChar])               
            j = j - 1
        elif pre[i][j] == 'D': #del
            if i > 1:
                firstChar = realword[i-2]
            else:
                firstChar = '#'
            secondChar = realword[i-1]
            bigram = firstChar+secondChar

            prob = prob * (0 if bigram not in confusionMatrix_del else confusionMatrix_del[bigram]/bigramDict[bigram])  
            i = i - 1
        elif pre[i][j] == 'T': # trans
            firstChar = realword[i-2]
            secondChar = realword[i-1]
            bigram = firstChar+secondChar
            prob = prob * (0 if bigram not in confusionMatrix_trans else confusionMatrix_trans[bigram]/bigramDict[bigram])                    
            i = i-2
            j = j-2
        else:  # N，未编辑
            if pre[i][j] == 'S': #sub
                firstChar = realword[i-1]
                secondChar = misword[j-1]
                bigram = firstChar + secondChar
                prob = prob * (0 if bigram not in confusionMatrix_sub else confusionMatrix_sub[bigram]/unigramDict[firstChar])    
            i = i-1
            j = j-1               

    return prob


# 更正一个句子，errorNum错误数量
def correctSentence(sentence,errorNum):
    tmp = nltk.word_tokenize(sentence)
    x = nltk.word_tokenize(sentence)
    #transtab = sentence.maketrans(sentence.lower(),sentence)
    #print(x)
    x = ['sos'] + x
    final = []
    maxWords = []
    for i in range(1,len(x)):
        result = []
        if x[i] in voc:
            continue        
        
        # 获取edit distance为1或2的所有候选单词
        cand = getCandidates(x[i])
        # 对每个候选单词，计算条件概率
        for c in cand:
            if str(c) == str(x[i]):
                prob = (len(x)-1-errorNum)/(len(x)-1)
            else:
                prob = calProbility(str(c),str(x[i]))
            bi = (str(x[i-1]),str(c))
            #rob = prob * (freq[bi] if bi in freq else 0.0)
            if bi in freq:
                prob = prob * freq[bi]
            else:
                prob = 0
            result.append((c,prob))

        
        # 有prob不为0的候选项， 若无，则优先选择P(x|w)
        flag = 0
        for rec,rep in result:
            if rep>0.0:
                flag = 1
                break
        result2 = []
        if flag == 0: #如果LM没有概率
            for rec,rep in result:
                prob = calProbility(str(rec),str(x[i]))
                result2.append((rec,prob))
            result = result2
        
        # 如果仍然未更正用LM更正
        flag = 0
        for rec,rep in result:
            if rep>0.0:
                flag = 1
                break
        result2 = []
        if flag==0: #如果P(x|w)=0
            for rec,rep in result:
                bi = (str(x[i-1]),str(rec))
                if bi in freq:
                    prob = freq[bi]
                else:
                    prob = 0
                result2.append((rec,prob))
            result = result2
        
        maxCorrect = ' '
        maxProb = -1
        for j1,j2 in result:
            if j2 > maxProb:
                maxProb = j2
                maxCorrect = j1
        
        if (maxProb>=0) and (maxCorrect in voc):
            maxWords.append((i,x[i],maxCorrect,maxProb))
    #print(maxWords)
    # maxWords按概率排序
    correctSentence = sentence
    if (len(maxWords) >= errorNum):
       
        maxWords.sort(key= lambda k:k[3])
        maxWords.reverse()   
        # 在sentence上更改misspelled   
        for i in range(errorNum):
            index,old,correct,prob = maxWords[i]            
            correctSentence = correctSentence.replace(str(old),str(correct))
    return correctSentence



# 入口
if __name__ == '__main__':
    

    # Bi-gram LM
    train_data,voc,freq = LM()

    # 导入语料，创建confusion Matrix
    errorCorpus = "/errors.txt"
    file = open(os.getcwd() + errorCorpus,'r')
    for line in file:
        parts = re.split(r'\W\s',line.lower())
        parts[-1] = parts[-1].strip()         
        createBigrams(parts[0])                  
        
        times = 1
        for i in range(1,len(parts)):
            misspelled = parts[i]
            pairNoWord = re.split(r'\*', misspelled) 
            createBigrams(pairNoWord[0].strip())
            if len(pairNoWord) > 1:
                misspelled = pairNoWord[0].strip()
                times = int(pairNoWord[1])
            for i in range(times):
                createConfusionMatrix(parts[0], misspelled)

    file.close()


    # 更正，写文件
    test_data = os.getcwd()+'/testdata.txt'

    fp = open(test_data,'r')
    newFile = open('result.txt','a+')

    sens = []
    for i in range(1000):
        tmp = fp.readline().strip()
        sens.append(tmp)
    for i in range(0,1000):
        print(i)
        tmp = str(sens[i])
        sentence = tmp.split('\t')[2]
        errorNum = tmp.split('\t')[1]
        correctedSentence = correctSentence(sentence,int(errorNum))
        newFile.write(str(i+1)+'\t'+correctedSentence+'\n') 
        newFile.flush()
    newFile.close()