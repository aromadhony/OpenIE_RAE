import os
import numpy as np
import gensim

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("D:\\vdi\\shared\\GoogleNews-vectors-negative300.bin.gz", binary=True)
index2wordw2v_set = set(word2vec_model.index2word)

def avg_feature_vector(words, model, num_features, index2word_set):
    #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    #list containing names of words in the vocabulary
    #index2word_set = set(model.index2word) this is moved as input param for performance reasons
    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])

    if(nwords>0):
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def load_and_get_avgvector(input_path):
    #jumlah sample masih hard-coded
    subj_vectors = np.zeros((70, 300))
    trigger_vectors = np.zeros((70, 300))
    obj_vectors = np.zeros((70, 300))
    with open(input_path,"r") as fi_reltuple:
        words = fi_reltuple.readlines();
        content = [x.strip() for x in words] 
        lenLines = len(content)
        for countLines in range(0,lenLines):
            arrLine = words[countLines].strip().split("\t")
            #print("wordLines = "+words[countLines])
            #print("counter_synset="+str(counter_synset))
            subj = arrLine[0];
            trigger = arrLine[1];
            obj = arrLine[2];
            subj_avg_vector = avg_feature_vector(subj.split(), model=word2vec_model, num_features=300, index2word_set = index2wordw2v_set)
            trigger_avg_vector = avg_feature_vector(trigger.split(), model=word2vec_model, num_features=300, index2word_set = index2wordw2v_set)
            obj_avg_vector = avg_feature_vector(obj.split(), model=word2vec_model, num_features=300, index2word_set = index2wordw2v_set)
            
            subj_vectors[countLines] = subj_avg_vector
            trigger_vectors[countLines] = trigger_avg_vector
            obj_vectors[countLines] = obj_avg_vector
    fi_reltuple.close()
    return subj_vectors, trigger_vectors, obj_vectors

def main():
    X1, X2, X3 = load_and_get_avgvector("astretest_reltuple_sample_70.txt")
    np.save("x1_astretestsample_70",X1)
    np.save("x2_astretestsample_70",X2)
    np.save("x3_astretestsample_70",X3)

if __name__ == "__main__": main()
