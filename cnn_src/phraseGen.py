from gensim.models import Word2Vec
import eval_data_helpers

#################### config ###################
modelfile = "../wvmodel/size300window5sg1min_count100negative10iter50.model"
pattern_file = "../data/eval_data.txt"
output_file = "../data/phrase_gen.txt"
############### end of config #################

model = Word2Vec.load(modelfile)

data_size, data, label = eval_data_helpers.load_data(pattern_file)

word_freq = {}
for sentence in data:
    for word in sentence.split():
        if word in word_freq:
            word_freq[word] = word_freq[word] + 1
        else:
            word_freq[word] = 1

word_mostsim = {}
for word, freq in word_freq.items():
    if word in model.wv.vocab:
        word_mostsim[word] = model.wv.most_similar(word)

with open(output_file, 'w') as f:
    for word, sim_word in word_mostsim.items():
        f.write(word)
        f.write('\t')
        f.write(str([item[0] for item in sim_word]))
        f.write('\n')

