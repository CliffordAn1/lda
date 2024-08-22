import numpy as np
import string
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

file = open("data.csv", "r", encoding='utf-8')

#stopwords (words which are shared by most documents, thus carry little semantic meaning) were manually entered, but it might be worthwhile to test if changing stopwords gives better results
stopwords = [
    "", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and",
    "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "january", "february", "march", "april", "may", "june", 
    "july", "august", "september", "october", "november", "december"
]

reports = []
#read file
lines = file.readlines()
for line in lines:
    report = line.split('\\')
    report[2] = report[2].lower()
    report[2] = report[2].translate(str.maketrans('', '', string.punctuation))
    report[2] = report[2].replace("\n", "")
    report[2] = report[2].split(" ")
    report[2] = [word for word in report[2] if (word not in stopwords and not word.isnumeric() and len(word) != 1)]
    reports.append(report)

iterations = 100 #number of iterations for Gibbs sampling
numtopics = 4
numdocs = 300
alpha = 0.05 #hyperparameter for document-topic distribution
beta = 0.05 #hyperparameter for word-topic distribution

ndk = np.zeros((numdocs, numtopics)) #number of words assigned to topic k in document d
nk = np.zeros(numtopics) #number of times any word is assigned to topic k
z = [] #topic allocation for every word in the corpus
words = [] #array of every single word in the corpus
temp = np.array([])
numwords = 0
    
for j in range(numdocs):
    words.append(reports[j][2])
    temp = np.append(temp, reports[j][2])

unique = set(temp)
test = list(unique)
nkw = np.zeros((len(unique), numtopics)) #number of times word w is assigned to topic k

#initialize ndk, nk, and nkw
for doc in range(numdocs):
    alloc = np.random.randint(0, numtopics, len(words[doc]))
    z.append(alloc)
    for word in range(len(words[doc])):
        numwords += 1
        ndk[doc][alloc[word]] += 1
        nk[alloc[word]] += 1
        nkw[test.index(words[doc][word])][alloc[word]] += 1

#Gibbs sampler (taken from pseudocode in paper)
for i in range(iterations):
    for j in range(numdocs):
        for k in range(len(z[j])):
            wordindex = test.index(words[j][k])
            ndk[j, z[j][k]] -= 1
            nk[z[j][k]] -= 1
            nkw[wordindex][z[j][k]] -= 1
            p = np.zeros(numtopics)
            for l in range(numtopics):
                p[l] = ((ndk[j, l] + alpha) * (nkw[wordindex][l] + beta)) / (nk[l] + (beta * numwords))
            p = p / np.sum(p)
            sample = np.random.choice(a=len(p), p=p)
            z[j][k] = sample
            ndk[j, sample] += 1
            nk[sample] += 1
            nkw[wordindex][sample] += 1

for i in range(numtopics):
    for rep in np.argpartition(ndk[:, i], -5)[-5:]:
        print(reports[rep][0])
    print()

for i in range(numtopics):
    for rep in np.argpartition(nkw[:, i], -5)[-5:]:
        print(test[rep])
        print(nkw[rep][i])
    print()

xb = np.array([])
yb = np.array([])
dz = np.array([])
numpoints = 0
x1 = np.array([])
y1 = np.array([])
x2 = np.array([])
y2 = np.array([])
x3 = np.array([])
y3 = np.array([])
x4 = np.array([])
y4 = np.array([])
for i in range(numdocs):
    ndk[i] = ndk[i] / np.sum(ndk[i])
for i in range(13):
    aggregate = np.zeros(numtopics)
    total = 0
    for j in range(numdocs):
        if int(reports[j][1]) >= 1971 + (2 * i) and int(reports[j][1]) < 1971 + (2 * (i + 1)):
            aggregate += ndk[j]
            total += 1
    aggregate = aggregate / total
    for k in range(numtopics):
        xb = np.append(xb, k)
        yb = np.append(yb, 1972 + (2 * i))
        dz = np.append(dz, aggregate[k])
        numpoints += 1
    x1 = np.append(x1, 1972 + (2 * i))
    y1 = np.append(y1, aggregate[0])
    x2 = np.append(x2, 1972 + (2 * i))
    y2 = np.append(y2, aggregate[1])
    x3 = np.append(x3, 1972 + (2 * i))
    y3 = np.append(y3, aggregate[2])
    x4 = np.append(x4, 1972 + (2 * i))
    y4 = np.append(y4, aggregate[3])
zb = np.zeros(numpoints)
dx = np.ones(numpoints) / 5
dy = np.ones(numpoints)

figs, axs = plt.subplots(2, 2)
axs[0, 0].bar(x1, y1, color='red')
axs[0, 0].set_title('Topic 1')
axs[0, 1].bar(x2, y2, color='blue')
axs[0, 1].set_title('Topic 2')
axs[1, 0].bar(x3, y3, color='green')
axs[1, 0].set_title('Topic 3')
axs[1, 1].bar(x4, y4, color='orange')
axs[1, 1].set_title('Topic 4')

for ax in axs.flat:
    ax.set(xlabel='Year', ylabel='Proportion to topic')
    ax.set_ylim(0, 1)

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

colors = ['red', 'blue', 'green', 'orange']
colormap = dict(zip(xb, colors))
barcolors = [colormap[xval] for xval in xb]
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

for xi, yi, zi, dxi, dyi, dzi, color in zip(xb, yb, zb, dx, dy, dz, barcolors):
    ax1.bar3d(xi, yi, zi, dxi, dyi, dzi, color=color)


ax1.set_xlabel('Topics')
ax1.set_ylabel('Time')
ax1.set_zlabel('Probability')

plt.show()