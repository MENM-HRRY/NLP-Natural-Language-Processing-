# NLP (Natural language processing)


#####  Key Concepts, text data cleaning

Corpus refers to a set of texts that are used to train a spam classification model.
```
corpus = ["Jack stole my tuna sandwich.", 
    "'Help!' I sobbed, sandwichlessly.", 
    "'Drop the sandwiches!' said the sandwich police."]
```
For example, you could use this corpus to train a text classification model so that it can distinguish between texts that talk about sandwiches and texts that don't.

### Convert a corpus of documents into a matrix of characteristics, that is, Words --> numbers.

If you want to apply a stemming algorithm, you can provide a suitable stemmer object as a value for this parameter. Some common stemmer options include:

PorterStemmer: Implements the Porter stemming algorithm, one of the oldest and most widely used stemming algorithms.
SnowballStemmer: Provides support for multiple languages ​​and stemming algorithms through the Snowball library.


The clipping of some words from the sublists is due to the stemming process you are performing in the our_tokenizer function using SnowballStemmer('english'). Stemming is the process of reducing words to their root or base, which often involves removing suffixes and prefixes.
  
       [['jack', 'stole', 'tuna', 'sandwich'],
        ['help', 'sobbed', 'sandwichlessly'],
        ['drop', 'sandwiches', 'said', 'sandwich', 'police']]

In the specific case you've shown, the word "sobbed" was reduced to "sob" and "sandwichlessly" was reduced to "sandwichless" after stemming was applied. This is normal in the stemming process, as it seeks to reduce words to their basic form.


```
[['jack', 'stole', 'tuna', 'sandwich'],
 ['help', 'sob', 'sandwichless'],
 ['drop', 'sandwich', 'said', 'sandwich', 'polic']]
```

####  Count Vectorizer, TFIDF 

Vocabulary:
```
['drop', 'help', 'jack',  'police', 'said', 'sandwich', 'sandwichlessly', 'sobbed', 'stole','tuna']

```

```
['jack', 'stole', 'sandwich','tuna']
[0, 0, 1, 1, 0, 0, 1, 0, 0, 1]

['help', 'sobbed', 'sandwichlessly']
[0, 1, 0, 0, 0, 0, 0, 1, 1, 0]

['drop', 'sandwich', 'said', 'sandwich', 'police']
[1, 0, 0, 0, 1, 1, 2, 0, 0, 0]
```

## Term frequency
$$
TF_{\text{word,document}} = \frac{\text{\N of times word appears in document}}{\text{total \N of words in document}}
$$


```
['jack', 'stole', 'sandwich', 'tuna']
[0, 0, 1/4, 1/4, 0, 0, 1/4, 0, 0, 1/4]

['help', 'sobbed', 'sandwichlessly']
[0, 1/3, 0, 0, 0, 0, 0, 1/3, 1/3, 0]

['drop', 'sandwich', 'said', 'sandwich', 'police']
[1/5, 0, 0, 0, 1/5, 1/5, 2/5, 0, 0, 0]
```
## Document frequency
$$
DF_{\text{word}} = \frac{\text{\N of documents containing word}}{\text{total \N of documents}}
$$


Vocabulary:
```
['drop', 'help', 'jack', 'police', 'said', 'sandwich', 'sandwichlessly', 'sobbed', 'stole', 'tuna']
```

Document frequency for each word:
```
[1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 2/3, 1/3, 1/3, 1/3]
```


## Inverse document frequency
$$ IDF_{word} = \log\left(\frac{total\_\#\_of\_documents}{\#\_of\_documents\_containing\_word}\right) $$


Vocabulary:
```
['drop', 'help', 'jack', 'police', 'said', 'sandwich', 'sandwichlessly', 'sobbed', 'stole','tuna']
```

IDF for each word:
```
[1.099, 1.099, 1.099, 1.099, 1.099, 1.099, 0.405, 1.099, 1.099, 1.099]
```
# TFIDF

Term Frequency-Inverse Document Frequency) es una técnica utilizada comúnmente en recuperación de información y minería de texto para evaluar la importancia relativa de un término dentro de un documento en una colección de documentos


Vocabulary:
```
['drop', 'help', 'jack', 'police', 'said', 'sandwich', 'sandwichlessly', 'sobbed', 'stole','tuna']
```
TF * IDF:

The relative importance of a term in a specific document compared to the entire collection of documents.


```
['jack', 'stole', 'sandwich', 'tuna']
[0, 0, 0.275, 0.275, 0, 0, 0.101, 0, 0, 0.275]


['help', 'sobbed', 'sandwichlessly']
[0, 0.366, 0, 0, 0, 0, 0, 0.366, 0.366, 0]

['drop', 'sandwich', 'said', 'sandwich', 'police']
[0.22, 0, 0, 0, 0.22, 0.22, 0.162, 0, 0, 0]
```
['jack', 'stole', 'sandwich', 'tuna']: This document has terms such as 'sandwich' and 'tuna' that are relatively uncommon in the full collection of documents but appear frequently in this particular document . Therefore, they have a higher TF-IDF value compared to other terms such as 'jack' and 'stole', which may be more common in other documents and therefore have a lower TF-IDF value.

['help', 'sobbed', 'sandwichlessly']: In this document, 'sobbed' and 'sandwichlessly' are terms that have a high TF-IDF value because they are relatively rare in the collection. 'help' also has a significant TF-IDF value, as it is less common but still appears in this document.

['drop', 'sandwich', 'said', 'sandwich', 'police']: Here, 'drop', 'said', 'sandwich', and 'police' have a higher TF-IDF value due to its rarity in the collection as a whole and its frequency in this particular document. 'sandwich' is common in this document, but it appears in other documents as well, so its TF-IDF value is lower compared to the other terms. Now that the DOCUMENTS have been converted to VECTORS, it is feasible to use them in any machine learning algorithm we need! We can use any type of similarity measure we want!


### Tweak model with Spam data

To evaluate and compare different classification models using confusion matrices, we consider the following main metrics: **accuracy**, **sensitivity (recall or TPR)**, **specificity (TNR)**, and **precision (PPV)**.

### Confusion Matrix

- **True Positives (TP)**: Elements correctly predicted as positive.
- **False Positives (FP)**: Elements incorrectly predicted as positive.
- **True Negatives (TN)**: Elements correctly predicted as negative.
- **False Negatives (FN)**: Elements incorrectly predicted as negative.

### Metrics

- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Sensitivity (Recall or TPR)**: TP / (TP + FN)
- **Precision (PPV)**: TP / (TP + FP)
- **Specificity (TNR)**: TN / (TN + FP)

#### Option a)

$$
\begin{bmatrix} 
1200 & 0 \\ 
37 & 156 
\end{bmatrix}
$$

- TP: 156, FP: 0, TN: 1200, FN: 37
- Accuracy: (1200 + 156) / (1200 + 156 + 0 + 37) = 0.973
- Recall: 156 / (156 + 37) = 0.808
- Precision: 156 / (156 + 0) = 1.0
- Specificity: 1200 / (1200 + 0) = 1.0

#### Option b)

$$
\begin{bmatrix} 
1194 & 3 \\ 
70 & 126 
\end{bmatrix}
$$

- TP: 126, FP: 3, TN: 1194, FN: 70
- Accuracy: (1194 + 126) / (1194 + 126 + 3 + 70) = 0.955
- Recall: 126 / (126 + 70) = 0.643
- Precision: 126 / (126 + 3) = 0.977
- Specificity: 1194 / (1194 + 3) = 0.998

#### Option c)

$$
\begin{bmatrix} 
1206 & 0 \\ 
119 & 68 
\end{bmatrix}
$$

- TP: 68, FP: 0, TN: 1206, FN: 119
- Accuracy: (1206 + 68) / (1206 + 68 + 0 + 119) = 0.915
- Recall: 68 / (68 + 119) = 0.364
- Precision: 68 / (68 + 0) = 1.0
- Specificity: 1206 / (1206 + 0) = 1.0

#### Option d)

$$
\begin{bmatrix} 
1209 & 0 \\ 
40 & 144 
\end{bmatrix}
$$

- TP: 144, FP: 0, TN: 1209, FN: 40
- Accuracy: (1209 + 144) / (1209 + 144 + 0 + 40) = 0.973
- Recall: 144 / (144 + 40) = 0.783
- Precision: 144 / (144 + 0) = 1.0
- Specificity: 1209 / (1209 + 0) = 1.0

#### Option e)

$$
\begin{bmatrix} 
1202 & 7 \\ 
33 & 151 
\end{bmatrix}
$$

- TP: 151, FP: 7, TN: 1202, FN: 33
- Accuracy: (1202 + 151) / (1202 + 151 + 7 + 33) = 0.975
- Recall: 151 / (151 + 33) = 0.821
- Precision: 151 / (151 + 7) = 0.956
- Specificity: 1202 / (1202 + 7) = 0.994

#### Option i)

$$
\begin{bmatrix} 
493 & 1 \\ 
3 & 61 
\end{bmatrix}
$$

- TP: 61, FP: 1, TN: 493, FN: 3
- Accuracy: (493 + 61) / (493 + 61 + 1 + 3) = 0.991
- Recall: 61 / (61 + 3) = 0.953
- Precision: 61 / (61 + 1) = 0.984
- Specificity: 493 / (493 + 1) = 0.998

### Model Evaluation

Comparing the calculated metrics, the model with the **highest accuracy** is the following:

- **Option i)**
    - Accuracy: 0.991
    - Recall: 0.953
    - Precision: 0.984
    - Specificity: 0.998

This model has the highest accuracy, high sensitivity, high precision, and high specificity. Therefore, **Option i)** is the best choice among the presented models.


## Pipeline with Spam data 

The model you are using to classify phrases as "spam" or "ham" (not spam) is making several incorrect predictions. Most phrases that would typically be considered spam are being classified as "ham." Here is a list of the phrases with their classification corrected for what is typically considered spam.
The vast majority of these phrases contain keywords and phrases that are typically associated with spam emails and messages, so they should be classified as "spam" by a well-trained classification model. If the model you are using is not providing the correct predictions, it may be necessary to adjust the model training or use a more robust data set to improve its accuracy.
