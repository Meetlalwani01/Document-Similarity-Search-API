# 20 Newsgroups Dataset 

This repository utilizes the **20 Newsgroups** dataset, a widely adopted benchmark for text classification and natural language processing tasks. The dataset comprises approximately 20,000 newsgroup posts, divided evenly among 20 different topics.
## Overview
The **20 Newsgroups** dataset provides a rich resource of textual data extracted from newsgroup posts. Each post contains unstructured text data that can include headers, the body of the message, and sometimes footers. This raw text data is typically pre-processed to remove noise (like headers or quotes) before being used in machine learning models.
## Dataset Structure
The key components of the dataset are as follows:
### 1. Data (Text)
- **Description:**  
  The `data` attribute is a list of raw text documents, where each document corresponds to a newsgroup post.
### 2. Target Labels
- **Description:**  
  The `target` attribute is an array of integer labels. Each integer (ranging from 0 to 19) represents a specific newsgroup category.
### 3. Target Names
- **Description:**  
  The `target_names` attribute is a list of the 20 newsgroup categories corresponding to the integer labels in `target`.
- **Categories Include:**
  - alt.atheism
  - comp.graphics
  - comp.os.ms-windows.misc
  - comp.sys.ibm.pc.hardware
  - comp.sys.mac.hardware
  - comp.windows.x
  - misc.forsale
  - rec.autos
  - rec.motorcycles
  - rec.sport.baseball
  - rec.sport.hockey
  - sci.crypt
  - sci.electronics
  - sci.med
  - sci.space
  - soc.religion.christian
  - talk.politics.guns
  - talk.politics.mideast
  - talk.politics.misc
  - talk.religion.misc
### 4. Additional Attributes
- **Filenames (Optional):**  
  Some versions of the dataset may include a `filenames` attribute, which provides file paths or unique identifiers for each document.
- **DESCR:**  
  The `DESCR` attribute contains an in-depth description of the dataset, detailing its origin, structure, and typical use cases.

## How to Load the Dataset

You can load the dataset using scikit-learn's `fetch_20newsgroups` function. Here’s a brief example:

```python
from sklearn.datasets import fetch_20newsgroups

# Load the training subset, removing headers, footers, and quotes for cleaner text data
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

# Display the number of training documents and available categories
print("Number of training documents:", len(newsgroups_train.data))
print("Categories:", newsgroups_train.target_names)
```

