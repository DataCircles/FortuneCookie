# FortuneCookies

## Run the Model

* `docker-compose up`

## Project Summary

Please checkout the presentation slides for a brief look to this project [here](https://github.com/ddong63/FortuneCookie/blob/master/FortuneCookieGenerator.pdf)

## How to get started

### Environment setup

Pull the docker image for this project.

### Data collection

Data were collected via multiple websites which contains fortune cookie quotes. First pass data processing was done. Data was saved in a csv format.

### Data preprocessing

Tokenization and padding were performed to this dataset. Data preprocessing functions were saved in `library/`, check it out for details.

### Model training

Three models were trained: GRU, GRU with GLoVe embeddings, GPT-2. Check the `scripts/` for more details.

### Model Deployment

An API is underdevelopment... Stay Tuned :-)


## Our progress tracking during the weekly meetups

If you are a beginner, below are our timeline for you as a reference. Each week, we met for 1.5 hours as a team of four to work on this project. Below tracks our progress for each week so you'll get an estimate for yourself to start a project like this. :-D

### Jan

Preparing slides for the talk

### Dec 12th thru Dec 27th

Implementation of GPT-2
Model validation

### Nov 14th thru Dec 5th

Move ipython notebook to a .py file, cleaned up the script.

To-do:
- make a python package
- add flask

### Nov 7th & Oct 30th

Added word embedding as an additional layer.

### Oct 17th

Things we've done:
- Tested the tensorflow character-level, turns out not work well.
- Added word embedding (glove) to the fortune cookie model

To-do:
- finish the word embedding

### Oct 10th

Things we've done:
* Went through fortunie cookie ipython notebook
* Figured out the tokenizer
  * tokenizer: strips out the punctuations
  * padding: addes 0's to the beginning of the sentences

To-do:
- Update ipython notebook to the new data.csv
- Some research to improve the model preformance
 - How to automatically stop a sentence
 - How to bring back the puncatuation (end stopping word) back to the sentence
 - How to fit the begin word smartly???
- Update the kares version
- Clean up the code and variable names

### September 26th

Things we talked about:
• Adding a metric to the results of the output of the model.
• We attempted to save our model as a static file for use later. We ran into issues in using different versions of keras.

We cleaned some data, and did a git commit to show those changes in the repo. We retrained our model on our newly cleaned data.

### September 12th
* Created docker image and a docker-compose file

### September 5th
* created a github repo
* developed project plan
* listed backlog items
* renamed .ipynb notebook
