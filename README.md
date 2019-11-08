# FortuneCookies

## Run the Model

* `docker-compose up`

## Machine Learning

* Refamiliarize ourselves with our current code
* Improve our model
  * Add word embedding?
  * Param Training
  * Data collection/refine trainset
  * Rank results / see improvement in the model

## Infrastructure

* Create a docker image/docker compose
* Continuous Integration
* Continuous Deployment

## App

* Render model results as plain text on an html page
* Build flask prototype
* Get a domain
* Website-to-model 
* Feedback from within UI


## Marketing

* Documentation of process/design/implementation
* Write blog(s)
  * decide where the blog belongs
* Polish git repo


## Things we've accomplished

### September 5th
* created a github repo
* developed project plan
* listed backlog items
* renamed .ipynb notebook

### September 12th
* Created docker image and a docker-compose file

### September 26th

Things we talked about:
• Adding a metric to the results of the output of the model.
• We attempted to save our model as a static file for use later. We ran into issues in using different versions of keras.

We cleaned some data, and did a git commit to show those changes in the repo. We retrained our model on our newly cleaned data.

TODO:
* Do a quick recap of our code
* First iteration of model improvement
 * one round of ranking

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

### Oct 17th

Things we've done:
- Tested the tensorflow character-level, turns out not work well.
- Added word embedding (glove) to the fortune cookie model

To-do:
- finish the word embedding

