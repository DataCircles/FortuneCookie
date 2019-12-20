# Fortune Cookie Generator

### Prerequistes

* [docker](https://www.docker.com/products/docker)
* [docker-compose](https://docs.docker.com/compose/)

### Run the Latest Model

1. Execute `docker-compose up` in your terminal.
2. [Open the jupyter notebook](https://github.com/jupyter/docker-stacks#quick-start) in your browser.

### Goal

Use a [recurrent neural network](https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9) to produce new fortune cookie fortunes. This repo shows three different attempts to produce a working model to produce new fortunes.

### Progress Thus Far

* Attempt One: GRU with corpus embedding. Largely based on this blog post. TODO: Add blog post link.
* Attempt Two: Pretrained embedding layer using [glove](https://nlp.stanford.edu/projects/glove/).
* Attempt Three: Fine tune GPT-2 within a docker container locally.

### Barriers

1. It has been difficult to find the volume of data required to train a natrual language processing model on fortune cookies. There are not very many fortune cookie databases out in the wild.
2. Fortune cookies are built from somewhat atypical English, which has made the lack of data more difficult.
