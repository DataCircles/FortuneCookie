# Build Docker Image Based on jupyter/base-notebook

Dockerfile —> Image —> Container
        (Build)   (Run)

## Step 1: Dockerfile
```
# build a image
docker build . -t test:test
# connect to a local port
docker run -p 8888:8888 test:test
# check 
docker compose ps
```

## Step 2: docker-compose.yml

```
version: "3.7"
services:

  <image-name-that-we-want>:
    build: ./
    volumes:
      - <local-file-system-folder-to-mount>/:<our-desired-location-to-mount-the-volume-inside-the-container>
    stdin_open: true
    tty: true
    ports:
      - "8888:8888"
```      

Run `docker-compose up` then the jupyter-notebook is hosted! 

# Other useful docker command

```
# print working directory
docker run -p 8888:8888 test:test pwd
# run docker image
docker-compose up
```

Some useful links:

https://container.training/intro-selfpaced.yml.html#1
https://medium.com/faun/the-missing-introduction-to-containerization-de1fbb73efc5
https://dev.to/azure/docker---from-the-beginning-part-i-28c6
