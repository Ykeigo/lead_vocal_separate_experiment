FROM mcr.microsoft.com/vscode/devcontainers/python:3.9-bullseye

RUN apt update
RUN sudo apt-get install libsndfile1
RUN python -m pip install --upgrade pip && python -m pip install poetry
