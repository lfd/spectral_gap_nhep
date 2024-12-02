# Contributing to From Hope to Heuristics

Contributions are highly welcome! :hugging_face:

Start of by..
1. Creating an issue using one of the templates (Bug Report, Feature Request)
   - let's discuss what's going wrong or what should be added
   - can you contribute with code? Great! Go ahead! :rocket: 
2. Forking the repository and working on your stuff. See the sections below for details on how to set things up.
3. Creating a pull request to the main repository

## Setup

Contributing to this project requires some more dependencies besides the "standard" packages.
Those are specified in the groups `dev`.
```
poetry install --with dev
```
These dependencies are essentially flake8, black, pre-commit and licensecheck which we use for quality assurance and code formatting.

Additionally, we have pre-commit hooks in place, which can be installed as follows: 
```
poetry run pre-commit autoupdate
poetry run pre-commit install
```

Currently the only purpose of the hook is to run Black on commit which will do some code formatting for you.
However be aware, that this might reject your commit and you have to re-do the commit.