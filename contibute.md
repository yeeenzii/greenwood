# Contributing

## Setting Up
Place .env file in root directory 
`
pip install -r requirements.txt
python -m pip freeze > requirments.txt
`

## Autodocumentaion
`
sphinx-apidoc -o docs greenwood
make clean html
`