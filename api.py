"""Ponto de entrada da API.

Local:   uvicorn api:app --reload        (documentação em http://127.0.0.1:8000/docs)
Render:  uvicorn api:app --host 0.0.0.0 --port $PORT   (Root Directory: ia)
"""

import logging

from aplicacao import criar_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = criar_app()
