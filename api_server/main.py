"""챗봇 서버에게 요청을 보내는 클라이언트

0. 프로젝트 루트 디렉토리에서 `python main.py --mode run_server`를 입력하여 모델 서버를 연다.
1. 어플리케이션 호출 스크립트 디렉토리 상에서 다음 명령어 입력: `uvicorn main:app --reload`
2. (기본 포트인 8000을 가정) `http://127.0.0.1:8000/predict/{query}`를 브라우저에 입력하면 응답을 받을 수 있다.
"""

import json
import os
import socket
import sys
sys.path.append(os.path.abspath(os.pardir))

from fastapi import FastAPI

from config import resources as rsc

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/predict/{query}")
async def predict(query):
    host = rsc.host
    port = rsc.port
    destination = (host, port)

    data = {
        'query': query,
    }
    msg = json.dumps(data)

    mysock = socket.socket()
    mysock.connect(destination)
    mysock.send(msg.encode())

    resp = mysock.recv(bufsize := 2048).decode()
    resp_data = json.loads(resp)

    return resp_data
