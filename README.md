# 프로젝트 설명 - 음식 주문 챗봇(Simple food order chatbot only for Korean)

자연어 처리 딥러닝 모델들 - 의도 분류기(Intent classifier), 개체명 인식기(Named entity recognizer)을 사용한 간단 음식 주문 처리 챗봇.

# 작동 과정

1. 사용자는 API 서버를 통해서 음식 주문 텍스트를 챗봇에게 전달한다.
2. 사용자가 입력한 텍스트에서 의도와 개체명을 엔진 내 딥러닝 모델들이 추출한다.
2. 추출된 이 두 조건으로 데이터베이스에서 질의하여 사전에 정의된 상황별 답변을 얻어낸다.
3. 얻어낸 답변을 사용자에게 반환.

# 빠른 시작

1. 프로젝트를 내려 받고, `requirements.txt` 파일을 이용해서 의존성 패키지를 설치한다.
2. 프로젝트 루트 디렉토리 상 터미널에서 `python main.py --mode run_server`를 입력하여 모델 서버를 연다.
3. 다른 터미널을 열고 어플리케이션 호출 스크립트 디렉토리 상에서 다음 명령어 입력: `uvicorn main:app --reload`
4. (로컬에서 요청한다고 가정) `http://127.0.0.1:8000/predict/질의텍스트` 를 브라우저에 입력하면 응답을 받을 수 있다.

# 프로젝트 구조:

- api_server: 프론트엔드 요청 서버 어플리케이션.
- config: 프로젝트에서 사용하는 각종 설정값들 모음.
- engine: 챗봇 엔진 소스 코드.
    - model: 자연어 처리 모델 훈련 스크립트 및 래핑 클래스
- models: 프로젝트 진행하면서 생성했던 텐서플로 케라스 모델들.
    - engine: 챗봇 엔진에 사용된 자연어 처리 모델들(의도 분류기, 개체명 인식기).
    - practice: 연습용 모델들(챗봇 엔진에 사용되지 않음).
- practice: 프로젝트 진행을 위해 필요한 사전 공부용 주피터 노트북들.
- test: `engine/` 내 코드들 테스트용(불완전).
- main.py: 챗봇 엔진 진입점 스크립트.
- requirements.txt: `pip freeze` 로 생성한 의존 패키지 모음.

# 주요 사용 툴:

- DB: [SQLite3](https://www.sqlite.org/index.html)
- 형태소 분석기: [Konlpy](https://konlpy.org/en/latest/) 패키지 내 [Komoran](https://docs.komoran.kr/index.html#)
- 백엔드 모델 서버: 파이썬 내장 `socket` 모듈
- API 서버: [FastAPI](https://fastapi.tiangolo.com/)
- 자연어 처리 모델: [TensorFlow](https://tensorflow.org/)
