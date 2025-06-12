FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# do stuff here

RUN apt-get update && apt-get install -y gcc

RUN pip install -U pip
RUN pip install pipenv 

COPY [ "Pipfile", "Pipfile.lock", "./" ]

# COPY [ "requirements.txt", "./" ]

RUN pipenv install --system --deploy

# RUN pip install -r requirements.txt

COPY ["score.py","script.sh","./"]

EXPOSE 4200

CMD ["bash", "script.sh"]