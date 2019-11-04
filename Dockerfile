FROM python:3

ADD . /T3
WORKDIR /T3

RUN pip install numpy cython

CMD ["python", "setup.py", "install"]