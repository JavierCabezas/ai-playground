docker build -t my-python-app .
docker run -it --rm -v $(pwd):/usr/src/app -w /usr/src/app my-python-app python [file.py]

Files:
transcript.py: Creates a subtitle file from the wav file given