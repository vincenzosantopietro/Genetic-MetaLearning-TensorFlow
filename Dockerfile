FROM python:3.6
WORKDIR /app
# copy environment definition
COPY resources/requirements.txt .

RUN pip install -r requirements.txt

# Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "aiml", "/bin/bash", "-c"]

RUN echo "Make sure TensorFlow is installed:"
RUN python -c "import tensorflow as tf; print(\"Using TensorFlow: {}\".format(tf.__version__))"

COPY src /app/src
# activate env
# RUN echo "conda activate aiml" > ~/.bashrc
# ENV PATH /opt/conda/envs/env/bin:$PATH

# run training
#ENTRYPOINT ["conda", "run", "-n", "aiml", "python","-u", "src/main.py"]
ENTRYPOINT ["python", "src/main.py"]