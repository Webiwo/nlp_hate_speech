from setuptools import setup, find_packages


REPO_NAME = "Kidney-Disease-Classification"
AUTHOR_USER_NAME = "Webiwo"
SRC_REPO = "nlp_hate_speech"
AUTHOR_EMAIL = "webiwo360@gmail.com"

setup(
    name="hate_speech_detection",
    version="0.1.0",
    author="Webiwo",
    author_email="webiwo360@gmail.com",
    description="A project for detecting hate speech using NLP techniques.",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    packages=find_packages(),
    install_requires=[],
)
