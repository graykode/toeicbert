from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup_info = dict(
    name='toeicbert',
    version='0.0.2',
    author='Tae Hwan Jung(@graykode)',
    author_email='nlkey2022@gmail.com',
    url='https://github.com/graykode/toeicbert',
    description='TOEIC blank problem solving using pytorch-pretrained-BERT model.',
    long_description=long_description,
    long_description_content_type='text/markdown',  # This is important!
    license='MIT',
    install_requires=[ 'tqdm', 'torch', 'pytorch_pretrained_bert', 'unidecode'],
    keywords='BERT TOEIC pytorch-pretrained-BERT bert nlp NLP',
    packages=["toeicbert"],
)

setup(**setup_info)