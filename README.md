# A Survey Study of Explainable AI for Clinical Outcome Prediction
This repository contains code to generate the XAI methods samples used in the survey, described in our AMIA Informatics Summit 2025 paper.

## Code setup
This code was developed in python 3.10 using the libraries listed in [Literature Augmented Clinical Outcome Prediction](https://arxiv.org/abs/2111.08374), which repository is [here](https://github.com/allenai/BEEP) and [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) (EMNLP 2019), which repository is [here](https://github.com/UKPLab/sentence-transformers). Please follow instruction of original repository to setup.

The folder named in the corresponding repository followed their structure and added the file required to replicate the generation of samples. After cloning the two repositories, replace BEEP/outcome-prediction and sentence-transformers/examples/applications/semantic-search/ with the two corresponding folders from this repository.

You will also need download the model weights trained by BEEP. These resources can be downloaded using the following command:

```aws s3 sync --no-sign-request s3://ai2-s2-beep models/```

Note that you need to have AWS CLI installed on your machine to execute this command. Move the pubmed_texts_and_dates.pkl file to the data directory.

The embedder we fine-tuned based on BEEP for the similar patient retrieval sample generation can be loaded from /training_stsbenchmark_continue_training-512_ulms_bert.

## Creating Mortality Prediction Datasets
In addition to environment setup, you will need access to the MIMIC-III dataset ([download here](https://physionet.org/content/mimiciii-demo/1.4/)). You will also need build the mortality datasets following the [van Aken et al (2021)](https://aclanthology.org/2021.eacl-main.75/), which can be obtained [here](https://github.com/bvanaken/clinical-outcome-prediction).

## Replicating XAI samples
The XAI samples -- LIME, Attention-based and Free-text Rationale -- generation notebook files are under the BEEP folder.
- LIME: The notebook is located at /BEEP-main/outcome-prediction/MOR_LIME.ipynb
- Attention-based: The notebook is located at /BEEP-main/outcome-prediction/MOR_Attention.ipynb
- Free-text Rationale: The notebook is located at /BEEP-main/outcome-prediction/MOR_GPT.ipynb. We use the Guidance wrapper, which can be installed by ```pip install guidance``` following the instruction at [here](https://github.com/guidance-ai/guidance). You will need GPT API to generate the sample.

The XAI sample, similar patient retrieval, generation file is under the sentence-transformer folder.
- Similar Patient Retrieval: You will need load fine-tuned embedder. The notebook is located at /sentence-transformers/examples/applications/semantic-search/MOR_STS_ner.ipynb.# XAI_MOR_Survey

## Replicating XAI Mortality Prediction Survey
We provide one version of the shuffled survey, "XAI Questionnaire A1 - Google Forms.pdf".
