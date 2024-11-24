# LlamaDx
LlamaDx is an advanced LLM technique, pretrained on LLama 3.2 models, designed to providing some predictions and medical knowledges for doctors, nurses and patients through clear and comprehensive explanations. It is an AI chatbot where the users can ask question and provide relevant information for disease diagnosis.

![Alt text](https://github.com/WGLab/LlamaDx/blob/main/llamadx.jpg)

1. Setup environments:
    a. Please follow the instructions how to setup LlamaStack with either [TogetherAI](https://github.com/meta-llama/llama-stack/blob/main/docs/source/distributions/self_hosted_distro/together.md) (what we are using), [Fireworks AI](https://github.com/meta-llama/llama-stack/blob/main/docs/source/distributions/self_hosted_distro/fireworks.md), or [Ollama](https://github.com/meta-llama/llama-stack/blob/main/docs/source/distributions/self_hosted_distro/ollama.md) from Meta.
    b. Setup [LlamaStack](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide)
    c. Obtain the [Groq API](https://console.groq.com/docs/overview)
    d. Install packages for the conda environments with requirements.txt
    e. Setup Streamlit account
2. Input Template:
    a. Regular Conversations
    b. Specialized Disease Diagnosis: Follow the template below:
    Age: 5 months old|Sex: Male|Ethnicity: Non-Hispanic|Race: Asian|Phenotypes: missing teeth, cleft lip, feeding difficulties, speech difficulties|\~~~|Candidate Genes from genome/exome sequencing tests: IRF6,PVRL1,GRHL3|~~~|What is the most probable diagnosis? (without "\")
    c. You can be able to upload the images (optional). If there is any information missing, please write the feature name and put "None".
3. Contact Developers:
Quan Minh Nguyen - PhD Student at the University of Pennsylvania (qmn103@seas.upenn.edu)

Dr. Kai Wang - Professor of Pathology and Laboratory Medicine at the Children's Hospital of Philadelphia (wangk@chop.edu)

