hdir=$(pwd)
cd ..

mkdir datasets
mkdir datasets/mcq_mix_v3
mkdir datasets/mcq_mix_v6
mkdir datasets/sci_retriever_r4
mkdir datasets/ranker_data

cd datasets

kaggle competitions download -c kaggle-llm-science-exam
unzip kaggle-llm-science-exam.zip -d kaggle-llm-science-exam
rm kaggle-llm-science-exam.zip

kaggle datasets download -d conjuring92/mcq-mix-v3
unzip mcq-mix-v3.zip -d ./mcq_mix_v3
rm mcq-mix-v3.zip

kaggle datasets download -d conjuring92/mcq-mix-v6
unzip mcq-mix-v6.zip -d ./mcq_mix_v6
rm mcq-mix-v6.zip

kaggle datasets download -d conjuring92/retriever-data-v4
unzip retriever-data-v4.zip -d ./sci_retriever_r4
rm retriever-data-v4.zip

kaggle datasets download -d conjuring92/ranker-dataset-vf
unzip ranker-dataset-vf.zip -d ./ranker_data
rm ranker-dataset-vf.zip

cd $hdir