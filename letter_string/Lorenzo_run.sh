#!/bin/bash

source ../.env
MODELS=( "ada"  "babbage" "curie" "davinci" "text-ada-001" "text-babbage-001" "text-curie-001" "text-davinci-001" "text-davinci-002" )

for MODEL in "${MODELS[@]}"; do
  echo "Running eval for model $MODEL"
  python eval_GPT3_letterstring_prob.py --api_key $OPENAI_API_KEY --engine $MODEL --N_trials 10
done

for MODEL in "${MODELS[@]}"; do
  echo "Running eval for model $MODEL"
  python eval_GPT3_letterstring_prob.py --api_key $OPENAI_API_KEY --engine $MODEL --N_trials 10 --sentence
done

for MODEL in "${MODELS[@]}"; do
  echo "Running eval for model $MODEL"
  python eval_GPT3_letterstring_prob.py --api_key $OPENAI_API_KEY --engine $MODEL --N_trials 10 --noprompt
done