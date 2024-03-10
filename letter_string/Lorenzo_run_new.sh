#!/bin/bash

source ../.env
MODELS=( "gpt-4-1106-preview" "gpt-3.5-turbo-0613" "gpt-4-0613" )

for MODEL in "${MODELS[@]}"; do
  echo "Running eval for model $MODEL"
  python eval_GPT3_letterstring_prob.py --api_key $OPENAI_API_KEY --engine $MODEL --N_trials 50
done

for MODEL in "${MODELS[@]}"; do
  echo "Running eval for model $MODEL"
  python eval_GPT3_letterstring_prob.py --api_key $OPENAI_API_KEY --engine $MODEL --N_trials 50 --sentence
done

for MODEL in "${MODELS[@]}"; do
  echo "Running eval for model $MODEL"
  python eval_GPT3_letterstring_prob.py --api_key $OPENAI_API_KEY --engine $MODEL --N_trials 50 --noprompt
done