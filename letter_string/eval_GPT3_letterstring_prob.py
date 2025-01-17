import openai
import numpy as np
import builtins
import argparse
import os
import time

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--sentence', action='store_true', help="Present problem in sentence format.")
parser.add_argument('--noprompt', action='store_true', help="Present problem without prompt.")
parser.add_argument('--api_key', type=str, required=True)
parser.add_argument('--engine', type=str, default='text-davinci-003')
parser.add_argument('--N_trials', type=int, default=50)
args = parser.parse_args()

save_fname = f'./gpt3_{args.engine}_letterstring_results'
if args.sentence:
	save_fname += '_sentence'
if args.noprompt:
	save_fname += '_noprompt'
save_fname += '.npz'

# GPT-3 settings
openai.api_key = args.api_key

N_trials_per_prob_type = args.N_trials
kwargs = {"temperature":0,  }

if "gpt-3.5" or "gpt-4" in args.engine:
	kwargs["model"] = args.engine
	kwargs["logprobs"] = False
	kwargs["max_tokens"] = 200
else:
	kwargs["engine"] = args.engine
	kwargs["logprobs"] = 1
	kwargs["echo"] = True
	kwargs["max_tokens"] = 40
	if args.sentence:
		kwargs["stop"]="\n"

# Load all problems
all_prob = np.load('./all_prob.npz', allow_pickle=True)['all_prob']
prob_types = builtins.list(all_prob.item().keys())
N_prob_types = len(prob_types)

# Evaluate
# N_trials_per_prob_type = 50
all_prob_type_responses = []
for p in range(N_prob_types):
	print('problem type' + str(p+1) + ' of ' + str(N_prob_types) + '...')
	prob_type_responses = []
	for t in range(N_trials_per_prob_type):
		print('trial ' + str(t+1) + ' of ' + str(N_trials_per_prob_type) + '...')
		# Generate prompt
		prob = all_prob.item()[prob_types[p]]['prob'][t]
		prompt = ''
		if not args.noprompt:
			prompt += "Let's try to complete the pattern:\n\n"
		if args.sentence:
			prompt += 'If '
			for i in range(len(prob[0][0])):
				prompt += str(prob[0][0][i])
				if i < len(prob[0][0]) - 1:
					prompt += ' '
			prompt += ' changes to '
			for i in range(len(prob[0][1])):
				prompt += str(prob[0][1][i])
				if i < len(prob[0][1]) - 1:
					prompt += ' '
			prompt += ', then '
			for i in range(len(prob[1][0])):
				prompt += str(prob[1][0][i])
				if i < len(prob[1][0]) - 1:
					prompt += ' '
			prompt += ' should change to '
		else:
			prompt += '['
			for i in range(len(prob[0][0])):
				prompt += str(prob[0][0][i])
				if i < len(prob[0][0]) - 1:
					prompt += ' '
			prompt += '] ['
			for i in range(len(prob[0][1])):
				prompt += str(prob[0][1][i])
				if i < len(prob[0][1]) - 1:
					prompt += ' '
			prompt += ']\n['
			for i in range(len(prob[1][0])):
				prompt += str(prob[1][0][i])
				if i < len(prob[1][0]) - 1:
					prompt += ' '
			prompt += '] ['
		# Get response
		response = []
		while len(response) == 0:
			try:
				if "gpt-3.5" or "gpt-4" in args.engine:
					response = openai.ChatCompletion.create(
							messages=[
								{"role": "system", "content": ""},
								{"role": "user", "content": prompt}
							], **kwargs)
				else:
					response = openai.Completion.create(prompt=prompt, **kwargs)
				# print(response)
				# print(response['choices'][0].message['content'])
			except:
				print('trying again...')
				time.sleep(5)
		if "gpt-3.5" or "gpt-4" in args.engine:
			prob_type_responses.append(response['choices'][0].message['content'])
		else:
			prob_type_responses.append(response['choices'][0]['text'])
	all_prob_type_responses.append(prob_type_responses)
	# Save
	# save_fname = './gpt3_letterstring_results'
	# if args.sentence:
	# 	save_fname += '_sentence'
	# if args.noprompt:
	# 	save_fname += '_noprompt'
	# save_fname += '.npz'
	np.savez(save_fname, all_prob_type_responses=all_prob_type_responses, allow_pickle=True)



