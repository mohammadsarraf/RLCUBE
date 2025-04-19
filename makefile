test:
	python test_rl_agent.py --scramble $(n) --use_pregenerated --tests 1000

train:
	python cube_rl.py --max_level 5 --max_episodes 50000 --use_pregenerated


