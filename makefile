test:
	python test_rl_agent.py --scramble $(n) --tests 1000

train:
	python cube_rl.py --max_level 5 --max_episodes 50000 --use_pregenerated --target_rate 90

input:
	python advanced_solver.py --interactive

advance:
	python advanced_solver.py --benchmark --scramble_moves $(n) --tests 1000

improve:
	python cube_rl.py --level $(n) --max_level $(n) --use_pregenerated --target_rate 95