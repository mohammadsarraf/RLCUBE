.PHONY: test train input advance improve train_all

test:
	python test_rl_agent.py --scramble $(n) --tests 10000

train:
	python cube_rl.py --level "$(n)" --max_level "$(n)" --min_rate "$(r)" --use_pregenerated --target_rate 100

input:
	python advanced_solver.py --interactive

advance:
	python advanced_solver.py --benchmark --scramble_moves $(n) --tests 100 --use_pregenerated

improve:
	python cube_rl.py --level "$(n)" --max_level "$(n)" --min_rate "$(r)" --batch_size "$(b)" --use_pregenerated --target_rate 100 --model "cube_solver_model_scramble_$(n).pt"

train_all:
	./train_all.sh

	