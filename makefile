.PHONY: train input advance improve train_all gen test parallel_train parallel_improve

# test:
# 	python test_rl_agent.py --scramble $(n) --tests "$(t)" --model "modelCheckpoints/cube_solver_model_scramble_$(n).pt"

test:
	@if [ -z "$(p)" ]; then \
		python test_rl_agent.py --scramble $(n) --tests "$(t)" --model "modelCheckpoints/cube_solver_model_scramble_$(m).pt"; \
	else \
		python test_rl_agent.py --scramble $(p) --tests "$(t)" --model "modelCheckpoints/cube_solver_model_scramble_$(m).pt" --use_pregenerated; \
	fi

train:
	python cube_rl.py --level $(n) --max_level $(n) --min_rate $(r) --use_pregenerated --target_rate 100 --min_episodes 25000 --batch_size 128 --recent_window 10000

input:
	python advanced_solver.py --interactive --model "modelCheckpoints/cube_solver_model_scramble_$(n).pt"

advance:
	python advanced_solver.py --benchmark --scramble_moves $(n) --tests "$(t)" --use_pregenerated

improve:
	python cube_rl.py --level "$(n)" --max_level "$(n)" --min_rate "$(r)" --batch_size 128 --use_pregenerated --target_rate 100 --model "modelCheckpoints/cube_solver_model_scramble_$(n).pt"

# New parallel training commands
parallel_train:
	python parallel_cube_rl.py --mode train --level $(s) --max_level $(e) --min_rate $(r) --use_pregenerated --target_rate 100 --min_episodes 25000 --batch_size 128 --recent_window $(w) --processes $(p)

parallel_improve:
	python parallel_cube_rl.py --mode improve --levels $(l) --min_rate $(r) --batch_size $(batch_size) --use_pregenerated --target_rate 100 --processes $(p) --recent_window $(w) --memory_size $(memory_size) --gamma $(gamma) --lr $(lr) --epsilon_min $(epsilon_min) --plat $(plat)

parallel_test:
	python parallel_cube_rl.py --mode test --test_level $(n) --num_tests $(t) --use_pregenerated

train_all:
	./train_all.sh

gen:
	@if [ -z "$(n)" ] || [ -z "$(m)" ]; then \
		echo "Usage: make gen n=<start> m=<end>"; \
		exit 1; \
	fi; \
	for i in $(shell seq $$(($(n))) $$(($(m)))); do \
		echo "python gen.py --level $$i --count 25000"; \
		python gen.py --level $$i --count 25000; \
	done
