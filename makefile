.PHONY: train input advance improve train_all gen test

# test:
# 	python test_rl_agent.py --scramble $(n) --tests "$(t)" --model "modelCheckpoints/cube_solver_model_scramble_$(n).pt"

test:
	@if [ -z "$(p)" ]; then \
		python test_rl_agent.py --scramble $(n) --tests "$(t)" --model "modelCheckpoints/cube_solver_model_scramble_$(m).pt"; \
	else \
		python test_rl_agent.py --scramble $(p) --tests "$(t)" --model "modelCheckpoints/cube_solver_model_scramble_$(m).pt" --use_pregenerated; \
	fi

train:
	python cube_rl.py --level $(n) --max_level $(n) --min_rate $(r) --use_pregenerated --target_rate 100 --min_episodes 50000 --batch_size 128 --recent_window 10000

input:
	python advanced_solver.py --interactive --model "modelCheckpoints/cube_solver_model_scramble_$(n).pt"

advance:
	python advanced_solver.py --benchmark --scramble_moves $(n) --tests "$(t)"

improve:
	python cube_rl.py --level "$(n)" --max_level "$(n)" --min_rate "$(r)" --batch_size 128 --use_pregenerated --target_rate 100 --model "modelCheckpoints/cube_solver_model_scramble_$(n).pt"

train_all:
	./train_all.sh

gen:
	@if [ -z "$(n)" ] || [ -z "$(m)" ]; then \
		echo "Usage: make gen n=<start> m=<end>"; \
		exit 1; \
	fi; \
	for i in $(shell seq $$(($(n))) $$(($(m)))); do \
		echo "python gen.py --level $$i --count 50000"; \
		python gen.py --level $$i --count 50000; \
	done
