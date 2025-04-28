

.PHONY: test
test:
	@if [ -z "$(p)" ]; then \
		python src/test_rl_agent.py --scramble $(n) --tests "$(t)" --model "data/modelCheckpoints/cube_solver_model_scramble_$(m).pt"; \
	else \
		python src/test_rl_agent.py --scramble $(p) --tests "$(t)" --model "data/modelCheckpoints/cube_solver_model_scramble_$(m).pt" --use_pregenerated; \
	fi


.PHONY: train
train:
	python src/rl_agent.py --level $(n) --max_level $(n) --min_rate $(r) --use_pregenerated --target_rate 100 --min_episodes 50000 --batch_size 128 --recent_window 10000


.PHONY: input
input:
	python src/advanced_solver.py --interactive --model "data/modelCheckpoints/cube_solver_model_scramble_$(n).pt"


.PHONY: solve
solve:
	python src/advanced_solver.py --benchmark --scramble_moves $(n) --tests "$(t)" --use_pregenerated


.PHONY: improve
improve:
	python src/rl_agent.py --level "$(n)" --max_level "$(n)" --min_rate "$(r)" --batch_size 128 --use_pregenerated --target_rate 100 --model "data/modelCheckpoints/cube_solver_model_scramble_$(n).pt"


# .PHONY: parallel_train
# parallel_train:
# 	python src/parallel_cube_rl.py --mode train --level $(s) --max_level $(e) --min_rate $(r) --use_pregenerated --target_rate 100 --min_episodes 50000 --batch_size 128 --recent_window $(w) --processes $(p)


# .PHONY: parallel_improve
# parallel_improve:
# 	python src/parallel_cube_rl.py --mode improve --levels $(l) --min_rate $(r) --batch_size 256 --use_pregenerated --target_rate 100 --processes $(p)


# .PHONY: parallel_test
# parallel_test:
# 	python src/parallel_cube_rl.py --mode test --test_level $(n) --num_tests $(t) --use_pregenerated


.PHONY: gen
gen:
	@if [ -z "$(n)" ] || [ -z "$(m)" ]; then \
		echo "Usage: make gen n=<start> m=<end>"; \
		exit 1; \
	fi; \
	for i in $(shell seq $$(($(n))) $$(($(m)))); do \
		echo "python src/scramble_generator.py --level $$i --count 50000"; \
		python src/scramble_generator.py --level $$i --count 50000; \
	done
