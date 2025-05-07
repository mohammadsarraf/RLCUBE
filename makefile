DOCKER ?= false

.PHONY: test
test:
	@if [ -z "$(p)" ]; then \
		python src/test_rl_agent.py --scramble $(n) --tests "$(t)" --model "data/modelCheckpoints/cube_solver_model_scramble_$(m).pt"; \
	else \
		python src/test_rl_agent.py --scramble $(p) --tests "$(t)" --model "data/modelCheckpoints/cube_solver_model_scramble_$(m).pt" --use_pregenerated; \
	fi

.PHONY: ctest
ctest:
	docker run -it --rm -v $(PWD):/app rlcube python src/test_rl_agent.py --scramble $(n) --tests "$(t)" --model "data/modelCheckpoints/localtraining/cube_solver_curriculum_all_6.pt" --use_pregenerated

.PHONY: train
train:
	python src/rl_agent.py --level $(n) --max_level $(n) --min_rate $(r) --use_pregenerated --target_rate 100 --min_episodes 50000 --batch_size 128 --recent_window 10000


.PHONY: ctrain
ctrain:
	@if [ -z "$(c)" ]; then \
		python -c "import sys; sys.path.append('src'); import helper, rl_agent; helper.continuous_curriculum_training(max_scramble=$(m), min_episodes=$(min), max_episodes=$(max), success_threshold=$(r), batch_size=$(b), use_pregenerated=True)"; \
	else \
		python -c "import sys; sys.path.append('src'); import helper, rl_agent; helper.continuous_curriculum_training(max_scramble=$(m), min_episodes=$(min), max_episodes=$(max), success_threshold=$(r), batch_size=$(b), checkpoint_path='$(c)', use_pregenerated=True, plateau_patience=$(p))"; \
	fi

.PHONY: input
input:
	python src/advanced_solver.py --interactive --model "data/modelCheckpoints/cube_solver_curriculum_all_1, 2, 3, 4, 5.pt"


.PHONY: cinput
cinput:
	docker run -it --rm -v $(PWD):/app rlcube python src/advanced_solver.py --interactive --model "data/modelCheckpoints/localtraining/cube_solver_curriculum_all_6.pt"


.PHONY: solve
solve:
	python src/advanced_solver.py --benchmark --scramble_moves $(n) --tests "$(t)" --use_pregenerated


.PHONY: csolve
csolve:
	@if [ -z "$(n)" ] || [ -z "$(t)" ]; then \
		echo "Usage: make csolve n=<scramble_moves> t=<num_tests>"; \
		exit 1; \
	fi; \
	docker run -it --rm -v $(PWD):/app rlcube python src/advanced_solver.py --benchmark --scramble_moves $(n) --tests "$(t)" --use_pregenerated --model "data/modelCheckpoints/cube_solver_curriculum_all_[1, 2, 3, 4, 5, 6].pt"

.PHONY: improve
improve:
	python src/rl_agent.py --level "$(n)" --max_level "$(n)" --min_rate "$(r)" --batch_size 128 --use_pregenerated --target_rate 100 --model "data/modelCheckpoints/cube_solver_model_scramble_$(n).pt"


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