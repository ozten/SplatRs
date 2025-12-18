# SplatRs Makefile
# Convenience targets for development workflows

.PHONY: help coverage coverage-test coverage-open coverage-clean profile profile-micro profile-clean

help:
	@echo "Available targets:"
	@echo "  coverage       - Generate HTML coverage report"
	@echo "  coverage-test  - Generate coverage report with test output"
	@echo "  coverage-open  - Generate coverage report and open in browser"
	@echo "  coverage-clean - Clean coverage artifacts"
	@echo ""
	@echo "  profile        - Profile training run (requires PROFILE_ARGS)"
	@echo "  profile-micro  - Profile 100-iteration micro training run"
	@echo "  profile-clean  - Clean profiling artifacts"

# Generate HTML coverage report for all features
coverage:
	cargo llvm-cov --html --all-features

# Generate coverage with test output (useful for debugging)
coverage-test:
	cargo llvm-cov --html --all-features -- --nocapture

# Generate coverage and open report in browser
coverage-open:
	cargo llvm-cov --html --open --all-features

# Clean coverage artifacts
coverage-clean:
	cargo llvm-cov clean --workspace
	rm -rf target/coverage/

# Profile a training run with custom arguments
# Usage: make profile PROFILE_ARGS="--preset micro --dataset-root datasets/scene"
profile:
ifndef PROFILE_ARGS
	@echo "Error: PROFILE_ARGS not set"
	@echo "Usage: make profile PROFILE_ARGS=\"--preset micro --dataset-root datasets/scene\""
	@exit 1
endif
	./scripts/profile-training.sh $(PROFILE_ARGS)

# Profile a micro training run (100 iterations)
# Requires: datasets/tandt_db/tandt/train dataset
profile-micro:
	@if [ ! -d "datasets/tandt_db/tandt/train" ]; then \
		echo "Error: T&T dataset not found at datasets/tandt_db/tandt/train"; \
		echo "Please download the dataset first."; \
		exit 1; \
	fi
	./scripts/profile-training.sh --preset micro --dataset-root datasets/tandt_db/tandt/train

# Clean profiling artifacts
profile-clean:
	rm -f flamegraph.svg perf.data perf.data.old *.folded
