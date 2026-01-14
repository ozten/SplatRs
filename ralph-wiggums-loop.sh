#!/bin/bash
# ralph-wiggum-codex.sh

MAX_ITERATIONS=${1:-25}
PROMPT_FILE="PROMPT.md"
PROGRESS_FILE="PROGRESS.txt"

iteration=0

while [ $iteration -lt $MAX_ITERATIONS ]; do
    echo "═══════════════════════════════════════════"
    echo "Iteration $((iteration + 1)) of $MAX_ITERATIONS"
    echo "═══════════════════════════════════════════"
    
    # Check for open tasks in beads
    OPEN_TASKS=$(bd ready --json 2>/dev/null | jq -r 'length')
    
    if [ "$OPEN_TASKS" -eq 0 ] || [ -z "$OPEN_TASKS" ]; then
        echo "No open tasks remaining. Exiting."
        break
    fi
    
    echo "Open tasks: $OPEN_TASKS"
    
    # Run Claude with the prompt
    claude -p "$(cat $PROMPT_FILE)" --dangerously-skip-permissions \
        --verbose --output-format stream-json \
        2>&1 | tee ./claude-iteration-$iteration.jsonl
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Codex exited with error code $EXIT_CODE"
        # Optionally continue or break
    fi
    
    ((iteration++))
    
    # Optional: Add a small delay to avoid rate limits
    sleep 2
done

echo "Loop completed after $iteration iterations"