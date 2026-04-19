#!/bin/bash
# ==============================================================================
# Parallel LIBERO Dataset Generation with Progress Tracking
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
LOG_DIR="logs/libero_generation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Task configurations
declare -A TASKS
# TASKS[libero_spatial]="/VLA-Data/scripts/lianqing/projects/vla/openpi/third_party/libero/libero/libero/../datasets/libero_spatial"
# TASKS[libero_object]="/VLA-Data/scripts/lianqing/projects/vla/openpi/third_party/libero/libero/libero/../datasets/libero_object"
# TASKS[libero_goal]="/VLA-Data/scripts/lianqing/projects/vla/openpi/third_party/libero/libero/libero/../datasets/libero_goal"
# TASKS[libero_10]="/VLA-Data/scripts/lianqing/projects/vla/openpi/third_party/libero/libero/libero/../datasets/libero_10"
# TASKS[libero_90]="/VLA-Data/scripts/lianqing/projects/vla/openpi/third_party/libero/libero/libero/../datasets/libero_90"

TARGET_BASE="/VLA-Data/scripts/lianqing/data/libero_xvla"

# Arrays to track PIDs and status
declare -A PIDS
declare -A STATUS
declare -A START_TIME
declare -A END_TIME

# Function to run a single task
run_task() {
    local task_name=$1
    local raw_dir=$2
    local target_dir="${TARGET_BASE}/${task_name}_no_noops"
    local log_file="${LOG_DIR}/${task_name}.log"
    local status_file="${LOG_DIR}/${task_name}.status"
    
    echo "RUNNING" > "$status_file"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting $task_name" >> "$log_file"
    
    python datasets/tools/regenerate_libero_dataset.py \
        --libero_task_suite "$task_name" \
        --libero_raw_data_dir "$raw_dir" \
        --libero_target_dir "$target_dir" \
        >> "$log_file" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "SUCCESS" > "$status_file"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Completed $task_name successfully" >> "$log_file"
    else
        echo "FAILED" > "$status_file"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Failed $task_name with exit code $exit_code" >> "$log_file"
    fi
    
    return $exit_code
}

# Function to check status of all tasks
check_status() {
    local all_done=true
    
    echo -e "\n${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}                    Task Status Overview                        ${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}\n"
    
    for task in "${!PIDS[@]}"; do
        local pid=${PIDS[$task]}
        local status_file="${LOG_DIR}/${task}.status"
        
        if [ -f "$status_file" ]; then
            local status=$(cat "$status_file")
        else
            local status="UNKNOWN"
        fi
        
        # Check if process is still running
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "  ${YELLOW}●${NC} ${task}: ${YELLOW}RUNNING${NC} (PID: $pid)"
            all_done=false
        else
            if [ "$status" = "SUCCESS" ]; then
                echo -e "  ${GREEN}✓${NC} ${task}: ${GREEN}SUCCESS${NC}"
            elif [ "$status" = "FAILED" ]; then
                echo -e "  ${RED}✗${NC} ${task}: ${RED}FAILED${NC}"
            else
                echo -e "  ${RED}✗${NC} ${task}: ${RED}UNKNOWN${NC}"
            fi
        fi
    done
    
    echo ""
    
    if [ "$all_done" = true ]; then
        return 0
    else
        return 1
    fi
}

# Function to display final summary
display_summary() {
    local total=${#TASKS[@]}
    local success=0
    local failed=0
    
    echo -e "\n${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}                    Final Summary                              ${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}\n"
    
    for task in "${!TASKS[@]}"; do
        local status_file="${LOG_DIR}/${task}.status"
        local status=$(cat "$status_file" 2>/dev/null || echo "UNKNOWN")
        
        if [ "$status" = "SUCCESS" ]; then
            ((success++))
            echo -e "  ${GREEN}✓${NC} ${task}: ${GREEN}SUCCESS${NC}"
            
            # Show log tail
            echo -e "    ${BLUE}Last 3 lines from log:${NC}"
            tail -n 3 "${LOG_DIR}/${task}.log" | sed 's/^/      /'
        else
            ((failed++))
            echo -e "  ${RED}✗${NC} ${task}: ${RED}FAILED${NC}"
            
            # Show error tail
            echo -e "    ${RED}Last 5 lines from log:${NC}"
            tail -n 5 "${LOG_DIR}/${task}.log" | sed 's/^/      /'
        fi
        echo ""
    done
    
    echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
    echo -e "  Total Tasks: ${total}"
    echo -e "  ${GREEN}Successful: ${success}${NC}"
    echo -e "  ${RED}Failed: ${failed}${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
    echo -e "\n  📁 Logs saved to: ${BLUE}${LOG_DIR}${NC}\n"
    
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}🎉 All tasks completed successfully!${NC}\n"
        return 0
    else
        echo -e "${RED}⚠️  Some tasks failed. Check logs for details.${NC}\n"
        return 1
    fi
}

# Main execution
main() {
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}      LIBERO Dataset Generation - Parallel Execution           ${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}\n"
    
    echo -e "📋 Starting ${#TASKS[@]} tasks in parallel..."
    echo -e "📁 Logs directory: ${BLUE}${LOG_DIR}${NC}\n"
    
    # Start all tasks in parallel
    for task in "${!TASKS[@]}"; do
        raw_dir="${TASKS[$task]}"
        echo -e "  ${YELLOW}▶${NC} Launching: ${task}"
        
        run_task "$task" "$raw_dir" &
        PIDS[$task]=$!
        START_TIME[$task]=$(date +%s)
        
        echo -e "    PID: ${PIDS[$task]}"
        echo -e "    Log: ${LOG_DIR}/${task}.log"
    done
    
    echo -e "\n${GREEN}✓${NC} All tasks launched!\n"
    
    # Monitor progress
    sleep 2
    while true; do
        if check_status; then
            break
        fi
        sleep 5
    done
    
    # Wait for all background jobs to finish
    for task in "${!PIDS[@]}"; do
        wait ${PIDS[$task]}
    done
    
    # Display final summary
    display_summary
}

# Run main function
main

# # libero_10
python datasets/tools/regenerate_libero_dataset.py \
    --libero_task_suite libero_10 \
    --libero_raw_data_dir /VLA-Data/scripts/lianqing/data/yifengzhu-hf/LIBERO-datasets/libero_10 \
    --libero_target_dir /VLA-Data/scripts/lianqing/data/libero_xvla/libero_10_no_noops_debug
    