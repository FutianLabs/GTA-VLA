cd /VLA-Data/scripts/lingyiran/x-vla-main
source .venv/bin/activate

# --- 运行模板 (取消注释你想测试的任务) ---

# 1. 基础 Carrot 任务
# uv run -m evaluation.simpler.WidowX.client_unified --task widowx_carrot_on_plate --connection_info logs/info.json

# 2. 9宫格随机相机
# uv run -m evaluation.simpler.WidowX.client_unified --task widowx_carrot_on_plate_9grid_random_camera --connection_info logs/info.json

# 3. 新模型抓取测试 (例如测试 10 个 episode)
# uv run -m evaluation.simpler.WidowX.client_unified --task widowx_grasp_new_models_eval --num_episodes 10 --connection_info logs/info.json

# 4. 带干扰物的抓取
# uv run -m evaluation.simpler.WidowX.client_unified --task widowx_grasp_new_models_with_distractors_random_instruction_eval --connection_info logs/info.json --seed 20260108
# uv run -m evaluation.simpler.WidowX.client_unified --task widowx_carrot_on_plate_9grid --connection_info logs/info.json --seed 20260108
# uv run -m evaluation.simpler.WidowX.client_unified --task widowx_carrot_on_plate --connection_info logs/info.json --seed 20260108


# --- 当前执行 ---
# uv run -m evaluation.simpler.WidowX.client_unified --task widowx_stack_cube_9grid --connection_info /VLA-Data/scripts/lingyiran/x-vla-main/logs/info.json --seed 20260108

# #Stack Cube Series
# uv run -m evaluation.simpler.WidowX.client_blocks_unified --task widowx_stack_cube_9grid_nearby_distractors --connection_info /VLA-Data/scripts/lingyiran/x-vla-main/logs/info.json --seed 20260108
# uv run -m evaluation.simpler.WidowX.client_blocks_unified --task widowx_stack_cube_9grid_distractors --connection_info /VLA-Data/scripts/lingyiran/x-vla-main/logs/info.json --seed 20260108
uv run -m evaluation.simpler.WidowX.client_blocks_unified --task widowx_stack_cube_9grid_bigger --connection_info /VLA-Data/scripts/lingyiran/x-vla-main/logs/info.json --seed 20260108
uv run -m evaluation.simpler.WidowX.client_blocks_unified --task widowx_stack_cube_9grid --connection_info /VLA-Data/scripts/lingyiran/x-vla-main/logs/info.json --seed 20260108

