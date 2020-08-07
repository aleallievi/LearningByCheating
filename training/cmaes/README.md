
# Generate Demonstration Data 

### Training command: 

python main.py --task train --algo ppo --env_id HopperBulletEnv-v0

### Sampling command:

python main.py --task sample --algo ppo --env_id HopperBulletEnv-v0 --load_model_path "path to checkpoint"
