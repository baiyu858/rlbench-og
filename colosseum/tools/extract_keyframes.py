"""
Extract keyframes from robot-colosseum demonstrations.

Usage:
    python extract_keyframes.py \
        --data_path=/home/hcp/robot-colosseum/data/original \
        --output_path=/home/hcp/robot-colosseum/data/keyframe \
        --tasks=solve_puzzle \
        --stopping_delta=0.1
"""

import os
import pickle
import json
import shutil
from typing import List
import numpy as np

from absl import app
from absl import flags

from rlbench.backend.const import *
from rlbench.demo import Demo

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=''):
        return iterable

try:
    import torch
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Language embeddings will not be generated.")


# ============================================================================
# CLIP Language Encoding
# ============================================================================

def clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(clip_model.dtype)
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)
    x = clip_model.ln_final(x).type(clip_model.dtype)
    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection
    return x, emb


def encode_language_goal(lang_goal, clip_model, device='cuda'):
    lang_goal_tokens = torch.tensor(
        clip.tokenize([lang_goal])[0].numpy(), 
        device=device
    )[None, ...].long()
    _, lang_goal_embs = clip_encode_text(clip_model, lang_goal_tokens)
    lang_goal_embs = lang_goal_embs.detach().float().cpu().numpy()
    return lang_goal_embs[0]


# ============================================================================
# Keypoint Discovery
# ============================================================================

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i != (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
            obs.gripper_open == demo[i - 1].gripper_open and
            demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
            next_is_not_final and gripper_state_no_change)
    return stopped


def keypoint_discovery(demo, stopping_delta: float=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    return episode_keypoints


# ============================================================================

FLAGS = flags.FLAGS

flags.DEFINE_string('data_path', '/home/hcp/robot-colosseum/data/original',
                    'Path to the original demonstration data.')
flags.DEFINE_string('output_path', '/home/hcp/robot-colosseum/data/keyframe',
                    'Path to save keyframe data.')
flags.DEFINE_list('tasks', [],
                  'Tasks to process. If empty, process all tasks.')
flags.DEFINE_float('stopping_delta', 0.1,
                   'Stopping delta parameter for keypoint discovery.')
flags.DEFINE_bool('overwrite', False,
                  'Whether to overwrite existing keyframe data.')
flags.DEFINE_bool('save_keypoint_json', True,
                  'Whether to save keypoint_indices.json.')
flags.DEFINE_bool('encode_language', True,
                  'Whether to encode language descriptions using CLIP.')
flags.DEFINE_string('clip_model', 'RN50',
                   'CLIP model to use.')
flags.DEFINE_string('device', 'cuda',
                   'Device for CLIP encoding.')


def check_and_make(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def copy_image_by_indices(src_folder, dst_folder, keypoint_indices):
    if not os.path.exists(src_folder):
        return
    
    check_and_make(dst_folder)
    
    # Always copy frame 0
    src_file_0 = os.path.join(src_folder, IMAGE_FORMAT % 0)
    dst_file_0 = os.path.join(dst_folder, IMAGE_FORMAT % 0)
    if os.path.exists(src_file_0):
        shutil.copy2(src_file_0, dst_file_0)
    
    # Copy keypoint frames
    for orig_idx in keypoint_indices:
        src_file = os.path.join(src_folder, IMAGE_FORMAT % orig_idx)
        dst_file = os.path.join(dst_folder, IMAGE_FORMAT % orig_idx)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
        else:
            print(f"Warning: Missing file {src_file}")


def process_episode(episode_path, output_episode_path, stopping_delta, variation_desc_path=None, clip_model=None):
    try:
        low_dim_path = os.path.join(episode_path, LOW_DIM_PICKLE)
        if not os.path.exists(low_dim_path):
            print(f"Warning: {low_dim_path} not found, skipping")
            return False
        
        # Load variation number if exists
        variation_number_path = os.path.join(episode_path, 'variation_number.pkl')
        variation_number = None
        if os.path.exists(variation_number_path):
            with open(variation_number_path, 'rb') as f:
                variation_number = pickle.load(f)
        
        # Load random seed if exists
        random_seed_path = os.path.join(episode_path, 'random_seed.pkl')
        random_seed = None
        if os.path.exists(random_seed_path):
            with open(random_seed_path, 'rb') as f:
                random_seed = pickle.load(f)
        
        with open(low_dim_path, 'rb') as f:
            obs_list = pickle.load(f)
        
        # Create Demo object
        demo = Demo(obs_list, variation_number=variation_number, random_seed=random_seed)
        
        # Discover keypoints
        keypoint_indices = keypoint_discovery(demo, stopping_delta=stopping_delta)
        
        # Ensure last frame is included
        last_frame_idx = len(demo) - 1
        if last_frame_idx not in keypoint_indices:
            keypoint_indices.append(last_frame_idx)
        
        keypoint_indices_for_json = [idx for idx in keypoint_indices if idx != 0]
        
        if len(keypoint_indices) == 0:
            print(f"Warning: No keypoints found in {episode_path}")
            return False
        
        print(f"  Found {len(keypoint_indices)} keypoints: {keypoint_indices}")
        
        # Create output directory
        check_and_make(output_episode_path)
        
        # Save Demo object
        with open(os.path.join(output_episode_path, LOW_DIM_PICKLE), 'wb') as f:
            pickle.dump(demo, f)
        
        # Save keypoint indices
        if FLAGS.save_keypoint_json:
            with open(os.path.join(output_episode_path, 'kfs.json'), 'w') as f:
                json.dump(keypoint_indices_for_json, f)
        
        # Copy variation number
        if os.path.exists(variation_number_path):
            shutil.copy2(
                variation_number_path,
                os.path.join(output_episode_path, 'variation_number.pkl')
            )
        
        # Copy and encode variation descriptions (from variation level)
        if variation_desc_path is not None and os.path.exists(variation_desc_path):
            shutil.copy2(
                variation_desc_path,
                os.path.join(output_episode_path, VARIATION_DESCRIPTIONS)
            )
            
            # Encode language description using CLIP if available
            if clip_model is not None and FLAGS.encode_language:
                try:
                    with open(variation_desc_path, 'rb') as f:
                        lang_descriptions = pickle.load(f)
                    
                    if isinstance(lang_descriptions, list) and len(lang_descriptions) > 0:
                        lang_goal = lang_descriptions[0]
                        lang_emb = encode_language_goal(lang_goal, clip_model, device=FLAGS.device)
                        
                        # Save language embedding
                        with open(os.path.join(output_episode_path, 'lang_emb.pkl'), 'wb') as f:
                            pickle.dump(lang_emb, f)
                        
                        print(f"    Encoded language: '{lang_goal}' -> shape {lang_emb.shape}")
                except Exception as e:
                    print(f"    Warning: Failed to encode language: {str(e)}")
        
        # Copy random seed
        if os.path.exists(random_seed_path):
            shutil.copy2(
                random_seed_path,
                os.path.join(output_episode_path, 'random_seed.pkl')
            )
        
        # Copy images for each camera
        camera_types = [
            (LEFT_SHOULDER_RGB_FOLDER, LEFT_SHOULDER_RGB_FOLDER),
            (LEFT_SHOULDER_DEPTH_FOLDER, LEFT_SHOULDER_DEPTH_FOLDER),
            (RIGHT_SHOULDER_RGB_FOLDER, RIGHT_SHOULDER_RGB_FOLDER),
            (RIGHT_SHOULDER_DEPTH_FOLDER, RIGHT_SHOULDER_DEPTH_FOLDER),
            (OVERHEAD_RGB_FOLDER, OVERHEAD_RGB_FOLDER),
            (OVERHEAD_DEPTH_FOLDER, OVERHEAD_DEPTH_FOLDER),
            (WRIST_RGB_FOLDER, WRIST_RGB_FOLDER),
            (WRIST_DEPTH_FOLDER, WRIST_DEPTH_FOLDER),
            (FRONT_RGB_FOLDER, FRONT_RGB_FOLDER),
            (FRONT_DEPTH_FOLDER, FRONT_DEPTH_FOLDER),
        ]
        
        for src_name, dst_name in camera_types:
            src_folder = os.path.join(episode_path, src_name)
            dst_folder = os.path.join(output_episode_path, dst_name)
            copy_image_by_indices(src_folder, dst_folder, keypoint_indices)
        
        return True
        
    except Exception as e:
        print(f"Error processing {episode_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def process_variation(variation_path, output_variation_path, stopping_delta, clip_model=None):
    """Process all episodes in a variation folder."""
    episodes_path = os.path.join(variation_path, 'episodes')
    if not os.path.exists(episodes_path):
        print(f"Warning: {episodes_path} not found")
        return 0
    
    output_episodes_path = os.path.join(output_variation_path, 'episodes')
    
    # Get variation_descriptions.pkl path from variation level
    var_desc_path = os.path.join(variation_path, 'variation_descriptions.pkl')
    
    # Copy variation_descriptions.pkl to output variation folder
    if os.path.exists(var_desc_path):
        check_and_make(output_variation_path)
        shutil.copy2(var_desc_path, os.path.join(output_variation_path, 'variation_descriptions.pkl'))
    
    # Get all episodes
    episode_folders = [
        f for f in os.listdir(episodes_path)
        if f.startswith('episode') and os.path.isdir(os.path.join(episodes_path, f))
    ]
    episode_folders.sort()
    
    success_count = 0
    
    # Pass variation_desc_path to each episode for lang_emb generation
    for episode_folder in episode_folders:
        episode_path = os.path.join(episodes_path, episode_folder)
        output_episode_path = os.path.join(output_episodes_path, episode_folder)
        
        if process_episode(episode_path, output_episode_path, stopping_delta, var_desc_path, clip_model):
            success_count += 1
    
    return success_count


def process_task(task_name, data_path, output_base_path, stopping_delta, clip_model=None):
    """Process all variations for a task."""
    task_input_path = os.path.join(data_path, task_name)
    task_output_path = os.path.join(output_base_path, task_name)
    
    if not os.path.exists(task_input_path):
        print(f"Warning: {task_input_path} does not exist, skipping")
        return 0
    
    # Check if output exists
    if os.path.exists(task_output_path) and not FLAGS.overwrite:
        print(f"Warning: {task_output_path} already exists. Use --overwrite to replace.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return 0
    
    print(f"\nProcessing task: {task_name}")
    print(f"  Input:  {task_input_path}")
    print(f"  Output: {task_output_path}")
    
    # Get all variation folders
    variation_folders = [
        f for f in os.listdir(task_input_path)
        if f.startswith('variation') and os.path.isdir(os.path.join(task_input_path, f))
    ]
    variation_folders.sort()
    
    print(f"  Found {len(variation_folders)} variations")
    
    total_success = 0
    
    for variation_folder in variation_folders:
        print(f"  Processing {variation_folder}...")
        variation_path = os.path.join(task_input_path, variation_folder)
        output_variation_path = os.path.join(task_output_path, variation_folder)
        
        success = process_variation(variation_path, output_variation_path, stopping_delta, clip_model)
        total_success += success
        print(f"    Processed {success} episodes")
    
    print(f"  Total: {total_success} episodes processed")
    return total_success


def main(argv):
    data_path = FLAGS.data_path
    output_path = FLAGS.output_path
    
    if not os.path.exists(data_path):
        raise ValueError(f"Data path {data_path} does not exist!")
    
    # Initialize CLIP
    clip_model = None
    if FLAGS.encode_language and CLIP_AVAILABLE:
        try:
            print(f"Loading CLIP model: {FLAGS.clip_model} on {FLAGS.device}")
            clip_model, _ = clip.load(FLAGS.clip_model, device=FLAGS.device)
            clip_model.eval()
            print("CLIP model loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load CLIP: {str(e)}")
    elif FLAGS.encode_language and not CLIP_AVAILABLE:
        print("CLIP not available. Skipping language encoding.")
    
    # Determine tasks
    if len(FLAGS.tasks) > 0:
        tasks = FLAGS.tasks
    else:
        tasks = [
            d for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        ]
    
    if len(tasks) == 0:
        print("No tasks found!")
        return
    
    print(f"Processing {len(tasks)} task(s): {', '.join(tasks)}")
    print(f"Stopping delta: {FLAGS.stopping_delta}")
    print(f"Output path: {output_path}")
    print(f"Language encoding: {'Enabled' if (FLAGS.encode_language and clip_model) else 'Disabled'}")
    print()
    
    total_success = 0
    
    for task in tasks:
        success = process_task(task, data_path, output_path, FLAGS.stopping_delta, clip_model)
        total_success += success
    
    print("\n" + "="*60)
    print(f"Keyframe extraction complete!")
    print(f"Total episodes processed: {total_success}")
    print("="*60)


if __name__ == '__main__':
    app.run(main)
