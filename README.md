# MuJoCo Playground GPU Training

GPU 가속을 활용한 MuJoCo Playground 강화학습 환경입니다.

## 📋 필수 요구사항

- Ubuntu 20.04 이상
- NVIDIA GPU (CUDA 12.x 지원)
- Anaconda/Miniconda
- Python 3.11

## 🔧 설치

```bash
# MuJoCo Playground 클론
git clone https://github.com/google-deepmind/mujoco_playground.git

# Conda 환경 생성
conda create -n mjx_playground python=3.11 -y
conda activate mjx_playground

# JAX with CUDA 12 설치
pip install -U "jax[cuda12]"

# 패키지 설치
cd mujoco_playground
pip install -e .
```

## 🚀 사용 방법

```bash
# 기본 학습
python learning/train_jax_ppo.py --env_name HumanoidStand

# 고급 옵션
python learning/train_jax_ppo.py \
    --env_name HumanoidStand \
    --num_timesteps 10000000 \
    --num_envs 2048 \
    --use_tb
```

## 🎮 주요 환경

- **Locomotion**: HumanoidStand, HumanoidWalk, Go1, Unitree
- **Manipulation**: PandaPickCube, PandaReach, PandaStack
- **DM Control**: CartpoleBalance, Reacher, Walker

## 📝 라이선스

MuJoCo Playground는 Apache 2.0 라이선스입니다.

---
