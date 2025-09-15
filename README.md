# Personalized Targeting Action Bank Generation

This project develops an algorithm that generates optimal marketing action banks from historical observation data to maximize user engagement in personalized targeting campaigns. The system includes a comprehensive simulation environment to validate algorithm performance before real-world deployment.

## Core Problem

Given historical data `D = {(x_i, a_i, y_i)}` where:
- `x_i` = user characteristics vector
- `a_i` = marketing action (email subject, content) from current action bank 
- `y_i ∈ {0,1}` = user response (click/open)

**Goal**: Generate new action bank `A^t = {a^t_1, ..., a^t_K}` that maximizes expected reward when deployed by the targeting strategy.

## Repository Structure

```
PersonalizedTargeting/
├── algorithm/                         # Core optimization algorithm
│   ├── optimization_algorithm.py     # Action bank generation from observations
│   └── ground_truth_evaluator.py     # Ground truth evaluation and metrics
│
├── simulation/                        # Company simulation environment
│   ├── user_generator.py             # User population with 8-dimensional features
│   ├── action_embedder.py            # OpenAI action creation and embedding
│   ├── ground_truth.py               # Ground truth preference models
│   └── company_simulator.py          # Simulation orchestrator
│
├── src/                              # Shared components
│   ├── data/                         # User and Action data structures
│   │   └── entities.py               # Core data classes
│   ├── models/                       # User preference models (reward models)
│   │   ├── base_user_preference_model.py      # Interface for all models
│   │   ├── neural_user_preference_model.py    # Neural network model
│   │   ├── bayesian_neural_user_preference_model.py  # Bayesian neural with uncertainty
│   │   ├── lightgbm_user_preference_model.py  # LightGBM model
│   │   ├── gaussian_process_user_preference_model.py  # GP model
│   │   └── linear_user_preference_model.py    # Linear/logistic model
│   ├── selection/                    # Action generation and selection
│   │   ├── action_generator.py       # LLM-based action generation
│   │   └── action_selector.py        # Greedy selection algorithm
│   └── strategies/                   # Company targeting strategies
│       ├── base_strategy.py          # Strategy interface
│       └── linucb_strategy.py        # LinUCB contextual bandit
│
├── results/                          # Generated results (auto-created)
│   ├── simulation_*/                 # Full simulation experiment results
│   │   ├── initialization/           # Configuration and setup
│   │   └── iteration_X/              # Per-iteration data and results
│   │       ├── users/                # Fresh user generation per iteration
│   │       ├── action_bank/          # Current marketing actions
│   │       ├── observations/         # Company targeting observations
│   │       └── new_action_bank/      # Algorithm-generated actions
│   └── simulation_from_data_*/       # Results from run_simulation_from_data.py
│       ├── run_from_data_config.json # Source data dir and algorithm config
│       ├── initialization/           # Copied from source data directory
│       └── iteration_X/              # Algorithm results on pre-generated data
    
│
├── run_full_simulation.py            # Main entry point
├── generate_simulation_data.py       # Generate reproducible simulation data
├── run_simulation_from_data.py       # Run algorithms on pre-generated data
├── test_preference_model.py          # User preference model testing
└── requirements.txt                  # Python dependencies
```

## Architecture Overview

The system follows a clean separation between simulation and algorithm components:

```
Observation Data (CSV) → Reward Model Training → Action Generation → Greedy Selection → New Action Bank
                ↑                                                                              ↓
        Company Simulation ←────────────────────────────────────────────────────────────────┘
        (LinUCB Strategy + Ground Truth Models)
```

### Key Components

#### 1. Algorithm (`algorithm/optimization_algorithm.py`)
The core algorithm that learns from observation data to generate new action banks:
- **Load Historical Data**: Process observation records from targeting campaigns
- **Train Reward Model**: Learn to predict `P(y=1|x,a)` from user-action features
- **Generate Action Pool**: Create large candidate set (~2000 actions) using LLM
- **Greedy Selection**: Pick top-K actions that maximize expected reward for user distribution

#### 2. Company Simulation (`simulation/`)
Simulates realistic company behavior for validation:
- **User Generation**: Creates diverse user populations with meaningful demographic features
- **Targeting Strategy**: LinUCB contextual bandit for action selection
- **Ground Truth Models**: Mixture of Experts or Gaussian Mixture Models for realistic preferences

#### 3. Reward Models (`src/models/`)
Various approaches to learn user preferences:
- **Neural Model**: Feedforward network with user-action feature engineering
- **Bayesian Neural Model**: MC Dropout-based neural network that provides both predictions and uncertainty estimates
- **LightGBM Model**: Gradient boosting for tabular data
- **Gaussian Process**: Probabilistic model with uncertainty quantification
- **Linear Model**: Baseline logistic regression
- **FT-Transformer Model**: Transformer-based encoder for tabular user features with fusion to action embeddings

#### 4. Targeting Strategies (`src/strategies/`)
Company strategy implementations:
- **LinUCB Strategy**: Linear upper confidence bound with PCA optimization
- **Base Strategy Interface**: Clean abstraction for different targeting approaches

## Quick Start

### Basic Usage

```bash
# Simple test run
python run_full_simulation.py --iterations 3 --users 100 --initial_actions 10

# Run with same action bank for all stages
python run_full_simulation.py \
    --iterations 4 \
    --users 10000 \
    --initial_actions 30 \
    --company_strategy linucb \
    --reward_model_type lightgbm \
    --action_bank_size 0

# Production configuration
python run_full_simulation.py \
    --iterations 4 \
    --users 10000 \
    --initial_actions 30 \
    --action_bank_size 10 \
    --company_strategy linucb \
    --reward_model_type bayesian_neural \
    --bnn_mc_samples 30 \
    --use_pca \
    --pca_components 128
```

## Configuration Options

### Core Parameters
- `--iterations`: Number of optimization iterations (default: 5)
- `--users`: User population size per iteration (default: 1000)
- `--initial_actions`: Starting action bank size (default: 30)
- `--results_dir`: Output directory (default: 'results')

### Reward Models
- `--reward_model_type`: Choose from 'neural', 'lightgbm', 'gaussian_process', 'bayesian_neural', 'ft_transformer'
- For LightGBM: `--lgb_n_estimators`, `--lgb_learning_rate`, `--lgb_num_leaves`
- For Bayesian Neural: `--bnn_mc_samples` (MC Dropout samples for uncertainty, default: 30)
  
FT-Transformer notes:
- Uses an FT-Transformer encoder for tabular user features and a lightweight MLP for action embeddings.
- Supports `fusion_type` of `'concat'` or `'attention'` (set via algorithm config if desired).

### Company Strategy
- `--company_strategy`: Choose from 'linucb', 'bootstrapped_dqn', 'legacy'
- For LinUCB: `--alpha` (exploration parameter, default: 1.0)
- PCA optimization: `--use_pca`, `--pca_components`

### Ground Truth Models
- `--ground_truth_type`: Choose from 'mixture_of_experts', 'gmm'
- For GMM: `--gmm_components`, `--density_scale_factor`, `--min_utility`, `--max_utility`

### Algorithm Tuning
- `--diversity_weight`: Penalty for similar actions (default: 0.15)
- `--action_pool_size`: Candidate actions to generate (default: 2000)
- `--action_bank_size`: Final action bank size per iteration (default: 20)
- `--use_uncertainty`: Enable uncertainty-aware action selection (for Bayesian models)
- `--uncertainty_weight`: Weight for uncertainty in action selection (default: 0.1)

## Reproducible Data Workflow

For fair algorithm comparisons, generate simulation data once and reuse it across multiple runs.

### 1. Generate Simulation Data

Create reproducible datasets with fixed user populations and initial action banks:

```bash
# Basic data generation
python generate_simulation_data.py \
  --users 10000 \
  --iterations 4 \
  --initial_actions 30

# Advanced data generation with custom ground truth
python generate_simulation_data.py \
  --users 1000 \
  --iterations 5 \
  --initial_actions 25 \
  --openai_api_key your_api_key
```

**Output Structure:**
```
data/simulation_data/<unique_id>/
├── data_config.json                 # Generation parameters
├── initialization/
│   └── action_bank/action_bank.json # Initial marketing actions
└── iteration_X/
    └── users/users.json             # User populations per iteration
```

### 2. Run Algorithms on Generated Data

Execute different algorithms on the same dataset for fair comparison:

```bash
# Baseline (same action bank across all stages)
python run_simulation_from_data.py \
  --data_dir data/simulation_data/<unique_id> \
  --company_strategy linucb \
  --reward_model_type lightgbm \
  --use_pca \
  --pca_components 128 \
  --action_bank_size 0

# Test LightGBM approach
python run_simulation_from_data.py \
  --data_dir data/simulation_data/<unique_id> \
  --company_strategy linucb \
  --reward_model_type lightgbm \
  --use_pca \
  --pca_components 128 \
  --action_bank_size 5
```

## Understanding Results

After running a simulation, results are stored in timestamped directories:

```
results/simulation_YYYYMMDD_HHMMSS/
├── simulation_config.json          # Complete configuration used
├── complete_simulation_results.json # Full results summary
├── initialization/                  # Initial setup data
│   └── initialization_summary.json
└── iteration_X/                     # Per-iteration results
    ├── observations.csv             # Training data generated
    ├── algorithm_results.json       # Algorithm performance metrics
    ├── ground_truth_evaluation.json # True preference analysis
    └── new_action_bank.json         # Generated marketing actions
```

### Key Metrics
- **Company Reward**: Average user engagement per iteration
- **Algorithm Expected Value**: Predicted performance of new action bank
- **Ground Truth Correlation**: How well the reward model predicts true preferences
- **Learning Progress**: Reward improvement over iterations
- **Uncertainty Analysis**: For Bayesian models, tracks prediction confidence and exploration vs exploitation
- **Action Bank Evolution**: Sequential action ID tracking and LinUCB knowledge preservation

## Development and Testing

### Test Individual Components

```bash
# Test all reward models
python test_preference_model.py --model_type compare_all

# Test specific model
python test_preference_model.py --model_type neural --users 500
```

### Component Architecture

The system uses clean interfaces for extensibility:

```python
# Add new reward model
class CustomUserPreferenceModel(BaseUserPreferenceModel):
    def fit(self, observations: List[CompanyObservation]):
        # Implementation here
        pass
    
    def predict_probabilities(self, users: List[User], actions: List[Action]):
        # Return probability matrix
        pass

# Add new targeting strategy  
class CustomStrategy(BaseCompanyStrategy):
    def select_action(self, user: MeaningfulUser, action_bank: List[EmbeddedAction]):
        # Implementation here
        pass
```

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## API Keys

For optimal performance, set up OpenAI API key:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Without an API key, the system uses fallback random embeddings for action generation.

This framework provides a complete testing environment for validating action bank generation algorithms before deployment in real marketing campaigns.
