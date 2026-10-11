# reward-hacking
hunting the reward hacking 

# ============================================================
# VEL TECH RL ALIGNMENT FINAL PROJECT - OPTION 3
# Hunting the Reward Hack
#
# Studies:
# 1. Different reward-model capacities
# 2. Different amounts of training data
# 3. Best-of-N reward over-optimization
# 4. Gold reward vs proxy reward
# 5. Length-control experiment
# 6. Confidence intervals and independent judge
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

SEED = 42
rng = np.random.default_rng(SEED)

# -----------------------------
# 1. Define the reward functions
# -----------------------------

def gold_reward(x):
    """
    Gold reward: the true objective, held out from model training.
    High quality is good.
    Excessive flattery and verbosity are undesirable.
    """
    quality = x[:, 0]
    flattery = x[:, 1]
    verbosity = x[:, 2]

    return quality - 0.5 * flattery**2 - 0.15 * verbosity


def proxy_reward(x):
    """
    Proxy reward: the imperfect objective used to train reward models.
    It mistakenly rewards flattery and verbosity.
    """
    quality = x[:, 0]
    flattery = x[:, 1]
    verbosity = x[:, 2]

    return (
        quality
        + 0.9 * flattery
        + 0.15 * flattery**2
        + 0.15 * verbosity
        + 0.1 * quality * flattery
    )


def generate_answers(n, rng):
    """Generate synthetic candidate answers."""
    quality = rng.normal(0, 1, n)
    flattery = np.abs(rng.normal(0, 1, n))
    verbosity = np.abs(rng.normal(0, 1, n))

    return np.column_stack([quality, flattery, verbosity])


# --------------------------------
# 2. Train reward models
# --------------------------------

def train_reward_model(n_samples, degree, seed):
    local_rng = np.random.default_rng(seed)

    X_train = generate_answers(n_samples, local_rng)

    # Training labels contain noise, as real preference labels do.
    y_train = proxy_reward(X_train) + local_rng.normal(
        0, 0.35, n_samples
    )

    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        Ridge(alpha=1.0)
    )

    model.fit(X_train, y_train)
    return model


# --------------------------------
# 3. Experimental settings
# --------------------------------

training_sizes = [100, 500, 2000]
model_degrees = [1, 2, 3]
candidate_counts = [1, 2, 4, 8, 16, 32, 64]

N_REPEATS = 250

results = []

print("Starting reward-hacking experiments...\n")


# --------------------------------
# 4. Run the experiments
# --------------------------------

for degree in model_degrees:
    for train_size in training_sizes:

        model = train_reward_model(
            n_samples=train_size,
            degree=degree,
            seed=SEED + degree * 10000 + train_size
        )

        for n_candidates in candidate_counts:

            raw_gold_scores = []
            raw_proxy_scores = []
            controlled_gold_scores = []
            controlled_judge_scores = []
            raw_flattery_scores = []
            raw_verbosity_scores = []

            for _ in range(N_REPEATS):

                # Generate a larger pool to support length control.
                pool_size = max(3 * n_candidates, n_candidates)
                X = generate_answers(pool_size, rng)

                predicted_proxy = model.predict(X)
                gold = gold_reward(X)

                # Independent judge: noisy gold evaluation.
                # It is NOT used to train or select the model.
                judge = gold + rng.normal(0, 0.25, pool_size)

                # Standard best-of-N selection using proxy reward.
                raw_indices = np.argsort(predicted_proxy)[
                    -n_candidates:
                ]
                raw_best = raw_indices[
                    np.argmax(predicted_proxy[raw_indices])
                ]

                raw_gold_scores.append(gold[raw_best])
                raw_proxy_scores.append(predicted_proxy[raw_best])
                raw_flattery_scores.append(X[raw_best, 1])
                raw_verbosity_scores.append(X[raw_best, 2])

                # Length-control intervention:
                # penalize verbosity more strongly during selection.
                controlled_score = (
                    predicted_proxy - 0.75 * X[:, 2]
                )

                controlled_indices = np.argsort(controlled_score)[
                    -n_candidates:
                ]
                controlled_best = controlled_indices[
                    np.argmax(controlled_score[controlled_indices])
                ]

                controlled_gold_scores.append(gold[controlled_best])
                controlled_judge_scores.append(
                    judge[controlled_best]
                )

            results.append({
                "Degree": degree,
                "Training Data": train_size,
                "Candidates": n_candidates,
                "Gold Reward": np.mean(raw_gold_scores),
                "Gold CI": (
                    1.96 * np.std(raw_gold_scores, ddof=1)
                    / np.sqrt(N_REPEATS)
                ),
                "Proxy Reward": np.mean(raw_proxy_scores),
                "Flattery": np.mean(raw_flattery_scores),
                "Verbosity": np.mean(raw_verbosity_scores),
                "Length-Controlled Gold": np.mean(
                    controlled_gold_scores
                ),
                "Independent Judge": np.mean(
                    controlled_judge_scores
                )
            })

df = pd.DataFrame(results)

print("Experiment completed successfully!")
print("\nFirst 10 result rows:")
print(df.head(10).round(3).to_string(index=False))


# --------------------------------
# 5. Plot gold reward vs optimization
# --------------------------------

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, degree in zip(axes, model_degrees):
    subset = df[df["Degree"] == degree]

    for train_size in training_sizes:
        group = subset[
            subset["Training Data"] == train_size
        ].sort_values("Candidates")

        x = group["Candidates"].to_numpy()
        y = group["Gold Reward"].to_numpy()
        ci = group["Gold CI"].to_numpy()

        ax.errorbar(
            x, y, yerr=ci,
            marker="o", capsize=3,
            label=f"Data = {train_size}"
        )

    ax.set_title(f"Reward model degree = {degree}")
    ax.set_xlabel("Number of candidate answers (N)")
    ax.set_ylabel("Mean gold reward")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle("Reward Hacking: Gold Reward vs Best-of-N")
plt.tight_layout()
plt.show()


# --------------------------------
# 6. Proxy reward vs gold reward
# --------------------------------

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, degree in zip(axes, model_degrees):
    subset = df[
        (df["Degree"] == degree)
        & (df["Training Data"] == 500)
    ].sort_values("Candidates")

    ax.plot(
        subset["Candidates"],
        subset["Proxy Reward"],
        marker="o",
        label="Proxy reward"
    )

    ax.plot(
        subset["Candidates"],
        subset["Gold Reward"],
        marker="s",
        label="Gold reward"
    )

    ax.set_title(f"Model degree {degree}, 500 samples")
    ax.set_xlabel("Number of candidates")
    ax.set_ylabel("Mean reward")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle("Proxy Optimization vs True Quality")
plt.tight_layout()
plt.show()


# --------------------------------
# 7. Diagnose reward hacking
# --------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

diagnostic = df[
    (df["Degree"] == 2)
    & (df["Training Data"] == 500)
].sort_values("Candidates")

axes[0].plot(
    diagnostic["Candidates"],
    diagnostic["Flattery"],
    marker="o"
)
axes[0].set_title("Flattery selected by the proxy")
axes[0].set_xlabel("Number of candidates")
axes[0].set_ylabel("Mean flattery")
axes[0].set_xscale("log", base=2)
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    diagnostic["Candidates"],
    diagnostic["Verbosity"],
    marker="o"
)
axes[1].set_title("Verbosity selected by the proxy")
axes[1].set_xlabel("Number of candidates")
axes[1].set_ylabel("Mean verbosity")
axes[1].set_xscale("log", base=2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# --------------------------------
# 8. Evaluate the length control
# --------------------------------

comparison = df[
    (df["Degree"] == 2)
    & (df["Training Data"] == 500)
].sort_values("Candidates")

plt.figure(figsize=(8, 5))

plt.plot(
    comparison["Candidates"],
    comparison["Gold Reward"],
    marker="o",
    label="Normal proxy optimization"
)

plt.plot(
    comparison["Candidates"],
    comparison["Length-Controlled Gold"],
    marker="s",
    label="With verbosity penalty"
)

plt.xlabel("Number of candidates")
plt.ylabel("Mean gold reward")
plt.title("Length-Control Intervention")
plt.xscale("log", base=2)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# --------------------------------
# 9. Independent judge evaluation
# --------------------------------

print("\nIndependent judge and length-control results:")
print(
    comparison[
        [
            "Candidates",
            "Gold Reward",
            "Length-Controlled Gold",
            "Independent Judge"
        ]
    ].round(3).to_string(index=False)
)


# --------------------------------
# 10. Find the best observed N
# --------------------------------

print("\nBest observed candidate count for each configuration:")

for degree in model_degrees:
    for train_size in training_sizes:
        group = df[
            (df["Degree"] == degree)
            & (df["Training Data"] == train_size)
        ]

        best_row = group.loc[group["Gold Reward"].idxmax()]

        print(
            f"Degree={degree}, data={train_size}: "
            f"best observed N={int(best_row['Candidates'])}, "
            f"gold reward={best_row['Gold Reward']:.3f}"
        )


# --------------------------------
# 11. Save results for the report
# --------------------------------

df.to_csv("option3_reward_hacking_results.csv", index=False)

print("\nCSV saved: option3_reward_hacking_results.csv")
print("All experiments and plots finished.")
