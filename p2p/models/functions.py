from p2p.models.constrained_cv import ConstrainedModelCV
from p2p.models.constrained_elasticnet import ConstrainedElasticNet
from p2p.models.constrained_lasso import ConstrainedLasso
from p2p.models.weighted_constrained_elasticnet import (
    WeightedConstrainedElasticNet,
)
from p2p.models.weighted_constrained_lasso import WeightedConstrainedLasso

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def fit_lasso_model(experiment, config, verbose=True):
    """
    Fit a constrained Lasso model using the custom `models` module.

    Args:
        experiment: Experiment object with `.get_dataframe_by_split()` and `.survey.original_to_binary_map`.
        config (dict): YAML config with keys: 'lasso', 'split_settings', etc.
        verbose (bool): Whether to print progress messages.

    Returns:
        model: Fitted ConstrainedLasso model.
        best_alpha: Best alpha value selected.
        diagnostics: Dictionary with performance metrics and alpha trajectory.
    """
    # --- Load config
    use_proxy_only = config["split_settings"]["use_proxy_only"]
    split_train, split_val = config["split_settings"]["train_val_split"]
    strategy = config["lasso"].get("validation", {}).get("strategy", "cv")
    cv_folds = config["lasso"].get("validation", {}).get("cv_folds", 5)
    alpha_expr = config["lasso"]["alpha_expr"]
    alphas = eval(alpha_expr, {"np": np})
    max_iter = config["lasso"].get("max_iter", 1000)
    zero_threshold = config["lasso"].get("zero_threshold", 1e-5)
    soft_group_sum = config["lasso"].get("soft_group_sum", True)
    group_penalty_strength = config["lasso"].get("group_penalty_strength", 100.0)

    block_weighting = config["lasso"].get("block_weighting", False)

    # --- Prepare data
    df_train = experiment.get_dataframe_by_split(split_train, proxy_only=use_proxy_only)
    df_val = experiment.get_dataframe_by_split(split_val, proxy_only=use_proxy_only)

    target_col = "aggregate"
    feature_cols = [c for c in df_train.columns if c != target_col]

    X_train = df_train[feature_cols].astype(float)
    y_train = df_train[target_col].astype(float).values
    X_val = df_val[feature_cols].astype(float)
    y_val = df_val[target_col].astype(float).values

    # --- Extract constraint groups
    constraint_groups = experiment.survey.original_to_binary_map

    diagnostics = {}


    # ------------------------------------------------------------------
    # Compute question weights if block_weighting is ON
    # Format required by WeightedConstrainedModelMixin:
    #       question_weights = {original_qid: weight}
    # ------------------------------------------------------------------
    if block_weighting:
        if verbose:
            print("[fit_lasso_model] Block weighting enabled.")
        question_weights = {
            original_qid: 1.0 / len(binary_ids)
            for original_qid, binary_ids in constraint_groups.items()
        }
    else:
        question_weights = None

    # ==================================================================
    #                       Cross-Validation Mode
    # ==================================================================
    if strategy == "cv":
        if verbose:
            print(f"[fit_lasso_model] Using {cv_folds}-fold cross-validation to select alpha.")

        # Combine data for CV
        X_all = pd.concat([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])

        # Select model class
        if block_weighting:
            if verbose:
                print("[CV] Using WeightedConstrainedLasso.")
            model_cls = WeightedConstrainedLasso
            extra_kwargs = {"question_weights": question_weights}
        else:
            if verbose:
                print("[CV] Using ConstrainedLasso (unweighted).")
            model_cls = ConstrainedLasso
            extra_kwargs = {}

        model_cv = ConstrainedModelCV(
            model_cls=model_cls,
            alphas=alphas,
            k=cv_folds,
            model_kwargs={
                "constraint_groups": constraint_groups,
                "zero_threshold": zero_threshold,
                "soft_group_sum": soft_group_sum,
                "group_penalty_strength": group_penalty_strength,
                **extra_kwargs,
            },
            verbose=verbose,
        )
        
        model_cv.fit(X_all, y_all)
        model = model_cv.get_best_model()
        best_alpha = model_cv.best_params_[0]

        diagnostics["mean_mse"] = model_cv.diagnostics_["mean_mse"]
        diagnostics["std_mse"] = model_cv.diagnostics_["std_mse"]
        diagnostics["alphas"] = model_cv.diagnostics_["alphas"]


    # ==================================================================
    #                     Hold-Out Validation Mode
    # ==================================================================
    else:
        if verbose:
            print("[fit_lasso_model] Using hold-out validation strategy.")

        train_errors = []
        val_errors = []

        for alpha in alphas:
            if block_weighting:
                model = WeightedConstrainedLasso(
                    alpha=alpha,
                    constraint_groups=constraint_groups,
                    zero_threshold=zero_threshold,
                    soft_group_sum=soft_group_sum,
                    group_penalty_strength=group_penalty_strength,
                    question_weights=question_weights,
                    verbose=verbose,
                )
            else:
                model = ConstrainedLasso(
                    alpha=alpha,
                    constraint_groups=constraint_groups,
                    zero_threshold=zero_threshold,
                    soft_group_sum=soft_group_sum,
                    group_penalty_strength=group_penalty_strength,
                    verbose=verbose,
                )
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_val_pred =  model.predict(X_val)

            train_errors.append(mean_squared_error(y_train, y_train_pred))
            val_errors.append(mean_squared_error(y_val, y_val_pred))

        best_idx = int(np.argmin(val_errors))
        best_alpha = alphas[best_idx]

        if verbose:
            print(f"[fit_lasso_model] Best alpha (Hold-out): {best_alpha:.2e}, MSE = {val_errors[best_idx]:.4f}")

        # Refit on all data using best alpha
        X_all = pd.concat([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])

        if block_weighting:
            model = WeightedConstrainedLasso(
                alpha=best_alpha,
                constraint_groups=constraint_groups,
                zero_threshold=zero_threshold,
                soft_group_sum=soft_group_sum,
                group_penalty_strength=group_penalty_strength,
                question_weights=question_weights,
                verbose=verbose,
            )
        else:
            model = ConstrainedLasso(
                alpha=best_alpha,
                constraint_groups=constraint_groups,
                zero_threshold=zero_threshold,
                soft_group_sum=soft_group_sum,
                group_penalty_strength=group_penalty_strength,
                verbose=verbose,
            )
        
        model.fit(X_all, y_all)

        diagnostics["train_errors"] = train_errors
        diagnostics["val_errors"] = val_errors
        diagnostics["alphas"] = alphas

    return model, best_alpha, diagnostics

def fit_elastic_net_model(experiment, config, verbose=True, random_seed=101):
    """
    Fit a constrained ElasticNet model using the custom `models` module.

    Args:
        experiment: Object with `.get_dataframe_by_split()` and `.survey.original_to_binary_map`.
        config (dict): YAML config with keys: 'elasticnet', 'split_settings', etc.
        verbose (bool): Whether to print status messages.
        random_seed (int): Random seed for reproducibility.

    Returns:
        model: Fitted ConstrainedElasticNet model.
        best_alpha: Selected alpha value.
        best_l1_ratio: Selected l1_ratio value.
        diagnostics: Dictionary with performance metrics and hyperparameter path.
    """
    # --- Load config
    use_proxy_only = config["split_settings"]["use_proxy_only"]
    split_train, split_val = config["split_settings"]["train_val_split"]
    strategy = config["elasticnet"].get("validation", {}).get("strategy", "cv")
    cv_folds = config["elasticnet"].get("validation", {}).get("cv_folds", 5)

    alpha_expr = config["elasticnet"]["alpha_expr"]
    alphas = eval(alpha_expr, {"np": np})
    max_iter = config["elasticnet"].get("max_iter", 1000)

    l1_ratio_expr = config["elasticnet"].get("l1_ratio_expr", "[0.1, 0.5, 0.9]")
    l1_ratios = eval(l1_ratio_expr, {"np": np})

    zero_threshold = config["elasticnet"].get("zero_threshold", 1e-5)
    soft_group_sum = config["elasticnet"].get("soft_group_sum", True)
    group_penalty_strength = config["elasticnet"].get("group_penalty_strength", 100.0)

    block_weighting = config["elasticnet"].get("block_weighting", False)

    # --- Prepare data
    df_train = experiment.get_dataframe_by_split(split_train, proxy_only=use_proxy_only)
    df_val = experiment.get_dataframe_by_split(split_val, proxy_only=use_proxy_only)

    target_col = "aggregate"
    feature_cols = [c for c in df_train.columns if c != target_col]

    X_train = df_train[feature_cols].astype(float)
    y_train = df_train[target_col].astype(float).values
    X_val = df_val[feature_cols].astype(float)
    y_val = df_val[target_col].astype(float).values

    # --- Extract constraint groups
    constraint_groups = experiment.survey.original_to_binary_map

    diagnostics = {}

    if strategy != "cv":
        raise NotImplementedError("Only CV-based selection is currently supported for ElasticNet.")

    if verbose:
        print(f"[fit_elastic_net_model] Using {cv_folds}-fold CV to select alpha and l1_ratio.")

    # Combine full data
    X_all = pd.concat([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    # ------------------------------------------------------------
    # Compute block weights (same structure as Lasso version)
    # Required format for WeightedConstrainedModelMixin:
    #     {original_qid: weight}
    # ------------------------------------------------------------
    if block_weighting:
        if verbose:
            print("[ElasticNet] Block weighting enabled.")

        question_weights = {
            original_qid: 1.0 / len(binary_ids)
            for original_qid, binary_ids in constraint_groups.items()
        }
    else:
        question_weights = None

    # ------------------------------------------------------------
    # Choose backbone model class
    # ------------------------------------------------------------
    if block_weighting:
        if verbose:
            print("[CV] Using WeightedConstrainedElasticNet.")
        model_cls = WeightedConstrainedElasticNet
        extra_kwargs = {"question_weights": question_weights}
    else:
        if verbose:
            print("[CV] Using ConstrainedElasticNet (unweighted).")
        model_cls = ConstrainedElasticNet
        extra_kwargs = {}

    # ------------------------------------------------------------
    # Run cross-validation
    # ------------------------------------------------------------
    model_cv = ConstrainedModelCV(
        model_cls=model_cls,
        alphas=alphas,
        l1_ratios=l1_ratios,
        k=cv_folds,
        model_kwargs={
            "constraint_groups": constraint_groups,
            "zero_threshold": zero_threshold,
            "soft_group_sum": soft_group_sum,
            "group_penalty_strength": group_penalty_strength,
            **extra_kwargs,
        },
        random_state=random_seed,
        verbose=verbose,
    )
    model_cv.fit(X_all, y_all)
    model = model_cv.get_best_model(X_all, y_all)
    best_alpha, best_l1_ratio = model_cv.best_params_

    if verbose:
        print(f"[fit_elastic_net_model] Best alpha: {best_alpha:.2e}, Best l1_ratio: {best_l1_ratio:.2f}")

    diagnostics["mean_mse"] = model_cv.diagnostics_["mean_mse"]
    diagnostics["std_mse"] = model_cv.diagnostics_["std_mse"]
    diagnostics["alphas"] = model_cv.diagnostics_["alphas"]
    diagnostics["l1_ratios"] = model_cv.diagnostics_["l1_ratios"]

    return model, best_alpha, best_l1_ratio, diagnostics
