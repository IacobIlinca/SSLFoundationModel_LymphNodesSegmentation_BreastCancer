import os
import pandas as pd

from src.VocoLarge.segmentation.multiclass_segmentation.config_multiclass import ConfigMulticlass


def spacing_dict_from_cfg(cfg: ConfigMulticlass):
    """
    Assumes cfg.target_spacing is ordered as (x, y, z).
    Example:
        cfg.target_spacing = (1.25, 1.25, 5.0)
    """
    sx, sy, sz = cfg.target_spacing

    return {
        "x": float(sx),
        "y": float(sy),
        "z": float(sz),
    }


def add_mm_columns(df: pd.DataFrame, spacing_by_axis: dict) -> pd.DataFrame:
    df = df.copy()

    df["spacing_mm"] = df["axis"].map(spacing_by_axis)

    error_columns = [
        "start_error",
        "end_error",
        "extent_error",
        "abs_start_error",
        "abs_end_error",
        "abs_extent_error",
    ]

    for col in error_columns:
        df[f"{col}_mm"] = df[col] * df["spacing_mm"]

    return df


def summarize_compact_mm(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[(df["gt_present"] == True) & (df["pred_present"] == True)].copy()

    summary = (
        valid
        .groupby(["axis", "class_name"])
        .agg(
            valid_cases=("case_id", "count"),
            mean_start_error_mm=("start_error_mm", "mean"),
            mean_end_error_mm=("end_error_mm", "mean"),
            mean_extent_error_mm=("extent_error_mm", "mean"),
            mean_abs_start_error_mm=("abs_start_error_mm", "mean"),
            mean_abs_end_error_mm=("abs_end_error_mm", "mean"),
            mean_abs_extent_error_mm=("abs_extent_error_mm", "mean"),
        )
        .reset_index()
    )

    return summary


def print_summary_mm(df: pd.DataFrame):
    valid = df[(df["gt_present"] == True) & (df["pred_present"] == True)].copy()

    print("\nAxis boundary error summary in millimetres")
    print("-" * 90)

    for axis in ["x", "y", "z"]:
        axis_df = valid[valid["axis"] == axis]

        print(f"\nAXIS: {axis}")
        print("-" * 90)

        for class_name in ["level2", "level3", "level4", "interpect"]:
            cls_df = axis_df[axis_df["class_name"] == class_name]

            if len(cls_df) == 0:
                print(f"{class_name}: no valid cases with both GT and prediction")
                continue

            print(f"\n{class_name}")
            print(f"  valid cases: {len(cls_df)}")
            print(f"  mean start error:      {cls_df['start_error_mm'].mean(): .2f} mm")
            print(f"  mean end error:        {cls_df['end_error_mm'].mean(): .2f} mm")
            print(f"  mean extent error:     {cls_df['extent_error_mm'].mean(): .2f} mm")
            print(f"  mean abs start error:  {cls_df['abs_start_error_mm'].mean(): .2f} mm")
            print(f"  mean abs end error:    {cls_df['abs_end_error_mm'].mean(): .2f} mm")
            print(f"  mean abs extent error: {cls_df['abs_extent_error_mm'].mean(): .2f} mm")

    print("-" * 90)


def main():
    cfg = ConfigMulticlass()

    spacing_by_axis = spacing_dict_from_cfg(cfg)

    input_csv = os.path.join(cfg.save_dir, "axis_boundary_errors.csv")
    output_csv = os.path.join(cfg.save_dir, "axis_boundary_errors_mm.csv")
    summary_csv = os.path.join(cfg.save_dir, "axis_boundary_summary_mm.csv")

    df = pd.read_csv(input_csv)

    df_mm = add_mm_columns(df, spacing_by_axis)

    df_mm.to_csv(output_csv, index=False)
    print(f"Saved mm-level per-case results to: {output_csv}")

    summary = summarize_compact_mm(df_mm)
    summary.to_csv(summary_csv, index=False)
    print(f"Saved compact mm summary to: {summary_csv}")

    print_summary_mm(df_mm)


if __name__ == "__main__":
    main()